#!/usr/bin/env python3
"""
SNR-binned SBC aggregation for ringdown inference calibration.

Per user's requirements:
- Bin runs by achieved SNR: <5, 5-10, 10-20, 20-40, >40
- Report per bin: PIT mean/std, 90% CI coverage, KS test for uniformity
- This is the PRIMARY calibration diagnostic (unconditional is secondary)

Usage:
    python aggregate_sbc_snr_binned.py --outdir out_sbc_v2 --n-runs 200
"""
import argparse
import json
import os
import numpy as np
from scipy import stats


# SNR bin edges per user spec
SNR_BINS = [
    (0, 5, "<5"),
    (5, 10, "5-10"),
    (10, 20, "10-20"),
    (20, 40, "20-40"),
    (40, np.inf, ">40"),
]


def load_sbc_runs(outdir, n_runs):
    """Load all SBC run summaries."""
    runs = []
    for run_id in range(n_runs):
        summary_path = os.path.join(outdir, f"run_{run_id:04d}", "sbc_summary.json")
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as fp:
            summary = json.load(fp)
            summary["run_id"] = run_id
            runs.append(summary)
    return runs


def bin_runs_by_snr(runs):
    """Bin runs by injected SNR."""
    binned = {label: [] for _, _, label in SNR_BINS}

    for run in runs:
        snr = run["injected"]["snr"]
        for lo, hi, label in SNR_BINS:
            if lo <= snr < hi:
                binned[label].append(run)
                break

    return binned


def compute_pit_stats(pit_values):
    """Compute PIT statistics and KS test for uniformity."""
    if len(pit_values) == 0:
        return None

    arr = np.array(pit_values)

    # KS test against Uniform(0, 1)
    ks_stat, ks_pvalue = stats.kstest(arr, 'uniform')

    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "expected_mean": 0.5,
        "expected_std": 1.0 / np.sqrt(12),  # ~0.289
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def compute_coverage(runs, ci_level=0.9):
    """Compute coverage fraction at given CI level."""
    if len(runs) == 0:
        return None

    coverage = {"Mf": 0, "chi": 0, "f0": 0}
    for run in runs:
        for param in coverage:
            if run["coverage"][param].get(str(ci_level), False):
                coverage[param] += 1

    n = len(runs)
    return {param: count / n for param, count in coverage.items()}


def analyze_snr_bin(runs, label):
    """Analyze a single SNR bin."""
    if len(runs) == 0:
        return None

    # Collect PITs
    pit_Mf = [r["pit"]["Mf"] for r in runs]
    pit_chi = [r["pit"]["chi"] for r in runs]
    pit_f0 = [r["pit"]["f0"] for r in runs]

    # SNR distribution within bin
    snrs = [r["injected"]["snr"] for r in runs]

    result = {
        "n_runs": len(runs),
        "snr_range": {
            "min": float(np.min(snrs)),
            "max": float(np.max(snrs)),
            "mean": float(np.mean(snrs)),
            "median": float(np.median(snrs)),
        },
        "pit": {
            "Mf": compute_pit_stats(pit_Mf),
            "chi": compute_pit_stats(pit_chi),
            "f0": compute_pit_stats(pit_f0),
        },
        "coverage_90": compute_coverage(runs, ci_level=0.9),
        "coverage_50": compute_coverage(runs, ci_level=0.5),
    }

    return result


def print_calibration_report(binned_results, total_runs):
    """Print formatted calibration report."""
    print()
    print("=" * 70)
    print("SNR-BINNED SBC CALIBRATION REPORT")
    print("=" * 70)
    print(f"Total runs loaded: {total_runs}")
    print()

    for _, _, label in SNR_BINS:
        result = binned_results.get(label)
        if result is None or result["n_runs"] == 0:
            print(f"SNR {label}: No runs")
            print()
            continue

        print(f"SNR {label}: {result['n_runs']} runs")
        print(f"  SNR range: {result['snr_range']['min']:.1f} - {result['snr_range']['max']:.1f}")
        print()

        # PIT stats
        print("  PIT Statistics (expected: mean=0.50, std=0.29):")
        for param in ["Mf", "chi", "f0"]:
            pit = result["pit"][param]
            if pit is None:
                continue
            ks_status = "PASS" if pit["ks_pvalue"] > 0.05 else "FAIL"
            print(f"    {param:5s}: mean={pit['mean']:.3f}, std={pit['std']:.3f}, "
                  f"KS p={pit['ks_pvalue']:.3f} [{ks_status}]")
        print()

        # Coverage
        cov90 = result["coverage_90"]
        cov50 = result["coverage_50"]
        print("  Coverage (expected = CI level):")
        for param in ["Mf", "chi", "f0"]:
            c90 = cov90[param] if cov90 else None
            c50 = cov50[param] if cov50 else None
            c90_str = f"{c90:.2f}" if c90 is not None else "N/A"
            c50_str = f"{c50:.2f}" if c50 is not None else "N/A"
            print(f"    {param:5s}: 50%={c50_str} (exp 0.50), 90%={c90_str} (exp 0.90)")
        print()
        print("-" * 70)
        print()


def compute_unconditional_sbc(runs):
    """Compute unconditional SBC statistics (all runs pooled)."""
    if len(runs) == 0:
        return None

    pit_Mf = [r["pit"]["Mf"] for r in runs]
    pit_chi = [r["pit"]["chi"] for r in runs]
    pit_f0 = [r["pit"]["f0"] for r in runs]

    return {
        "n_runs": len(runs),
        "pit": {
            "Mf": compute_pit_stats(pit_Mf),
            "chi": compute_pit_stats(pit_chi),
            "f0": compute_pit_stats(pit_f0),
        },
        "coverage_90": compute_coverage(runs, ci_level=0.9),
        "coverage_50": compute_coverage(runs, ci_level=0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out_sbc_v2",
                        help="SBC output directory")
    parser.add_argument("--n-runs", type=int, default=200,
                        help="Number of SBC runs to check")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: outdir/sbc_snr_binned.json)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.outdir, "sbc_snr_binned.json")

    print(f"Loading SBC runs from {args.outdir}...")
    runs = load_sbc_runs(args.outdir, args.n_runs)
    print(f"Loaded {len(runs)} runs")

    if len(runs) == 0:
        print("No runs found!")
        return

    # Bin by SNR
    binned = bin_runs_by_snr(runs)

    # Analyze each bin
    binned_results = {}
    for _, _, label in SNR_BINS:
        binned_results[label] = analyze_snr_bin(binned[label], label)

    # Print report
    print_calibration_report(binned_results, len(runs))

    # Unconditional stats (secondary)
    print("UNCONDITIONAL SBC (secondary - dominated by low-SNR draws):")
    print("-" * 70)
    unconditional = compute_unconditional_sbc(runs)
    if unconditional:
        for param in ["Mf", "chi", "f0"]:
            pit = unconditional["pit"][param]
            print(f"  {param:5s}: mean={pit['mean']:.3f}, std={pit['std']:.3f}, "
                  f"KS p={pit['ks_pvalue']:.3f}")
        print()
        cov90 = unconditional["coverage_90"]
        print(f"  90% coverage: Mf={cov90['Mf']:.2f}, chi={cov90['chi']:.2f}, f0={cov90['f0']:.2f}")
    print()

    # Save results
    output_data = {
        "snr_binned": binned_results,
        "unconditional": unconditional,
        "snr_bins": [(lo, hi, label) for lo, hi, label in SNR_BINS],
        "total_runs": len(runs),
    }

    with open(args.output, "w") as fp:
        json.dump(output_data, fp, indent=2, default=str)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
