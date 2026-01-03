#!/usr/bin/env python3
"""
Collect and analyze null test results with evidence-quality tracking.

Computes per-run:
- gps, t_start, duration, fmin, fmax, df
- ln_BF, logZ_H0, logZ_H1, logZerr_H0, logZerr_H1
- logLmax (if available from sampler)
- sigma_lnBF = sqrt(logZerr_H0^2 + logZerr_H1^2)
- inconclusive flag: |ln_BF| < 3 * sigma_lnBF
"""

import argparse
import os
import json
import glob
import numpy as np


def load_bilby_result(result_dir, label):
    """Load bilby result JSON and extract evidence info."""
    # Bilby saves results as {label}_result.json
    json_path = os.path.join(result_dir, f"{label}_result.json")
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    result = {
        "log_evidence": data.get("log_evidence"),
        "log_evidence_err": data.get("log_evidence_err"),
    }

    # Try to get logLmax from nested sampler results
    if "nested_samples" in data:
        try:
            # Maximum log-likelihood from samples
            logl_samples = [s.get("log_likelihood", -np.inf) if isinstance(s, dict) else -np.inf
                          for s in data["nested_samples"]]
            if logl_samples:
                result["logLmax"] = max(logl_samples)
        except (TypeError, AttributeError):
            pass

    # Alternative: check for maximum_log_likelihood in result
    if "maximum_log_likelihood" in data:
        result["logLmax"] = data["maximum_log_likelihood"]

    return result


def collect_run_results(outdir):
    """Collect results from a null test run directory."""
    results = {}

    # Check for summary file first
    summary_path = os.path.join(outdir, "mscf_train_null_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            # File may be truncated or incomplete
            print(f"WARNING: Could not parse {summary_path} (truncated?)")
            # Try analysis_settings.json as fallback
            alt_path = os.path.join(outdir, "analysis_settings.json")
            if os.path.exists(alt_path):
                try:
                    with open(alt_path) as f:
                        summary = json.load(f).get("results_summary", {})
                        if summary:
                            summary = {"aligned": summary.get("aligned", {}),
                                      "timeslid": summary.get("timeslid", {})}
                except json.JSONDecodeError:
                    return results
            else:
                return results

        # Extract aligned results
        if "aligned" in summary:
            al = summary["aligned"]
            sigma_lnBF = np.sqrt(al.get("logZ_H0_err", 0)**2 + al.get("logZ_H1_err", 0)**2)
            ln_BF = al.get("ln_BF", 0)
            inconclusive = abs(ln_BF) < 3 * sigma_lnBF

            results["aligned"] = {
                "ln_BF": float(ln_BF),
                "logZ_H0": al.get("logZ_H0"),
                "logZ_H1": al.get("logZ_H1"),
                "logZerr_H0": al.get("logZ_H0_err"),
                "logZerr_H1": al.get("logZ_H1_err"),
                "sigma_lnBF": float(sigma_lnBF),
                "inconclusive": bool(inconclusive),
            }

        # Extract time-slid results
        if "timeslid" in summary:
            ts = summary["timeslid"]
            sigma_lnBF = np.sqrt(ts.get("logZ_H0_err", 0)**2 + ts.get("logZ_H1_err", 0)**2)
            ln_BF = ts.get("ln_BF", 0)
            inconclusive = abs(ln_BF) < 3 * sigma_lnBF

            results["timeslid"] = {
                "time_slide_s": summary.get("time_slide_L1_s"),
                "ln_BF": float(ln_BF),
                "logZ_H0": ts.get("logZ_H0"),
                "logZ_H1": ts.get("logZ_H1"),
                "logZerr_H0": ts.get("logZ_H0_err"),
                "logZerr_H1": ts.get("logZ_H1_err"),
                "sigma_lnBF": float(sigma_lnBF),
                "inconclusive": bool(inconclusive),
            }

        # Analysis settings
        results["settings"] = {
            "fmin": summary.get("fmin"),
            "fmax": summary.get("fmax"),
            "nlive": summary.get("nlive"),
            "seed": summary.get("seed"),
            "data_dir": summary.get("data"),
        }

    # Try to get logLmax from individual bilby results
    for label in ["h0_aligned", "h1_mscf_train_aligned", "h0_timeslid", "h1_mscf_train_timeslid"]:
        bilby_result = load_bilby_result(outdir, label)
        if bilby_result and "logLmax" in bilby_result:
            if "aligned" in label:
                key = "aligned"
                model = "H0" if label.startswith("h0") else "H1"
            else:
                key = "timeslid"
                model = "H0" if label.startswith("h0") else "H1"

            if key in results:
                results[key][f"logLmax_{model}"] = bilby_result["logLmax"]

    return results


def format_result_line(name, data):
    """Format a single result for display."""
    ln_BF = data.get("ln_BF", np.nan)
    sigma = data.get("sigma_lnBF", np.nan)
    inconclusive = data.get("inconclusive", False)

    flag = " [INCONCLUSIVE]" if inconclusive else ""

    return f"  {name}: ln_BF = {ln_BF:+.2f} ± {sigma:.2f}{flag}"


def main():
    parser = argparse.ArgumentParser(description="Collect null test results with evidence quality")
    parser.add_argument("--outdir", default="out_mscf_train_null_test",
                       help="Output directory with results")
    parser.add_argument("--scan-subdirs", action="store_true",
                       help="Also scan subdirectories for additional runs")
    parser.add_argument("--output-json", default=None,
                       help="Save collected results to JSON file")
    args = parser.parse_args()

    all_results = {}

    # Collect from main directory
    main_results = collect_run_results(args.outdir)
    if main_results:
        all_results["main"] = main_results

    # Scan subdirectories if requested
    if args.scan_subdirs:
        for subdir in glob.glob(os.path.join(args.outdir, "*")):
            if os.path.isdir(subdir):
                sub_results = collect_run_results(subdir)
                if sub_results:
                    all_results[os.path.basename(subdir)] = sub_results

    # Print summary
    print("=" * 70)
    print("NULL TEST RESULTS - EVIDENCE QUALITY ANALYSIS")
    print("=" * 70)
    print()
    print("Criterion: |ln_BF| < 3*sigma_lnBF => INCONCLUSIVE")
    print()

    n_total = 0
    n_inconclusive = 0
    ln_BF_values = {"aligned": [], "timeslid": []}

    for run_name, run_data in sorted(all_results.items()):
        print(f"Run: {run_name}")
        if "settings" in run_data:
            s = run_data["settings"]
            print(f"  Settings: fmin={s.get('fmin')}, fmax={s.get('fmax')}, nlive={s.get('nlive')}")

        for key in ["aligned", "timeslid"]:
            if key in run_data:
                data = run_data[key]
                print(format_result_line(key, data))
                n_total += 1
                if data.get("inconclusive"):
                    n_inconclusive += 1
                if data.get("ln_BF") is not None:
                    ln_BF_values[key].append(data["ln_BF"])
        print()

    # Summary statistics
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total runs analyzed: {n_total}")
    print(f"Inconclusive (|ln_BF| < 3σ): {n_inconclusive}")

    for key in ["aligned", "timeslid"]:
        vals = ln_BF_values[key]
        if vals:
            print(f"\n{key.upper()} ln_BF distribution:")
            print(f"  N = {len(vals)}")
            print(f"  Mean = {np.mean(vals):+.2f}")
            print(f"  Std  = {np.std(vals):.2f}")
            print(f"  Min  = {np.min(vals):+.2f}")
            print(f"  Max  = {np.max(vals):+.2f}")

    # Save to JSON if requested
    if args.output_json:
        output = {
            "runs": all_results,
            "summary": {
                "n_total": n_total,
                "n_inconclusive": n_inconclusive,
                "aligned_ln_BF_values": ln_BF_values["aligned"],
                "timeslid_ln_BF_values": ln_BF_values["timeslid"],
            }
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
