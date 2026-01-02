#!/usr/bin/env python3
"""
GW150914 ringdown echo analysis pipeline.

Runs the full analysis workflow:
  2A: H0 ringdown-only sanity check
  2B: H1 vs H0 Bayes factor on same segment
  2C: Robustness sweep (t_start grid, duration, band)

Usage:
  python scripts/run_gw150914_analysis.py --mode sanity
  python scripts/run_gw150914_analysis.py --mode bf
  python scripts/run_gw150914_analysis.py --mode sweep
  python scripts/run_gw150914_analysis.py --mode all
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime

# GW150914 parameters
GPS_MERGER = 1126259462.4
IFOS = "H1,L1"

# Default analysis settings
DEFAULT_T_START_MS = 3.0  # 3 ms after merger
DEFAULT_DURATION_MS = 100.0  # 100 ms segment
DEFAULT_FMIN = 150.0
DEFAULT_FMAX = 400.0
DEFAULT_NLIVE = 500
DEFAULT_SEED = 42

# Sweep grid
T_START_GRID_MS = [1, 2, 3, 4, 5, 6]
DURATION_GRID_MS = [100, 200]
# Frequency bands: [default, alternative]
FREQ_BANDS = [
    (150, 400, "default"),
    (130, 450, "wider"),
]


def ms_label(x):
    """Format ms value for paths/labels (0.1 ms resolution)."""
    return f"{x:.1f}".replace(".", "p")


def fmt2(x):
    """Safely format float to 2 decimal places, return N/A if None."""
    return "N/A" if x is None else f"{float(x):.2f}"


def get_data_dir(base_outdir, t_start_ms, duration_ms):
    """Consistent data directory naming including duration."""
    return os.path.join(base_outdir, "data", f"t{ms_label(t_start_ms)}ms_d{ms_label(duration_ms)}ms")


def get_event_name(t_start_ms, duration_ms):
    """Event name for this configuration."""
    return f"GW150914_t{ms_label(t_start_ms)}ms_d{ms_label(duration_ms)}ms"


def fetch_data(base_outdir, t_start_ms, duration_ms, force=False):
    """Fetch ringdown-gated data for given configuration."""
    data_dir = get_data_dir(base_outdir, t_start_ms, duration_ms)
    event = get_event_name(t_start_ms, duration_ms)
    sentinel = os.path.join(data_dir, "FETCH_DONE.json")

    if os.path.exists(sentinel) and not force:
        print(f"  Data already fetched: {data_dir}")
        return True

    os.makedirs(data_dir, exist_ok=True)

    t_start_s = t_start_ms / 1000.0
    duration_s = duration_ms / 1000.0

    cmd = [
        sys.executable,
        "scripts/fetch_ringdown_gated.py",
        "--event", event,
        "--gps", str(GPS_MERGER),
        "--t-start", str(t_start_s),
        "--gate-duration", str(duration_s),
        "--ifos", IFOS,
        "--outdir", data_dir,
    ]

    print(f"  Fetching: {' '.join(cmd)}")

    # Write logs to file
    log_path = os.path.join(data_dir, "fetch.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print(f"  FETCH FAILED (see {log_path})")
        return False

    # Write our own sentinel file
    sentinel_data = {
        "event": event,
        "t_start_ms": t_start_ms,
        "duration_ms": duration_ms,
        "gps_merger": GPS_MERGER,
        "timestamp": datetime.now().isoformat(),
    }
    with open(sentinel, "w") as f:
        json.dump(sentinel_data, f, indent=2)

    print(f"  Fetch complete: {data_dir}")
    return True


def run_bf_analysis(base_outdir, t_start_ms, duration_ms, fmin, fmax, nlive, seed, result_subdir):
    """Run full H0 vs H1 Bayes factor analysis."""
    data_dir = get_data_dir(base_outdir, t_start_ms, duration_ms)
    event = get_event_name(t_start_ms, duration_ms)
    result_dir = os.path.join(base_outdir, result_subdir)

    os.makedirs(result_dir, exist_ok=True)

    t_start_s = t_start_ms / 1000.0

    # run_bayes_factor_gated.py is at repo root (not in scripts/)
    cmd = [
        sys.executable,
        "run_bayes_factor_gated.py",
        "--event", event,
        "--gps", str(GPS_MERGER),
        "--t-start", str(t_start_s),
        "--ifos", IFOS,
        "--outdir", data_dir,
        "--resultdir", result_dir,
        "--fmin", str(fmin),
        "--fmax", str(fmax),
        "--nlive", str(nlive),
        "--seed", str(seed),
    ]

    print(f"  Running BF: {' '.join(cmd)}")

    # Write logs to file
    log_path = os.path.join(result_dir, "run.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        print(f"  BF RUN FAILED (see {log_path})")
        return None

    # Load result - the script outputs {event}_bf_summary.json
    result_file = os.path.join(result_dir, f"{event}_bf_summary.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)

    print(f"  Warning: No result file found: {result_file}")
    return None


def run_sanity_check(args):
    """2A: H0 ringdown-only sanity check.

    Runs full BF analysis but focuses on H0 results for sanity checking.
    """
    print("\n" + "=" * 70)
    print("2A: H0 RINGDOWN-ONLY SANITY CHECK")
    print("=" * 70)

    t_start_ms = args.t_start_ms
    duration_ms = args.duration_ms
    base_outdir = args.base_outdir

    # Fetch data
    print(f"\nFetching data (t_start={t_start_ms}ms, duration={duration_ms}ms)...")
    if not fetch_data(base_outdir, t_start_ms, duration_ms):
        print("FAILED: Could not fetch data")
        return False

    # Run BF analysis (includes H0)
    result_subdir = f"sanity/t{ms_label(t_start_ms)}ms_d{ms_label(duration_ms)}ms"
    print(f"\nRunning analysis -> {os.path.join(base_outdir, result_subdir)}")

    result = run_bf_analysis(
        base_outdir, t_start_ms, duration_ms,
        args.fmin, args.fmax,
        args.nlive, args.seed,
        result_subdir
    )

    if result is None:
        print("FAILED: Analysis did not complete")
        return False

    print("\n" + "-" * 50)
    print("SANITY CHECK RESULTS (H0 posteriors):")
    print("-" * 50)
    print(f"  log Z(H0) = {fmt2(result.get('log_evidence_H0'))}")
    print(f"  t_start = {fmt2(result.get('t_start_ms'))} ms")
    print(f"  segment = {fmt2(result.get('segment_duration_ms'))} ms")
    print(f"  Results saved to: {os.path.join(base_outdir, result_subdir)}")
    print("-" * 50)

    return True


def run_bayes_factor(args):
    """2B: H1 vs H0 Bayes factor on same segment."""
    print("\n" + "=" * 70)
    print("2B: H1 vs H0 BAYES FACTOR")
    print("=" * 70)

    t_start_ms = args.t_start_ms
    duration_ms = args.duration_ms
    base_outdir = args.base_outdir

    # Fetch data
    print(f"\nFetching data (t_start={t_start_ms}ms, duration={duration_ms}ms)...")
    if not fetch_data(base_outdir, t_start_ms, duration_ms):
        print("FAILED: Could not fetch data")
        return False

    # Run BF analysis
    result_subdir = f"bf/t{ms_label(t_start_ms)}ms_d{ms_label(duration_ms)}ms"
    print(f"\nRunning BF analysis -> {os.path.join(base_outdir, result_subdir)}")

    result = run_bf_analysis(
        base_outdir, t_start_ms, duration_ms,
        args.fmin, args.fmax,
        args.nlive, args.seed,
        result_subdir
    )

    if result is None:
        print("FAILED: BF analysis did not complete")
        return False

    print("\n" + "-" * 50)
    print("BAYES FACTOR RESULTS:")
    print("-" * 50)
    print(f"  log Z(H0) = {fmt2(result.get('log_evidence_H0'))}")
    print(f"  log Z(H1) = {fmt2(result.get('log_evidence_H1'))}")
    print(f"  ln BF(H1/H0) = {fmt2(result.get('ln_BF'))}")
    print(f"  log10 BF = {fmt2(result.get('log10_BF'))}")
    print("-" * 50)

    return True


def run_robustness_sweep(args):
    """2C: Robustness sweep over t_start, duration, frequency band."""
    print("\n" + "=" * 70)
    print("2C: ROBUSTNESS SWEEP")
    print("=" * 70)

    base_outdir = args.base_outdir
    results = []
    seed = args.seed  # Fixed seed for reproducibility

    for t_start_ms in T_START_GRID_MS:
        for duration_ms in DURATION_GRID_MS:
            for fmin, fmax, band_label in FREQ_BANDS:
                config_label = f"t{ms_label(t_start_ms)}ms_d{ms_label(duration_ms)}ms_{band_label}"
                print(f"\n--- {config_label} ---")

                # Fetch data
                if not fetch_data(base_outdir, t_start_ms, duration_ms):
                    print(f"  SKIPPED: fetch failed")
                    continue

                # Run BF
                result_subdir = f"sweep/{config_label}"
                result = run_bf_analysis(
                    base_outdir, t_start_ms, duration_ms,
                    fmin, fmax,
                    args.nlive, seed,
                    result_subdir
                )

                if result is None:
                    print(f"  SKIPPED: BF failed")
                    continue

                result_entry = {
                    "t_start_ms": t_start_ms,
                    "duration_ms": duration_ms,
                    "fmin": fmin,
                    "fmax": fmax,
                    "band_label": band_label,
                    "ln_BF": result.get("ln_BF"),
                    "log10_BF": result.get("log10_BF"),
                    "log_evidence_H0": result.get("log_evidence_H0"),
                    "log_evidence_H1": result.get("log_evidence_H1"),
                }
                results.append(result_entry)

                print(f"  ln BF = {fmt2(result.get('ln_BF'))}")

    # Save sweep results
    sweep_dir = os.path.join(base_outdir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    sweep_summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "nlive": args.nlive,
        "results": results,
    }

    with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
        json.dump(sweep_summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("ROBUSTNESS SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'ln BF':>10} {'log10 BF':>12}")
    print("-" * 60)
    for r in results:
        label = f"t{ms_label(r['t_start_ms'])}ms_d{ms_label(r['duration_ms'])}ms_{r['band_label']}"
        print(f"{label:<35} {fmt2(r.get('ln_BF')):>10} {fmt2(r.get('log10_BF')):>12}")
    print("-" * 60)

    # Check stability
    if len(results) >= 2:
        ln_bfs = [r["ln_BF"] for r in results if r.get("ln_BF") is not None]
        if len(ln_bfs) >= 2:
            mean_bf = statistics.mean(ln_bfs)
            std_bf = statistics.pstdev(ln_bfs)
            print(f"\nln BF: mean = {mean_bf:.2f}, std = {std_bf:.2f}")
            if std_bf < 1.0:
                print("STABLE: BF consistent across analysis choices")
            else:
                print("UNSTABLE: BF varies significantly with analysis choices")

    return True


def main():
    parser = argparse.ArgumentParser(description="GW150914 ringdown echo analysis")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["sanity", "bf", "sweep", "all"],
                        help="Analysis mode: sanity (2A), bf (2B), sweep (2C), or all")
    parser.add_argument("--base-outdir", type=str, default="out_gw150914",
                        help="Base output directory for all data and results")
    parser.add_argument("--t-start-ms", type=float, default=DEFAULT_T_START_MS,
                        help=f"Time after merger to start gate (ms, default {DEFAULT_T_START_MS})")
    parser.add_argument("--duration-ms", type=float, default=DEFAULT_DURATION_MS,
                        help=f"Gate duration (ms, default {DEFAULT_DURATION_MS})")
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN)
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX)
    parser.add_argument("--nlive", type=int, default=DEFAULT_NLIVE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"GW150914 Ringdown Echo Analysis")
    print(f"Mode: {args.mode}")
    print(f"GPS merger: {GPS_MERGER}")
    print(f"Base output: {args.base_outdir}")
    print(f"Settings: t_start={args.t_start_ms}ms, duration={args.duration_ms}ms")
    print(f"Frequency band: {args.fmin}-{args.fmax} Hz")
    print(f"Sampler: nlive={args.nlive}, seed={args.seed}")

    if args.mode == "sanity" or args.mode == "all":
        run_sanity_check(args)

    if args.mode == "bf" or args.mode == "all":
        run_bayes_factor(args)

    if args.mode == "sweep" or args.mode == "all":
        run_robustness_sweep(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
