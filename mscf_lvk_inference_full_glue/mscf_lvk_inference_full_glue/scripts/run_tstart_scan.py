#!/usr/bin/env python3
"""
Scan over t_start values to test robustness of echo BF.

If echo evidence only appears when including earlier times (close to merger),
it's likely contamination from inspiral/merger. If it persists for late-time
windows, that's more meaningful.

Default scan: t_start = {5, 10, 15, 20, 25} ms after merger
"""
import argparse
import subprocess
import os
import sys
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gps", type=float, default=1126259462.4,
                   help="Merger GPS time")
    p.add_argument("--event-base", type=str, default="GW150914",
                   help="Base event name")
    p.add_argument("--t-starts", type=str, default="5,10,15,20,25",
                   help="Comma-separated t_start values in ms")
    p.add_argument("--gate-duration", type=float, default=0.2,
                   help="Gate duration in seconds (default 200 ms)")
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir-base", type=str, default="out_gated_scan")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nlive", type=int, default=400)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    t_starts_ms = [float(x) for x in args.t_starts.split(",")]
    os.makedirs(args.outdir_base, exist_ok=True)

    results = []

    for t_start_ms in t_starts_ms:
        t_start_s = t_start_ms / 1000.0
        event_name = f"{args.event_base}_t{int(t_start_ms)}ms"
        outdir = os.path.join(args.outdir_base, f"t{int(t_start_ms)}ms")
        os.makedirs(outdir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"t_start = {t_start_ms} ms")
        print(f"{'='*60}")

        # Step 1: Fetch gated data
        fetch_cmd = [
            sys.executable, "scripts/fetch_ringdown_gated.py",
            "--event", event_name,
            "--gps", str(args.gps),
            "--t-start", str(t_start_s),
            "--gate-duration", str(args.gate_duration),
            "--ifos", args.ifos,
            "--outdir", outdir
        ]
        print(f"[1] Fetching: {' '.join(fetch_cmd)}")
        if not args.dry_run:
            result = subprocess.run(fetch_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: {result.stderr}")
                continue
            print(result.stdout)

        # Step 2: Run gated BF analysis
        bf_cmd = [
            sys.executable, "run_bayes_factor_gated.py",
            "--event", event_name,
            "--gps", str(args.gps),
            "--t-start", str(t_start_s),
            "--ifos", args.ifos,
            "--outdir", outdir,
            "--resultdir", outdir,
            "--nlive", str(args.nlive)
        ]
        if args.seed is not None:
            bf_cmd.extend(["--seed", str(args.seed)])

        print(f"[2] Running BF: {' '.join(bf_cmd)}")
        if not args.dry_run:
            result = subprocess.run(bf_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            # Load summary
            summary_path = os.path.join(outdir, f"{event_name}_bf_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                results.append({
                    "t_start_ms": t_start_ms,
                    "ln_BF": summary["ln_BF"],
                    "log10_BF": summary["log10_BF"]
                })

    # Print summary
    print(f"\n{'='*60}")
    print("t_start SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"{'t_start (ms)':<15} {'ln BF':<12} {'log10 BF':<12}")
    print("-" * 40)
    for r in results:
        print(f"{r['t_start_ms']:<15.0f} {r['ln_BF']:<12.3f} {r['log10_BF']:<12.3f}")

    # Save combined results
    combined_path = os.path.join(args.outdir_base, "tstart_scan_summary.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scan summary to {combined_path}")

    # Interpretation
    if results:
        print(f"\n>>> INTERPRETATION <<<")
        max_bf = max(r["log10_BF"] for r in results)
        min_bf = min(r["log10_BF"] for r in results)
        if max_bf > 1 and min_bf < 0.5:
            print("  WARNING: BF varies significantly with t_start")
            print("  High BF at early times may indicate inspiral contamination")
        elif max_bf > 1:
            print("  BF consistently high - but check if physically reasonable")
        else:
            print("  No strong echo evidence at any t_start")


if __name__ == "__main__":
    main()
