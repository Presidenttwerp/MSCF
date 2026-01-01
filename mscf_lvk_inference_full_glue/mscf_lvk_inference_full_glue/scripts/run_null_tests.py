#!/usr/bin/env python3
"""
Null tests: Run Bayes factor pipeline on 5 off-source segments (no GW signal).

These segments are far from any known GW events, so they contain only detector noise.
If the pipeline finds strong evidence for echoes (log10 BF > 1) in pure noise,
that indicates model flexibility/leakage issues.

Off-source GPS times chosen to be ~1000s, 2000s, 3000s, 4000s, 5000s before GW150914,
which are well away from any detected events.
"""
import argparse
import subprocess
import os
import sys

# GW150914 GPS time
GW150914_GPS = 1126259462.4

# Choose 5 off-source times ~1000-5000s before the event
# These times have no known GW signals
OFF_SOURCE_OFFSETS = [-1000, -2000, -3000, -4000, -5000]  # seconds before GW150914

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-segments", type=int, default=5, help="Number of off-source segments")
    p.add_argument("--outdir", type=str, default="out_null")
    p.add_argument("--seed-base", type=int, default=42, help="Base seed for reproducibility")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    n_segs = min(args.n_segments, len(OFF_SOURCE_OFFSETS))

    for i in range(n_segs):
        offset = OFF_SOURCE_OFFSETS[i]
        gps = GW150914_GPS + offset
        seg_name = f"null_seg{i+1}"
        seg_outdir = os.path.join(args.outdir, seg_name)
        os.makedirs(seg_outdir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"NULL TEST {i+1}/{n_segs}: GPS {gps:.1f} (offset {offset}s from GW150914)")
        print(f"{'='*60}")

        # Step 1: Fetch data and PSD for this off-source segment
        fetch_cmd = [
            sys.executable, "scripts/fetch_and_make_psd.py",
            "--event", seg_name,
            "--gps", str(gps),
            "--duration", "32.0",
            "--psd-duration", "512.0",
            "--sample-rate", "4096",
            "--ifos", "H1,L1",
            "--outdir", seg_outdir
        ]

        print(f"\n[1] Fetching data: {' '.join(fetch_cmd)}")
        if not args.dry_run:
            result = subprocess.run(fetch_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR fetching data:\n{result.stderr}")
                continue
            print(result.stdout)

        # Step 2: Run Bayes factor analysis
        # Use the same GPS as t0 center (even though there's no signal,
        # we still need a reference time for the prior)
        seed = args.seed_base + i
        bf_cmd = [
            sys.executable, "run_bayes_factor.py",
            "--event", seg_name,
            "--gps", str(gps),
            "--ifos", "H1,L1",
            "--outdir", seg_outdir,
            "--resultdir", seg_outdir,
            "--seed", str(seed)
        ]

        print(f"\n[2] Running Bayes factor: {' '.join(bf_cmd)}")
        if not args.dry_run:
            result = subprocess.run(bf_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            # Extract the Bayes factor from output
            for line in result.stdout.split('\n'):
                if 'ln BF_10' in line or 'log10 BF_10' in line:
                    print(f"\n>>> RESULT: {line}")

        print(f"\nCompleted segment {i+1}/{n_segs}")

    print(f"\n{'='*60}")
    print("NULL TEST SUMMARY")
    print(f"{'='*60}")
    print("Check each segment's log10(BF). If ANY has log10(BF) > 1, there's a problem!")


if __name__ == "__main__":
    main()
