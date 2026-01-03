#!/usr/bin/env python3
"""
PARALLEL null distribution for MSCF echo model validation.

Runs multiple tests simultaneously using process pool.
Each run is completely independent - parallelization does NOT affect accuracy.

Resource usage per run:
- ~300-400 MB RAM
- 1 CPU core at 100%

Safe parallelism: 8-16 workers on a 32-core machine with 32GB RAM
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# GPS offsets to test
GPS_OFFSETS = [
    -300, -250, -200, -180, -150, -120, -100, -80, -60, -50, -40, -30, -20, -10,
    10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300
]

# Time-slides to test
TIME_SLIDES = [0.1, 0.2, 0.3, 0.5, 1.0]


def find_available_offsets(data_base_dir):
    """Find all available GPS offset directories."""
    pattern = os.path.join(data_base_dir, "GW150914_offsource_gps*_joint")
    dirs = glob.glob(pattern)

    offsets = []
    import re
    for d in dirs:
        basename = os.path.basename(d)
        match = re.search(r'gps([+-]?\d+)s', basename)
        if match:
            offsets.append(int(match.group(1)))

    return sorted(offsets)


def run_single_test(args_tuple):
    """
    Run a single test. Designed to be called from ProcessPoolExecutor.
    Returns (offset, slide, result_dict or None, error_msg or None)
    """
    data_dir, time_slide, outdir, nlive, seed, offset = args_tuple

    try:
        cmd = [
            sys.executable,
            "scripts/test_mscf_train_null.py",
            "--data-dir", data_dir,
            "--time-slide", str(time_slide),
            "--outdir", outdir,
            "--nlive", str(nlive),
            "--seed", str(seed),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            return (offset, time_slide, None, result.stderr[-500:])

        # Parse the summary JSON
        summary_path = os.path.join(outdir, "mscf_train_null_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
                return (offset, time_slide, data, None)

        return (offset, time_slide, None, "No summary file found")

    except subprocess.TimeoutExpired:
        return (offset, time_slide, None, "TIMEOUT after 1 hour")
    except Exception as e:
        return (offset, time_slide, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="PARALLEL null distribution"
    )
    parser.add_argument("--data-base-dir",
                       default="out_gw150914_offsource/data",
                       help="Base directory containing off-source data")
    parser.add_argument("--outdir",
                       default="out_null_distribution_parallel",
                       help="Output directory")
    parser.add_argument("--offsets", default=None,
                       help="Comma-separated GPS offsets (default: auto-detect)")
    parser.add_argument("--time-slides", default=None,
                       help="Comma-separated time-slides (default: 0.1,0.2,0.3,0.5,1.0)")
    parser.add_argument("--nlive", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers (default: 8, max recommended: 16)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Limit workers to reasonable range
    max_workers = min(args.workers, multiprocessing.cpu_count() - 2, 16)
    print(f"Using {max_workers} parallel workers")

    os.makedirs(args.outdir, exist_ok=True)

    # Find/parse offsets
    if args.offsets:
        offsets = [int(x) for x in args.offsets.split(",")]
    else:
        offsets = find_available_offsets(args.data_base_dir)

    # Parse time-slides
    if args.time_slides:
        time_slides = [float(x) for x in args.time_slides.split(",")]
    else:
        time_slides = TIME_SLIDES

    total_runs = len(offsets) * len(time_slides)

    print("=" * 70)
    print("PARALLEL NULL DISTRIBUTION")
    print("=" * 70)
    print(f"\nData directory: {args.data_base_dir}")
    print(f"GPS offsets: {len(offsets)} offsets")
    print(f"Time-slides: {time_slides}")
    print(f"Total runs: {total_runs}")
    print(f"Workers: {max_workers}")
    print(f"nlive: {args.nlive}")
    print(f"Estimated time: ~{total_runs * 7 / max_workers / 60:.1f} hours")
    print()

    if args.dry_run:
        print("DRY RUN - not executing")
        return

    # Find completed runs if resuming
    completed = set()
    if args.resume:
        for d in glob.glob(os.path.join(args.outdir, "offset*")):
            summary = os.path.join(d, "mscf_train_null_summary.json")
            if os.path.exists(summary):
                # Extract offset and slide from directory name
                import re
                match = re.search(r'offset([+-]?\d+)_slide([\d.]+)s', os.path.basename(d))
                if match:
                    completed.add((int(match.group(1)), float(match.group(2))))
        print(f"Resuming: {len(completed)}/{total_runs} already completed")

    # Build task list
    tasks = []
    for offset in offsets:
        data_dir = os.path.join(
            args.data_base_dir,
            f"GW150914_offsource_gps{offset:+d}s_joint"
        )

        if not os.path.exists(data_dir):
            continue

        for slide in time_slides:
            if (offset, slide) in completed:
                continue

            run_outdir = os.path.join(
                args.outdir,
                f"offset{offset:+d}_slide{slide}s"
            )
            os.makedirs(run_outdir, exist_ok=True)

            seed = args.seed_base + abs(offset) * 10 + int(slide * 10)
            tasks.append((data_dir, slide, run_outdir, args.nlive, seed, offset))

    print(f"Tasks to run: {len(tasks)}")
    print()

    # Run in parallel
    all_results = []
    errors = []
    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_test, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures)):
            offset, slide, result, error = future.result()

            elapsed = (datetime.now() - start_time).total_seconds() / 60
            completed_count = len(all_results) + len(errors) + len(completed) + 1
            rate = completed_count / max(elapsed, 1) * 60  # per hour

            if error:
                errors.append((offset, slide, error))
                print(f"[{completed_count}/{total_runs}] offset{offset:+d}_slide{slide}s: ERROR - {error[:50]}")
            else:
                result["gps_offset"] = offset
                result["time_slide"] = slide
                all_results.append(result)

                aligned_bf = result["aligned"]["ln_BF"]
                slid_bf = result["timeslid"]["ln_BF"]
                print(f"[{completed_count}/{total_runs}] offset{offset:+d}_slide{slide}s: "
                      f"aligned={aligned_bf:.1f}, timeslid={slid_bf:.1f} "
                      f"({elapsed:.1f}min, ~{rate:.1f}/hr)")

            # Save intermediate results every 10 completions
            if completed_count % 10 == 0:
                with open(os.path.join(args.outdir, "all_results.json"), "w") as f:
                    json.dump(all_results, f, indent=2)

    # Final save
    with open(os.path.join(args.outdir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Calculate statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Completed: {len(all_results)}/{total_runs}")
    print(f"Errors: {len(errors)}")
    print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")

    aligned_bfs = [r["aligned"]["ln_BF"] for r in all_results]
    slid_bfs = [r["timeslid"]["ln_BF"] for r in all_results]

    if aligned_bfs:
        aligned = np.array(aligned_bfs)
        print(f"\nALIGNED (n={len(aligned)}):")
        print(f"  median: {np.median(aligned):.2f}")
        print(f"  mean ± std: {np.mean(aligned):.2f} ± {np.std(aligned):.2f}")
        print(f"  min / max: {np.min(aligned):.2f} / {np.max(aligned):.2f}")
        print(f"  P(ln_BF > 0):  {100*np.mean(aligned > 0):.1f}%")
        print(f"  P(ln_BF > 10): {100*np.mean(aligned > 10):.1f}%")
        print(f"  P(ln_BF > 20): {100*np.mean(aligned > 20):.1f}%")

        if np.any(aligned > 10):
            print(f"  *** WARNING: {np.sum(aligned > 10)} runs with lnBF > 10 ***")

    if slid_bfs:
        slid = np.array(slid_bfs)
        print(f"\nTIME-SLID (n={len(slid)}):")
        print(f"  median: {np.median(slid):.2f}")
        print(f"  mean ± std: {np.mean(slid):.2f} ± {np.std(slid):.2f}")
        print(f"  min / max: {np.min(slid):.2f} / {np.max(slid):.2f}")
        print(f"  P(ln_BF > 0):  {100*np.mean(slid > 0):.1f}%")
        print(f"  P(ln_BF > 10): {100*np.mean(slid > 10):.1f}%")
        print(f"  P(ln_BF > 20): {100*np.mean(slid > 20):.1f}%")

        if np.any(slid > 10):
            print(f"  *** WARNING: {np.sum(slid > 10)} runs with lnBF > 10 ***")

    # Decision gate check
    print("\n" + "=" * 70)
    print("DECISION GATE CHECK")
    print("=" * 70)

    aligned_pass = len(aligned_bfs) > 0 and np.mean(np.array(aligned_bfs) > 10) < 0.02
    slid_pass = len(slid_bfs) > 0 and np.mean(np.array(slid_bfs) > 10) < 0.02

    if aligned_pass and slid_pass:
        print("PASS: Both aligned and time-slid meet criteria (P(lnBF>10) < 2%)")
    else:
        print("FAIL: Decision gate not met")
        if not aligned_pass:
            print(f"  - Aligned: P(lnBF>10) = {100*np.mean(np.array(aligned_bfs) > 10):.1f}%")
        if not slid_pass:
            print(f"  - Time-slid: P(lnBF>10) = {100*np.mean(np.array(slid_bfs) > 10):.1f}%")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for offset, slide, err in errors[:10]:
            print(f"  offset{offset:+d}_slide{slide}s: {err[:80]}")

    print(f"\nResults saved to: {args.outdir}")


if __name__ == "__main__":
    main()
