#!/usr/bin/env python3
"""
Rolling summary monitor for null distribution runs.

Computes incremental statistics from completed runs to audit progress
without "vibes" - just hard numbers.

LOCKED DECISION GATES (defined upfront, no rationalization):
============================================================

1. NULL BF CRITERIA (H1_coh vs H0):
   - P(lnBF > 10) < 1-2%  (ideally 0)
   - max(lnBF) should not repeatedly exceed 10-20
   - If violated: FAIL - model still overfits

2. COHERENCE DISCRIMINATOR (H1_coh vs H1_incoh on time-slides):
   - lnBF_coh/incoh <= 0 almost always
   - Should be strongly negative (we're seeing -100 to -700)
   - If violated: FAIL - coherence test broken

If BOTH hold across 140 runs: PROCEED to on-source robustness
If EITHER fails: STOP and debug before proceeding
"""

import os
import json
import glob
import numpy as np
from datetime import datetime


def load_null_results(outdir="out_null_distribution_full"):
    """Load all completed null distribution results."""
    results = []
    pattern = os.path.join(outdir, "*/mscf_train_null_summary.json")

    for path in glob.glob(pattern):
        try:
            with open(path, 'r') as f:
                content = f.read()
                # Handle truncated JSON (bool serialization bug)
                if '"PASS_aligned":' in content and not content.strip().endswith('}'):
                    # Truncated - extract what we can
                    pass
                d = json.loads(content)

                run_name = os.path.basename(os.path.dirname(path))

                results.append({
                    'run': run_name,
                    'data': d.get('data', ''),
                    'time_slide': d.get('time_slide_L1_s', 0),

                    # Aligned data (no time-slide applied)
                    'aligned_logZ_H0': d['aligned']['logZ_H0'],
                    'aligned_logZ_H0_err': d['aligned']['logZ_H0_err'],
                    'aligned_logZ_H1': d['aligned']['logZ_H1'],
                    'aligned_logZ_H1_err': d['aligned']['logZ_H1_err'],
                    'aligned_lnBF': d['aligned']['ln_BF'],
                    'aligned_lnBF_err': d['aligned']['ln_BF_err'],

                    # Time-slid data
                    'timeslid_logZ_H0': d['timeslid']['logZ_H0'],
                    'timeslid_logZ_H0_err': d['timeslid']['logZ_H0_err'],
                    'timeslid_logZ_H1': d['timeslid']['logZ_H1'],
                    'timeslid_logZ_H1_err': d['timeslid']['logZ_H1_err'],
                    'timeslid_lnBF': d['timeslid']['ln_BF'],
                    'timeslid_lnBF_err': d['timeslid']['ln_BF_err'],
                })
        except (json.JSONDecodeError, KeyError) as e:
            # Truncated or malformed - try to extract partial data
            try:
                with open(path, 'r') as f:
                    content = f.read()

                run_name = os.path.basename(os.path.dirname(path))

                # Parse manually for key values
                import re

                def extract_value(pattern, text):
                    match = re.search(pattern, text)
                    return float(match.group(1)) if match else None

                # Extract aligned section
                aligned_match = re.search(r'"aligned":\s*\{([^}]+)\}', content, re.DOTALL)
                timeslid_match = re.search(r'"timeslid":\s*\{([^}]+)\}', content, re.DOTALL)

                if aligned_match and timeslid_match:
                    aligned_section = aligned_match.group(1)
                    timeslid_section = timeslid_match.group(1)

                    results.append({
                        'run': run_name,
                        'data': '',
                        'time_slide': extract_value(r'"time_slide_L1_s":\s*([\d.]+)', content) or 0,

                        'aligned_logZ_H0': extract_value(r'"logZ_H0":\s*([-\d.]+)', aligned_section),
                        'aligned_logZ_H0_err': extract_value(r'"logZ_H0_err":\s*([-\d.]+)', aligned_section),
                        'aligned_logZ_H1': extract_value(r'"logZ_H1":\s*([-\d.]+)', aligned_section),
                        'aligned_logZ_H1_err': extract_value(r'"logZ_H1_err":\s*([-\d.]+)', aligned_section),
                        'aligned_lnBF': extract_value(r'"ln_BF":\s*([-\d.]+)', aligned_section),
                        'aligned_lnBF_err': extract_value(r'"ln_BF_err":\s*([-\d.]+)', aligned_section),

                        'timeslid_logZ_H0': extract_value(r'"logZ_H0":\s*([-\d.]+)', timeslid_section),
                        'timeslid_logZ_H0_err': extract_value(r'"logZ_H0_err":\s*([-\d.]+)', timeslid_section),
                        'timeslid_logZ_H1': extract_value(r'"logZ_H1":\s*([-\d.]+)', timeslid_section),
                        'timeslid_logZ_H1_err': extract_value(r'"logZ_H1_err":\s*([-\d.]+)', timeslid_section),
                        'timeslid_lnBF': extract_value(r'"ln_BF":\s*([-\d.]+)', timeslid_section),
                        'timeslid_lnBF_err': extract_value(r'"ln_BF_err":\s*([-\d.]+)', timeslid_section),
                    })
            except Exception:
                print(f"  [WARN] Could not parse: {path}")
                continue

    return results


def load_coherence_results(pattern="out_coherence_test_*/coherence_*.json"):
    """Load coherence test results."""
    results = []

    for path in glob.glob(pattern):
        try:
            with open(path, 'r') as f:
                d = json.load(f)
                results.append({
                    'gps_offset': d.get('gps_offset', 0),
                    'time_slide': d.get('time_slide', 0),
                    'logZ_coh': d['logZ_coh'],
                    'logZ_coh_err': d['logZ_coh_err'],
                    'logZ_incoh': d['logZ_incoh'],
                    'logZ_incoh_err': d['logZ_incoh_err'],
                    'lnBF_coh_incoh': d['ln_BF_coh_incoh'],
                    'lnBF_coh_incoh_err': d['ln_BF_coh_incoh_err'],
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Could not parse: {path}")
            continue

    return results


def compute_stats(values):
    """Compute summary statistics for an array of values."""
    if len(values) == 0:
        return None

    arr = np.array(values)
    return {
        'count': len(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'p90': np.percentile(arr, 90),
        'p95': np.percentile(arr, 95),
        'p99': np.percentile(arr, 99),
        'frac_gt_0': np.mean(arr > 0),
        'frac_gt_10': np.mean(arr > 10),
        'frac_gt_20': np.mean(arr > 20),
        'n_gt_10': np.sum(arr > 10),
        'n_gt_20': np.sum(arr > 20),
    }


def print_decision_gates():
    """Print the locked decision gates."""
    print("=" * 70)
    print("LOCKED DECISION GATES (no rationalization)")
    print("=" * 70)
    print()
    print("1. NULL BF CRITERIA (H1_coh vs H0):")
    print("   - P(lnBF > 10) < 1-2%  (ideally 0)")
    print("   - max(lnBF) should not repeatedly exceed 10-20")
    print("   - If violated: FAIL")
    print()
    print("2. COHERENCE DISCRIMINATOR (H1_coh vs H1_incoh):")
    print("   - lnBF_coh/incoh <= 0 almost always")
    print("   - Should be strongly negative")
    print("   - If violated: FAIL")
    print()
    print("PASS = Both criteria met across all runs")
    print("=" * 70)
    print()


def print_null_summary(results):
    """Print summary of null distribution results."""
    if not results:
        print("No null distribution results yet.")
        return

    # Separate aligned and time-slid
    aligned_lnBFs = [r['aligned_lnBF'] for r in results if r['aligned_lnBF'] is not None]
    timeslid_lnBFs = [r['timeslid_lnBF'] for r in results if r['timeslid_lnBF'] is not None]

    print(f"NULL DISTRIBUTION SUMMARY ({len(results)}/140 runs complete)")
    print("-" * 70)

    if aligned_lnBFs:
        stats = compute_stats(aligned_lnBFs)
        print(f"\nALIGNED DATA (pure noise, no time-slide):")
        print(f"  Count:    {stats['count']}")
        print(f"  Mean:     {stats['mean']:.2f}")
        print(f"  Median:   {stats['median']:.2f}")
        print(f"  Std:      {stats['std']:.2f}")
        print(f"  Min/Max:  {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"  90/95/99%: {stats['p90']:.2f} / {stats['p95']:.2f} / {stats['p99']:.2f}")
        print(f"  P(lnBF > 0):  {stats['frac_gt_0']*100:.1f}% ({int(stats['count']*stats['frac_gt_0'])}/{stats['count']})")
        print(f"  P(lnBF > 10): {stats['frac_gt_10']*100:.1f}% ({stats['n_gt_10']}/{stats['count']})")
        print(f"  P(lnBF > 20): {stats['frac_gt_20']*100:.1f}% ({stats['n_gt_20']}/{stats['count']})")

        # Decision gate check
        if stats['frac_gt_10'] > 0.02:
            print(f"  *** FAIL: P(lnBF > 10) = {stats['frac_gt_10']*100:.1f}% > 2% ***")
        elif stats['n_gt_10'] > 0:
            print(f"  ** WARNING: {stats['n_gt_10']} runs with lnBF > 10 **")
        else:
            print(f"  PASS: No runs with lnBF > 10")

    if timeslid_lnBFs:
        stats = compute_stats(timeslid_lnBFs)
        print(f"\nTIME-SLID DATA (incoherent noise):")
        print(f"  Count:    {stats['count']}")
        print(f"  Mean:     {stats['mean']:.2f}")
        print(f"  Median:   {stats['median']:.2f}")
        print(f"  Std:      {stats['std']:.2f}")
        print(f"  Min/Max:  {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"  90/95/99%: {stats['p90']:.2f} / {stats['p95']:.2f} / {stats['p99']:.2f}")
        print(f"  P(lnBF > 0):  {stats['frac_gt_0']*100:.1f}% ({int(stats['count']*stats['frac_gt_0'])}/{stats['count']})")
        print(f"  P(lnBF > 10): {stats['frac_gt_10']*100:.1f}% ({stats['n_gt_10']}/{stats['count']})")
        print(f"  P(lnBF > 20): {stats['frac_gt_20']*100:.1f}% ({stats['n_gt_20']}/{stats['count']})")

        # Decision gate check
        if stats['frac_gt_10'] > 0.02:
            print(f"  *** FAIL: P(lnBF > 10) = {stats['frac_gt_10']*100:.1f}% > 2% ***")
        elif stats['n_gt_10'] > 0:
            print(f"  ** WARNING: {stats['n_gt_10']} runs with lnBF > 10 **")
        else:
            print(f"  PASS: No runs with lnBF > 10")

    # List individual results
    print(f"\nINDIVIDUAL RESULTS:")
    print(f"{'Run':<30} {'Aligned lnBF':>15} {'Time-slid lnBF':>15}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x['run']):
        aligned = f"{r['aligned_lnBF']:.2f}" if r['aligned_lnBF'] is not None else "N/A"
        timeslid = f"{r['timeslid_lnBF']:.2f}" if r['timeslid_lnBF'] is not None else "N/A"
        print(f"{r['run']:<30} {aligned:>15} {timeslid:>15}")


def print_coherence_summary(results):
    """Print summary of coherence test results."""
    if not results:
        print("\nNo coherence test results found.")
        return

    print(f"\nCOHERENCE DISCRIMINATOR SUMMARY ({len(results)} tests)")
    print("-" * 70)

    lnBFs = [r['lnBF_coh_incoh'] for r in results]
    stats = compute_stats(lnBFs)

    print(f"  Count:    {stats['count']}")
    print(f"  Mean:     {stats['mean']:.2f}")
    print(f"  Median:   {stats['median']:.2f}")
    print(f"  Min/Max:  {stats['min']:.2f} / {stats['max']:.2f}")
    print(f"  P(lnBF > 0): {stats['frac_gt_0']*100:.1f}%")

    # Decision gate check
    if stats['max'] > 0:
        print(f"  ** WARNING: max lnBF_coh/incoh = {stats['max']:.2f} > 0 **")
    else:
        print(f"  PASS: All lnBF_coh/incoh <= 0 (incoherent wins)")

    print(f"\n  {'Time-slide':<12} {'lnBF_coh/incoh':>18} {'Status':<10}")
    print("  " + "-" * 45)
    for r in sorted(results, key=lambda x: x['time_slide']):
        status = "PASS" if r['lnBF_coh_incoh'] <= 0 else "FAIL"
        print(f"  {r['time_slide']:<12.1f} {r['lnBF_coh_incoh']:>18.2f} {status:<10}")


def identify_outliers(results, threshold=10):
    """Identify runs with lnBF > threshold for Step 2 forensics."""
    outliers = []

    for r in results:
        if r['aligned_lnBF'] is not None and r['aligned_lnBF'] > threshold:
            outliers.append({
                'run': r['run'],
                'type': 'aligned',
                'lnBF': r['aligned_lnBF'],
                'data': r['data'],
            })
        if r['timeslid_lnBF'] is not None and r['timeslid_lnBF'] > threshold:
            outliers.append({
                'run': r['run'],
                'type': 'timeslid',
                'lnBF': r['timeslid_lnBF'],
                'data': r['data'],
            })

    return outliers


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor null distribution progress")
    parser.add_argument("--outdir", default="out_null_distribution_full",
                       help="Null distribution output directory")
    parser.add_argument("--watch", action="store_true",
                       help="Watch mode - update every 60 seconds")
    args = parser.parse_args()

    while True:
        os.system('clear' if os.name == 'posix' else 'cls')

        print(f"MSCF Null Distribution Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        print_decision_gates()

        # Load and summarize null distribution
        null_results = load_null_results(args.outdir)
        print_null_summary(null_results)

        # Load and summarize coherence tests
        coherence_results = load_coherence_results()
        print_coherence_summary(coherence_results)

        # Check for outliers
        outliers = identify_outliers(null_results, threshold=10)
        if outliers:
            print(f"\n*** OUTLIERS DETECTED (lnBF > 10) - Step 2 forensics needed ***")
            print("-" * 70)
            for o in outliers:
                print(f"  {o['run']} ({o['type']}): lnBF = {o['lnBF']:.2f}")
                print(f"    Data: {o['data']}")
            print()
            print("Run per-IFO attribution on these segments:")
            print("  python scripts/run_detector_attribution.py --data-dir <data> --time-slide <slide>")

        print()
        print("=" * 70)

        if not args.watch:
            break

        print("Refreshing in 60 seconds... (Ctrl+C to exit)")
        try:
            import time
            time.sleep(60)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
