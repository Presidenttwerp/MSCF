#!/usr/bin/env python
"""Debug script to check:
1. Whether coverage computation uses correct quantiles
2. Whether noise scaling / chi^2 is correct
3. Whether SNR is computed consistently
"""

import numpy as np
import json
import os
from pathlib import Path
import sys

def check_single_run_coverage(run_dir):
    """Check coverage for a single run using actual percentiles vs Gaussian approximation."""

    summary_file = Path(run_dir) / "injection_summary.json"
    if not summary_file.exists():
        print(f"No summary file in {run_dir}")
        return None

    with open(summary_file) as f:
        summary = json.load(f)

    # Get injected values
    Mf_inj = summary['injected']['Mf']
    chi_inj = summary['injected']['chi']
    f0_inj = summary['injected']['f0']

    # Get recovered statistics
    rec = summary['recovered']

    # Method 1: Gaussian approximation (what aggregate_ensemble_v2.py uses)
    # 90% CI = median ± 1.645 * std
    Mf_lo_gauss = rec['Mf_median'] - 1.645 * rec['Mf_std']
    Mf_hi_gauss = rec['Mf_median'] + 1.645 * rec['Mf_std']

    chi_lo_gauss = rec['chi_median'] - 1.645 * rec['chi_std']
    chi_hi_gauss = rec['chi_median'] + 1.645 * rec['chi_std']

    f0_lo_gauss = rec['f0_median'] - 1.645 * rec['f0_std']
    f0_hi_gauss = rec['f0_median'] + 1.645 * rec['f0_std']

    # Method 2: Use 16/84 percentiles scaled to 90%
    # This is approximate but uses the actual percentile spread
    # 68% CI -> 90% CI scale factor is 1.645/1 = 1.645
    Mf_spread_68 = (rec['Mf_84'] - rec['Mf_16']) / 2
    Mf_lo_scaled = rec['Mf_median'] - 1.645 * Mf_spread_68
    Mf_hi_scaled = rec['Mf_median'] + 1.645 * Mf_spread_68

    chi_spread_68 = (rec['chi_84'] - rec['chi_16']) / 2
    chi_lo_scaled = rec['chi_median'] - 1.645 * chi_spread_68
    chi_hi_scaled = rec['chi_median'] + 1.645 * chi_spread_68

    # Check coverage
    cover_Mf_gauss = Mf_lo_gauss <= Mf_inj <= Mf_hi_gauss
    cover_chi_gauss = chi_lo_gauss <= chi_inj <= chi_hi_gauss
    cover_f0_gauss = f0_lo_gauss <= f0_inj <= f0_hi_gauss

    cover_Mf_scaled = Mf_lo_scaled <= Mf_inj <= Mf_hi_scaled
    cover_chi_scaled = chi_lo_scaled <= chi_inj <= chi_hi_scaled

    print(f"\n{'='*60}")
    print(f"Run: {run_dir}")
    print(f"{'='*60}")

    print(f"\nInjected: Mf={Mf_inj:.2f}, chi={chi_inj:.3f}, f0={f0_inj:.1f}")
    print(f"Recovered: Mf={rec['Mf_median']:.2f}±{rec['Mf_std']:.2f}, "
          f"chi={rec['chi_median']:.3f}±{rec['chi_std']:.3f}, "
          f"f0={rec['f0_median']:.1f}±{rec['f0_std']:.1f}")

    print(f"\nMf 90% CI (Gaussian): [{Mf_lo_gauss:.2f}, {Mf_hi_gauss:.2f}]")
    print(f"Mf 90% CI (scaled):   [{Mf_lo_scaled:.2f}, {Mf_hi_scaled:.2f}]")
    print(f"Mf injected={Mf_inj:.2f} -> Cover(gauss)={cover_Mf_gauss}, Cover(scaled)={cover_Mf_scaled}")

    print(f"\nchi 90% CI (Gaussian): [{chi_lo_gauss:.3f}, {chi_hi_gauss:.3f}]")
    print(f"chi 90% CI (scaled):   [{chi_lo_scaled:.3f}, {chi_hi_scaled:.3f}]")
    print(f"chi injected={chi_inj:.3f} -> Cover(gauss)={cover_chi_gauss}, Cover(scaled)={cover_chi_scaled}")

    print(f"\nf0 90% CI (Gaussian): [{f0_lo_gauss:.1f}, {f0_hi_gauss:.1f}]")
    print(f"f0 injected={f0_inj:.1f} -> Cover(gauss)={cover_f0_gauss}")

    # Compute how many sigma away the injected value is
    Mf_z = abs(rec['Mf_median'] - Mf_inj) / rec['Mf_std'] if rec['Mf_std'] > 0 else np.inf
    chi_z = abs(rec['chi_median'] - chi_inj) / rec['chi_std'] if rec['chi_std'] > 0 else np.inf
    f0_z = abs(rec['f0_median'] - f0_inj) / rec['f0_std'] if rec['f0_std'] > 0 else np.inf

    print(f"\nZ-scores (|median - inj| / std):")
    print(f"  Mf:  z = {Mf_z:.2f} (expect z < 1.645 for 90% coverage)")
    print(f"  chi: z = {chi_z:.2f}")
    print(f"  f0:  z = {f0_z:.2f}")

    return {
        'Mf_z': Mf_z, 'chi_z': chi_z, 'f0_z': f0_z,
        'cover_Mf': cover_Mf_gauss, 'cover_chi': cover_chi_gauss, 'cover_f0': cover_f0_gauss
    }


def compute_actual_percentile_coverage(base_dir, snr):
    """Load actual posterior samples and compute true 5/95 percentiles."""
    import bilby

    snr_dir = Path(base_dir) / f"snr{snr}"
    if not snr_dir.exists():
        print(f"No directory for SNR={snr}")
        return

    # Pick first available run
    for noise_dir in sorted(snr_dir.iterdir()):
        result_file = noise_dir / "h0_injection_test_result.json"
        if result_file.exists():
            print(f"\nLoading full posterior from {result_file}...")
            result = bilby.result.read_in_result(filename=str(result_file))

            Mf_post = result.posterior["Mf"].values
            chi_post = result.posterior["chi"].values

            # Compute f0 for each sample
            f0_post = np.array([qnm_220_freq_tau(m, c)[0] for m, c in zip(Mf_post, chi_post)])

            # Get injected values from summary
            summary_file = noise_dir / "injection_summary.json"
            with open(summary_file) as f:
                summary = json.load(f)

            Mf_inj = summary['injected']['Mf']
            chi_inj = summary['injected']['chi']
            f0_inj = summary['injected']['f0']

            # Actual 5/95 percentiles
            Mf_5, Mf_95 = np.percentile(Mf_post, [5, 95])
            chi_5, chi_95 = np.percentile(chi_post, [5, 95])
            f0_5, f0_95 = np.percentile(f0_post, [5, 95])

            # Gaussian approximation
            Mf_lo_gauss = np.median(Mf_post) - 1.645 * np.std(Mf_post)
            Mf_hi_gauss = np.median(Mf_post) + 1.645 * np.std(Mf_post)

            chi_lo_gauss = np.median(chi_post) - 1.645 * np.std(chi_post)
            chi_hi_gauss = np.median(chi_post) + 1.645 * np.std(chi_post)

            f0_lo_gauss = np.median(f0_post) - 1.645 * np.std(f0_post)
            f0_hi_gauss = np.median(f0_post) + 1.645 * np.std(f0_post)

            print(f"\n{'='*60}")
            print(f"ACTUAL vs GAUSSIAN 90% CI COMPARISON")
            print(f"{'='*60}")

            print(f"\nMf:")
            print(f"  Injected: {Mf_inj:.2f}")
            print(f"  Actual 5/95:  [{Mf_5:.2f}, {Mf_95:.2f}]  width={Mf_95-Mf_5:.2f}")
            print(f"  Gaussian:     [{Mf_lo_gauss:.2f}, {Mf_hi_gauss:.2f}]  width={Mf_hi_gauss-Mf_lo_gauss:.2f}")
            print(f"  Cover(actual): {Mf_5 <= Mf_inj <= Mf_95}")
            print(f"  Cover(gauss):  {Mf_lo_gauss <= Mf_inj <= Mf_hi_gauss}")

            print(f"\nchi:")
            print(f"  Injected: {chi_inj:.3f}")
            print(f"  Actual 5/95:  [{chi_5:.3f}, {chi_95:.3f}]  width={chi_95-chi_5:.3f}")
            print(f"  Gaussian:     [{chi_lo_gauss:.3f}, {chi_hi_gauss:.3f}]  width={chi_hi_gauss-chi_lo_gauss:.3f}")
            print(f"  Cover(actual): {chi_5 <= chi_inj <= chi_95}")
            print(f"  Cover(gauss):  {chi_lo_gauss <= chi_inj <= chi_hi_gauss}")

            print(f"\nf0:")
            print(f"  Injected: {f0_inj:.1f}")
            print(f"  Actual 5/95:  [{f0_5:.1f}, {f0_95:.1f}]  width={f0_95-f0_5:.1f}")
            print(f"  Gaussian:     [{f0_lo_gauss:.1f}, {f0_hi_gauss:.1f}]  width={f0_hi_gauss-f0_lo_gauss:.1f}")
            print(f"  Cover(actual): {f0_5 <= f0_inj <= f0_95}")
            print(f"  Cover(gauss):  {f0_lo_gauss <= f0_inj <= f0_hi_gauss}")

            # Check for multimodality
            print(f"\nPosterior shape diagnostics:")
            print(f"  N samples: {len(Mf_post)}")
            print(f"  Mf skewness: {(np.mean(Mf_post) - np.median(Mf_post)) / np.std(Mf_post):.3f}")
            print(f"  chi skewness: {(np.mean(chi_post) - np.median(chi_post)) / np.std(chi_post):.3f}")
            print(f"  f0 skewness: {(np.mean(f0_post) - np.median(f0_post)) / np.std(f0_post):.3f}")

            return

    print(f"No result files found in {snr_dir}")


def check_all_runs_z_scores(base_dir, snr):
    """Check z-scores for all runs at given SNR."""
    snr_dir = Path(base_dir) / f"snr{snr}"
    if not snr_dir.exists():
        print(f"No directory for SNR={snr}")
        return

    z_Mf = []
    z_chi = []
    z_f0 = []

    for noise_dir in sorted(snr_dir.iterdir()):
        summary_file = noise_dir / "injection_summary.json"
        if not summary_file.exists():
            continue

        with open(summary_file) as f:
            summary = json.load(f)

        Mf_inj = summary['injected']['Mf']
        chi_inj = summary['injected']['chi']
        f0_inj = summary['injected']['f0']
        rec = summary['recovered']

        if rec['Mf_std'] > 0:
            z_Mf.append(abs(rec['Mf_median'] - Mf_inj) / rec['Mf_std'])
        if rec['chi_std'] > 0:
            z_chi.append(abs(rec['chi_median'] - chi_inj) / rec['chi_std'])
        if rec['f0_std'] > 0:
            z_f0.append(abs(rec['f0_median'] - f0_inj) / rec['f0_std'])

    print(f"\n{'='*60}")
    print(f"Z-SCORE DISTRIBUTION FOR SNR={snr} ({len(z_Mf)} runs)")
    print(f"{'='*60}")
    print(f"\nIf posteriors are well-calibrated, expect:")
    print(f"  - median(z) ~ 0.67 (half-normal distribution)")
    print(f"  - P(z < 1.645) ~ 0.90 (i.e., 90% coverage)")
    print(f"  - P(z < 2) ~ 0.95")

    if z_Mf:
        print(f"\nMf z-scores:")
        print(f"  median = {np.median(z_Mf):.2f}")
        print(f"  mean = {np.mean(z_Mf):.2f}")
        print(f"  P(z < 1.645) = {np.mean(np.array(z_Mf) < 1.645):.2%}")
        print(f"  P(z < 2) = {np.mean(np.array(z_Mf) < 2):.2%}")

    if z_chi:
        print(f"\nchi z-scores:")
        print(f"  median = {np.median(z_chi):.2f}")
        print(f"  mean = {np.mean(z_chi):.2f}")
        print(f"  P(z < 1.645) = {np.mean(np.array(z_chi) < 1.645):.2%}")
        print(f"  P(z < 2) = {np.mean(np.array(z_chi) < 2):.2%}")

    if z_f0:
        print(f"\nf0 z-scores:")
        print(f"  median = {np.median(z_f0):.2f}")
        print(f"  mean = {np.mean(z_f0):.2f}")
        print(f"  P(z < 1.645) = {np.mean(np.array(z_f0) < 1.645):.2%}")
        print(f"  P(z < 2) = {np.mean(np.array(z_f0) < 2):.2%}")

    return z_Mf, z_chi, z_f0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='out_ensemble_v2')
    parser.add_argument('--snr', type=int, default=30)
    parser.add_argument('--check-all', action='store_true', help='Check z-scores for all runs')
    parser.add_argument('--check-posteriors', action='store_true', help='Load actual posteriors for comparison')
    args = parser.parse_args()

    if args.check_all:
        for snr in [20, 30, 40, 50]:
            check_all_runs_z_scores(args.base_dir, snr)
    elif args.check_posteriors:
        compute_actual_percentile_coverage(args.base_dir, args.snr)
    else:
        # Check first available run at given SNR
        snr_dir = Path(args.base_dir) / f"snr{args.snr}"
        if snr_dir.exists():
            for noise_dir in sorted(snr_dir.iterdir())[:3]:  # Check first 3 runs
                check_single_run_coverage(noise_dir)
