#!/usr/bin/env python
"""Aggregate population ensemble results and compute pass fractions + bias distributions."""

import json
import os
import numpy as np
from pathlib import Path

def load_results(base_dir="out_ensemble"):
    """Load all injection_summary.json files from ensemble runs."""
    results = {20: [], 30: [], 40: [], 50: []}

    for snr in [20, 30, 40, 50]:
        snr_dir = Path(base_dir) / f"snr{snr}"
        if not snr_dir.exists():
            continue

        for noise_dir in sorted(snr_dir.iterdir()):
            summary_file = noise_dir / "injection_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    data['noise_seed'] = noise_dir.name
                    results[snr].append(data)

    return results

def compute_statistics(results):
    """Compute pass fractions and bias distributions for each SNR."""
    stats = {}

    for snr, runs in results.items():
        if not runs:
            stats[snr] = None
            continue

        n_runs = len(runs)

        # Pass fractions
        n_pass_Mf = sum(1 for r in runs if r['pass_conditions']['Mf_within_2sigma'])
        n_pass_chi = sum(1 for r in runs if r['pass_conditions']['chi_within_2sigma'])
        n_pass_f0 = sum(1 for r in runs if r['pass_conditions']['f0_within_20Hz'])
        n_pass_overall = sum(1 for r in runs if r['pass_conditions']['overall_pass'])

        # Injected values (should be same for all)
        Mf_inj = runs[0]['injected']['Mf']
        chi_inj = runs[0]['injected']['chi']
        f0_inj = runs[0]['injected']['f0']

        # Recovered values
        Mf_recovered = [r['recovered']['Mf_median'] for r in runs]
        chi_recovered = [r['recovered']['chi_median'] for r in runs]
        f0_recovered = [r['recovered']['f0_median'] for r in runs]

        # Biases
        Mf_bias = [m - Mf_inj for m in Mf_recovered]
        chi_bias = [c - chi_inj for c in chi_recovered]
        f0_bias = [f - f0_inj for f in f0_recovered]

        # Fractional biases
        Mf_frac_bias = [(m - Mf_inj) / Mf_inj * 100 for m in Mf_recovered]
        chi_frac_bias = [(c - chi_inj) / chi_inj * 100 for c in chi_recovered]
        f0_frac_bias = [(f - f0_inj) / f0_inj * 100 for f in f0_recovered]

        stats[snr] = {
            'n_runs': n_runs,
            'pass_fraction': {
                'Mf': n_pass_Mf / n_runs,
                'chi': n_pass_chi / n_runs,
                'f0': n_pass_f0 / n_runs,
                'overall': n_pass_overall / n_runs,
            },
            'n_pass': {
                'Mf': n_pass_Mf,
                'chi': n_pass_chi,
                'f0': n_pass_f0,
                'overall': n_pass_overall,
            },
            'injected': {
                'Mf': Mf_inj,
                'chi': chi_inj,
                'f0': f0_inj,
            },
            'bias': {
                'Mf': {'mean': np.mean(Mf_bias), 'std': np.std(Mf_bias), 'values': Mf_bias},
                'chi': {'mean': np.mean(chi_bias), 'std': np.std(chi_bias), 'values': chi_bias},
                'f0': {'mean': np.mean(f0_bias), 'std': np.std(f0_bias), 'values': f0_bias},
            },
            'fractional_bias_percent': {
                'Mf': {'mean': np.mean(Mf_frac_bias), 'std': np.std(Mf_frac_bias)},
                'chi': {'mean': np.mean(chi_frac_bias), 'std': np.std(chi_frac_bias)},
                'f0': {'mean': np.mean(f0_frac_bias), 'std': np.std(f0_frac_bias)},
            },
            'recovered_median': {
                'Mf': {'mean': np.mean(Mf_recovered), 'std': np.std(Mf_recovered)},
                'chi': {'mean': np.mean(chi_recovered), 'std': np.std(chi_recovered)},
                'f0': {'mean': np.mean(f0_recovered), 'std': np.std(f0_recovered)},
            }
        }

    return stats

def print_summary(stats):
    """Print human-readable summary."""
    print("=" * 80)
    print("POPULATION ENSEMBLE STUDY RESULTS")
    print("Injection: Mf=67.8 M☉, χ=0.68, f0=251 Hz")
    print("=" * 80)

    # Pass fraction table
    print("\n📊 PASS FRACTIONS BY SNR")
    print("-" * 60)
    print(f"{'SNR':>5} | {'N runs':>7} | {'Mf':>8} | {'χ':>8} | {'f0':>8} | {'Overall':>8}")
    print("-" * 60)

    for snr in [20, 30, 40, 50]:
        if stats[snr] is None:
            print(f"{snr:>5} | {'N/A':>7}")
            continue
        s = stats[snr]
        pf = s['pass_fraction']
        print(f"{snr:>5} | {s['n_runs']:>7} | {pf['Mf']*100:>7.1f}% | {pf['chi']*100:>7.1f}% | {pf['f0']*100:>7.1f}% | {pf['overall']*100:>7.1f}%")
    print("-" * 60)

    # Bias table
    print("\n📐 PARAMETER BIASES (recovered - injected)")
    print("-" * 70)
    print(f"{'SNR':>5} | {'Mf bias':>12} | {'χ bias':>12} | {'f0 bias (Hz)':>14}")
    print("-" * 70)

    for snr in [20, 30, 40, 50]:
        if stats[snr] is None:
            continue
        s = stats[snr]
        b = s['bias']
        print(f"{snr:>5} | {b['Mf']['mean']:>+6.1f}±{b['Mf']['std']:<5.1f} | {b['chi']['mean']:>+6.3f}±{b['chi']['std']:<5.3f} | {b['f0']['mean']:>+7.1f}±{b['f0']['std']:<5.1f}")
    print("-" * 70)

    # Fractional bias
    print("\n📈 FRACTIONAL BIASES (% of injected value)")
    print("-" * 60)
    print(f"{'SNR':>5} | {'Mf %':>12} | {'χ %':>12} | {'f0 %':>12}")
    print("-" * 60)

    for snr in [20, 30, 40, 50]:
        if stats[snr] is None:
            continue
        s = stats[snr]
        fb = s['fractional_bias_percent']
        print(f"{snr:>5} | {fb['Mf']['mean']:>+6.1f}±{fb['Mf']['std']:<5.1f} | {fb['chi']['mean']:>+6.1f}±{fb['chi']['std']:<5.1f} | {fb['f0']['mean']:>+6.1f}±{fb['f0']['std']:<5.1f}")
    print("-" * 60)

    print("\n📋 INTERPRETATION:")
    print("- Pass criteria: Mf and χ within 2σ, f0 within 20Hz")
    print("- Low pass fraction indicates noise-realization sensitivity, NOT a bug")
    print("- Consistent bias direction across SNRs may indicate degeneracy structure")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='out_ensemble', help='Base directory for ensemble results')
    args = parser.parse_args()

    results = load_results(args.base_dir)

    # Report counts
    print("\nLoaded results:")
    for snr in [20, 30, 40, 50]:
        print(f"  SNR={snr}: {len(results[snr])} runs")

    stats = compute_statistics(results)
    print_summary(stats)

    # Save detailed stats
    output_file = f"{args.base_dir}/ensemble_summary.json"

    # Convert numpy arrays to lists for JSON serialization
    stats_json = {}
    for snr, s in stats.items():
        if s is None:
            stats_json[snr] = None
            continue
        stats_json[snr] = {
            'n_runs': s['n_runs'],
            'pass_fraction': s['pass_fraction'],
            'n_pass': s['n_pass'],
            'injected': s['injected'],
            'bias': {
                param: {'mean': float(b['mean']), 'std': float(b['std'])}
                for param, b in s['bias'].items()
            },
            'fractional_bias_percent': {
                param: {'mean': float(b['mean']), 'std': float(b['std'])}
                for param, b in s['fractional_bias_percent'].items()
            },
            'recovered_median': {
                param: {'mean': float(b['mean']), 'std': float(b['std'])}
                for param, b in s['recovered_median'].items()
            }
        }

    with open(output_file, 'w') as f:
        json.dump(stats_json, f, indent=2)
    print(f"\nSaved detailed stats to {output_file}")
