#!/usr/bin/env python3
"""
Validate the greybody factor solver.

Runs 6 validation gates and produces diagnostic plots.
"""

import os
import sys
import time

import numpy as np

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR
from mscf_derived.greybody import (
    greybody_factor, greybody_spectrum, compute_and_cache_greybody,
)
from mscf_derived.validation import run_all_validations


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("GREYBODY FACTOR VALIDATION")
    print("=" * 70)

    # --- Run validation gates ---
    t0 = time.time()
    all_pass, gates = run_all_validations(l=2, verbose=True)
    dt_validate = time.time() - t0
    print(f"\nValidation time: {dt_validate:.1f} s")

    # --- Precompute and cache greybody spectrum ---
    print("\nPrecomputing greybody cache (l=2, odd)...")
    t0 = time.time()
    cached = compute_and_cache_greybody(l=2, parity='odd', n_grid=500)
    dt_cache = time.time() - t0
    print(f"  Cached {len(cached['Momega'])} points in {dt_cache:.1f} s")

    # Also cache Zerilli for isospectrality comparison
    print("Precomputing greybody cache (l=2, even)...")
    cached_z = compute_and_cache_greybody(l=2, parity='even', n_grid=500)

    # --- Print key values ---
    print("\n" + "=" * 70)
    print("KEY GREYBODY VALUES (l=2)")
    print("=" * 70)

    test_points = [0.1, 0.2, 0.3, 0.3737, 0.5, 0.7, 1.0, 2.0, 3.0]
    print(f"{'Mω':>8s}  {'|R_b|²':>10s}  {'|T_b|²':>10s}  {'|R_b|':>8s}  {'|T_b|':>8s}  {'flux err':>10s}")
    for om in test_points:
        res = greybody_factor(om, l=2)
        print(f"  {om:6.4f}  {res['Rb2']:10.6f}  {res['Tb2']:10.6f}  "
              f"{res['Rb']:8.4f}  {res['Tb']:8.4f}  {res['flux_error']:10.2e}")

    # --- Generate plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        Momega = cached['Momega']
        Rb2 = cached['Rb2']
        Tb2 = cached['Tb2']

        # (a) Greybody factors vs Mω
        ax = axes[0, 0]
        ax.plot(Momega, Rb2, 'b-', linewidth=2, label=r'$|R_b|^2$')
        ax.plot(Momega, Tb2, 'r-', linewidth=2, label=r'$|T_b|^2$')
        ax.plot(Momega, Rb2 + Tb2, 'k--', linewidth=1, alpha=0.5, label=r'$|R_b|^2 + |T_b|^2$')
        ax.axvline(0.3737, color='gray', linestyle=':', label='QNM (Mω=0.374)')
        ax.set_xlabel(r'$M\omega$')
        ax.set_ylabel('Greybody factor')
        ax.set_title('Regge-Wheeler l=2 greybody factors')
        ax.legend()
        ax.set_xlim(0, 2)
        ax.grid(True, alpha=0.3)

        # (b) Log-log showing power-law regime
        ax = axes[0, 1]
        ax.loglog(Momega, Tb2, 'r-', linewidth=2, label=r'$|T_b|^2$ (RW)')
        ax.loglog(cached_z['Momega'], cached_z['Tb2'], 'b--', linewidth=1.5,
                  label=r'$|T_b|^2$ (Zerilli)')
        # Reference slope (Mω)^6
        ref_om = np.logspace(-2, -0.5, 50)
        ref_Tb2 = (ref_om / 0.3)**6 * 0.59  # Normalized at QNM
        ax.loglog(ref_om, ref_Tb2, 'k:', alpha=0.5, label=r'$(M\omega)^6$ reference')
        ax.set_xlabel(r'$M\omega$')
        ax.set_ylabel(r'$|T_b|^2$')
        ax.set_title('Low-frequency power law & isospectrality')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (c) Echo amplitudes A_n(ω) for first 6 echoes
        ax = axes[1, 0]
        N_echo = 6
        for n in range(1, N_echo + 1):
            if n == 1:
                A_n = Tb2  # A_1 = T_b²
            else:
                A_n = Tb2 * Rb2**(n - 1)  # A_n = T_b² × R_b^{n-1}
            ax.plot(Momega, A_n, linewidth=1.5, label=f'$A_{n}$')
        ax.axvline(0.3737, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(r'$M\omega$')
        ax.set_ylabel('Echo amplitude')
        ax.set_title('Derived echo amplitudes $A_n(\\omega) = T_b^2 R_b^{n-1}$')
        ax.legend(ncol=2)
        ax.set_xlim(0, 1.5)
        ax.grid(True, alpha=0.3)

        # (d) Flux conservation error
        ax = axes[1, 1]
        flux_err = np.abs(1.0 - Rb2 - Tb2)
        ax.semilogy(Momega, flux_err, 'k-', linewidth=1)
        ax.set_xlabel(r'$M\omega$')
        ax.set_ylabel(r'$|1 - |R_b|^2 - |T_b|^2|$')
        ax.set_title('Flux conservation error')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(RESULTS_DIR, 'greybody_validation.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"\nPlot saved: {outfile}")

    except ImportError:
        print("\nmatplotlib not available — skipping plots")

    # --- Summary ---
    print("\n" + "=" * 70)
    verdict = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
