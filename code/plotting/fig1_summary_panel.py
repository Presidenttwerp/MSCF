#!/usr/bin/env python3
"""
Figure 1: Transfer function and Bogoliubov phase through the MSCF bounce.

Four-panel 2x2 figure:
  (a) Bogoliubov phase arg(alpha_k) across full k-range
  (b) CMB-range zoom showing phase flatness
  (c) Multipole phase variation relative to ell=2
  (d) Landau-Zener coupling ratio (coupled/uncoupled T^2)

Data loaded from precomputed results of coupled_mode_evolution.py and
phase_correlation_test.py.

References:
    MSCF v2.1.7, Section IX.E, Figure 1.

Output:
    paper/figures/fig1_summary_panel.png
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE_DIR = REPO_ROOT / "results" / "cmb_comparison"
COUPLED_DIR = REPO_ROOT / "results" / "transfer_functions"
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})


def main() -> None:
    # ================================================================
    # Load data
    # ================================================================

    # Phase function
    phase_dat = np.loadtxt(PHASE_DIR / "phase_function.dat")
    k_phase = phase_dat[:, 0]
    arg_alpha = phase_dat[:, 1]

    # Full results JSON
    with open(PHASE_DIR / "phase_results.json") as f:
        phase_json = json.load(f)

    # Uncoupled phases
    unc_modes = phase_json.get('mode_results_uncoupled', [])
    k_unc = np.array([m['k'] for m in unc_modes])
    arg_alpha_unc = np.array([m['arg_alpha'] for m in unc_modes])

    # Coupled evolution results
    with open(COUPLED_DIR / "full_coupled_results.json") as f:
        coupled_json = json.load(f)

    k_coupled = np.array([m['k'] for m in coupled_json['coupled']])
    T2_coupled = np.array([m['T2'] for m in coupled_json['coupled']])
    k_uncoupled = np.array([m['k'] for m in coupled_json['uncoupled']])
    T2_uncoupled = np.array([m['T2'] for m in coupled_json['uncoupled']])

    # Multipole phases
    mp_data = phase_json['multipole_phases']
    kappa_best = '0.000562'

    # ================================================================
    # Combined 2x2 summary panel
    # ================================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel (a): Phase function across full k range ---
    ax = axes[0, 0]
    ax.plot(k_phase, arg_alpha, 'C0-', lw=2, label='Coupled', zorder=3)
    ax.plot(k_unc, arg_alpha_unc, 'C1--', lw=1.5, label='Uncoupled', zorder=2)
    ax.axvspan(1e-8, 1e-4, alpha=0.07, color='steelblue', zorder=0)
    ax.axvspan(0.01, 10, alpha=0.07, color='firebrick', zorder=0)
    ax.axhline(-np.pi/2, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax.axhline(np.pi/2, color='gray', ls=':', lw=0.7, alpha=0.4)
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [Planck units]')
    ax.set_ylabel(r'$\arg(\alpha_k)$ [rad]')
    ax.set_title(r'(a) Bogoliubov phase $\arg(\alpha_k)$ vs wavenumber')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_xlim(1e-8, 20)
    ax.set_ylim(-3.5, 2.8)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.text(3e-7, 2.3, 'CMB scales', color='steelblue', fontsize=9,
            ha='center', fontweight='bold')
    ax.text(0.3, 2.3, r'$\Xi$-well region', color='firebrick', fontsize=9,
            ha='center', fontweight='bold')
    ax.annotate(r'$\Delta\phi \approx 330°$', xy=(0.1, -2.8),
                fontsize=9, color='firebrick', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='firebrick', alpha=0.7))

    # --- Panel (b): CMB zoom ---
    ax = axes[0, 1]
    cmb_mask = k_phase <= 1e-4
    ax.plot(k_phase[cmb_mask], arg_alpha[cmb_mask], 'C0-', lw=2.5,
            label='Coupled')
    unc_cmb = k_unc <= 1e-4
    ax.plot(k_unc[unc_cmb], arg_alpha_unc[unc_cmb], 'C1--', lw=1.8,
            label='Uncoupled')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [Planck units]')
    ax.set_ylabel(r'$\arg(\alpha_k)$ [rad]')
    ax.set_title(r'(b) CMB range: $\Delta\phi = 2.9\times10^{-3}$ rad ($0.17\degree$)')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, lw=0.5)
    y_cmb_coupled = arg_alpha[cmb_mask]
    ax.set_xlim(k_phase[cmb_mask][0], k_phase[cmb_mask][-1])
    textstr = (r'$\arg(\alpha_k) \approx -\pi/2$' + '\n'
               r'$\delta\phi_{\ell=2 \to \ell=30} = 0.008\degree$')
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='goldenrod', alpha=0.9))

    # --- Panel (c): Multipole phases relative to ell=2 ---
    ax = axes[1, 0]
    for kappa_str, label, color, ms in [
            ('0.000316', r'$\kappa = 3.16\times10^{-4}$', 'C0', 'o'),
            ('0.000562', r'$\kappa = 5.62\times10^{-4}$', 'C1', 's'),
            ('0.001', r'$\kappa = 1.00\times10^{-3}$', 'C2', '^')]:
        mp = mp_data[kappa_str]
        ells = sorted([int(e) for e in mp.keys()])
        phi_vals = np.array([mp[str(e)]['phi_arg_alpha'] for e in ells])
        phi_rel = phi_vals - phi_vals[0]
        ax.plot(ells, np.degrees(phi_rel) * 3600, f'{ms}-', markersize=3,
                label=label, color=color, lw=1.5)

    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$\phi_\ell - \phi_{\ell=2}$  [arcsec]')
    ax.set_title(r'(c) Multipole phase variation')
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(1, 31)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.annotate(r'Planck Q-O anomaly: $\sim 10\degree = 36\,000$ arcsec',
                xy=(16, -5), fontsize=8, color='red', ha='center',
                style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='red', alpha=0.8))
    ax.text(28, -42, r'$\times 700$ to Planck',
            fontsize=7, color='red', ha='right', va='top', style='italic')

    # --- Panel (d): Coupled/uncoupled T^2 ratio ---
    ax = axes[1, 1]
    n = min(len(k_coupled), len(k_uncoupled))
    ratio = T2_coupled[:n] / T2_uncoupled[:n]
    ax.plot(k_coupled[:n], ratio, 'C3o-', ms=5, lw=2, zorder=3)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, zorder=1)
    ax.fill_between([k_coupled[0], k_coupled[n-1]], [1.0, 1.0],
                    [0.001, 0.001], alpha=0.03, color='C3', zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$ [Planck units]')
    ax.set_ylabel(r'$T^2_{\rm coupled}\, /\, T^2_{\rm uncoupled}$')
    ax.set_title(r'(d) Landau-Zener coupling ratio')
    ax.set_ylim(1e-3, 2)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.axhline(ratio[0], color='C0', ls=':', lw=0.8, alpha=0.4)
    ax.annotate(f'Low-$k$ plateau: {ratio[0]:.3f}\n(32% suppression)',
                xy=(k_coupled[2], ratio[2]), xytext=(0.03, 0.12),
                arrowprops=dict(arrowstyle='->', color='C3', lw=1.2),
                fontsize=8, color='C3',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='C3',
                          alpha=0.8))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(OUTPUT_DIR / "fig1_summary_panel.png")
    print(f"Saved: {OUTPUT_DIR / 'fig1_summary_panel.png'}")
    plt.close(fig)


if __name__ == '__main__':
    main()
