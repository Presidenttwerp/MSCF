#!/usr/bin/env python3
"""
Figure 3: Echo amplitude inversion, the MSCF discriminant.

Generic ECO models (single reflective surface) produce monotonically
decaying echo trains: A1 > A2 > A3 > ...

MSCF (two-surface interior) produces an inverted pattern: A2 > A1,
because A1 is a weak horizon reflection while A2 is a strong barrier
reflection after full cavity transit.

References:
    MSCF v2.1.7, Section XI.C, Figure 3.

Output:
    paper/figures/fig3_echo_inversion.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.0,
    'mathtext.fontset': 'cm',
})


def main() -> None:
    N_echoes = 8
    n = np.arange(1, N_echoes + 1)

    # Generic single-surface ECO: monotonic geometric decay
    R_generic = 0.65
    A_generic_0 = 1.0
    A_generic = A_generic_0 * R_generic ** (n - 1)

    # MSCF two-surface (Theorem 11.1: T = 1 - R1, amplitude convention)
    R_horizon = 0.15
    T_horizon = 1 - R_horizon  # amplitude transmission (paper Eq. 34)
    R_barrier = 1.0            # total reflection at inversion barrier

    A_mscf = np.zeros(N_echoes)
    A_mscf[0] = R_horizon                      # A1: horizon reflection
    A_mscf[1] = T_horizon**2 * R_barrier       # A2: transmit-reflect-transmit
    for i in range(2, N_echoes):
        A_mscf[i] = A_mscf[1] * R_horizon ** (i - 1)

    A_generic = A_generic / A_generic[0]
    A_mscf = A_mscf / A_mscf.max()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    C_GEN = '#888888'
    C_MSCF = '#D94A4A'
    C_MSCF_DARK = '#A03030'

    ax.plot(n, A_generic, 'o--', color=C_GEN, ms=10, lw=1.8,
            markeredgecolor='white', markeredgewidth=1.2,
            label='Generic ECO (single surface)', zorder=5)

    ax.plot(n, A_mscf, 's-', color=C_MSCF, ms=11, lw=2.2,
            markeredgecolor='white', markeredgewidth=1.2,
            label='MSCF (two surfaces)', zorder=6)

    ax.annotate('',
                xy=(2, A_mscf[1] + 0.02), xytext=(1, A_mscf[0] + 0.02),
                arrowprops=dict(arrowstyle='->', color=C_MSCF_DARK, lw=2.5,
                                connectionstyle='arc3,rad=-0.3'),
                zorder=10)
    ax.text(1.5, 0.52, r'$A_2 > A_1$' + '\ninversion',
            fontsize=12, ha='center', va='center', color=C_MSCF_DARK,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF0F0', ec=C_MSCF_DARK,
                      alpha=0.95, lw=1.5))

    ax.text(1.3, A_mscf[0] + 0.04, r'$A_1$' + ' (horizon reflection)',
            fontsize=8, ha='left', va='bottom', color=C_MSCF_DARK, zorder=20,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.text(2.55, A_mscf[1] - 0.02, r'$A_2$' + '\n(barrier\nreflection)',
            fontsize=8, ha='left', va='top', color=C_MSCF_DARK, zorder=20,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))

    ax.text(3.5, A_generic[2] + 0.05, 'monotonic\ndecay',
            fontsize=9, ha='center', va='bottom', color=C_GEN,
            style='italic', zorder=20)

    ax.set_xlabel('Echo number  ' + r'$n$', fontsize=14)
    ax.set_ylabel('Relative amplitude  ' + r'$A_n / A_{\max}$', fontsize=14)
    ax.set_xlim(0.4, N_echoes + 0.6)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(n)
    ax.set_xticklabels([r'$%d$' % i for i in n])

    leg = ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    leg.set_zorder(20)
    ax.grid(True, alpha=0.15, lw=0.5)

    ax.text(0.5, 0.02,
            r'If $A_2 > A_1$: two-surface interior (MSCF)'
            r'$\qquad$'
            r'If $A_1 > A_2 > A_3$: single-surface ECO',
            transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='#F8F8F8', ec='#888',
                      alpha=0.95, lw=1.0))

    fig.savefig(OUTPUT_DIR / "fig3_echo_inversion.png")
    print(f"Saved: {OUTPUT_DIR / 'fig3_echo_inversion.png'}")
    plt.close(fig)


if __name__ == '__main__':
    main()
