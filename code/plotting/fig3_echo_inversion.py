#!/usr/bin/env python3
"""
Figure 3: Echo amplitude trains — MSCF discriminant.

Three curves:
1. Generic ECO (single reflective surface): monotonic, gradual decay
2. Classical MSCF (derived): strong first echo, steep drop by factor ~11 per echo
   (primary prediction — barrier nearly transparent at QNM, |R_b|^2 ~ 0.008)
3. Quantum MSCF (dashed): possible inversion if R_1 > 0 at horizon
   (secondary possibility — horizon acquires reflectivity from quantum corrections)

References:
    MSCF v2.1.7, Section XI.C, Figure 3.
    Derived greybody factors: mscf_derived_echo/ pipeline results.

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

    # ---------------------------------------------------------------
    # 1. Generic single-surface ECO: monotonic geometric decay
    #    R ~ 0.65 typical for firewall/fuzzball models
    # ---------------------------------------------------------------
    R_generic = 0.65
    A_generic = R_generic ** (n - 1)

    # ---------------------------------------------------------------
    # 2. Classical MSCF (derived): barrier nearly transparent at QNM
    #    From greybody computation: |T_b| = 0.996, |R_b| = 0.087
    #    a_n = |T_b| * |R_b|^{n-1}
    #    Ratio A1/A2 ~ 11.5
    # ---------------------------------------------------------------
    Tb = 0.996   # barrier transmissivity at QNM
    Rb = 0.087   # barrier reflectivity at QNM
    A_classical = Tb * Rb ** (n - 1)

    # ---------------------------------------------------------------
    # 3. Quantum MSCF (dashed): horizon acquires reflectivity R_1 > 0
    #    from quantum corrections. This restores the inversion signature.
    #    A_1 = R_1 (horizon reflection, weak)
    #    A_2 = T_1^2 * R_barrier (transmit, reflect off barrier, transmit back)
    #    A_n>2 = A_2 * R_1^{n-2}
    # ---------------------------------------------------------------
    R1_quantum = 0.15  # illustrative horizon reflectivity
    T1_quantum = 1.0 - R1_quantum  # amplitude transmission
    A_quantum = np.zeros(N_echoes)
    A_quantum[0] = R1_quantum                        # A1: horizon reflection
    A_quantum[1] = T1_quantum**2                     # A2: full cavity transit
    for i in range(2, N_echoes):
        A_quantum[i] = A_quantum[1] * R1_quantum ** (i - 1)

    # Normalize all to their own max for comparison
    A_generic_norm = A_generic / A_generic[0]
    A_classical_norm = A_classical / A_classical[0]
    A_quantum_norm = A_quantum / A_quantum.max()

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))

    C_GEN = '#888888'
    C_CLASS = '#2266AA'
    C_QUANT = '#D94A4A'

    # Generic ECO
    ax.plot(n, A_generic_norm, 'o--', color=C_GEN, ms=10, lw=1.8,
            markeredgecolor='white', markeredgewidth=1.2,
            label='Generic ECO (single surface)', zorder=5)

    # Classical MSCF — primary prediction (solid, prominent)
    ax.plot(n, A_classical_norm, 's-', color=C_CLASS, ms=11, lw=2.5,
            markeredgecolor='white', markeredgewidth=1.2,
            label=r'Classical MSCF ($|R_b|^2 = 0.008$)', zorder=7)

    # Quantum MSCF — secondary possibility (dashed)
    ax.plot(n, A_quantum_norm, 'D--', color=C_QUANT, ms=9, lw=1.8,
            markeredgecolor='white', markeredgewidth=1.2,
            alpha=0.75,
            label=r'Quantum MSCF ($R_1 > 0$, if present)', zorder=6)

    # Annotations for classical MSCF
    ax.annotate(r'$A_1 / A_2 \approx 11$',
                xy=(2, A_classical_norm[1]),
                xytext=(3.2, 0.38),
                arrowprops=dict(arrowstyle='->', color=C_CLASS, lw=1.5,
                                connectionstyle='arc3,rad=0.2'),
                fontsize=11, ha='center', va='center', color=C_CLASS,
                fontweight='bold', zorder=20,
                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F0FF', ec=C_CLASS,
                          alpha=0.95, lw=1.2))

    ax.annotate('steep drop\n(barrier transparent\nat QNM)',
                xy=(3, A_classical_norm[2]),
                xytext=(4.5, 0.20),
                arrowprops=dict(arrowstyle='->', color=C_CLASS, lw=1.0,
                                connectionstyle='arc3,rad=-0.2'),
                fontsize=9, ha='center', va='center', color=C_CLASS,
                style='italic', zorder=20)

    # Annotation for quantum MSCF inversion
    ax.annotate(r'$A_2 > A_1$',
                xy=(1.5, (A_quantum_norm[0] + A_quantum_norm[1]) / 2),
                xytext=(1.1, 0.55),
                arrowprops=dict(arrowstyle='->', color=C_QUANT, lw=1.2,
                                connectionstyle='arc3,rad=0.3'),
                fontsize=10, ha='center', va='center', color=C_QUANT,
                fontweight='bold', alpha=0.75, zorder=20,
                bbox=dict(boxstyle='round,pad=0.25', fc='#FFF0F0', ec=C_QUANT,
                          alpha=0.7, lw=1.0))

    # Generic ECO label
    ax.text(4.5, A_generic_norm[3] + 0.05, 'gradual\ndecay',
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

    # Decision rule box at bottom
    ax.text(0.5, 0.02,
            'Classical MSCF: strong $A_1$, steep decay (primary prediction)'
            r'$\qquad$'
            'Quantum: inversion $A_2 > A_1$ (if $R_1 > 0$)',
            transform=ax.transAxes,
            fontsize=8.5, ha='center', va='bottom', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='#F8F8F8', ec='#888',
                      alpha=0.95, lw=1.0))

    fig.savefig(OUTPUT_DIR / "fig3_echo_inversion.png")
    print(f"Saved: {OUTPUT_DIR / 'fig3_echo_inversion.png'}")
    plt.close(fig)


if __name__ == '__main__':
    main()
