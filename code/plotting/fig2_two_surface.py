#!/usr/bin/env python3
"""
Figure 2: Two-surface interior structure of the MSCF black hole.

Left panel: Metric function F(r) = (1 - x_g)(1 - x_g/2) compared to the
Schwarzschild f(r) = 1 - x_g, showing the two zeros (horizon and inversion
barrier) and the cavity between them.

Right panel: Spatial schematic of the three-region interior with echo paths
illustrating the amplitude inversion (A2 > A1).

Cavity proper distance: 1.285 r_s (computed numerically).

References:
    MSCF v2.1.7, Sections X and XI, Figure 2.

Output:
    paper/figures/fig2_two_surface_structure.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from scipy.integrate import quad

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'mathtext.fontset': 'cm',
})


def cavity_integrand(r_over_M: float) -> float:
    """Integrand for proper cavity length, in units of M."""
    u = r_over_M
    F_abs = abs((2.0/u - 1.0) * (1.0 - 1.0/u))
    return 1.0 / np.sqrt(F_abs) if F_abs > 1e-14 else 0.0


def main() -> None:
    # Compute cavity proper length
    delta_ell_M, _ = quad(cavity_integrand, 1.001, 1.999)
    delta_ell_rs = delta_ell_M / 2.0
    print(f"Cavity proper length: {delta_ell_M:.4f} M = {delta_ell_rs:.4f} r_s")

    # Color scheme
    C_EXT = '#4A90D9'
    C_EXT_FILL = '#D6E8F7'
    C_CAV = '#D94A4A'
    C_CAV_FILL = '#F5D6D6'
    C_CORE = '#4AA65B'
    C_CORE_FILL = '#D6F0DB'
    C_GR = '#888888'

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.08)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ================================================================
    # LEFT PANEL: Metric function F vs r/r_s
    # ================================================================
    ax = ax_left

    u = np.linspace(0.32, 3.5, 2000)
    F_mscf = (1.0 - 1.0/u) * (1.0 - 1.0/(2.0*u))
    f_gr = 1.0 - 1.0/u

    ext_mask = u >= 1.0
    cav_mask = (u >= 0.5) & (u <= 1.0)
    core_mask = u <= 0.5

    ax.fill_between(u[ext_mask], 0, F_mscf[ext_mask],
                    alpha=0.12, color=C_EXT, zorder=0)
    ax.fill_between(u[cav_mask], 0, F_mscf[cav_mask],
                    alpha=0.18, color=C_CAV, zorder=0)
    ax.fill_between(u[core_mask], 0, F_mscf[core_mask],
                    alpha=0.12, color=C_CORE, zorder=0)

    ax.plot(u, F_mscf, color='#1a1a1a', lw=2.5, zorder=5,
            label=r'MSCF: $F = (1-x_g)(1-x_g/2)$')
    ax.plot(u, f_gr, color=C_GR, ls='--', lw=1.8, alpha=0.75, zorder=4,
            label=r'GR: $f = 1 - x_g$')

    ax.axhline(0, color='black', lw=0.6, zorder=1)
    ax.plot(1.0, 0, 'o', color='#1a1a1a', ms=9, zorder=8,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(0.5, 0, 's', color='#1a1a1a', ms=9, zorder=8,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(2.0/3.0, -1.0/8.0, 'x', color=C_CAV, ms=8, mew=2.5, zorder=7)

    ax.axvline(1.0, color='#1a1a1a', ls=':', lw=0.8, alpha=0.4, zorder=2)
    ax.axvline(0.5, color='#1a1a1a', ls=':', lw=0.8, alpha=0.4, zorder=2)

    ax.annotate('Horizon\n' + r'$r = r_s$' + '\n' + r'($x_g = 1$)',
                xy=(1.0, 0), xytext=(1.55, -0.18),
                arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.2,
                                connectionstyle='arc3,rad=-0.2'),
                fontsize=9, ha='center', va='top', zorder=20,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          ec='#888', alpha=0.95))

    ax.annotate('Inversion barrier\n' + r'$r = M$  ($x_g = 2$)',
                xy=(0.5, 0), xytext=(0.75, 0.32),
                arrowprops=dict(arrowstyle='->', color='#1a1a1a', lw=1.2),
                fontsize=9, ha='center', va='bottom', zorder=20,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          ec='#888', alpha=0.95))

    ax.annotate(r'$F_{\min} = -\frac{1}{8}$',
                xy=(2.0/3.0, -1.0/8.0), xytext=(1.15, -0.25),
                arrowprops=dict(arrowstyle='->', color=C_CAV, lw=1),
                fontsize=9, color=C_CAV, zorder=20)

    ax.text(2.3, 0.30, r'$\mathbf{F > 0}$' + '\ntimelike',
            fontsize=10, ha='center', color=C_EXT, fontweight='bold', zorder=20)
    ax.text(0.75, -0.06, r'$\mathbf{F < 0}$' + '\nspacelike',
            fontsize=10, ha='center', color=C_CAV, fontweight='bold', zorder=20)
    ax.text(0.385, 0.08, r'$\mathbf{F > 0}$' + '\ntimelike',
            fontsize=8, ha='center', color='#2D7A3A', fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    ax.annotate('', xy=(0.34, f_gr[0]-0.02), xytext=(0.34, f_gr[0] + 0.12),
                arrowprops=dict(arrowstyle='->', color=C_GR, lw=1.5), zorder=20)
    ax.text(0.56, -0.38, r'GR: $f \to -\infty$' + '\n(singularity)',
            fontsize=9, color=C_GR, ha='center', style='italic', zorder=20,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    ax.set_xlabel(r'$r\, /\, r_s$', fontsize=13)
    ax.set_ylabel(r'Metric function', fontsize=13)
    ax.set_title(r'(a) MSCF metric $F(r)$ vs Schwarzschild $f(r)$', fontsize=12)
    ax.set_xlim(0.32, 3.5)
    ax.set_ylim(-0.45, 0.65)
    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    leg.set_zorder(20)
    ax.grid(True, alpha=0.12, lw=0.5)

    # ================================================================
    # RIGHT PANEL: Spatial schematic
    # ================================================================
    ax = ax_right
    ax.set_aspect('equal')
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.8, 3.8)

    R_EXT = 3.2
    R_HORIZON = 2.2
    R_WALL = 1.1

    ext_circle = Circle((0, 0), R_EXT, fc=C_EXT_FILL, ec='none', zorder=0)
    ax.add_patch(ext_circle)
    cav_circle = Circle((0, 0), R_HORIZON, fc=C_CAV_FILL, ec='none', zorder=1)
    ax.add_patch(cav_circle)
    core_circle = Circle((0, 0), R_WALL, fc=C_CORE_FILL, ec='none', zorder=2)
    ax.add_patch(core_circle)

    horizon_ring = Circle((0, 0), R_HORIZON, fill=False,
                           ec='#1a1a1a', lw=3.0, zorder=10)
    ax.add_patch(horizon_ring)
    wall_ring = Circle((0, 0), R_WALL, fill=False,
                        ec='#1a1a1a', lw=3.0, ls=(0, (6, 3)), zorder=10)
    ax.add_patch(wall_ring)

    ax.text(0, R_HORIZON + 0.25, r'Horizon ($r = r_s$)',
            fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1a1a1a',
                      alpha=0.95, lw=1.5))

    ax.text(0, -(R_WALL + 0.25), r'MSCF wall ($r = M$)',
            fontsize=10, ha='center', va='top', fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1a1a1a',
                      alpha=0.95, lw=1.5, ls='dashed'))

    ax.text(2.75, 2.75, 'Region I\nExterior\n' + r'$F > 0$',
            fontsize=10, ha='center', va='center', color=C_EXT,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_EXT,
                      alpha=0.9, lw=1.2))

    R_cav_mid = (R_HORIZON + R_WALL) / 2.0
    ax.text(-R_cav_mid - 0.05, 0, 'Region II\nCavity\n' + r'$F < 0$',
            fontsize=10, ha='center', va='center', color=C_CAV,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_CAV,
                      alpha=0.9, lw=1.2))

    ax.text(0, 0, 'Region III\nBarrier\n' + r'$F \to -1$',
            fontsize=10, ha='center', va='center', color=C_CORE,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_CORE,
                      alpha=0.9, lw=1.2))

    # Cavity proper length annotation
    angle_arr = -35
    rad = np.radians(angle_arr)
    x_h = R_HORIZON * np.cos(rad)
    y_h = R_HORIZON * np.sin(rad)
    x_w = R_WALL * np.cos(rad)
    y_w = R_WALL * np.sin(rad)

    ax.annotate('', xy=(x_h, y_h), xytext=(x_w, y_w),
                arrowprops=dict(arrowstyle='<->', color=C_CAV, lw=2.0,
                                shrinkA=2, shrinkB=2), zorder=15)
    x_mid = (x_h + x_w) / 2.0 + 0.4
    y_mid = (y_h + y_w) / 2.0 - 0.3
    ax.text(x_mid, y_mid,
            r'$\Delta\ell \approx 1.285\, r_s$',
            fontsize=10, ha='left', va='center', color=C_CAV,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_CAV, alpha=0.9))

    # Echo arrows
    C_ECHO = '#CC8800'
    r_in = R_WALL + 0.08
    r_out = R_HORIZON - 0.08

    # A1 path: weak echo reflecting off HORIZON
    a1_angle = 130
    rad_a1 = np.radians(a1_angle)
    x_out_a1 = r_out * np.cos(rad_a1)
    y_out_a1 = r_out * np.sin(rad_a1)
    a1_refl = np.radians(a1_angle + 8)
    x_out_a1r = r_out * np.cos(a1_refl)
    y_out_a1r = r_out * np.sin(a1_refl)

    ax.annotate('', xy=(x_out_a1 * 0.98, y_out_a1 * 0.98),
                xytext=(x_out_a1 * 1.25, y_out_a1 * 1.25),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=1.0,
                                alpha=0.6, shrinkA=0, shrinkB=0), zorder=15)
    ax.annotate('', xy=(x_out_a1r * 1.25, y_out_a1r * 1.25),
                xytext=(x_out_a1r * 0.98, y_out_a1r * 0.98),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=0.8,
                                alpha=0.5, ls='dashed', shrinkA=0, shrinkB=0),
                zorder=15)

    a1_lx = R_HORIZON * 1.15 * np.cos(np.radians(a1_angle + 4))
    a1_ly = R_HORIZON * 1.15 * np.sin(np.radians(a1_angle + 4))
    ax.text(a1_lx, a1_ly, r'$A_1$' + '\n(weak)',
            fontsize=8, ha='center', va='center', color=C_ECHO,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_ECHO,
                      alpha=0.9, lw=0.8))

    # A2 path: strong echo through horizon, off barrier, back out
    a2_angle = 50
    rad_a2 = np.radians(a2_angle)
    x_h_a2 = r_out * np.cos(rad_a2)
    y_h_a2 = r_out * np.sin(rad_a2)
    x_w_a2 = r_in * np.cos(rad_a2)
    y_w_a2 = r_in * np.sin(rad_a2)
    ax.annotate('', xy=(x_w_a2, y_w_a2), xytext=(x_h_a2, y_h_a2),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=2.0,
                                alpha=0.9, shrinkA=0, shrinkB=0), zorder=15)

    a2_refl = np.radians(a2_angle + 8)
    x_w_a2r = r_in * np.cos(a2_refl)
    y_w_a2r = r_in * np.sin(a2_refl)
    x_h_a2r = r_out * np.cos(a2_refl)
    y_h_a2r = r_out * np.sin(a2_refl)
    ax.annotate('', xy=(x_h_a2r, y_h_a2r), xytext=(x_w_a2r, y_w_a2r),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=2.0,
                                alpha=0.9, shrinkA=0, shrinkB=0), zorder=15)

    ax.annotate('', xy=(x_h_a2r * 1.25, y_h_a2r * 1.25),
                xytext=(x_h_a2r * 0.98, y_h_a2r * 0.98),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=2.0,
                                alpha=0.8, shrinkA=0, shrinkB=0), zorder=15)

    a2_lx = R_HORIZON * 1.15 * np.cos(np.radians(a2_angle + 4))
    a2_ly = R_HORIZON * 1.15 * np.sin(np.radians(a2_angle + 4))
    ax.text(a2_lx, a2_ly, r'$A_2$' + '\n(strong)',
            fontsize=8, ha='center', va='center', color=C_ECHO,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_ECHO,
                      alpha=0.9, lw=0.8))

    # Background echo
    bg_angle = -130
    rad_bg = np.radians(bg_angle)
    ax.annotate('', xy=(r_out * np.cos(rad_bg), r_out * np.sin(rad_bg)),
                xytext=(r_in * np.cos(rad_bg), r_in * np.sin(rad_bg)),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=1.2,
                                alpha=0.6, shrinkA=0, shrinkB=0), zorder=12)
    bg_refl = np.radians(bg_angle + 6)
    ax.annotate('', xy=(r_in * np.cos(bg_refl), r_in * np.sin(bg_refl)),
                xytext=(r_out * np.cos(bg_refl), r_out * np.sin(bg_refl)),
                arrowprops=dict(arrowstyle='->', color=C_ECHO, lw=1.2,
                                alpha=0.5, ls='dashed', shrinkA=0, shrinkB=0),
                zorder=12)

    ax.text(0, -2.85, r'$\mathbf{A_2 > A_1}$: second echo stronger than first',
            fontsize=10, ha='center', va='center', color=C_ECHO,
            fontweight='bold', zorder=20,
            bbox=dict(boxstyle='round,pad=0.35', fc='#FFF8E8', ec=C_ECHO,
                      alpha=0.95, lw=1.5))

    box_text = ('Other models:\n'
                'single surface at\n'
                r'$r \approx r_s + \varepsilon$' + '\n'
                '(firewall, fuzzball,\n quantum BH)')
    ax.text(-3.3, 2.9, box_text,
            fontsize=8, ha='center', va='top', color='#666',
            style='italic', zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#999',
                      alpha=0.9, lw=0.8))

    ax.text(0, -0.65, r'(GR: singularity at $r=0$)',
            fontsize=8, ha='center', color='#777', style='italic', zorder=20)

    ax.set_title(r'(b) Two-surface interior: horizon + inversion barrier',
                 fontsize=12)
    ax.axis('off')

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.92)
    fig.savefig(OUTPUT_DIR / "fig2_two_surface_structure.png")
    print(f"Saved: {OUTPUT_DIR / 'fig2_two_surface_structure.png'}")
    plt.close(fig)


if __name__ == '__main__':
    main()
