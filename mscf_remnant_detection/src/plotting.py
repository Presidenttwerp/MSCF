"""
Publication figures for MSCF remnant detection analysis.

6 figures at 300 dpi:
  1. Exclusion plot (σ_SI vs M_DM)
  2. Recoil spectrum comparison
  3. Number density vs mass
  4. Detection channel summary
  5. Energy deposition per transit
  6. Derivation chain flowchart
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from . import constants as c
from . import cross_section as xs
from . import limits
from . import rate
from . import alternatives as alt
from . import remnant


# Global style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

FIG_DIR = "figures"


def fig1_exclusion_plot(outdir=FIG_DIR):
    """
    σ_SI vs M_DM exclusion plot.

    Shows LZ, XENONnT, PandaX-4T limits, neutrino floor,
    and the MSCF remnant prediction.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    M = np.logspace(0, 22, 500)  # 1 GeV to 10^22 GeV

    # Limits
    ax.plot(M, limits.lz_limit(M), "b-", lw=2, label="LZ (2024)")
    ax.plot(M, limits.xenonnt_limit(M), "r--", lw=1.5, label="XENONnT (2024)")
    ax.plot(M, limits.pandax_limit(M), "g-.", lw=1.5, label="PandaX-4T (2024)")
    ax.fill_between(M, limits.neutrino_floor(M), 1e-60,
                     alpha=0.15, color="orange", label=r"Neutrino floor")

    # MSCF remnant point
    result_lz = xs.compute_all("LZ")
    M_rem = c.M_REM_GEV
    sigma_rem = result_lz["sigma_per_nucleon_cm2"]

    ax.plot(M_rem, sigma_rem, "k*", ms=15, zorder=10,
            label=f"MSCF remnant ({sigma_rem:.1e} cm$^2$)")

    # Arrow showing gap
    ax.annotate("", xy=(M_rem, sigma_rem), xytext=(M_rem, limits.lz_limit(np.array([M_rem]))[0]),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    gap = np.log10(limits.lz_limit(np.array([M_rem]))[0] / sigma_rem)
    ax.text(M_rem * 3, 1e-55, f"{gap:.0f} orders\nof magnitude",
            fontsize=9, color="gray", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e22)
    ax.set_ylim(1e-65, 1e-35)
    ax.set_xlabel(r"$M_{\rm DM}$ [GeV]", fontsize=13)
    ax.set_ylabel(r"$\sigma_{\rm SI}$ [cm$^2$]", fontsize=13)
    ax.set_title("MSCF Remnant vs Direct Detection Limits", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig1_exclusion_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig1_exclusion_plot.png"


def fig2_recoil_spectrum(outdir=FIG_DIR):
    """
    Recoil energy spectrum: WIMP (exponential) vs gravitational Rutherford (1/E²).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    E_keV = np.logspace(-1, 2, 200)  # 0.1 to 100 keV
    E_J = E_keV * 1e3 * c.EV_TO_J

    m_Xe = c.DETECTORS["LZ"]["m_nucleus_kg"]

    # Gravitational Rutherford: dσ/dE ∝ 1/E²
    dsig_grav = np.array([xs.dsigma_dER_rutherford(E, m_Xe) for E in E_J])
    dsig_grav_norm = dsig_grav / dsig_grav[np.searchsorted(E_keV, 10)]

    # WIMP (50 GeV, SI): dR/dE ∝ exp(-E/E_0) with E_0 ~ 20 keV for Xe
    E_0 = 20  # keV
    dsig_wimp = np.exp(-E_keV / E_0)
    dsig_wimp_norm = dsig_wimp / dsig_wimp[np.searchsorted(E_keV, 10)]

    ax.plot(E_keV, dsig_grav_norm, "b-", lw=2, label=r"Gravitational ($\propto 1/E_R^2$)")
    ax.plot(E_keV, dsig_wimp_norm, "r--", lw=2, label=r"WIMP 50 GeV ($\propto e^{-E_R/E_0}$)")

    ax.axvline(1.5, color="gray", ls=":", label="LZ threshold (1.5 keV)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(1e-4, 1e4)
    ax.set_xlabel(r"$E_R$ [keV]", fontsize=13)
    ax.set_ylabel(r"$d\sigma/dE_R$ (normalized at 10 keV)", fontsize=13)
    ax.set_title("Recoil Spectrum: Gravitational vs WIMP", fontsize=14)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig2_recoil_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig2_recoil_spectrum.png"


def fig3_number_density(outdir=FIG_DIR):
    """
    Number density n = ρ_DM/M vs DM mass, with remnant and WIMP marked.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    M_GeV = np.logspace(-1, 22, 500)
    M_kg = M_GeV * c.GEV_TO_KG
    n_m3 = c.RHO_DM_SI / M_kg
    n_cm3 = n_m3 * 1e-6

    ax.plot(M_GeV, n_cm3, "k-", lw=2)

    # Mark remnant
    n_rem = c.RHO_DM_SI / c.M_REM_KG * 1e-6
    ax.plot(c.M_REM_GEV, n_rem, "b*", ms=15, label=f"MSCF remnant ($n \\approx$ {n_rem:.1e} cm$^{{-3}}$)")

    # Mark 100 GeV WIMP
    n_wimp = c.RHO_DM_SI / (100 * c.GEV_TO_KG) * 1e-6
    ax.plot(100, n_wimp, "rs", ms=10, label=f"100 GeV WIMP ($n \\approx$ {n_wimp:.1e} cm$^{{-3}}$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm DM}$ [GeV]", fontsize=13)
    ax.set_ylabel(r"$n$ [cm$^{-3}$]", fontsize=13)
    ax.set_title("DM Number Density vs Mass", fontsize=14)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig3_number_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig3_number_density.png"


def fig4_channel_summary(outdir=FIG_DIR):
    """
    Bar chart of log₁₀(σ_pred / σ_sens) for each detection channel.
    All bars should be deeply negative.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    result_lz = xs.compute_all("LZ")
    sigma_n = result_lz["sigma_per_nucleon_cm2"]
    M_rem = c.M_REM_GEV

    comparison = limits.compare_with_limits(sigma_n, M_rem)

    # Additional channels
    lens = alt.einstein_radius(3e19)
    femto = alt.femtolensing_check()
    pta = alt.pulsar_timing_shapiro(c.L_PLANCK, 1e13)

    channels = {
        "LZ": comparison["ratio_LZ"],
        "XENONnT": comparison["ratio_XENONnT"],
        "PandaX-4T": comparison["ratio_PandaX"],
        r"$\nu$ floor": comparison["ratio_nu_floor"],
        "Mica": comparison["ratio_mica"],
        "Lensing\n" + r"($\theta_E$)": lens["theta_E_rad"] / 1e-6,  # vs microarcsec
        "Femto-\nlensing": femto["r_s_over_lambda_MeV"],
        "PTA": pta["ratio_dt"],
    }

    names = list(channels.keys())
    values = [np.log10(max(v, 1e-100)) for v in channels.values()]

    colors = ["#1f77b4"] * 5 + ["#ff7f0e"] * 3
    bars = ax.bar(names, values, color=colors, edgecolor="black", lw=0.8)

    ax.axhline(0, color="red", lw=2, ls="--", label="Detectable")
    ax.set_ylabel(r"$\log_{10}(\sigma_{\rm pred} / \sigma_{\rm sens})$", fontsize=13)
    ax.set_title("MSCF Remnant: Sensitivity Gap by Channel", fontsize=14)
    ax.legend(fontsize=10)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                f"{val:.0f}", ha="center", va="top", fontsize=9, color="white",
                fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig4_channel_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig4_channel_summary.png"


def fig5_energy_deposition(outdir=FIG_DIR):
    """
    Energy deposition per transit through different detector materials.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    detectors = ["LZ", "DarkSide-20k", "SuperCDMS"]
    labels = ["LZ (Xe)", "DarkSide-20k (Ar)", "SuperCDMS (Ge)"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    L_range = np.logspace(-2, 1, 100)  # 0.01 to 10 m

    for det_name, label, color in zip(detectors, labels, colors):
        dE_info = rate.energy_deposit_per_transit(det_name, L_det_m=1.0)
        dE_dx = dE_info["dE_dx_eV_m"]
        dE_total = dE_dx * L_range

        ax.plot(L_range, dE_total, lw=2, color=color, label=label)

    # Thresholds
    for det_name, label, color in zip(detectors, labels, colors):
        E_th = c.DETECTORS[det_name]["E_th_keV"] * 1e3  # eV
        ax.axhline(E_th, color=color, ls=":", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Path length through detector [m]", fontsize=13)
    ax.set_ylabel("Energy deposited [eV]", fontsize=13)
    ax.set_title("Energy Deposition per Remnant Transit", fontsize=14)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig5_energy_deposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig5_energy_deposition.png"


def fig6_derivation_chain(outdir=FIG_DIR):
    """
    Derivation chain flowchart: Axiom 5 → κ_max → M_min → T_eff → remnant → σ → verdict.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, "Axiom 5\n" + r"$\rho_{\rm crit} = c^7/(\hbar G^2)$"),
        (2.0, 1.5, "Eq. 38\n" + r"$\kappa_{\rm max} = c^2/(2l_P)$"),
        (3.5, 1.5, "Eq. 39\n" + r"$M_{\rm min} = M_P/2$"),
        (5.0, 1.5, "Eq. 43\n" + r"$T_{\rm eff} = 0$"),
        (6.5, 1.5, "Remnant\nStable"),
        (8.0, 1.5, r"$\sigma_{\rm grav}$" + "\n" + r"$\sim 10^{-63}$ cm$^2$"),
        (9.3, 1.5, "28 orders\nbelow LZ"),
    ]

    for x, y, text in boxes:
        bbox = FancyBboxPatch((x - 0.55, y - 0.5), 1.1, 1.0,
                               boxstyle="round,pad=0.1",
                               facecolor="#e8f0fe", edgecolor="#1a73e8",
                               lw=1.5)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold")

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.55
        x2 = boxes[i+1][0] - 0.55
        ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#1a73e8", lw=1.5))

    ax.set_title("MSCF Derivation Chain: Axiom 5 to Detection Verdict",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(f"{outdir}/fig6_derivation_chain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{outdir}/fig6_derivation_chain.png"


def generate_all(outdir=FIG_DIR):
    """Generate all 6 publication figures."""
    import os
    os.makedirs(outdir, exist_ok=True)

    figs = []
    figs.append(fig1_exclusion_plot(outdir))
    figs.append(fig2_recoil_spectrum(outdir))
    figs.append(fig3_number_density(outdir))
    figs.append(fig4_channel_summary(outdir))
    figs.append(fig5_energy_deposition(outdir))
    figs.append(fig6_derivation_chain(outdir))
    return figs
