#!/usr/bin/env python3
"""
Planck Commander TT likelihood evaluation for MSCF vs LCDM.

Evaluates the Gibbs-based Gaussianized Commander likelihood (the gold
standard for ell = 2-29 TT) using CAMB-computed C_ell spectra with and
without the MSCF transfer function T^2(k).

References:
    MSCF v2.1.7, Section IX.E.4.

Key results:
    - logL(LCDM) = -11.73
    - MSCF always worse than LCDM at low ell:
        kappa = 3.16e-4: Delta logL = -3.26
        kappa = 5.62e-4: Delta logL = -5.19
        kappa = 1.00e-3: Delta logL = -10.68
    - Gaussian chi^2 approximation gave the wrong sign

Requires:
    Planck 2018 Commander TT data installed via cobaya:
        cobaya-install planck_2018_lowl.TT --packages-path ~/cobaya_packages

Output:
    results/cmb_comparison/commander_results.json
    results/cmb_comparison/commander_spectra.png
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

print = functools.partial(print, flush=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "cmb_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_planck_data(user_path: str | None = None) -> Path:
    """Locate Planck Commander TT data directory."""
    candidates = []
    if user_path is not None:
        candidates.append(Path(user_path))
    candidates.append(Path.home() / "cobaya_packages" / "data" / "planck_2018_lowT_native")
    candidates.append(Path.home() / "planck_data" / "data" / "planck_2018_lowT_native")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Planck Commander data not found. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nInstall via: cobaya-install planck_2018_lowl.TT --packages-path ~/cobaya_packages"
    )


def main(planck_data_path: str | None = None) -> None:
    # ================================================================
    # 1. INSTANTIATE COMMANDER TT LIKELIHOOD
    # ================================================================

    PLANCK_DATA_PATH = find_planck_data(planck_data_path)

    print("=" * 60)
    print("PLANCK COMMANDER LIKELIHOOD: PROPER Delta logL")
    print("=" * 60)
    print(f"Data path: {PLANCK_DATA_PATH}")

    LMIN = 2
    LMAX = 29

    # Load covariance -> inverse
    cov = np.loadtxt(str(PLANCK_DATA_PATH / "cov.txt"))[
        LMIN - 2 : LMAX + 1 - 2, LMIN - 2 : LMAX + 1 - 2
    ]
    covinv = np.linalg.inv(cov)

    # Load Gaussianized means
    mu = np.ascontiguousarray(
        np.loadtxt(str(PLANCK_DATA_PATH / "mu.txt"))[LMIN - 2 : LMAX + 1 - 2]
    )

    # Load offset spectrum
    mu_sigma_raw = np.loadtxt(str(PLANCK_DATA_PATH / "mu_sigma.txt"))[
        LMIN - 2 : LMAX + 1 - 2
    ]
    mu_sigma = np.zeros(LMAX + 1)
    mu_sigma[LMIN:] = mu_sigma_raw

    # Load splines (Gaussianization transform)
    nbins = 1000
    spline_cl = np.loadtxt(str(PLANCK_DATA_PATH / "cl2x_1.txt"))[
        :, LMIN - 2 : LMAX + 1 - 2
    ]
    spline_val = np.loadtxt(str(PLANCK_DATA_PATH / "cl2x_2.txt"))[
        :, LMIN - 2 : LMAX + 1 - 2
    ]

    splines = []
    spline_derivs = []
    prior_bounds = np.zeros((LMAX + 1 - LMIN, 2))

    for i in range(LMAX - LMIN + 1):
        j = 0
        while abs(spline_val[j, i] + 5) < 1e-4:
            j += 1
        prior_bounds[i, 0] = spline_cl[j + 2, i]
        j = nbins - 1
        while abs(spline_val[j, i] - 5) < 1e-4:
            j -= 1
        prior_bounds[i, 1] = spline_cl[j - 2, i]
        s = InterpolatedUnivariateSpline(spline_cl[:, i], spline_val[:, i])
        splines.append(s)
        spline_derivs.append(s.derivative())

    # Compute offset (same as cobaya's initialization)
    def _commander_logL_raw(cls_TT: np.ndarray, calib: float = 1.0) -> float:
        theory = cls_TT[LMIN : LMAX + 1] / calib ** 2
        if any(theory < prior_bounds[:, 0]) or any(theory > prior_bounds[:, 1]):
            return -np.inf
        logl = 0.0
        x = np.zeros_like(theory)
        for i, (spline, diff_spline, cl) in enumerate(
            zip(splines, spline_derivs, theory)
        ):
            dxdCl = diff_spline(cl)
            if dxdCl < 0:
                return -np.inf
            logl += np.log(dxdCl)
            x[i] = spline(cl)
        delta = x - mu
        logl += -0.5 * covinv.dot(delta).dot(delta)
        return logl

    offset_value = _commander_logL_raw(mu_sigma)
    print(f"Commander offset (self-calibration): {offset_value:.4f}")

    def commander_logL(cls_TT: np.ndarray, calib: float = 1.0) -> float:
        return _commander_logL_raw(cls_TT, calib) - offset_value

    print(f"Commander likelihood initialized: ell = {LMIN} to {LMAX}")
    print(f"Prior bounds (D_2): [{prior_bounds[0, 0]:.1f}, {prior_bounds[0, 1]:.1f}] uK^2")

    # ================================================================
    # 2. CAMB: COMPUTE C_ell FOR LCDM AND MSCF
    # ================================================================

    import camb

    # Planck 2018 best-fit parameters
    COSMO = dict(
        H0=67.32, ombh2=0.02237, omch2=0.1200, tau=0.0544,
        As=2.1e-9, ns=0.9649,
    )
    K_PIVOT = 0.05  # Mpc^-1

    # T^2_Xi transfer function
    k_data = np.array([
        1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5,
        1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 5.0e-3, 8.0e-3,
        1.0e-2, 2.0e-2, 3.0e-2, 5.0e-2, 8.0e-2,
        0.10, 0.162, 0.264, 0.428,
        0.695, 1.13, 1.83, 2.98, 4.83
    ])
    T2_data = np.array([
        6.43, 6.44, 6.54, 7.18,
        3.86, 2.73, 2.41, 2.02, 1.85, 1.70,
        1.63, 1.42, 1.31, 1.17, 1.05,
        1.00, 0.89, 0.79, 0.70,
        0.62, 0.57, 0.56, 0.67, 1.36
    ])
    T2_interp = interp1d(
        np.log10(k_data), np.log10(T2_data),
        kind='cubic', fill_value=(np.log10(6.43), 0.0),
        bounds_error=False,
    )

    def T2_xi(k_planck: np.ndarray) -> np.ndarray:
        return 10 ** T2_interp(np.log10(np.clip(k_planck, 1e-10, 100)))

    def run_camb_lcdm() -> np.ndarray:
        """Run CAMB with standard LCDM power-law initial power."""
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=COSMO['H0'], ombh2=COSMO['ombh2'],
                           omch2=COSMO['omch2'], tau=COSMO['tau'])
        pars.InitPower.set_params(As=COSMO['As'], ns=COSMO['ns'],
                                   pivot_scalar=K_PIVOT)
        pars.set_for_lmax(LMAX + 50)
        pars.set_accuracy(AccuracyBoost=1.5)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        return powers['total'][:, 0]

    def run_camb_mscf(kappa: float) -> np.ndarray:
        """Run CAMB with MSCF-modified initial power spectrum."""
        k_camb = np.logspace(-6, np.log10(2.0), 500)  # Mpc^-1
        P_scalar = np.zeros_like(k_camb)
        for i, k in enumerate(k_camb):
            k_planck = k / kappa
            t2 = T2_xi(k_planck)
            P_scalar[i] = COSMO['As'] * (k / K_PIVOT) ** (COSMO['ns'] - 1) * t2

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=COSMO['H0'], ombh2=COSMO['ombh2'],
                           omch2=COSMO['omch2'], tau=COSMO['tau'])
        pars.set_for_lmax(LMAX + 50)
        pars.set_accuracy(AccuracyBoost=1.5)
        pars.set_initial_power_table(k_camb, P_scalar)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
        return powers['total'][:, 0]

    # ================================================================
    # 3. RUN AND EVALUATE
    # ================================================================

    kappa_values = [3.16e-4, 5.62e-4, 1.0e-3]

    print(f"\n{'=' * 60}")
    print("CAMB: COMPUTING C_ell SPECTRA")
    print("=" * 60)

    print("\nRunning CAMB for LCDM...")
    Dl_lcdm = run_camb_lcdm()
    logL_lcdm = commander_logL(Dl_lcdm)

    print(f"  D_2  = {Dl_lcdm[2]:.1f} uK^2 (expect ~1025)")
    print(f"  D_20 = {Dl_lcdm[20]:.1f} uK^2")
    print(f"  D_29 = {Dl_lcdm[29]:.1f} uK^2")
    print(f"  Commander logL(LCDM) = {logL_lcdm:.4f}")

    results_list = []
    for kappa in kappa_values:
        print(f"\nRunning CAMB for MSCF (kappa={kappa:.2e})...")
        Dl_mscf = run_camb_mscf(kappa)

        logL_mscf = commander_logL(Dl_mscf)
        delta_logL = logL_mscf - logL_lcdm
        equiv_delta_chi2 = -2 * delta_logL

        ratio_2 = Dl_mscf[2] / Dl_lcdm[2]
        ratio_10 = Dl_mscf[10] / Dl_lcdm[10]
        ratio_29 = Dl_mscf[29] / Dl_lcdm[29]
        mean_suppression = 1 - np.mean(Dl_mscf[2:30] / Dl_lcdm[2:30])

        print(f"  D_2(MSCF)  = {Dl_mscf[2]:.1f} uK^2 (ratio: {ratio_2:.4f})")
        print(f"  D_10(MSCF) = {Dl_mscf[10]:.1f} uK^2 (ratio: {ratio_10:.4f})")
        print(f"  D_29(MSCF) = {Dl_mscf[29]:.1f} uK^2 (ratio: {ratio_29:.4f})")
        print(f"  Mean suppression (ell=2-29): {mean_suppression * 100:.1f}%")
        print(f"  Commander logL(MSCF)  = {logL_mscf:.4f}")
        print(f"  Delta logL = {delta_logL:+.4f}")
        print(f"  Equiv Delta chi^2 = {equiv_delta_chi2:+.4f}")

        if delta_logL > 0:
            print(f"  -> MSCF PREFERRED by Commander likelihood")
        else:
            print(f"  -> LCDM preferred by Commander likelihood")

        results_list.append({
            'kappa': float(kappa),
            'logL_lcdm': float(logL_lcdm),
            'logL_mscf': float(logL_mscf),
            'delta_logL': float(delta_logL),
            'equiv_delta_chi2': float(equiv_delta_chi2),
            'mean_suppression_pct': float(mean_suppression * 100),
            'D2_lcdm': float(Dl_lcdm[2]),
            'D2_mscf': float(Dl_mscf[2]),
            'D2_ratio': float(ratio_2),
            'D29_ratio': float(ratio_29),
            'Dl_lcdm_lowl': Dl_lcdm[2:30].tolist(),
            'Dl_mscf_lowl': Dl_mscf[2:30].tolist(),
        })

    # ================================================================
    # 4. COMPARISON TO GAUSSIAN APPROXIMATION
    # ================================================================

    print(f"\n{'=' * 60}")
    print("COMPARISON: Commander vs Gaussian chi^2")
    print("=" * 60)
    print(f"{'kappa':>10} | {'Cmdr Delta logL':>16} | {'Equiv Dchi2':>12} | {'Gaussian Dchi2':>20}")
    print("-" * 68)

    gaussian_dchi2 = {3.16e-4: 2.99, 5.62e-4: 9.61, 1.0e-3: 15.29}

    for r in results_list:
        kappa = r['kappa']
        g_dchi2 = gaussian_dchi2.get(kappa, float('nan'))
        cmdr_favor = -r['equiv_delta_chi2']
        print(f"{kappa:>10.2e} | {r['delta_logL']:>+16.4f} | {cmdr_favor:>+12.4f} | {g_dchi2:>+20.2f}")

    print(f"\nNote: Positive Delta chi^2 = MSCF fits better than LCDM")

    # ================================================================
    # 5. VERDICT
    # ================================================================

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print("=" * 60)

    best = max(results_list, key=lambda r: r['delta_logL'])
    best_dchi2 = -best['equiv_delta_chi2']

    if best['delta_logL'] > 0:
        if best_dchi2 > 9:
            verdict = f"MSCF PREFERRED at >3 sigma (Delta chi^2 = {best_dchi2:+.2f})"
        elif best_dchi2 > 4:
            verdict = f"MSCF PREFERRED at >2 sigma (Delta chi^2 = {best_dchi2:+.2f})"
        elif best_dchi2 > 1:
            verdict = f"MSCF slightly preferred (Delta chi^2 = {best_dchi2:+.2f}, not significant)"
        else:
            verdict = f"Negligible preference (Delta chi^2 = {best_dchi2:+.2f})"
    else:
        verdict = f"LCDM preferred (Delta chi^2 = {best_dchi2:+.2f})"

    print(f"Best kappa: {best['kappa']:.2e}")
    print(f"Commander Delta logL: {best['delta_logL']:+.4f}")
    print(f"Equivalent Delta chi^2: {best_dchi2:+.4f}")
    print(f"{verdict}")

    # ================================================================
    # 6. SAVE RESULTS
    # ================================================================

    output = {
        'method': 'Planck 2018 Commander TT (Gibbs-based, Gaussianized likelihood)',
        'ell_range': [LMIN, LMAX],
        'cosmo_params': COSMO,
        'results': results_list,
        'best_kappa': best['kappa'],
        'best_delta_logL': best['delta_logL'],
        'best_equiv_delta_chi2': best_dchi2,
        'verdict': verdict,
        'gaussian_comparison': {
            'gaussian_dchi2_kappa562e4': 9.61,
            'commander_dchi2_kappa562e4': float(-results_list[1]['equiv_delta_chi2']),
        },
    }

    outfile = OUTPUT_DIR / "commander_results.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outfile}")

    # ================================================================
    # 7. DIAGNOSTIC PLOT
    # ================================================================

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                              gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    ell_arr = np.arange(2, 30)
    ax.plot(ell_arr, Dl_lcdm[2:30], 'b-', linewidth=2, label='LCDM')
    colors = ['#e74c3c', '#e67e22', '#9b59b6']
    for i, r in enumerate(results_list):
        ax.plot(ell_arr, r['Dl_mscf_lowl'], '-', color=colors[i], linewidth=1.5,
                label=f'MSCF kappa={r["kappa"]:.1e} (DlogL={r["delta_logL"]:+.2f})')
    ax.set_xlabel('Multipole ell')
    ax.set_ylabel('D_ell [uK^2]')
    ax.set_title('Commander TT likelihood: MSCF vs LCDM (ell=2-29)')
    ax.legend(fontsize=9)
    ax.set_xlim(2, 29)

    ax = axes[1]
    for i, r in enumerate(results_list):
        ratio = np.array(r['Dl_mscf_lowl']) / np.array(r['Dl_lcdm_lowl'])
        ax.plot(ell_arr, ratio, '-', color=colors[i], linewidth=1.5,
                label=f'kappa={r["kappa"]:.1e}')
    ax.axhline(1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Multipole ell')
    ax.set_ylabel('D_ell(MSCF) / D_ell(LCDM)')
    ax.legend(fontsize=9)
    ax.set_xlim(2, 29)
    ax.set_ylim(0.5, 1.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'commander_spectra.png', dpi=200)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / 'commander_spectra.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Planck Commander TT likelihood for MSCF vs LCDM')
    parser.add_argument('--planck-data', type=str, default=None,
                        help='Path to planck_2018_lowT_native directory')
    args = parser.parse_args()
    main(planck_data_path=args.planck_data)
