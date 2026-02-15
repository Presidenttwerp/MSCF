#!/usr/bin/env python3
"""
Phase correlation test: multipole alignment from a deterministic bounce.

A deterministic bounce maps input phases to output phases through a
k-dependent but fixed Bogoliubov transformation B(k) = {alpha_k, beta_k}.
This script extracts the complex Bogoliubov coefficients, maps them to
CMB multipole space, and tests whether the resulting phase correlations
can explain Planck anomalies (quadrupole-octupole alignment, axis of evil).

References:
    MSCF v2.1.7, Section IX.E.4.

Key results:
    - Phase variation across CMB k-range: 2.9e-3 rad (0.17 deg)
    - At multipole level: phi_ell range = 1.38e-4 rad (0.008 deg)
    - Verdict: NULL_SCALE_SEPARATION
    - Root cause: all CMB modes are superhorizon throughout the bounce

Code duplication with coupled_mode_evolution.py is intentional:
each script is independently verifiable.

Requires:
    numpy >= 2.0 (uses np.trapezoid)

Outputs:
    results/cmb_comparison/phase_results.json
    results/cmb_comparison/phase_function.dat
"""

import numpy as np
import json
import time
import functools
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.special import spherical_jn

print = functools.partial(print, flush=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "cmb_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# CONSTANTS
# ================================================================
M2: float = 1.0 / (8.0 * np.pi)
alpha_coeff: float = 1.0 / (1728.0 * np.pi**2)
rho_c: float = 1.0
eps_reg: float = 1e-12
TAU2_MIN: float = 1e-8
XI_EPS: float = 0.01

# QR parameters
QR_MAX_GROWTH: float = 1.0
QR_SAFETY: float = 0.5
QR_RTOL: float = 1e-10
QR_ATOL: float = 1e-12


def save_json(path: Path, data: dict) -> None:
    """Save JSON with numpy type handling."""
    def handler(x):
        if isinstance(x, np.floating):
            if np.isnan(x):
                return None
            if np.isinf(x):
                return "inf" if x > 0 else "-inf"
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=handler)


# ================================================================
# BACKGROUND + PUMP
# ================================================================

def solve_background(
    w: float,
    tau_max: float = 200,
    N_points: int = 500000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve MSCF background through bounce."""
    def rhs(t, y):
        x, H, lna = y
        x = np.clip(x, 1e-30, 1.0 - 1e-15)
        dH = -4.0 * np.pi * (1.0 + w) * x * (1.0 - 2.0 * x)
        dx = -3.0 * H * (1.0 + w) * x
        dlna = H
        return [dx, dH, dlna]

    x0 = 0.001
    H0 = -np.sqrt(8.0 * np.pi / 3.0 * x0 * (1.0 - x0))

    sol = solve_ivp(rhs, (-tau_max, tau_max), [x0, H0, 0.0],
                    t_eval=np.linspace(-tau_max, tau_max, N_points),
                    method='Radau', rtol=1e-13, atol=1e-16)

    t = sol.t
    x = sol.y[0]
    H = sol.y[1]
    a = np.exp(sol.y[2])

    i_bounce = np.argmin(np.abs(H))

    dt = np.diff(t)
    deta = dt / a[:-1]
    eta = np.concatenate([[0], np.cumsum(deta)])
    eta -= eta[i_bounce]

    return t, x, H, a, eta, i_bounce


def compute_xi_pump_arrays(
    t: np.ndarray,
    a: np.ndarray,
    i_bounce: int,
    w: float,
    rho_c_val: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Xi-corrected scalar and tensor pump potentials (analytic)."""
    alpha = 1.0 / (3.0 * (1.0 + w))
    H_c = np.sqrt(8.0 * np.pi * rho_c_val / 3.0)
    sigma = 3.0 * (1.0 + w) * H_c / 2.0
    a_b = a[i_bounce]
    t_bounce = t[i_bounce]

    N = len(t)
    U_scalar = np.zeros(N)
    U_tensor = np.zeros(N)

    for i in range(N):
        tau = sigma * (t[i] - t_bounce)
        tau2_raw = tau * tau
        tau2 = max(tau2_raw, TAU2_MIN)
        u = 1.0 + tau2

        u_alpha = u ** alpha
        Omega = sigma * a_b * u_alpha
        Omega_tau = 2.0 * alpha * sigma * a_b * tau * u ** (alpha - 1.0)

        tau2m1 = tau2 - 1.0
        abs_tau2m1 = abs(tau2m1)

        if abs_tau2m1 > XI_EPS:
            xi_tau = tau / tau2m1
            xi_tautau = -(tau2 + 1.0) / (tau2m1 * tau2m1)
        else:
            xi_tau = 0.0
            xi_tautau = -2.0 / (XI_EPS * XI_EPS)

        L_s_tau = 2.0 * alpha * tau / u - tau / tau2 + xi_tau
        L_s_tautau = 2.0 * alpha * (1.0 - tau2) / (u * u) + 1.0 / tau2 + xi_tautau

        U_scalar[i] = Omega * Omega * (L_s_tautau + L_s_tau * L_s_tau) \
                     + Omega * Omega_tau * L_s_tau

        L_t_tau = 2.0 * alpha * tau / u
        L_t_tautau = 2.0 * alpha * (1.0 - tau2) / (u * u)

        U_tensor[i] = Omega * Omega * (L_t_tautau + L_t_tau * L_t_tau) \
                     + Omega * Omega_tau * L_t_tau

    return U_scalar, U_tensor


def compute_kinetic_matrix(
    x: np.ndarray,
    H: np.ndarray,
    a: np.ndarray,
    w: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2x2 kinetic matrix for mixing angle."""
    cs2_m = w
    Hdot = -4.0 * np.pi * (1.0 + w) * x * (1.0 - 2.0 * x)

    V_pp = 108.0 * alpha_coeff * H**2
    A = M2 / 3.0 + V_pp / 2.0
    B = 2.0 * M2 / 3.0 + V_pp / 2.0
    G_grav = np.where(np.abs(B) > 1e-30,
                      3.0 * M2 * A * V_pp / (2.0 * B**2), 0.0)

    H2_reg = H**2 + eps_reg**2
    eps_m = -Hdot / H2_reg
    G_matt = eps_m * M2 / cs2_m

    return G_grav, G_matt


def compute_mixing_angle(
    G_grav: np.ndarray,
    G_matt: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixing angle theta(eta) from eigenvalue fractions."""
    theta = np.arctan2(np.sqrt(np.abs(G_grav)),
                       np.sqrt(np.abs(G_matt) + 1e-60))
    dtheta_deta = np.gradient(theta, eta)
    dtheta_deta = gaussian_filter1d(dtheta_deta, sigma=5)
    return theta, dtheta_deta


# ================================================================
# QR-STABILIZED MODE EVOLUTION (complex Bogoliubov extraction)
# ================================================================

def evolve_mode_complex(
    k: float,
    U_grav_interp: interp1d,
    U_matt_interp: interp1d,
    theta_dot_interp: interp1d,
    w: float,
    eta_start: float,
    eta_end: float,
) -> dict | None:
    """
    QR-stabilized 4-component coupled evolution returning COMPLEX
    Bogoliubov coefficients.

    exp(logS_common) is REAL and POSITIVE, so phases of alpha_k and
    beta_k in the rescaled coordinates ARE the physical phases.
    """
    cs2_m = w
    cs2_grav = 1.0

    def rhs_16(eta, Y_flat):
        U1 = float(U_matt_interp(eta))
        U2 = float(U_grav_interp(eta))
        td = float(theta_dot_interp(eta))

        omega1_sq = cs2_m * k * k - U1
        omega2_sq = cs2_grav * k * k - U2

        dY = np.empty(16)
        for col in range(4):
            off = col * 4
            v1 = Y_flat[off]
            v1p = Y_flat[off + 1]
            v2 = Y_flat[off + 2]
            v2p = Y_flat[off + 3]
            dY[off] = v1p
            dY[off + 1] = -omega1_sq * v1 + td * (2.0 * v2p + td * v2)
            dY[off + 2] = v2p
            dY[off + 3] = -omega2_sq * v2 - td * (2.0 * v1p + td * v1)
        return dY

    # Bunch-Davies vacuum ICs in the adiabatic (matter) mode
    omega_init = np.sqrt(cs2_m) * k
    v1_init = 1.0 / np.sqrt(2.0 * omega_init)
    v1p_init = -1j * omega_init * v1_init

    c_re = np.array([np.real(v1_init), np.real(v1p_init), 0.0, 0.0])
    c_im = np.array([np.imag(v1_init), np.imag(v1p_init), 0.0, 0.0])

    Q = np.eye(4)
    z_re = c_re.copy()
    z_im = c_im.copy()

    norm_re = np.linalg.norm(z_re)
    norm_im = np.linalg.norm(z_im)
    logS_re = np.log(norm_re) if norm_re > 1e-300 else -690.0
    logS_im = np.log(norm_im) if norm_im > 1e-300 else -690.0
    z_re /= max(norm_re, 1e-300)
    z_im /= max(norm_im, 1e-300)

    eta = eta_start
    n_segments = 0

    while eta < eta_end - 1e-15:
        U1 = float(U_matt_interp(eta))
        U2 = float(U_grav_interp(eta))

        kappa2_m = U1 - cs2_m * k * k
        kappa2_g = U2 - cs2_grav * k * k
        kappa2 = max(kappa2_m, kappa2_g)

        if kappa2 > 1.0:
            kappa = np.sqrt(kappa2)
            seg_len = QR_MAX_GROWTH / kappa * QR_SAFETY
        else:
            omega2 = max(-kappa2_m, -kappa2_g, 0.01)
            omega = np.sqrt(omega2)
            seg_len = 50.0 * 2.0 * np.pi / omega

        d = max(abs(eta), 0.01)
        seg_len = min(seg_len, d * 0.3, 50.0, eta_end - eta)
        seg_len = max(seg_len, 1e-12)

        if eta_end - eta < 1e-15:
            break
        seg_end = min(eta + seg_len, eta_end)

        transition_kappa2 = max(cs2_m * k * k * 0.1, 10.0)
        method = "Radau" if kappa2 > -transition_kappa2 else "DOP853"

        Y_flat = Q.flatten(order='F')

        try:
            sol = solve_ivp(rhs_16, [eta, seg_end], Y_flat,
                            method=method, rtol=QR_RTOL, atol=QR_ATOL)
        except Exception:
            return None

        if not sol.success:
            return None

        Phi_raw = sol.y[:, -1].reshape(4, 4, order='F')
        Q_new, R_seg = np.linalg.qr(Phi_raw)

        for j in range(4):
            if R_seg[j, j] < 0:
                Q_new[:, j] *= -1
                R_seg[j, :] *= -1

        z_re_new = R_seg @ z_re
        s = np.linalg.norm(z_re_new)
        if s > 1e-300:
            z_re = z_re_new / s
            logS_re += np.log(s)

        z_im_new = R_seg @ z_im
        s = np.linalg.norm(z_im_new)
        if s > 1e-300:
            z_im = z_im_new / s
            logS_im += np.log(s)

        Q = Q_new
        eta = seg_end
        n_segments += 1

    # Reconstruct physical state at eta_end
    logS_common = (logS_re + logS_im) / 2.0

    y_re_local = Q @ z_re * np.exp(logS_re - logS_common)
    y_im_local = Q @ z_im * np.exp(logS_im - logS_common)

    v1_final = y_re_local[0] + 1j * y_im_local[0]
    v1p_final = y_re_local[1] + 1j * y_im_local[1]
    v2_final = y_re_local[2] + 1j * y_im_local[2]

    # Bogoliubov extraction (rescaled coordinates)
    omega_final = np.sqrt(cs2_m) * k
    alpha_k = (np.sqrt(omega_final / 2.0) * v1_final
               + 1j * v1p_final / np.sqrt(2.0 * omega_final))
    beta_k = (np.sqrt(omega_final / 2.0) * v1_final
              - 1j * v1p_final / np.sqrt(2.0 * omega_final))

    T2_rescaled = abs(alpha_k)**2 + abs(beta_k)**2
    T2_physical = T2_rescaled * np.exp(2.0 * logS_common)

    iso_leakage = abs(v2_final)**2 / (abs(v1_final)**2 + 1e-60)

    arg_alpha = float(np.angle(alpha_k))
    arg_beta = float(np.angle(beta_k))

    phase_osc = np.exp(2j * omega_final * eta_end)
    phase_shift = float(np.angle(alpha_k + beta_k * phase_osc))

    return {
        'k': float(k),
        'T2': float(T2_physical),
        'alpha2': float(abs(alpha_k)**2 * np.exp(2.0 * logS_common)),
        'beta2': float(abs(beta_k)**2 * np.exp(2.0 * logS_common)),
        'iso_leakage': float(iso_leakage),
        'logS': float(logS_common),
        'n_segments': n_segments,
        'arg_alpha': arg_alpha,
        'arg_beta': arg_beta,
        'alpha_re': float(np.real(alpha_k)),
        'alpha_im': float(np.imag(alpha_k)),
        'beta_re': float(np.real(beta_k)),
        'beta_im': float(np.imag(beta_k)),
        'phase_shift': phase_shift,
        'alpha_beta_rel_phase': float(arg_alpha - arg_beta),
        'beta_over_alpha_abs': float(abs(beta_k) / (abs(alpha_k) + 1e-300)),
    }


# ================================================================
# PHASE -> MULTIPOLE MAPPING
# ================================================================

def bogoliubov_phase_to_multipole_phase(
    k_arr: np.ndarray,
    arg_alpha_arr: np.ndarray,
    arg_beta_arr: np.ndarray,
    phase_shift_arr: np.ndarray,
    ell_max: int = 30,
    kappa: float = 5.62e-4,
) -> dict[int, dict]:
    """
    Map k-dependent Bogoliubov phases to CMB multipole phases.

    The effective phase for multipole ell is the transfer-function-weighted
    average of the k-dependent phase:
        phi_ell = integral dk w_ell(k) Delta_phi(k)
    where w_ell(k) ~ j_l(k eta_0)^2 peaks at k ~ ell/eta_0.
    """
    k_mpc = k_arr / kappa
    eta_0 = 14000.0  # Mpc, comoving distance to recombination

    log_k_mpc = np.log10(k_mpc)
    k_fine = np.logspace(np.max([log_k_mpc.min(), -6.0]),
                         np.min([log_k_mpc.max(), 0.0]),
                         10000)

    phase_interps: dict[str, interp1d] = {}
    for name, arr in [('arg_alpha', arg_alpha_arr),
                      ('arg_beta', arg_beta_arr),
                      ('phase_shift', phase_shift_arr)]:
        phase_interps[name] = interp1d(log_k_mpc, arr, kind='linear',
                                        fill_value='extrapolate',
                                        bounds_error=False)

    results: dict[int, dict] = {}

    for ell in range(2, ell_max + 1):
        x = k_fine * eta_0

        jl = np.array([spherical_jn(ell, xi) for xi in x])
        window = jl**2 * k_fine

        norm = np.trapezoid(window, np.log(k_fine))
        if norm < 1e-30:
            results[ell] = {
                'ell': ell,
                'phi_arg_alpha': 0.0,
                'phi_arg_beta': 0.0,
                'phi_shift': 0.0,
                'k_eff_mpc': float(ell / eta_0),
                'k_eff_planck': float(ell / eta_0 * kappa),
                'window_norm': float(norm),
            }
            continue

        window /= norm

        log_k_fine = np.log10(k_fine)
        phi_alpha = np.trapezoid(
            window * phase_interps['arg_alpha'](log_k_fine), np.log(k_fine))
        phi_beta = np.trapezoid(
            window * phase_interps['arg_beta'](log_k_fine), np.log(k_fine))
        phi_shift = np.trapezoid(
            window * phase_interps['phase_shift'](log_k_fine), np.log(k_fine))

        k_eff = np.exp(np.trapezoid(window * np.log(k_fine), np.log(k_fine)))

        results[ell] = {
            'ell': ell,
            'phi_arg_alpha': float(phi_alpha),
            'phi_arg_beta': float(phi_beta),
            'phi_shift': float(phi_shift),
            'k_eff_mpc': float(k_eff),
            'k_eff_planck': float(k_eff * kappa),
            'window_norm': float(norm),
        }

    return results


# ================================================================
# ALIGNMENT STATISTICS
# ================================================================

def compute_alignment_statistics(
    phi_ell_dict: dict[int, dict],
    phase_key: str = 'phi_arg_alpha',
) -> dict:
    """
    Compute multipole alignment statistics from bounce-predicted phases.

    Statistics:
    1. Phase coherence between adjacent multipoles
    2. Quadrupole-octupole phase alignment
    3. Even-odd parity asymmetry
    4. Phase smoothness (roughness of phi(ell))
    """
    ells = sorted(phi_ell_dict.keys())
    phases = np.array([phi_ell_dict[ell][phase_key] for ell in ells])

    delta_phi = np.diff(phases)
    delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi
    coherence = float(np.mean(np.cos(delta_phi)))

    if 2 in phi_ell_dict and 3 in phi_ell_dict:
        phi_2 = phi_ell_dict[2][phase_key]
        phi_3 = phi_ell_dict[3][phase_key]
        delta_23 = ((phi_3 - phi_2) + np.pi) % (2 * np.pi) - np.pi
    else:
        delta_23 = np.pi

    even_phases = [phi_ell_dict[ell][phase_key] for ell in ells if ell % 2 == 0]
    odd_phases = [phi_ell_dict[ell][phase_key] for ell in ells if ell % 2 == 1]
    mean_even = np.mean(even_phases)
    mean_odd = np.mean(odd_phases)
    parity_offset = ((mean_odd - mean_even) + np.pi) % (2 * np.pi) - np.pi

    if len(phases) > 4:
        d2phi = np.diff(phases, n=2)
        d2phi = (d2phi + np.pi) % (2 * np.pi) - np.pi
        smoothness = 1.0 / (np.var(d2phi) + 1e-30)
    else:
        smoothness = 0.0

    return {
        'phase_coherence': float(coherence),
        'delta_phi_23': float(delta_23),
        'delta_phi_23_degrees': float(np.degrees(abs(delta_23))),
        'parity_offset': float(parity_offset),
        'parity_offset_degrees': float(np.degrees(abs(parity_offset))),
        'smoothness': float(smoothness),
        'phase_range': float(np.max(phases) - np.min(phases)),
        'phase_std': float(np.std(phases)),
        'all_phases': {int(ell): float(phi_ell_dict[ell][phase_key])
                       for ell in ells},
        'all_k_eff': {int(ell): float(phi_ell_dict[ell]['k_eff_mpc'])
                      for ell in ells},
    }


# ================================================================
# MONTE CARLO SIGNIFICANCE
# ================================================================

def monte_carlo_significance(alignment_stats: dict, N_mc: int = 10000) -> dict:
    """Compare MSCF phase predictions to random-phase null hypothesis."""
    ells = sorted(alignment_stats['all_phases'].keys())
    N_ell = len(ells)
    rng = np.random.default_rng(42)

    coherence_null = np.zeros(N_mc)
    delta23_null = np.zeros(N_mc)
    parity_null = np.zeros(N_mc)
    smoothness_null = np.zeros(N_mc)

    for i in range(N_mc):
        rand_phases = rng.uniform(0, 2 * np.pi, N_ell)

        dp = np.diff(rand_phases)
        dp = (dp + np.pi) % (2 * np.pi) - np.pi
        coherence_null[i] = np.mean(np.cos(dp))

        d23 = ((rand_phases[1] - rand_phases[0]) + np.pi) % (2 * np.pi) - np.pi
        delta23_null[i] = abs(d23)

        even_p = rand_phases[0::2]
        odd_p = rand_phases[1::2]
        po = ((np.mean(odd_p) - np.mean(even_p)) + np.pi) % (2 * np.pi) - np.pi
        parity_null[i] = abs(po)

        if N_ell > 4:
            d2 = np.diff(rand_phases, n=2)
            d2 = (d2 + np.pi) % (2 * np.pi) - np.pi
            smoothness_null[i] = 1.0 / (np.var(d2) + 1e-30)

    p_coherence = float(np.mean(coherence_null >= alignment_stats['phase_coherence']))
    p_delta23 = float(np.mean(delta23_null <= abs(alignment_stats['delta_phi_23'])))
    p_parity = float(np.mean(parity_null >= abs(alignment_stats['parity_offset'])))
    p_smoothness = float(np.mean(smoothness_null >= alignment_stats['smoothness']))

    return {
        'N_mc': N_mc,
        'p_coherence': p_coherence,
        'p_delta23': p_delta23,
        'p_parity': p_parity,
        'p_smoothness': p_smoothness,
        'null_coherence_mean': float(np.mean(coherence_null)),
        'null_coherence_std': float(np.std(coherence_null)),
        'null_delta23_mean_deg': float(np.degrees(np.mean(delta23_null))),
        'null_smoothness_mean': float(np.mean(smoothness_null)),
        'null_smoothness_std': float(np.std(smoothness_null)),
    }


# ================================================================
# PLANCK COMPARISON
# ================================================================

def compare_to_planck(alignment_stats: dict, mc_results: dict) -> dict:
    """Compare MSCF predictions to Planck anomaly measurements."""

    print(f"\n{'='*70}")
    print("COMPARISON TO PLANCK ANOMALY MEASUREMENTS")
    print(f"{'='*70}")

    print(f"\n{'Observable':<40} {'Planck':>10} {'MSCF':>10} {'Random':>10}")
    print("-" * 70)

    print(f"{'Q-O alignment angle (deg)':<40} "
          f"{'~10':>10} "
          f"{alignment_stats['delta_phi_23_degrees']:>10.4f} "
          f"{'~60':>10}")

    print(f"{'Phase coherence':<40} "
          f"{'(N/A)':>10} "
          f"{alignment_stats['phase_coherence']:>10.6f} "
          f"{mc_results['null_coherence_mean']:>10.6f}")

    print(f"{'Parity offset (deg)':<40} "
          f"{'large':>10} "
          f"{alignment_stats['parity_offset_degrees']:>10.4f} "
          f"{'~60':>10}")

    print(f"{'Phase range (rad)':<40} "
          f"{'(N/A)':>10} "
          f"{alignment_stats['phase_range']:>10.6f} "
          f"{'~3.6':>10}")

    print(f"\n{'p-value (vs random)':<40} {'Value':>10}")
    print("-" * 50)
    print(f"{'Phase coherence':<40} {mc_results['p_coherence']:>10.4f}")
    print(f"{'Q-O phase alignment':<40} {mc_results['p_delta23']:>10.4f}")
    print(f"{'Parity asymmetry':<40} {mc_results['p_parity']:>10.4f}")
    print(f"{'Phase smoothness':<40} {mc_results['p_smoothness']:>10.4f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    phase_range = alignment_stats['phase_range']

    if phase_range < 0.01:
        print(f"\nPHASE RANGE: {phase_range:.2e} rad ({np.degrees(phase_range):.4f} deg)")
        print("  Phase is FLAT across all CMB multipoles.")
        print("  All multipoles get the same phase shift from the bounce.")
        print("  CONSEQUENCE: zero relative phase correlation.")
        print("  Root cause: SCALE SEPARATION.")
        print("  The bounce barrier (Delta_eta ~ 0.3 Planck) is ~60 orders of")
        print("  magnitude smaller than CMB wavelengths (k ~ 1e-5 Mpc^-1).")
        verdict = "NULL_SCALE_SEPARATION"
        interpretation = ("Bounce phase is flat across CMB scales. "
                          f"Phase range = {phase_range:.2e} rad. "
                          "Scale separation kills all phase correlations.")

    elif mc_results['p_coherence'] < 0.05:
        if alignment_stats['delta_phi_23_degrees'] < 30:
            verdict = "PROMISING"
            interpretation = ("Phase coherence significant and Q-O offset consistent "
                              "with Planck. Proceed to full a_lm computation.")
        else:
            verdict = "PARTIAL"
            interpretation = ("Phase coherence significant but Q-O offset too large. "
                              f"Delta_23 = {alignment_stats['delta_phi_23_degrees']:.1f} deg.")
    else:
        verdict = "NULL_NO_COHERENCE"
        interpretation = ("Phase coherence not significant. "
                          f"p = {mc_results['p_coherence']:.3f}.")

    print(f"\nFinal verdict: {verdict}")
    print(f"  {interpretation}")

    return {
        'verdict': verdict,
        'interpretation': interpretation,
    }


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    print("=" * 70)
    print("MSCF PHASE CORRELATION TEST")
    print("Deterministic bounce -> correlated a_lm phases -> multipole alignment")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    t_total = time.time()

    # ============================================================
    # Step 1: Background + pump field
    # ============================================================

    print("\n[1/6] Setting up background and pump field...")

    w = 1.0  # stiff matter

    t_bg, x, H, a, eta, i_bounce = solve_background(w)
    print(f"  Bounce at t={t_bg[i_bounce]:.4f}, x={x[i_bounce]:.6f}")

    U_scalar, U_tensor = compute_xi_pump_arrays(t_bg, a, i_bounce, w)

    G_grav, G_matt = compute_kinetic_matrix(x, H, a, w)
    theta, dtheta_deta = compute_mixing_angle(G_grav, G_matt, eta)

    theta_dot_max = float(np.max(np.abs(dtheta_deta)))
    print(f"  Max |dtheta/deta| = {theta_dot_max:.4f}")

    interp_kw = dict(kind='linear', bounds_error=False)
    U_grav_interp = interp1d(eta, U_tensor,
                             fill_value=(U_tensor[0], U_tensor[-1]), **interp_kw)
    U_matt_interp = interp1d(eta, U_scalar,
                             fill_value=(U_scalar[0], U_scalar[-1]), **interp_kw)
    theta_dot_interp = interp1d(eta, dtheta_deta,
                                fill_value=0.0, **interp_kw)
    zero_interp = interp1d(eta, np.zeros_like(eta), kind='linear',
                           fill_value=0.0, bounds_error=False)

    eta_start = max(eta[2], -20.0)
    eta_end = min(eta[-3], 20.0)
    print(f"  Integration range: eta in [{eta_start:.2f}, {eta_end:.2f}]")

    # ============================================================
    # Step 2: Dense k-scan with complex Bogoliubov extraction
    # ============================================================

    print("\n[2/6] Evolving modes (dense k-grid for phase extraction)...")

    k_values = np.sort(np.unique(np.concatenate([
        np.logspace(-8, -5, 20),
        np.logspace(-5, -3, 15),
        np.logspace(-3, -1, 15),
        np.logspace(-1, 1, 15),
    ])))

    print(f"  {len(k_values)} k-values from {k_values[0]:.1e} to {k_values[-1]:.1e}")

    print("\n  --- COUPLED evolution (theta' != 0) ---")
    results_coupled = []
    for i, k in enumerate(k_values):
        t0 = time.time()
        if i % 10 == 0:
            print(f"  k = {k:.4e} ({i+1}/{len(k_values)})...", end='')

        res = evolve_mode_complex(k, U_grav_interp, U_matt_interp,
                                  theta_dot_interp, w, eta_start, eta_end)
        dt = time.time() - t0

        if res is not None:
            results_coupled.append(res)
            if i % 10 == 0:
                print(f" arg(a)={res['arg_alpha']:+.6f}  T2={res['T2']:.2e}  ({dt:.1f}s)")
        else:
            if i % 10 == 0:
                print(f" FAILED ({dt:.1f}s)")

    print("\n  --- UNCOUPLED evolution (theta' = 0) ---")
    results_uncoupled = []
    for i, k in enumerate(k_values):
        t0 = time.time()
        if i % 10 == 0:
            print(f"  k = {k:.4e} ({i+1}/{len(k_values)})...", end='')

        res = evolve_mode_complex(k, U_grav_interp, U_matt_interp,
                                  zero_interp, w, eta_start, eta_end)
        dt = time.time() - t0

        if res is not None:
            results_uncoupled.append(res)
            if i % 10 == 0:
                print(f" arg(a)={res['arg_alpha']:+.6f}  ({dt:.1f}s)")
        else:
            if i % 10 == 0:
                print(f" FAILED ({dt:.1f}s)")

    # ============================================================
    # Step 3: Phase diagnostics
    # ============================================================

    print(f"\n[3/6] Phase structure diagnostics...")

    k_arr = np.array([r['k'] for r in results_coupled])
    arg_alpha_arr = np.array([r['arg_alpha'] for r in results_coupled])
    arg_beta_arr = np.array([r['arg_beta'] for r in results_coupled])
    phase_shift_arr = np.array([r['phase_shift'] for r in results_coupled])
    T2_arr = np.array([r['T2'] for r in results_coupled])

    print(f"\n  {'k':>10} {'arg(a)':>10} {'arg(b)':>10} {'phi_shift':>10} "
          f"{'|b/a|':>8} {'T2':>10}")
    print("  " + "-" * 60)
    for r in results_coupled[::4]:
        print(f"  {r['k']:10.3e} {r['arg_alpha']:+10.6f} {r['arg_beta']:+10.6f} "
              f"{r['phase_shift']:+10.6f} "
              f"{r['beta_over_alpha_abs']:8.4f} {r['T2']:10.3e}")

    cmb_mask = (k_arr >= 1e-8) & (k_arr <= 1e-4)
    range_alpha = 0.0
    if np.any(cmb_mask):
        arg_alpha_cmb = arg_alpha_arr[cmb_mask]
        k_cmb = k_arr[cmb_mask]

        range_alpha = np.max(arg_alpha_cmb) - np.min(arg_alpha_cmb)
        range_beta = np.max(arg_beta_arr[cmb_mask]) - np.min(arg_beta_arr[cmb_mask])
        range_shift = np.max(phase_shift_arr[cmb_mask]) - np.min(phase_shift_arr[cmb_mask])

        print(f"\n  === CMB k-range [{k_cmb[0]:.1e}, {k_cmb[-1]:.1e}] ({np.sum(cmb_mask)} modes) ===")
        print(f"  arg(alpha) range: {range_alpha:.6e} rad ({np.degrees(range_alpha):.6f} deg)")
        print(f"  arg(beta)  range: {range_beta:.6e} rad ({np.degrees(range_beta):.6f} deg)")
        print(f"  phase_shift range: {range_shift:.6e} rad ({np.degrees(range_shift):.6f} deg)")

        if len(k_cmb) > 3:
            log_k_cmb = np.log10(k_cmb)
            fit = np.polyfit(log_k_cmb, arg_alpha_cmb, 1)
            print(f"  Linear fit arg(alpha) vs log10(k): slope = {fit[0]:.6e}, intercept = {fit[1]:.6f}")

        if range_alpha < 0.01:
            print(f"\n  *** SCALE SEPARATION DETECTED ***")
            print(f"  Phase variation {range_alpha:.2e} rad << 0.01 rad")
            print(f"  All CMB multipoles get approximately the same phase shift.")

    well_mask = (k_arr >= 0.01) & (k_arr <= 10)
    if np.any(well_mask):
        arg_alpha_well = arg_alpha_arr[well_mask]
        range_well = np.max(arg_alpha_well) - np.min(arg_alpha_well)
        print(f"\n  Well k-range [0.01, 10]: arg(alpha) range = {range_well:.4f} rad "
              f"({np.degrees(range_well):.2f} deg)")

    if results_uncoupled:
        k_unc = np.array([r['k'] for r in results_uncoupled])
        arg_alpha_unc = np.array([r['arg_alpha'] for r in results_uncoupled])

        common_mask = np.isin(k_arr, k_unc)
        if np.any(common_mask):
            coupled_phases = arg_alpha_arr[common_mask]
            uncoupled_phases = arg_alpha_unc[np.isin(k_unc, k_arr)]
            phase_diff = coupled_phases - uncoupled_phases
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

            print(f"\n  Coupled vs uncoupled phase difference:")
            print(f"  Max |Delta arg(alpha)|: {np.max(np.abs(phase_diff)):.6f} rad "
                  f"({np.degrees(np.max(np.abs(phase_diff))):.4f} deg)")
            print(f"  Mean |Delta arg(alpha)|: {np.mean(np.abs(phase_diff)):.6f} rad")

    # ============================================================
    # Step 4: Map to multipoles
    # ============================================================

    print(f"\n[4/6] Mapping Bogoliubov phases to multipole space...")

    kappa_values = [3.16e-4, 5.62e-4, 1.00e-3]
    multipole_results: dict[float, dict[int, dict]] = {}

    for kappa in kappa_values:
        print(f"\n  --- kappa = {kappa:.2e} ---")
        phi_ell = bogoliubov_phase_to_multipole_phase(
            k_arr, arg_alpha_arr, arg_beta_arr, phase_shift_arr,
            ell_max=30, kappa=kappa)

        multipole_results[kappa] = phi_ell

        print(f"  {'ell':>4} {'k_eff (Mpc^-1)':>14} {'k_eff (Planck)':>14} "
              f"{'phi_alpha':>12} {'phi_shift':>12}")
        print("  " + "-" * 60)
        for ell in range(2, 11):
            d = phi_ell[ell]
            print(f"  {ell:>4} {d['k_eff_mpc']:>14.4e} {d['k_eff_planck']:>14.4e} "
                  f"{d['phi_arg_alpha']:>12.8f} {d['phi_shift']:>12.8f}")

        phi_range = max(phi_ell[ell]['phi_arg_alpha'] for ell in range(2, 31)) - \
                    min(phi_ell[ell]['phi_arg_alpha'] for ell in range(2, 31))
        print(f"  Phase range ell=2-30: {phi_range:.2e} rad ({np.degrees(phi_range):.6f} deg)")

    # ============================================================
    # Step 5: Alignment statistics
    # ============================================================

    print(f"\n[5/6] Computing alignment statistics...")

    kappa_best = 5.62e-4
    phi_ell_best = multipole_results[kappa_best]

    stats_alpha = compute_alignment_statistics(phi_ell_best, phase_key='phi_arg_alpha')
    stats_shift = compute_alignment_statistics(phi_ell_best, phase_key='phi_shift')

    print(f"\n  Using arg(alpha) phases:")
    print(f"    Phase coherence: {stats_alpha['phase_coherence']:.8f}")
    print(f"    Q-O phase offset: {stats_alpha['delta_phi_23_degrees']:.6f} deg")
    print(f"    Parity offset: {stats_alpha['parity_offset_degrees']:.6f} deg")
    print(f"    Phase range: {stats_alpha['phase_range']:.2e} rad")
    print(f"    Phase std: {stats_alpha['phase_std']:.2e} rad")

    print(f"\n  Using phase_shift phases:")
    print(f"    Phase coherence: {stats_shift['phase_coherence']:.8f}")
    print(f"    Q-O phase offset: {stats_shift['delta_phi_23_degrees']:.6f} deg")

    # ============================================================
    # Step 6: Monte Carlo significance
    # ============================================================

    print(f"\n[6/6] Monte Carlo significance test (N=10000)...")
    mc = monte_carlo_significance(stats_alpha, N_mc=10000)

    print(f"\n  p(coherence): {mc['p_coherence']:.4f}")
    print(f"  p(Q-O alignment): {mc['p_delta23']:.4f}")
    print(f"  p(parity): {mc['p_parity']:.4f}")
    print(f"  p(smoothness): {mc['p_smoothness']:.4f}")

    # ============================================================
    # Final comparison
    # ============================================================

    comparison = compare_to_planck(stats_alpha, mc)

    # ============================================================
    # Save everything
    # ============================================================

    dt_total = time.time() - t_total

    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_s': float(dt_total),
        'w': float(w),
        'eta_range': [float(eta_start), float(eta_end)],
        'n_modes': len(results_coupled),

        'mode_results': [{
            'k': r['k'],
            'T2': r['T2'],
            'arg_alpha': r['arg_alpha'],
            'arg_beta': r['arg_beta'],
            'phase_shift': r['phase_shift'],
            'beta_over_alpha_abs': r['beta_over_alpha_abs'],
            'alpha_re': r['alpha_re'],
            'alpha_im': r['alpha_im'],
            'beta_re': r['beta_re'],
            'beta_im': r['beta_im'],
            'logS': r['logS'],
            'n_segments': r['n_segments'],
        } for r in results_coupled],

        'mode_results_uncoupled': [{
            'k': r['k'],
            'arg_alpha': r['arg_alpha'],
            'arg_beta': r['arg_beta'],
            'phase_shift': r['phase_shift'],
        } for r in results_uncoupled],

        'cmb_phase_diagnostics': {
            'k_range': [float(k_arr[cmb_mask][0]), float(k_arr[cmb_mask][-1])]
                if np.any(cmb_mask) else None,
            'arg_alpha_range_rad': float(range_alpha) if np.any(cmb_mask) else None,
            'arg_alpha_range_deg': float(np.degrees(range_alpha))
                if np.any(cmb_mask) else None,
        },

        'multipole_phases': {
            str(kappa): {int(ell): d for ell, d in phi_ell.items()}
            for kappa, phi_ell in multipole_results.items()
        },

        'alignment_stats_arg_alpha': stats_alpha,
        'alignment_stats_phase_shift': stats_shift,
        'monte_carlo': mc,
        'comparison': comparison,
    }

    json_path = OUTPUT_DIR / "phase_results.json"
    save_json(json_path, output)
    print(f"\nResults saved to: {json_path}")

    dat_path = OUTPUT_DIR / "phase_function.dat"
    phase_data = np.column_stack([
        k_arr,
        arg_alpha_arr,
        arg_beta_arr,
        phase_shift_arr,
        T2_arr,
    ])
    np.savetxt(dat_path, phase_data,
               header='k_planck  arg_alpha  arg_beta  phase_shift  T2',
               fmt='%.8e')
    print(f"Phase function saved to: {dat_path}")

    print(f"\nTotal time: {dt_total:.1f}s")


if __name__ == '__main__':
    main()
