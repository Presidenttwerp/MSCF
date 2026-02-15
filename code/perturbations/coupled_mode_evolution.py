#!/usr/bin/env python3
"""
Full 2x2 coupled mode evolution through the MSCF bounce.

Computes the transfer function T^2(k) for adiabatic perturbations including
Landau-Zener coupling between matter and gravitational eigenmodes, and an
asymmetric equation-of-state transition (w = 1 to w = 1/3).

References:
    MSCF v2.1.7, Sections IX.E.1 and IX.E.3.
    Equations for the Xi-corrected pump potential and 4x4 QR-stabilized
    fundamental matrix are derived in that paper.

Key results:
    - Coupled/uncoupled ratio = 0.678 at low k (32% suppression from coupling)
    - Isocurvature leakage = 0.65 at low k
    - T^2(k) monotonic (no oscillatory crossings of unity)
    - Asymmetric w transition does not alter results at CMB scales

Outputs:
    results/transfer_functions/full_coupled_results.json
    results/transfer_functions/T2_coupled_w1.dat
"""

import numpy as np
import json
import time
import functools
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

print = functools.partial(print, flush=True)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "transfer_functions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# CONSTANTS
# ================================================================
M2: float = 1.0 / (8.0 * np.pi)
alpha_coeff: float = 1.0 / (1728.0 * np.pi**2)
rho_c: float = 1.0
eps_reg: float = 1e-12   # H^2 regularization (for kinetic matrix / mixing angle only)
TAU2_MIN: float = 1e-8   # bounce regularization (matches modes.py)
XI_EPS: float = 0.01     # Xi-well regularization (caps well depth for Radau tractability)


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
# ANALYTIC PUMP POTENTIALS
# ================================================================
# The Xi-corrected scalar pump includes (1/2)ln|tau^2-1| which
# creates wells at tau=+-1 (where Xi=0, rho=rho_c/2). These wells
# destructively interfere with the bounce barrier.

def compute_xi_pump_arrays(
    t: np.ndarray,
    a: np.ndarray,
    i_bounce: int,
    w: float,
    rho_c_val: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Xi-corrected scalar and tensor pump potentials z''/z analytically.

    Scalar: L_xi = const + alpha*ln(u) - ln|tau| + (1/2)*ln|tau^2-1|
    Tensor: L_t  = const + alpha*ln(u)           (= a''/a)

    Returns (U_scalar, U_tensor) arrays on the t grid.
    """
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

        # --- Xi-corrected scalar pump ---
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

        # --- Tensor pump (no Xi correction) ---
        L_t_tau = 2.0 * alpha * tau / u
        L_t_tautau = 2.0 * alpha * (1.0 - tau2) / (u * u)

        U_tensor[i] = Omega * Omega * (L_t_tautau + L_t_tau * L_t_tau) \
                     + Omega * Omega_tau * L_t_tau

    return U_scalar, U_tensor


def compute_xi_pump_general(
    t: np.ndarray,
    x: np.ndarray,
    H: np.ndarray,
    a: np.ndarray,
    eta: np.ndarray,
    i_bounce: int,
    rho_c_val: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Xi-corrected pump from numerical background (for varying w).

    z_xi = a * sqrt(|Xi|) * sqrt(2*rho) / H
    Tensor: z_T = a
    z''/z = L'' + L'^2  (conformal time derivatives via np.gradient)
    """
    H_c = np.sqrt(8.0 * np.pi * rho_c_val / 3.0)
    H2_floor = H_c**2 * TAU2_MIN

    Xi = np.abs(1.0 - 2.0 * x)

    L_s = np.log(a) \
        + 0.5 * np.log(np.maximum(Xi, XI_EPS)) \
        + 0.5 * np.log(2.0 * np.maximum(x, 1e-30) * rho_c_val) \
        - 0.5 * np.log(H**2 + H2_floor)

    L_t = np.log(a)

    dL_s = np.gradient(L_s, eta)
    d2L_s = np.gradient(dL_s, eta)
    U_scalar = d2L_s + dL_s**2

    dL_t = np.gradient(L_t, eta)
    d2L_t = np.gradient(dL_t, eta)
    U_tensor = d2L_t + dL_t**2

    return U_scalar, U_tensor


# ================================================================
# STEP 1: BACKGROUND EVOLUTION
# ================================================================

def solve_background(
    w: float,
    tau_max: float = 200,
    N_points: int = 500000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Solve MSCF background through bounce.
    Returns t, x, H, a, eta, i_bounce arrays.
    """
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


def compute_kinetic_matrix(
    x: np.ndarray,
    H: np.ndarray,
    a: np.ndarray,
    w: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2x2 kinetic matrix components at each time step.
    Used for mixing angle computation (NOT for pump potentials).
    """
    cs2_m = w
    Hdot = -4.0 * np.pi * (1.0 + w) * x * (1.0 - 2.0 * x)
    Xi = 1.0 - 2.0 * x

    V_pp = 108.0 * alpha_coeff * H**2
    A = M2 / 3.0 + V_pp / 2.0
    B = 2.0 * M2 / 3.0 + V_pp / 2.0
    G_grav = np.where(np.abs(B) > 1e-30, 3.0 * M2 * A * V_pp / (2.0 * B**2), 0.0)

    H2_reg = H**2 + eps_reg**2
    eps_m = -Hdot / H2_reg
    G_matt = eps_m * M2 / cs2_m

    F_grav = -M2 * np.ones_like(x)
    F_matt = eps_m * M2

    return G_grav, G_matt, F_grav, F_matt, eps_m, Xi, Hdot


def compute_mixing_angle(
    G_grav: np.ndarray,
    G_matt: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the mixing angle theta(eta) and dtheta/deta from eigenvalue fractions.
    theta = arctan2(sqrt(|G_grav|), sqrt(|G_matt|))
    0 = pure matter, pi/2 = pure gravity
    """
    theta = np.arctan2(np.sqrt(np.abs(G_grav)),
                       np.sqrt(np.abs(G_matt) + 1e-60))
    dtheta_deta = np.gradient(theta, eta)
    dtheta_deta = gaussian_filter1d(dtheta_deta, sigma=5)
    return theta, dtheta_deta


# ================================================================
# STEP 2: MODE EVOLUTION (QR-stabilized)
# ================================================================
# Evolves a 4x4 fundamental matrix with periodic QR
# reorthogonalization to prevent the stiff z''/z ~ 2/eta^2 barrier
# from numerically killing subdominant modes.

# QR parameters
QR_MAX_GROWTH: float = 1.0
QR_SAFETY: float = 0.5
QR_RTOL: float = 1e-10
QR_ATOL: float = 1e-12


def evolve_coupled_mode(
    k: float,
    eta_arr: np.ndarray,
    U_grav_interp: interp1d,
    U_matt_interp: interp1d,
    theta_dot_interp: interp1d,
    w: float,
    eta_start: float,
    eta_end: float,
) -> dict | None:
    """
    QR-stabilized evolution of the full 4-component coupled system.

    State: Y = [v_matt, v_matt', v_grav, v_grav']

    v_m'' + (cs2_m k^2 - U_m) v_m = theta' (2 v_g' + theta' v_g)
    v_g'' + (k^2 - U_g) v_g = -theta' (2 v_m' + theta' v_m)

    Returns transfer coefficient T^2 for the adiabatic mode.
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

    # Bogoliubov extraction (rescaled)
    omega_final = np.sqrt(cs2_m) * k
    alpha_k = (np.sqrt(omega_final / 2.0) * v1_final
               + 1j * v1p_final / np.sqrt(2.0 * omega_final))
    beta_k = (np.sqrt(omega_final / 2.0) * v1_final
              - 1j * v1p_final / np.sqrt(2.0 * omega_final))

    T2_rescaled = abs(alpha_k)**2 + abs(beta_k)**2
    T2_physical = T2_rescaled * np.exp(2.0 * logS_common)

    iso_leakage = abs(v2_final)**2 / (abs(v1_final)**2 + 1e-60)

    return {
        'k': float(k),
        'T2': float(T2_physical),
        'alpha2': float(abs(alpha_k)**2 * np.exp(2.0 * logS_common)),
        'beta2': float(abs(beta_k)**2 * np.exp(2.0 * logS_common)),
        'iso_leakage': float(iso_leakage),
        'logS': float(logS_common),
        'n_segments': n_segments,
    }


# ================================================================
# STEP 3: FULL PIPELINE (Part A)
# ================================================================

def run_coupled_evolution(
    w: float = 1.0,
    label: str = "symmetric",
) -> tuple[list[dict], list[dict], dict]:
    """Run the full coupled mode evolution for a given w."""

    print(f"\n{'='*60}")
    print(f"COUPLED MODE EVOLUTION: w={w}, {label}")
    print(f"{'='*60}")

    # Background
    print("\n[1/4] Solving background...")
    t, x, H, a, eta, i_bounce = solve_background(w)
    print(f"  Bounce at t={t[i_bounce]:.4f}, x={x[i_bounce]:.6f}")

    # Kinetic matrix (for mixing angle only)
    print("[2/4] Computing kinetic matrix and pump potentials...")
    G_grav, G_matt, F_grav, F_matt, eps_m, Xi, Hdot = \
        compute_kinetic_matrix(x, H, a, w)

    # Xi-corrected analytic pump potentials
    U_scalar, U_tensor = compute_xi_pump_arrays(t, a, i_bounce, w)

    # Mixing angle from eigenvalue fractions
    theta, dtheta_deta = compute_mixing_angle(G_grav, G_matt, eta)

    theta_dot_max = float(np.max(np.abs(dtheta_deta)))
    print(f"  Max |dtheta/deta| = {theta_dot_max:.4f}")
    print(f"  theta range: [{np.min(theta):.4f}, {np.max(theta):.4f}]")

    # Build interpolators
    interp_kw = dict(kind='linear', bounds_error=False)
    U_grav_interp = interp1d(eta, U_tensor,
                             fill_value=(U_tensor[0], U_tensor[-1]), **interp_kw)
    U_matt_interp = interp1d(eta, U_scalar,
                             fill_value=(U_scalar[0], U_scalar[-1]), **interp_kw)
    theta_dot_interp = interp1d(eta, dtheta_deta,
                                fill_value=0.0, **interp_kw)

    # Integration range
    eta_margin = 20.0
    eta_start = max(eta[2], -eta_margin)
    eta_end = min(eta[-3], eta_margin)
    print(f"  Integration range: eta in [{eta_start:.2f}, {eta_end:.2f}]")

    # k grid: 15 points spanning the key range
    print("[3/4] Evolving modes (coupled)...")
    k_values = np.logspace(-4, 1.5, 15)

    results_coupled = []
    for i, k in enumerate(k_values):
        t0 = time.time()
        print(f"  k = {k:.4e} ({i+1}/{len(k_values)})...", end='')
        res = evolve_coupled_mode(k, eta, U_grav_interp, U_matt_interp,
                                  theta_dot_interp, w, eta_start, eta_end)
        dt = time.time() - t0
        if res is not None:
            results_coupled.append(res)
            print(f" T2={res['T2']:.4e}  ({dt:.1f}s)")
        else:
            print(f" FAILED ({dt:.1f}s)")

    # Uncoupled comparison (theta' = 0)
    print("\n[3b/4] Running uncoupled comparison (theta'=0)...")
    zero_interp = interp1d(eta, np.zeros_like(eta), kind='linear',
                           fill_value=0.0, bounds_error=False)

    results_uncoupled = []
    for i, k in enumerate(k_values):
        t0 = time.time()
        print(f"  k = {k:.4e} ({i+1}/{len(k_values)})...", end='')
        res = evolve_coupled_mode(k, eta, U_grav_interp, U_matt_interp,
                                  zero_interp, w, eta_start, eta_end)
        dt = time.time() - t0
        if res is not None:
            results_uncoupled.append(res)
            print(f" T2={res['T2']:.4e}  ({dt:.1f}s)")
        else:
            print(f" FAILED ({dt:.1f}s)")

    # Print comparison
    print(f"\n[4/4] Results:")
    print(f"{'k':>10} {'T2_coupled':>12} {'T2_uncoupled':>14} {'ratio':>8} {'iso_leak':>10}")
    print("-" * 60)
    for rc, ru in zip(results_coupled, results_uncoupled):
        ratio = rc['T2'] / (ru['T2'] + 1e-60)
        print(f"{rc['k']:10.4e} {rc['T2']:12.4f} {ru['T2']:14.4f} "
              f"{ratio:8.3f} {rc['iso_leakage']:10.4e}")

    return results_coupled, results_uncoupled, {
        'w': float(w),
        'label': label,
        'theta_dot_max': theta_dot_max,
        'eta_range': [float(eta_start), float(eta_end)],
    }


# ================================================================
# Part B: ASYMMETRIC w TRANSITION
# ================================================================

def solve_background_asymmetric(
    w_bounce: float = 1.0,
    w_post: float = 1.0/3.0,
    Delta_t_w: float = 5.0,
    t_transition: float = 10.0,
    tau_max: float = 200,
    N_points: int = 500000,
) -> tuple | None:
    """
    Solve background with w transitioning from w_bounce to w_post.

    w(t) = (w_bounce + w_post)/2 + (w_bounce - w_post)/2 * tanh((t - t_trans)/Delta_t)
    """
    def w_of_t(t):
        return ((w_bounce + w_post) / 2.0
                + (w_bounce - w_post) / 2.0 * np.tanh((t - t_transition) / Delta_t_w))

    def rhs(t, y):
        x, H, lna = y
        x = np.clip(x, 1e-30, 1.0 - 1e-15)
        w_t = w_of_t(t)
        dH = -4.0 * np.pi * (1.0 + w_t) * x * (1.0 - 2.0 * x)
        dx = -3.0 * H * (1.0 + w_t) * x
        dlna = H
        return [dx, dH, dlna]

    x0 = 0.001
    H0 = -np.sqrt(8.0 * np.pi / 3.0 * x0 * (1.0 - x0))

    sol = solve_ivp(rhs, (-tau_max, tau_max), [x0, H0, 0.0],
                    t_eval=np.linspace(-tau_max, tau_max, N_points),
                    method='Radau', rtol=1e-13, atol=1e-16)

    if not sol.success:
        return None

    t = sol.t
    x = sol.y[0]
    H = sol.y[1]
    a = np.exp(sol.y[2])
    w_arr = np.array([w_of_t(ti) for ti in t])
    cs2_arr = w_arr.copy()

    i_bounce = np.argmin(np.abs(H))

    dt = np.diff(t)
    deta = dt / a[:-1]
    eta = np.concatenate([[0], np.cumsum(deta)])
    eta -= eta[i_bounce]

    return t, x, H, a, eta, i_bounce, w_arr, cs2_arr


def run_asymmetric_evolution() -> dict[str, list[dict]]:
    """Run mode evolution with asymmetric w transition."""

    print(f"\n{'='*60}")
    print("ASYMMETRIC w TRANSITION: w=1 (bounce) -> w=1/3 (radiation)")
    print(f"{'='*60}")

    scenarios = [
        {'w_bounce': 1.0, 'w_post': 1.0/3.0, 'Delta_t': 2.0, 't_trans': 5.0,
         'label': 'fast_early'},
        {'w_bounce': 1.0, 'w_post': 1.0/3.0, 'Delta_t': 10.0, 't_trans': 10.0,
         'label': 'slow_mid'},
        {'w_bounce': 1.0, 'w_post': 1.0/3.0, 'Delta_t': 5.0, 't_trans': 20.0,
         'label': 'mid_late'},
    ]

    all_results: dict[str, list[dict]] = {}

    for scen in scenarios:
        print(f"\n  Scenario: {scen['label']} "
              f"(Delta_t={scen['Delta_t']}, t_trans={scen['t_trans']})")

        bg_result = solve_background_asymmetric(
            w_bounce=scen['w_bounce'], w_post=scen['w_post'],
            Delta_t_w=scen['Delta_t'], t_transition=scen['t_trans'])

        if bg_result is None:
            print(f"    FAILED: background solve failed")
            all_results[scen['label']] = []
            continue

        t, x, H, a, eta, i_bounce, w_arr, cs2_arr = bg_result

        print(f"    Bounce at t={t[i_bounce]:.4f}")
        print(f"    w at bounce: {w_arr[i_bounce]:.4f}")
        i_50 = np.argmin(np.abs(t - 50))
        print(f"    w at t=50: {w_arr[i_50]:.4f}")

        # Kinetic matrix (for mixing angle)
        Hdot = np.zeros_like(t)
        for i in range(len(t)):
            Hdot[i] = -4.0 * np.pi * (1.0 + w_arr[i]) * x[i] * (1.0 - 2.0 * x[i])

        H2_reg = H**2 + eps_reg**2
        eps_m = -Hdot / H2_reg
        V_pp = 108.0 * alpha_coeff * H**2
        A_arr = M2 / 3.0 + V_pp / 2.0
        B_arr = 2.0 * M2 / 3.0 + V_pp / 2.0
        G_grav = np.where(np.abs(B_arr) > 1e-30,
                          3.0 * M2 * A_arr * V_pp / (2.0 * B_arr**2), 0.0)
        G_matt = eps_m * M2 / cs2_arr

        # Xi-corrected pump from numerical background (handles varying w)
        U_scalar, U_tensor = compute_xi_pump_general(
            t, x, H, a, eta, i_bounce, rho_c)

        # Mixing angle
        theta, dtheta_deta = compute_mixing_angle(G_grav, G_matt, eta)

        # Build interpolators
        interp_kw = dict(kind='linear', bounds_error=False)
        U_grav_interp = interp1d(eta, U_tensor,
                                 fill_value=(U_tensor[0], U_tensor[-1]), **interp_kw)
        U_matt_interp = interp1d(eta, U_scalar,
                                 fill_value=(U_scalar[0], U_scalar[-1]), **interp_kw)
        theta_dot_interp = interp1d(eta, dtheta_deta,
                                    fill_value=0.0, **interp_kw)

        eta_start = max(eta[2], -20.0)
        eta_end = min(eta[-3], 20.0)

        # k scan (15 points)
        k_values = np.logspace(-4, 1, 15)

        print(f"    Evolving {len(k_values)} modes...")
        results = []
        for k in k_values:
            res = evolve_coupled_mode(k, eta, U_grav_interp, U_matt_interp,
                                      theta_dot_interp,
                                      w_arr[0],  # use initial w for ICs
                                      eta_start, eta_end)
            if res is not None:
                results.append(res)

        print(f"\n    {'k':>10} {'T2':>10} {'iso_leak':>10}")
        print("    " + "-" * 35)
        for r in results:
            print(f"    {r['k']:10.4e} {r['T2']:10.4f} {r['iso_leakage']:10.4e}")

        all_results[scen['label']] = results

    return all_results


# ================================================================
# Part C: ANALYSIS & OUTPUT
# ================================================================

def main() -> None:
    print("=" * 70)
    print("MSCF Full 2x2 Coupled Mode Evolution")
    print("  + Asymmetric w Transition")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Xi-well regularization: XI_EPS = {XI_EPS}")
    print("=" * 70)

    t_total = time.time()

    # ---- Part A: Full coupled evolution with w=1 (symmetric) ----
    results_coupled, results_uncoupled, meta = \
        run_coupled_evolution(w=1.0, label="symmetric_w1")

    # ---- Part B: Asymmetric w transition ----
    results_asymmetric = run_asymmetric_evolution()

    # ============================================================
    # ANALYSIS: Does T^2(k) oscillate?
    # ============================================================

    dt_total = time.time() - t_total

    print(f"\n{'='*60}")
    print("KEY QUESTION: Does T^2(k) oscillate?")
    print(f"{'='*60}")

    if results_coupled:
        k_arr = np.array([r['k'] for r in results_coupled])
        T2_arr = np.array([r['T2'] for r in results_coupled])
        T2_unc = np.array([r['T2'] for r in results_uncoupled])

        crossings = int(np.sum(np.diff(np.sign(T2_arr - 1.0)) != 0))
        dT2 = np.diff(T2_arr)
        sign_changes = int(np.sum(np.diff(np.sign(dT2)) != 0))

        print(f"\nCoupled T^2(k):")
        print(f"  Range: [{np.min(T2_arr):.4f}, {np.max(T2_arr):.4f}]")
        print(f"  Crossings of T^2=1: {crossings}")
        print(f"  Sign changes in dT^2/dk: {sign_changes}")
        print(f"  Oscillatory? {'YES' if sign_changes > 3 else 'NO (monotonic-ish)'}")

        print(f"\nCoupling effect (coupled vs uncoupled):")
        ratio = T2_arr / (T2_unc + 1e-60)
        coupling_significant = bool(np.max(np.abs(ratio - 1.0)) > 0.1)
        print(f"  Max ratio: {np.max(ratio):.4f}")
        print(f"  Min ratio: {np.min(ratio):.4f}")
        print(f"  Coupling changes T^2 by >10%: "
              f"{'YES' if coupling_significant else 'NO'}")
    else:
        crossings = 0
        sign_changes = 0
        coupling_significant = False

    asym_oscillatory = False
    for name, res_list in results_asymmetric.items():
        if res_list:
            T2_a = np.array([r['T2'] for r in res_list])
            cross_a = int(np.sum(np.diff(np.sign(T2_a - 1.0)) != 0))
            sc_a = int(np.sum(np.diff(np.sign(np.diff(T2_a))) != 0))
            print(f"\n  Asymmetric {name}:")
            print(f"    T^2 range: [{np.min(T2_a):.4f}, {np.max(T2_a):.4f}]")
            print(f"    T^2=1 crossings: {cross_a}")
            print(f"    dT^2/dk sign changes: {sc_a}")
            if cross_a > 2:
                asym_oscillatory = True

    any_oscillatory = crossings > 2 or asym_oscillatory

    if coupling_significant and any_oscillatory:
        verdict = "OSCILLATORY_COUPLING"
        interpretation = ("Coupling produces oscillatory T^2(k) crossing unity. "
                          "This could improve low-ell fit vs both LCDM and smooth suppression.")
    elif coupling_significant:
        verdict = "COUPLING_SIGNIFICANT"
        interpretation = ("Coupling changes T^2 by >10% but no oscillatory crossings. "
                          "Monotonic modification only.")
    elif any_oscillatory:
        verdict = "OSCILLATORY_ASYMMETRIC"
        interpretation = ("Asymmetric w transition produces oscillatory T^2 "
                          "even without coupling. Worth investigating further.")
    else:
        verdict = "NULL"
        interpretation = ("Neither coupling nor asymmetric w produces significant "
                          "oscillatory features. Both approximations were adequate.")

    print(f"\n  Verdict: {verdict}")
    print(f"  {interpretation}")
    print(f"\n  Total time: {dt_total:.1f}s")

    # ---- Save JSON ----
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'coupled': results_coupled,
        'uncoupled': results_uncoupled,
        'asymmetric': {k: v for k, v in results_asymmetric.items()},
        'meta': meta,
        'oscillation_diagnostics': {
            'coupled_crossings': crossings,
            'coupled_sign_changes': sign_changes,
            'coupling_significant': coupling_significant,
            'any_oscillatory': any_oscillatory,
        },
        'verdict': verdict,
        'interpretation': interpretation,
        'total_time_s': float(dt_total),
    }

    json_path = OUTPUT_DIR / "full_coupled_results.json"
    save_json(json_path, output)
    print(f"\n  Results saved to: {json_path}")

    # ---- Save T^2(k) for CAMB injection ----
    if results_coupled:
        k_out = np.array([r['k'] for r in results_coupled])
        T2_out = np.array([r['T2'] for r in results_coupled])

        dat_path = OUTPUT_DIR / "T2_coupled_w1.dat"
        np.savetxt(dat_path, np.column_stack([k_out, T2_out]),
                   header='k_internal  T2_coupled', fmt='%.8e')
        print(f"  T^2(k) saved to: {dat_path}")

    for name, res_list in results_asymmetric.items():
        if res_list:
            k_a = np.array([r['k'] for r in res_list])
            T2_a = np.array([r['T2'] for r in res_list])
            dat_path = OUTPUT_DIR / f"T2_asym_{name}.dat"
            np.savetxt(dat_path, np.column_stack([k_a, T2_a]),
                       header=f'k_internal  T2_{name}', fmt='%.8e')

    if any_oscillatory:
        print("\nOscillatory features found! Next steps:")
        print("  1. Feed T2_coupled_w1.dat into Commander pipeline")
        print("  2. Re-run cobaya with Commander likelihood")
        print("  3. Compare Delta_lnL vs LCDM")


if __name__ == '__main__':
    main()
