"""
Greybody factor solver for Schwarzschild angular momentum barrier.

Solves the Regge-Wheeler scattering problem:
    d²ψ/dr*² + [ω² - V(r*)] ψ = 0

with purely ingoing initial condition at the horizon (r* → -∞):
    ψ = exp(-iωr*),  ψ' = -iω exp(-iωr*)

At large r* the solution decomposes as:
    ψ ~ A_out exp(+iωr*) + A_in exp(-iωr*)

The reflection and transmission coefficients are:
    |R_b|² = |A_in|² / |A_out|²    (reflected fraction at barrier)
    |T_b|² = 1 / |A_out|²          (transmitted fraction through barrier)

Flux conservation: |R_b|² + |T_b|² = 1

For MSCF echo amplitudes: A_n(ω) = T_b²(ω) × R_b^{n-1}(ω)
(first echo transmits twice through barrier, each subsequent echo
reflects once off the barrier from inside the cavity).
"""

import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .config import (
    RSTAR_INFINITY, SOLVER_RTOL, SOLVER_ATOL,
    GREYBODY_N_GRID, GREYBODY_MOMEGA_MIN, GREYBODY_MOMEGA_MAX,
    MSUN_S, CACHE_DIR,
)


# ---- Potential evaluation ----

def _regge_wheeler_potential(r, l):
    """V_RW(r) = (1 - 2/r) [l(l+1)/r² - 6/r³], geometric units M=1."""
    if r <= 2.0:
        return 0.0
    f = 1.0 - 2.0 / r
    return f * (l * (l + 1) / r**2 - 6.0 / r**3)


def _zerilli_potential(r, l):
    """V_Z(r), isospectral with RW. Geometric units M=1."""
    if r <= 2.0:
        return 0.0
    f = 1.0 - 2.0 / r
    Lambda = (l - 1) * (l + 2) / 2.0
    numer = (2.0 * Lambda**2 * (Lambda + 1) * r**3
             + 6.0 * Lambda**2 * r**2
             + 18.0 * Lambda * r + 18.0)
    denom = r**3 * (Lambda * r + 3.0)**2
    return f * numer / denom


def _schwarzschild_r_from_rstar(rstar):
    """Invert r*(r) = r + 2 ln|r/2 - 1| via Newton iteration."""
    if rstar > 10.0:
        r = rstar
    elif rstar < -5.0:
        r = 2.0 + 2.0 * np.exp((rstar - 2.0) / 2.0)
    else:
        r = max(2.5, rstar + 2.0)

    for _ in range(60):
        x = r / 2.0 - 1.0
        if abs(x) < 1e-15:
            r += 1e-12
            continue
        f = r + 2.0 * np.log(abs(x)) - rstar
        fp = r / (r - 2.0)
        if abs(fp) < 1e-30:
            break
        dr = f / fp
        r -= dr
        if r <= 2.0:
            r = 2.0 + abs(dr) * 0.1
        if abs(dr) < 1e-14 * max(1.0, abs(r)):
            break
    return r


def _build_potential_interpolator(l, parity='odd',
                                  rstar_min=-50.0,
                                  rstar_max=RSTAR_INFINITY,
                                  n_grid=10000):
    """Build cubic interpolator for V(r*) on a dense grid."""
    rstar_arr = np.linspace(rstar_min, rstar_max, n_grid)
    V_func = _regge_wheeler_potential if parity == 'odd' else _zerilli_potential

    r_arr = np.array([_schwarzschild_r_from_rstar(rs) for rs in rstar_arr])
    V_arr = np.array([V_func(r, l) for r in r_arr])

    return interp1d(rstar_arr, V_arr, kind='cubic',
                    bounds_error=False, fill_value=0.0)


# Module-level cache for interpolators
_V_interp_cache = {}


def _get_V_interp(l, parity='odd'):
    """Get or build cached potential interpolator."""
    key = (l, parity)
    if key not in _V_interp_cache:
        _V_interp_cache[key] = _build_potential_interpolator(l, parity)
    return _V_interp_cache[key]


def _find_integration_start(V_interp, omega, threshold=1e-10):
    """
    Find r* where V(r*)/ω² < threshold on the horizon side.

    The potential decays exponentially as r* → -∞ (near horizon).
    Starting too far left wastes integration effort and accumulates
    numerical error through 100s of oscillation cycles.
    """
    omega2 = omega * omega
    # Search leftward from barrier peak (r*≈0 for Schwarzschild)
    for rstar in np.arange(-5.0, -50.0, -1.0):
        V = float(V_interp(rstar))
        if abs(V) / omega2 < threshold:
            return rstar
    return -50.0


# ---- Core scattering solver ----

def greybody_factor(omega, l=2, parity='odd'):
    """
    Compute greybody (reflection/transmission) factors at frequency omega.

    Solves the scattering problem with purely ingoing BC at horizon.

    Parameters
    ----------
    omega : float
        Frequency in geometric units (Mω, dimensionless). Must be > 0.
    l : int
        Angular momentum quantum number.
    parity : str
        'odd' (Regge-Wheeler) or 'even' (Zerilli).

    Returns
    -------
    dict with keys:
        'Rb2'  : float — |R_b|² (barrier reflectivity)
        'Tb2'  : float — |T_b|² (barrier transmissivity)
        'Rb'   : float — |R_b| (amplitude)
        'Tb'   : float — |T_b| (amplitude)
        'A_in' : complex — ingoing amplitude at infinity
        'A_out': complex — outgoing amplitude at infinity
        'flux_error' : float — |1 - Rb2 - Tb2| (should be < 1e-8)
    """
    if omega <= 0:
        raise ValueError(f"omega must be > 0, got {omega}")

    V_interp = _get_V_interp(l, parity)
    omega2 = omega * omega

    # Adaptive starting point: start close enough to the barrier
    # that V/ω² is negligible, but not so far that we waste effort.
    rstar_0 = _find_integration_start(V_interp, omega)

    # Initial conditions: purely ingoing wave ψ = exp(-iωr*)
    phase0 = -omega * rstar_0
    psi0 = complex(np.cos(phase0), np.sin(phase0))
    dpsi0 = -1j * omega * psi0

    y0 = [psi0.real, psi0.imag, dpsi0.real, dpsi0.imag]

    def rhs(rstar, y):
        psi = complex(y[0], y[1])
        dpsi = complex(y[2], y[3])
        V = float(V_interp(rstar))
        ddpsi = (V - omega2) * psi
        return [dpsi.real, dpsi.imag, ddpsi.real, ddpsi.imag]

    rstar_f = RSTAR_INFINITY

    # Adaptive max_step: ≤ 1/10 of wavelength = 2π/ω
    wavelength = 2.0 * np.pi / omega
    max_step = min(wavelength / 10.0, (rstar_f - rstar_0) / 500)

    sol = solve_ivp(
        rhs, [rstar_0, rstar_f], y0,
        method='DOP853',
        rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed at omega={omega}: {sol.message}")

    # Extract final solution
    psi_f = complex(sol.y[0, -1], sol.y[1, -1])
    dpsi_f = complex(sol.y[2, -1], sol.y[3, -1])

    # Asymptotic decomposition at r* = rstar_f:
    # ψ ~ A_out exp(+iωr*) + A_in exp(-iωr*)
    # ψ' = iω A_out exp(+iωr*) - iω A_in exp(-iωr*)
    # => A_out = (ωψ - iψ') exp(-iωr*) / (2ω)
    # => A_in  = (ωψ + iψ') exp(+iωr*) / (2ω)
    exp_plus = np.exp(1j * omega * rstar_f)
    exp_minus = np.exp(-1j * omega * rstar_f)

    A_out = (omega * psi_f - 1j * dpsi_f) * exp_minus / (2.0 * omega)
    A_in = (omega * psi_f + 1j * dpsi_f) * exp_plus / (2.0 * omega)

    A_out_sq = abs(A_out)**2
    A_in_sq = abs(A_in)**2

    # Greybody factors
    #
    # With IC at horizon: ψ_hor = exp(-iωr*) [ingoing at horizon, B_in=1, B_out=0]
    # At infinity: ψ ~ A_in exp(-iωr*) + A_out exp(+iωr*)
    #
    # Standard scattering interpretation (wave incident from infinity):
    #   Transmission:  T = 1/A_in    →  |T_b|² = 1/|A_in|²
    #   Reflection:    R = A_out/A_in →  |R_b|² = |A_out|²/|A_in|²
    #
    # Flux conservation: |T_b|² + |R_b|² = 1
    Tb2 = 1.0 / A_in_sq if A_in_sq > 0 else 0.0
    Rb2 = A_out_sq / A_in_sq if A_in_sq > 0 else 1.0

    flux_error = abs(1.0 - Rb2 - Tb2)

    return {
        'Rb2': Rb2,
        'Tb2': Tb2,
        'Rb': np.sqrt(Rb2),
        'Tb': np.sqrt(Tb2),
        'A_in': A_in,
        'A_out': A_out,
        'flux_error': flux_error,
    }


def greybody_spectrum(omega_arr, l=2, parity='odd'):
    """
    Compute greybody factors over a frequency array.

    Parameters
    ----------
    omega_arr : array_like
        Array of frequencies Mω (dimensionless).
    l : int
        Angular momentum quantum number.
    parity : str
        'odd' or 'even'.

    Returns
    -------
    dict with keys:
        'Rb2' : np.ndarray — |R_b|²(ω)
        'Tb2' : np.ndarray — |T_b|²(ω)
        'Rb'  : np.ndarray — |R_b|(ω)
        'Tb'  : np.ndarray — |T_b|(ω)
        'flux_error' : np.ndarray
    """
    omega_arr = np.asarray(omega_arr)
    n = len(omega_arr)
    Rb2 = np.zeros(n)
    Tb2 = np.zeros(n)
    flux_err = np.zeros(n)

    for i, om in enumerate(omega_arr):
        if om <= 0:
            Rb2[i] = 1.0
            Tb2[i] = 0.0
            flux_err[i] = 0.0
            continue
        res = greybody_factor(om, l=l, parity=parity)
        Rb2[i] = res['Rb2']
        Tb2[i] = res['Tb2']
        flux_err[i] = res['flux_error']

    return {
        'Rb2': Rb2,
        'Tb2': Tb2,
        'Rb': np.sqrt(Rb2),
        'Tb': np.sqrt(Tb2),
        'flux_error': flux_err,
    }


def greybody_at_physical_freq(f_Hz, Mf_msun, l=2, parity='odd'):
    """
    Compute greybody factors at physical frequency f [Hz] for a BH of mass Mf.

    Parameters
    ----------
    f_Hz : float or array
        Frequency in Hz.
    Mf_msun : float
        Black hole mass in solar masses.
    l : int
        Angular momentum quantum number.
    parity : str
        'odd' or 'even'.

    Returns
    -------
    dict
        Same as greybody_factor/greybody_spectrum, with added 'Momega' key.
    """
    M_sec = Mf_msun * MSUN_S
    f_Hz = np.atleast_1d(np.asarray(f_Hz, dtype=float))
    Momega = 2.0 * np.pi * f_Hz * M_sec

    if len(Momega) == 1:
        result = greybody_factor(float(Momega[0]), l=l, parity=parity)
        result['Momega'] = float(Momega[0])
        return result

    result = greybody_spectrum(Momega, l=l, parity=parity)
    result['Momega'] = Momega
    return result


# ---- Caching ----

def compute_and_cache_greybody(l=2, parity='odd',
                               n_grid=GREYBODY_N_GRID,
                               Momega_min=GREYBODY_MOMEGA_MIN,
                               Momega_max=GREYBODY_MOMEGA_MAX):
    """
    Precompute greybody factors on a log-spaced grid and save to .npz.

    The cached file is universal (depends only on l and parity, not on BH mass).

    Returns
    -------
    dict with 'Momega', 'Rb2', 'Tb2', 'Rb', 'Tb', 'flux_error' arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'greybody_l{l}_{parity}.npz')

    omega_arr = np.logspace(np.log10(Momega_min), np.log10(Momega_max), n_grid)
    result = greybody_spectrum(omega_arr, l=l, parity=parity)
    result['Momega'] = omega_arr

    np.savez(cache_file, **result)
    return result


def load_cached_greybody(l=2, parity='odd'):
    """
    Load precomputed greybody factors from cache.

    Returns None if cache doesn't exist.
    """
    cache_file = os.path.join(CACHE_DIR, f'greybody_l{l}_{parity}.npz')
    if not os.path.exists(cache_file):
        return None
    data = np.load(cache_file)
    return {k: data[k] for k in data.files}


def get_greybody_interpolator(l=2, parity='odd'):
    """
    Get interpolation functions for greybody factors.

    Computes and caches if needed. Returns a dict of interpolation functions.

    Returns
    -------
    dict with keys 'Rb2_interp', 'Tb2_interp' — callable(Momega) → float/array.
    """
    cached = load_cached_greybody(l, parity)
    if cached is None:
        cached = compute_and_cache_greybody(l, parity)

    Momega = cached['Momega']
    Rb2_interp = interp1d(Momega, cached['Rb2'], kind='cubic',
                          bounds_error=False,
                          fill_value=(1.0, 0.0))  # Rb2→1 at low ω, →0 at high ω
    Tb2_interp = interp1d(Momega, cached['Tb2'], kind='cubic',
                          bounds_error=False,
                          fill_value=(0.0, 1.0))  # Tb2→0 at low ω, →1 at high ω

    return {
        'Rb2_interp': Rb2_interp,
        'Tb2_interp': Tb2_interp,
        'Momega_grid': Momega,
        'Rb2_grid': cached['Rb2'],
        'Tb2_grid': cached['Tb2'],
    }
