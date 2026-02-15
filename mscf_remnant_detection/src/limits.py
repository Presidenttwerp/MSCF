"""
Experimental exclusion limits and neutrino floor.

Parameterized curves for LZ, XENONnT, PandaX-4T, neutrino fog,
and super-heavy DM constraints (ancient mica, MACRO).

All limits expressed as σ_SI (per-nucleon spin-independent) in cm².
"""
import numpy as np
from . import constants as c


def lz_limit(M_DM_GeV):
    """
    LZ 2024 exclusion limit on σ_SI.

    Low mass: steeply rising (kinematic suppression).
    Minimum: σ ≈ 9.2e-48 cm² at M ≈ 36 GeV.
    High mass: σ ∝ M_DM (constant rate → σ ∝ 1/n ∝ M).

    Parameters
    ----------
    M_DM_GeV : float or array
        DM mass in GeV.

    Returns
    -------
    sigma_cm2 : float or array
        90% CL upper limit in cm².
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma = np.full_like(M, np.inf)

    sigma_min = 9.2e-48  # cm² at 36 GeV
    M_min = 36.0         # GeV

    # Low mass (M < 36 GeV): approximate as power law
    low = M < M_min
    sigma[low] = sigma_min * (M_min / M[low])**4

    # High mass (M >= 36 GeV): linear scaling
    high = M >= M_min
    sigma[high] = sigma_min * (M[high] / M_min)

    return sigma


def xenonnt_limit(M_DM_GeV):
    """
    XENONnT 2024 exclusion limit on σ_SI.

    Minimum: σ ≈ 2.6e-47 cm² at M ≈ 28 GeV.
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma = np.full_like(M, np.inf)

    sigma_min = 2.6e-47
    M_min = 28.0

    low = M < M_min
    sigma[low] = sigma_min * (M_min / M[low])**4

    high = M >= M_min
    sigma[high] = sigma_min * (M[high] / M_min)

    return sigma


def pandax_limit(M_DM_GeV):
    """
    PandaX-4T 2024 exclusion limit.

    Minimum: σ ≈ 3.8e-47 cm² at M ≈ 40 GeV.
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma = np.full_like(M, np.inf)

    sigma_min = 3.8e-47
    M_min = 40.0

    low = M < M_min
    sigma[low] = sigma_min * (M_min / M[low])**4

    high = M >= M_min
    sigma[high] = sigma_min * (M[high] / M_min)

    return sigma


def neutrino_floor(M_DM_GeV):
    """
    Neutrino fog/floor for Xe-based detectors.

    Approximate: σ_ν ≈ 10⁻⁴⁹ × (M/GeV) cm² at high mass,
    with minimum ~10⁻⁴⁹ cm² around 6 GeV.
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma = np.full_like(M, np.inf)

    sigma_min = 1e-49  # cm²
    M_min = 6.0        # GeV

    low = M < M_min
    sigma[low] = sigma_min * (M_min / M[low])**3

    high = M >= M_min
    sigma[high] = sigma_min * (M[high] / M_min)

    return sigma


def ancient_mica_limit(M_DM_GeV):
    """
    Ancient mica track constraints for super-heavy DM (M > 10^10 GeV).

    σ ≲ 10⁻²⁸ cm² for M ~ 10¹⁵ GeV (Price & Salamon 1986).
    Scales as σ ∝ M at high mass.
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma_ref = 1e-28   # cm² at M_ref
    M_ref = 1e15        # GeV
    return sigma_ref * (M / M_ref)


def macro_limit(M_DM_GeV):
    """
    MACRO experiment constraints for magnetic monopoles / super-heavy DM.

    σ ≲ 10⁻²⁵ cm² for M ~ 10¹⁶ GeV.
    """
    M = np.asarray(M_DM_GeV, dtype=float)
    sigma_ref = 1e-25
    M_ref = 1e16
    return sigma_ref * (M / M_ref)


def compare_with_limits(sigma_n_cm2, M_DM_GeV):
    """
    Compare a per-nucleon cross-section with all limits.

    Returns dict of ratios σ_pred / σ_limit.
    """
    M = float(M_DM_GeV)
    return {
        "sigma_n_cm2": sigma_n_cm2,
        "M_DM_GeV": M,
        "ratio_LZ": sigma_n_cm2 / lz_limit(np.array([M]))[0],
        "ratio_XENONnT": sigma_n_cm2 / xenonnt_limit(np.array([M]))[0],
        "ratio_PandaX": sigma_n_cm2 / pandax_limit(np.array([M]))[0],
        "ratio_nu_floor": sigma_n_cm2 / neutrino_floor(np.array([M]))[0],
        "ratio_mica": sigma_n_cm2 / ancient_mica_limit(np.array([M]))[0],
    }
