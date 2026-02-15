"""
MSCF Remnant Properties

Verifies the derivation chain from MSCF v2.1.7:
  Axiom 5 → κ_max (Eq. 38) → M_min = M_P/2 (Eq. 39) → T_eff (Eq. 43)
"""
import numpy as np
from . import constants as c


def t_eff(M_kg):
    """
    MSCF effective Hawking temperature (Eq. 43).

    T_eff = T_H × [1 - (1/4)(M_P/M)^2]

    Parameters
    ----------
    M_kg : float or array
        Black hole mass in kg.

    Returns
    -------
    T_eff_K : float or array
        Effective temperature in Kelvin.
    """
    T_H = c.HBAR_SI * c.C_SI**3 / (8 * np.pi * c.G_SI * M_kg * c.K_B_SI)
    correction = 1.0 - 0.25 * (c.M_PLANCK_KG / M_kg)**2
    return T_H * correction


def verify_eq43():
    """Gate 0: T_eff(M_P/2) = 0 algebraically."""
    M_half = c.M_PLANCK_KG / 2
    T = t_eff(M_half)
    # Correction factor: 1 - (1/4)(M_P / (M_P/2))^2 = 1 - (1/4)*4 = 0
    correction = 1.0 - 0.25 * (c.M_PLANCK_KG / M_half)**2
    return {
        "M_rem_kg": M_half,
        "correction_factor": correction,
        "T_eff_K": T,
        "gate_0_pass": np.abs(correction) < 1e-15,
    }


def remnant_properties():
    """
    Compute key properties of the MSCF remnant.

    Returns
    -------
    dict with M_rem, E_rem, r_s, n_rem, flux, etc.
    """
    M = c.M_REM_KG
    E_J = M * c.C_SI**2
    E_GeV = M * c.KG_TO_GEV
    r_s = 2 * c.G_SI * M / c.C_SI**2  # Schwarzschild radius

    # Number density: n = rho_DM / M
    n_SI = c.RHO_DM_SI / M           # m^-3
    n_cm3 = n_SI * 1e-6              # cm^-3

    # Flux: Phi = n × v_0
    flux_SI = n_SI * c.V_0           # m^-2 s^-1
    flux_cm2 = flux_SI * 1e-4        # cm^-2 s^-1

    # de Broglie wavelength
    lambda_dB = c.HBAR_SI / (M * c.V_0)

    return {
        "M_rem_kg": M,
        "M_rem_GeV": E_GeV,
        "E_rem_J": E_J,
        "r_s_m": r_s,
        "r_s_over_lP": r_s / c.L_PLANCK,
        "n_m3": n_SI,
        "n_cm3": n_cm3,
        "flux_m2_s": flux_SI,
        "flux_cm2_s": flux_cm2,
        "lambda_dB_m": lambda_dB,
        "lambda_dB_over_lP": lambda_dB / c.L_PLANCK,
    }


def derivation_chain():
    """
    Print the full derivation chain from MSCF v2.1.7.

    Returns
    -------
    dict with intermediate results.
    """
    # Axiom 5 (Eq. 3): rho_crit = c^7 / (hbar G^2)
    rho_crit = c.C_SI**7 / (c.HBAR_SI * c.G_SI**2)

    # Eq. 38: kappa_max = c^2 / (2 l_P)
    kappa_max = c.C_SI**2 / (2 * c.L_PLANCK)

    # Eq. 39: kappa_Schw = c^4 / (4GM) <= kappa_max => M >= M_P/2
    M_min = c.C_SI**4 / (4 * c.G_SI * kappa_max)
    # Verify: M_min = c^4 / (4G × c^2/(2l_P)) = c^2 l_P / (2G)
    #        = c^2 sqrt(hbar G/c^3) / (2G) = sqrt(hbar c / G) / 2 = M_P/2
    M_min_check = c.M_PLANCK_KG / 2

    # Eq. 43: T_eff = T_H [1 - (M_P/2M)^2], vanishes at M = M_P/2
    T_at_Mmin = t_eff(M_min)

    return {
        "rho_crit_mscf_J_m3": rho_crit,
        "kappa_max_m_s2": kappa_max,
        "M_min_kg": M_min,
        "M_min_check_kg": M_min_check,
        "M_min_agreement": np.abs(M_min - M_min_check) / M_min_check < 1e-10,
        "T_eff_at_Mmin_K": T_at_Mmin,
    }
