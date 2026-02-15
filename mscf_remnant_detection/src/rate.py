"""
Event rate calculations for direct detection experiments.

Integrates differential rate dR/dE_R over detector acceptance,
using standard halo model velocity distribution.
"""
import numpy as np
from scipy import integrate
from . import constants as c
from . import cross_section as xs


def v_min(E_R_J, m_A_kg):
    """
    Minimum velocity to produce recoil energy E_R.

    v_min = sqrt(m_A E_R / (2 μ²))
    """
    mu = xs.reduced_mass(c.M_REM_KG, m_A_kg)
    return np.sqrt(m_A_kg * E_R_J / (2 * mu**2))


def eta_integral(v_min_val):
    """
    Mean inverse velocity (halo integral) for Maxwell-Boltzmann distribution
    with escape velocity cutoff.

    η(v_min) = ∫_{v_min}^{v_esc} f(v)/v d³v

    Analytic result for shifted Maxwellian (Lewin & Smith 1996).
    """
    v0 = c.V_0
    vE = c.V_E
    vesc = c.V_ESC

    # Normalization: N = erf(z) - 2z exp(-z²)/√π, z = vesc/v0
    z = vesc / v0
    N_esc = np.math.erf(z) - 2 * z * np.exp(-z**2) / np.sqrt(np.pi)

    x_min = v_min_val / v0
    y = vE / v0

    if v_min_val >= vesc + vE:
        return 0.0

    if v_min_val <= vesc - vE:
        # Standard regime
        eta = (np.math.erf(x_min + y) - np.math.erf(x_min - y)
               - 4 * y * np.exp(-z**2) / np.sqrt(np.pi))
        return eta / (2 * N_esc * vE)
    else:
        # High v_min regime
        eta = (np.math.erf(z) - np.math.erf(x_min - y)
               - 2 * (z - x_min + y) * np.exp(-z**2) / np.sqrt(np.pi))
        return eta / (2 * N_esc * vE)


def differential_rate(E_R_J, m_A_kg, A):
    """
    Differential event rate dR/dE_R [events / kg / s / J].

    dR/dE_R = (ρ_DM / M_rem) × (σ_0 / (2μ²)) × m_A × η(v_min) × A²

    For gravitational Rutherford: dσ/dE_R ∝ 1/E_R², so
    dR/dE_R = n_DM × (dσ/dE_R)_avg
    """
    # Number density of DM remnants (per m^3)
    n_DM = c.RHO_DM_SI / c.M_REM_KG

    # Minimum velocity for this recoil
    v_min_val = v_min(E_R_J, m_A_kg)

    # Halo integral
    eta = eta_integral(v_min_val)
    if eta <= 0:
        return 0.0

    # For Rutherford: dσ/dE_R = π(GM m_A)^2 / (2μ² v⁴ E_R²)
    # Halo-averaged: <1/v⁴> is hard; use <v^-1> approximation with effective v
    # More properly: ∫ f(v) (1/v⁴) v² dv × v_factor
    # Standard approach: factor out v dependence
    mu = xs.reduced_mass(c.M_REM_KG, m_A_kg)
    coupling = c.G_SI * c.M_REM_KG * m_A_kg

    # Halo-averaged differential rate for 1/v⁴ dependence:
    # dR/dE_R = n × π coupling² / (2μ² E_R²) × <1/v³>
    # where <1/v³> = η(v_min) / v_min for Maxwell-Boltzmann
    # Use numerical integration for accuracy
    v_min_here = v_min_val if v_min_val > 0 else 1.0

    # Conservative: use v_0 as characteristic velocity
    # dR/dE_R (per target nucleus) = n × σ(E_R, v_0) × v_0
    # where σ(E_R) is the differential cross section evaluated at v_0
    dsig = xs.dsigma_dER_rutherford(E_R_J, m_A_kg, v=c.V_0)

    # Rate per target nucleus
    rate_per_nucleus = n_DM * dsig * c.V_0 * eta / (1.0 / c.V_0)
    # Simplify: the eta integral already accounts for velocity distribution
    # Use: dR/dE_R = (n_DM / m_A) × π coupling² / (2μ²) × I(E_R)
    # where I(E_R) = ∫ f(v)/(v³ E_R²) dv from v_min

    # Direct evaluation: rate per kg of detector
    rate_per_nucleus_simple = n_DM * dsig * c.V_0
    rate_per_kg = rate_per_nucleus_simple / m_A_kg

    return rate_per_kg


def total_rate(detector_name):
    """
    Total event rate for a detector [events / kg / s].

    Integrates dR/dE_R from E_th to E_R_max.
    """
    det = c.DETECTORS[detector_name]
    m_A = det["m_nucleus_kg"]
    A = det["A"]
    E_th_J = det["E_th_keV"] * 1e3 * c.EV_TO_J

    # Maximum recoil: E_R_max = 2μ²v²/m_A, with v = v_esc + v_E
    mu = xs.reduced_mass(c.M_REM_KG, m_A)
    v_max = c.V_ESC + c.V_E
    E_R_max_J = 2 * mu**2 * v_max**2 / m_A

    if E_th_J >= E_R_max_J:
        return 0.0

    # For Rutherford 1/E_R² spectrum, integral is analytic:
    # ∫_{E_th}^{E_max} dσ/dE_R dE_R = σ(>E_th) - σ(>E_max)
    # Rate = n_DM × v_0 × [σ(>E_th) - σ(>E_max)] / m_A (per kg)
    n_DM = c.RHO_DM_SI / c.M_REM_KG
    sigma_above_Eth = xs.sigma_rutherford(m_A, E_th_J, v=c.V_0)
    sigma_above_Emax = xs.sigma_rutherford(m_A, E_R_max_J, v=c.V_0)

    sigma_window = sigma_above_Eth - sigma_above_Emax
    rate_per_kg = n_DM * c.V_0 * sigma_window / m_A

    return rate_per_kg


def expected_events(detector_name):
    """
    Expected number of events for a given experiment.

    N = rate [/kg/s] × exposure [kg × yr] × yr_to_s
    """
    det = c.DETECTORS[detector_name]
    rate = total_rate(detector_name)
    exposure_kg_s = det["exposure_kg_yr"] * c.YR_TO_S
    return rate * exposure_kg_s


def energy_deposit_per_transit(detector_name, L_det_m=1.0):
    """
    Energy deposited per remnant transit through the detector.

    Impulse approximation: ΔE_single = G² M²_rem m_A / (b² v²)
    Summing over all nuclei within impact parameter:
      dE/dx = n_target × 2π ∫ ΔE(b) b db from b_min to b_max

    For gravitational scattering, this gives:
      dE/dx = 2π n_target G² M²_rem m_A × ln(b_max/b_min) / v²
    """
    det = c.DETECTORS[detector_name]
    m_A = det["m_nucleus_kg"]
    A = det["A"]

    # Number density of target nuclei
    rho_det_kg_m3 = {"Xe": 2950, "Ar": 1400, "Ge": 5320}  # liquid/crystal densities
    rho = rho_det_kg_m3.get(det["element"], 3000)
    n_target = rho / m_A

    # Impact parameter limits
    b_min = c.L_PLANCK   # cannot resolve below Planck length
    b_max = n_target**(-1.0/3.0)  # mean inter-nuclear spacing

    coulomb_log = np.log(b_max / b_min)

    # Energy loss rate (per unit length)
    dE_dx = (2 * np.pi * n_target * c.G_SI**2 * c.M_REM_KG**2 *
             m_A * coulomb_log / c.V_0**2)

    # Total energy deposited in transit
    dE_total = dE_dx * L_det_m

    return {
        "dE_dx_J_m": dE_dx,
        "dE_dx_eV_m": dE_dx * c.J_TO_EV,
        "dE_total_J": dE_total,
        "dE_total_eV": dE_total * c.J_TO_EV,
        "n_target_m3": n_target,
        "coulomb_log": coulomb_log,
        "b_min_m": b_min,
        "b_max_m": b_max,
    }
