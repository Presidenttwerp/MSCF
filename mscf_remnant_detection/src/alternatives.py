"""
Alternative detection channels for Planck-mass gravitational relics.

Channels: gravitational microlensing, femtolensing, pulsar timing,
dynamical constraints (disk heating, NS capture, overclosure).
"""
import numpy as np
from . import constants as c


def einstein_radius(D_l_m, D_s_m=None):
    """
    Einstein radius for gravitational microlensing.

    θ_E = √(4GM / c² × D_ls / (D_l × D_s))

    For a Planck-mass remnant, this is absurdly small.

    Parameters
    ----------
    D_l_m : float
        Distance to lens in meters.
    D_s_m : float, optional
        Distance to source (default: 2 × D_l).

    Returns
    -------
    dict with theta_E_rad, theta_E_arcsec, comparison values.
    """
    if D_s_m is None:
        D_s_m = 2 * D_l_m
    D_ls = D_s_m - D_l_m

    r_E_sq = 4 * c.G_SI * c.M_REM_KG * D_ls / (c.C_SI**2 * D_s_m)
    r_E = np.sqrt(np.abs(r_E_sq)) * D_l_m  # physical Einstein radius at lens
    theta_E = np.sqrt(np.abs(4 * c.G_SI * c.M_REM_KG * D_ls /
                              (c.C_SI**2 * D_l_m * D_s_m)))

    # Schwarzschild radius of remnant
    r_s = 2 * c.G_SI * c.M_REM_KG / c.C_SI**2

    return {
        "theta_E_rad": theta_E,
        "theta_E_arcsec": theta_E * 206265,
        "r_E_m": r_E,
        "r_s_m": r_s,
        "r_s_over_lP": r_s / c.L_PLANCK,
        "theta_E_over_microarcsec": theta_E * 206265e6,
    }


def femtolensing_check():
    """
    Femtolensing constraint check.

    Femtolensing requires r_s ~ λ_γ for diffraction effects.
    For remnant: r_s ~ l_P ≈ 1.6e-35 m.
    Gamma-ray wavelength: λ_γ ~ 10⁻¹² m (for 1 MeV).

    Returns
    -------
    dict with comparison of length scales.
    """
    r_s = 2 * c.G_SI * c.M_REM_KG / c.C_SI**2

    # Gamma-ray wavelength at different energies
    E_keV = 1.0   # keV
    E_MeV = 1.0   # MeV
    E_GeV = 1.0   # GeV

    lambda_keV = c.HBAR_SI * c.C_SI / (E_keV * 1e3 * c.EV_TO_J)
    lambda_MeV = c.HBAR_SI * c.C_SI / (E_MeV * 1e6 * c.EV_TO_J)
    lambda_GeV = c.HBAR_SI * c.C_SI / (E_GeV * 1e9 * c.EV_TO_J)

    # Fresnel scale for lensing: r_F = √(λ D / 2π)
    D_lens = 1e22  # 1 kpc
    r_F_MeV = np.sqrt(lambda_MeV * D_lens / (2 * np.pi))

    return {
        "r_s_m": r_s,
        "l_P_m": c.L_PLANCK,
        "lambda_keV_m": lambda_keV,
        "lambda_MeV_m": lambda_MeV,
        "lambda_GeV_m": lambda_GeV,
        "r_s_over_lambda_MeV": r_s / lambda_MeV,
        "r_F_MeV_m": r_F_MeV,
        "r_s_over_r_F": r_s / r_F_MeV,
        "detectable": r_s > lambda_MeV,  # should be False
    }


def pulsar_timing_shapiro(b_min_m, b_max_m, D_pulsar_m=3e19):
    """
    Shapiro delay from remnant transit near line of sight.

    Δt = (4GM/c³) × ln(b_max/b_min)

    For Planck-mass object, this is incredibly small.

    Parameters
    ----------
    b_min_m : float
        Minimum impact parameter (≥ l_P).
    b_max_m : float
        Maximum impact parameter.
    D_pulsar_m : float
        Distance to pulsar (default: 1 kpc).
    """
    dt_shapiro = 4 * c.G_SI * c.M_REM_KG / c.C_SI**3 * np.log(b_max_m / b_min_m)

    # Current PTA sensitivity
    dt_pta_s = 1e-7  # ~100 ns (current best)

    # Event rate: N ~ n_DM × v × π b_max² × T_obs
    n_DM = c.RHO_DM_SI / c.M_REM_KG
    T_obs = 10 * c.YR_TO_S  # 10 years
    rate = n_DM * c.V_0 * np.pi * b_max_m**2 * T_obs

    return {
        "dt_shapiro_s": dt_shapiro,
        "dt_pta_s": dt_pta_s,
        "ratio_dt": dt_shapiro / dt_pta_s,
        "N_transits": rate,
    }


def disk_heating_constraint():
    """
    Galactic disk heating constraint.

    DM remnants passing through the disk deposit kinetic energy via
    gravitational dynamical friction on disk stars.

    Energy transfer per passage through disk of thickness H:
      ΔE = 4π G² M²_rem ρ_disk H ln(Λ) / v

    Lacey & Ostriker (1985): disk heating rate must be < ~10⁻²⁶ W/m².
    Relevant for M_DM > 10⁶ M_sun; for M_P ~ 10⁻⁸ kg this is negligible.
    """
    H_disk = 300 * 3.086e16    # disk scale height ~300 pc in m
    rho_disk = 0.1 * c.M_SOLAR_KG / (3.086e16)**3  # ~0.1 M_sun/pc³

    # Coulomb log: ln(b_max/b_min), b_max ~ H, b_min ~ GM/v²
    b_min = c.G_SI * c.M_REM_KG / c.V_0**2
    b_max = H_disk
    coulomb_log = np.log(b_max / b_min)

    # Energy deposited per passage via dynamical friction
    dE_per_passage = (4 * np.pi * c.G_SI**2 * c.M_REM_KG**2 *
                      rho_disk * H_disk * coulomb_log / c.V_0)

    # Heating rate = flux × ΔE
    n_DM = c.RHO_DM_SI / c.M_REM_KG
    flux = n_DM * c.V_0  # m⁻² s⁻¹
    epsilon_mscf = flux * dE_per_passage  # W/m²

    # Observed disk heating: ~10⁻²⁶ W/m² (Lacey & Ostriker 1985)
    epsilon_obs = 1e-26  # W/m²

    return {
        "epsilon_obs_W_m2": epsilon_obs,
        "epsilon_mscf_W_m2": epsilon_mscf,
        "ratio": epsilon_mscf / epsilon_obs,
        "constrained": epsilon_mscf > epsilon_obs,
    }


def ns_capture_constraint():
    """
    Neutron star capture constraint.

    If DM accumulates in NS, it could cause collapse.
    For gravitational-only interaction, capture rate is negligible.
    """
    # NS parameters
    M_NS = 1.4 * c.M_SOLAR_KG
    R_NS = 1e4  # 10 km in meters
    v_esc_NS = np.sqrt(2 * c.G_SI * M_NS / R_NS)

    # Geometric capture cross-section (gravitational focusing)
    sigma_geom = np.pi * R_NS**2 * (1 + v_esc_NS**2 / c.V_0**2)

    # Capture rate
    n_DM = c.RHO_DM_SI / c.M_REM_KG
    rate = n_DM * c.V_0 * sigma_geom  # captures per second

    # Mass accumulated over NS lifetime
    t_NS = 1e10 * c.YR_TO_S
    M_acc = rate * c.M_REM_KG * t_NS

    return {
        "sigma_geom_m2": sigma_geom,
        "capture_rate_per_s": rate,
        "M_accumulated_kg": M_acc,
        "M_acc_over_M_NS": M_acc / M_NS,
        "constrained": M_acc > 0.01 * M_NS,
    }


def overclosure_check():
    """
    Verify remnants don't overclose the universe.

    If all DM is remnants at the cosmic average density:
      Ω_rem = Ω_DM (by construction, since we assume ρ_rem = ρ_DM).

    The key check: can M_rem particles with the correct abundance exist?
    Ω_DM < 1, so no overclosure.

    Note: RHO_DM_SI is the LOCAL density (0.3 GeV/cm³ ≈ 5e-22 kg/m³),
    which is ~10⁵× the cosmic average Ω_DM × ρ_crit ≈ 2e-27 kg/m³.
    """
    # Cosmic average DM density
    rho_DM_cosmic = c.OMEGA_DM * c.RHO_CRIT_COSMO
    n_rem_cosmic = rho_DM_cosmic / c.M_REM_KG
    Omega_rem = c.OMEGA_DM  # by construction

    # Local density (for rate calculations)
    n_rem_local = c.RHO_DM_SI / c.M_REM_KG

    return {
        "n_rem_cosmic_m3": n_rem_cosmic,
        "n_rem_local_m3": n_rem_local,
        "rho_DM_cosmic_kg_m3": rho_DM_cosmic,
        "rho_DM_local_kg_m3": c.RHO_DM_SI,
        "local_over_cosmic": c.RHO_DM_SI / rho_DM_cosmic,
        "Omega_rem": Omega_rem,
        "Omega_DM": c.OMEGA_DM,
        "consistent": Omega_rem <= 1.0,  # not overclosed
    }


def summary():
    """Summarize all alternative channel sensitivities."""
    # Lensing at 1 kpc
    lens_1kpc = einstein_radius(3e19)  # 1 kpc in m

    # Femtolensing
    femto = femtolensing_check()

    # Pulsar timing
    pta = pulsar_timing_shapiro(c.L_PLANCK, 1e13)  # b_max = 0.001 pc

    # Disk heating
    disk = disk_heating_constraint()

    # NS capture
    ns = ns_capture_constraint()

    # Overclosure
    oc = overclosure_check()

    return {
        "lensing_theta_E_rad": lens_1kpc["theta_E_rad"],
        "femtolensing_detectable": femto["detectable"],
        "pta_dt_ratio": pta["ratio_dt"],
        "disk_constrained": disk["constrained"],
        "ns_constrained": ns["constrained"],
        "overclosure_consistent": oc["consistent"],
    }
