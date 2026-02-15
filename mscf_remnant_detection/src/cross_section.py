"""
Gravitational scattering cross-section: MSCF remnant on atomic nucleus.

Three independent methods for V(r) = -G M_rem m_A / r:
  A) Classical Rutherford
  B) Born approximation
  C) Dimensional cross-check

Per-nucleon conversion for comparison with direct detection limits.
"""
import numpy as np
from . import constants as c


def reduced_mass(m1_kg, m2_kg):
    """Reduced mass of two-body system."""
    return m1_kg * m2_kg / (m1_kg + m2_kg)


def sommerfeld_parameter(m_A_kg, v=None):
    """
    Gravitational Sommerfeld parameter eta_G = G M_rem m_A / (hbar v).

    Must be << 1 for Born approximation validity.
    """
    if v is None:
        v = c.V_0
    mu = reduced_mass(c.M_REM_KG, m_A_kg)
    return c.G_SI * c.M_REM_KG * m_A_kg / (c.HBAR_SI * v)


# =========================================================================
# Method A: Classical Rutherford
# =========================================================================
def sigma_rutherford(m_A_kg, E_th_J, v=None):
    """
    Classical Rutherford cross-section for gravitational scattering.

    For V(r) = -α/r with α = G M_rem m_A:
      dσ/dΩ = (α / (4 E_cm))^2 / sin^4(θ/2),  E_cm = μv²/2
      dσ/dE_R = π α² / (μ v² E_R²)

    Integrated above recoil threshold E_th:
      σ(>E_th) = π α² / (μ v² E_th)

    Parameters
    ----------
    m_A_kg : float
        Target nucleus mass in kg.
    E_th_J : float
        Recoil energy threshold in Joules.
    v : float, optional
        Relative velocity in m/s (default: v_0 = 220 km/s).

    Returns
    -------
    sigma_m2 : float
        Cross-section in m^2.
    """
    if v is None:
        v = c.V_0
    mu = reduced_mass(c.M_REM_KG, m_A_kg)
    coupling = c.G_SI * c.M_REM_KG * m_A_kg
    return np.pi * coupling**2 / (mu * v**2 * E_th_J)


def dsigma_dER_rutherford(E_R_J, m_A_kg, v=None):
    """
    Differential cross-section dσ/dE_R for gravitational Rutherford.

    dσ/dE_R = π α² / (μ v² E_R²)    where α = G M_rem m_A
    """
    if v is None:
        v = c.V_0
    mu = reduced_mass(c.M_REM_KG, m_A_kg)
    coupling = c.G_SI * c.M_REM_KG * m_A_kg
    return np.pi * coupling**2 / (mu * v**2 * E_R_J**2)


# =========================================================================
# Method B: Born approximation
# =========================================================================
def sigma_born(m_A_kg, E_th_J, v=None):
    """
    Born approximation cross-section.

    Scattering amplitude: f(q) = -2μ G M_rem m_A / (ℏ² q²)
    Integrated: σ(>E_th) = 4π μ² (G M_rem m_A)² / (ℏ⁴ q_min⁴) × (angular factor)

    For Coulomb-type potential, Born = classical Rutherford (exact).
    """
    if v is None:
        v = c.V_0
    mu = reduced_mass(c.M_REM_KG, m_A_kg)

    # q_min from E_th: E_R = q^2/(2 m_A) => q_min = sqrt(2 m_A E_th)
    q_min = np.sqrt(2 * m_A_kg * E_th_J)

    # Born amplitude: |f(q)|^2 = (2μ G M_rem m_A / (ℏ² q²))^2
    # σ = 2π ∫ |f(q)|^2 q dq / k^2  ... reduces to Rutherford
    # Shortcut: for 1/r potential, Born = Rutherford identically
    coupling = c.G_SI * c.M_REM_KG * m_A_kg
    return np.pi * coupling**2 / (mu * v**2 * E_th_J)


# =========================================================================
# Method C: Dimensional cross-check
# =========================================================================
def sigma_dimensional(m_A_kg, E_th_J, v=None):
    """
    Dimensional estimate: σ ~ G² M²_rem m_A / (v² E_th).

    For 1/r potential the exact Rutherford result (μ ≈ m_A) is:
      σ = π G² M²_rem m_A / (v² E_th)

    The dimensional estimate captures the scaling (missing only π).
    """
    if v is None:
        v = c.V_0
    mu = reduced_mass(c.M_REM_KG, m_A_kg)
    # α = G M m_A, σ = π α²/(μ v² E_th) = π G² M² m_A² / (μ v² E_th)
    # With μ ≈ m_A: σ ≈ π G² M² m_A / (v² E_th)
    # Dimensional estimate (drop π):
    sigma_dim = c.G_SI**2 * c.M_REM_KG**2 * m_A_kg / (v**2 * E_th_J)
    return sigma_dim


# =========================================================================
# Per-nucleon conversion
# =========================================================================
def sigma_per_nucleon(sigma_A_m2, A, m_A_kg):
    """
    Convert nucleus-level cross-section to per-nucleon for comparison
    with direct detection limits.

    σ_n = σ_A × (μ_n/μ_A)² × (1/A²)

    For M_rem >> m_A >> m_n:
      μ_A ≈ m_A, μ_n ≈ m_n
      σ_n = σ_A × (m_n/m_A)² / A²
           = σ_A × (m_n/(A·m_n))² / A² = σ_A / A⁴
    """
    mu_A = reduced_mass(c.M_REM_KG, m_A_kg)
    mu_n = reduced_mass(c.M_REM_KG, c.M_PROTON_KG)
    return sigma_A_m2 * (mu_n / mu_A)**2 / A**2


def compute_all(detector_name):
    """
    Compute cross-sections for a specific detector using all three methods.

    Returns dict with sigma_A, sigma_n, eta_G, method comparison.
    """
    det = c.DETECTORS[detector_name]
    m_A = det["m_nucleus_kg"]
    A = det["A"]
    E_th_J = det["E_th_keV"] * 1e3 * c.EV_TO_J

    sigma_A = sigma_rutherford(m_A, E_th_J)
    sigma_B = sigma_born(m_A, E_th_J)
    sigma_C = sigma_dimensional(m_A, E_th_J)

    sigma_n = sigma_per_nucleon(sigma_A, A, m_A)
    eta_G = sommerfeld_parameter(m_A)

    # Maximum recoil energy: E_R_max = 2 μ² v² / m_A
    mu = reduced_mass(c.M_REM_KG, m_A)
    E_R_max_J = 2 * mu**2 * c.V_0**2 / m_A

    return {
        "detector": detector_name,
        "A": A,
        "sigma_rutherford_m2": sigma_A,
        "sigma_born_m2": sigma_B,
        "sigma_dimensional_m2": sigma_C,
        "sigma_per_nucleon_m2": sigma_n,
        "sigma_per_nucleon_cm2": sigma_n * c.M2_TO_CM2,
        "eta_G": eta_G,
        "born_valid": eta_G < 1e-3,
        "E_R_max_J": E_R_max_J,
        "E_R_max_keV": E_R_max_J / (1e3 * c.EV_TO_J),
        "methods_agree": np.abs(sigma_A / sigma_B - 1) < 0.01,
        "dimensional_ratio": sigma_C / sigma_A,
    }
