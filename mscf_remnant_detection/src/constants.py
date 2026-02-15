"""
Physical constants, MSCF parameters, detector specifications, and halo model.

Sources:
  - CODATA 2018 recommended values
  - MSCF v2.1.7: Eqs. 36-43, Axiom 5, Theorem 12.2
  - Particle Data Group 2022 for nuclear masses
  - Standard halo model (Lewin & Smith 1996)
"""
import numpy as np

# =============================================================================
# Fundamental Constants (SI)
# =============================================================================
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
C_SI = 2.99792458e8         # m/s (exact)
HBAR_SI = 1.054571817e-34   # J s
K_B_SI = 1.380649e-23       # J/K (exact)
E_CHARGE = 1.602176634e-19  # C (exact)

# =============================================================================
# Planck Units (derived, not hardcoded)
# =============================================================================
M_PLANCK_KG = np.sqrt(HBAR_SI * C_SI / G_SI)
L_PLANCK = np.sqrt(HBAR_SI * G_SI / C_SI**3)
T_PLANCK = np.sqrt(HBAR_SI * G_SI / C_SI**5)
E_PLANCK_J = M_PLANCK_KG * C_SI**2

# =============================================================================
# MSCF Parameters (Theorem 12.2, Eq. 39)
# =============================================================================
M_REM_KG = M_PLANCK_KG / 2                     # M_min = M_P/2 (Eq. 39)
RHO_CRIT_MSCF = C_SI**7 / (HBAR_SI * G_SI**2)  # Axiom 5 (Eq. 3)
KAPPA_MAX = C_SI**2 / (2 * L_PLANCK)            # Eq. 38

# =============================================================================
# Unit Conversions
# =============================================================================
GEV_TO_KG = 1.78266192e-27
KG_TO_GEV = 1.0 / GEV_TO_KG
EV_TO_J = E_CHARGE   # 1 eV in Joules
GEV_TO_J = EV_TO_J * 1e9
J_TO_EV = 1.0 / EV_TO_J
YR_TO_S = 3.15576e7  # Julian year
M_SOLAR_KG = 1.98892e30
CM_TO_M = 1e-2
M_TO_CM = 1e2
KG_TO_G = 1e3
CM2_TO_M2 = 1e-4
M2_TO_CM2 = 1e4

# Planck mass in GeV
M_PLANCK_GEV = M_PLANCK_KG * KG_TO_GEV
M_REM_GEV = M_REM_KG * KG_TO_GEV

# =============================================================================
# Nuclear Masses (SI)
# =============================================================================
M_PROTON_KG = 1.67262192e-27
M_AMU_KG = 1.66053907e-27
M_NEUTRON_KG = 1.67492750e-27

# =============================================================================
# Detector Specifications
# =============================================================================
# Format: (name, target_element, A, m_nucleus_kg, E_th_keV, exposure_kg_yr)
DETECTORS = {
    "LZ": {
        "element": "Xe",
        "A": 131,
        "m_nucleus_kg": 131 * M_AMU_KG,
        "E_th_keV": 1.5,       # nuclear recoil threshold
        "exposure_kg_yr": 5.5e3,  # 5.5 tonne-year (1000 live days × 5.5t)
    },
    "XENONnT": {
        "element": "Xe",
        "A": 131,
        "m_nucleus_kg": 131 * M_AMU_KG,
        "E_th_keV": 1.0,
        "exposure_kg_yr": 4.0e3,
    },
    "PandaX-4T": {
        "element": "Xe",
        "A": 131,
        "m_nucleus_kg": 131 * M_AMU_KG,
        "E_th_keV": 1.1,
        "exposure_kg_yr": 3.7e3,
    },
    "DarkSide-20k": {
        "element": "Ar",
        "A": 40,
        "m_nucleus_kg": 40 * M_AMU_KG,
        "E_th_keV": 7.0,
        "exposure_kg_yr": 2.0e5,  # 200 tonne-year
    },
    "SuperCDMS": {
        "element": "Ge",
        "A": 73,
        "m_nucleus_kg": 73 * M_AMU_KG,
        "E_th_keV": 0.04,
        "exposure_kg_yr": 44.0,  # modest exposure
    },
}

# =============================================================================
# Standard Halo Model (Lewin & Smith 1996)
# =============================================================================
RHO_DM_GEV_CM3 = 0.3          # GeV/cm^3 local DM density
RHO_DM_SI = RHO_DM_GEV_CM3 * GEV_TO_KG * 1e6  # kg/m^3
V_0 = 220e3                    # m/s circular velocity
V_ESC = 544e3                  # m/s galactic escape velocity
V_E = 232e3                    # m/s Earth velocity (annual average)

# =============================================================================
# Cosmological
# =============================================================================
H_0_SI = 67.4e3 / 3.0857e22   # s^-1
OMEGA_DM = 0.265
RHO_CRIT_COSMO = 3 * H_0_SI**2 / (8 * np.pi * G_SI)  # kg/m^3
