"""Tests for physical constants and derived quantities."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import constants as c


def test_planck_mass():
    """Gate 1: M_P matches CODATA to 0.01%."""
    M_P_expected = 2.176434e-8  # kg (CODATA 2018)
    assert abs(c.M_PLANCK_KG - M_P_expected) / M_P_expected < 1e-4


def test_planck_length():
    """Gate 1: l_P matches CODATA to 0.01%."""
    l_P_expected = 1.616255e-35  # m
    assert abs(c.L_PLANCK - l_P_expected) / l_P_expected < 1e-4


def test_remnant_mass_exact():
    """M_rem = M_P/2 exactly by construction."""
    assert c.M_REM_KG == c.M_PLANCK_KG / 2


def test_planck_units_derived():
    """Planck units are derived from G, hbar, c — not hardcoded."""
    M_P = np.sqrt(c.HBAR_SI * c.C_SI / c.G_SI)
    assert abs(c.M_PLANCK_KG - M_P) / M_P < 1e-15

    l_P = np.sqrt(c.HBAR_SI * c.G_SI / c.C_SI**3)
    assert abs(c.L_PLANCK - l_P) / l_P < 1e-15


def test_mscf_rho_crit():
    """Axiom 5: ρ_crit = c^7/(ℏG²)."""
    rho = c.C_SI**7 / (c.HBAR_SI * c.G_SI**2)
    assert abs(c.RHO_CRIT_MSCF - rho) / rho < 1e-15


def test_kappa_max():
    """Eq. 38: κ_max = c²/(2l_P)."""
    kappa = c.C_SI**2 / (2 * c.L_PLANCK)
    assert abs(c.KAPPA_MAX - kappa) / kappa < 1e-15


def test_remnant_mass_gev():
    """M_rem ≈ 6.1e18 GeV."""
    assert 5e18 < c.M_REM_GEV < 7e18


def test_unit_conversions():
    """Round-trip unit conversions."""
    M_gev = c.M_REM_KG * c.KG_TO_GEV
    M_kg_back = M_gev * c.GEV_TO_KG
    assert abs(M_kg_back - c.M_REM_KG) / c.M_REM_KG < 1e-5


def test_detectors_present():
    """All five detectors defined."""
    for name in ["LZ", "XENONnT", "PandaX-4T", "DarkSide-20k", "SuperCDMS"]:
        assert name in c.DETECTORS
        det = c.DETECTORS[name]
        assert det["A"] > 0
        assert det["m_nucleus_kg"] > 0
        assert det["E_th_keV"] > 0
        assert det["exposure_kg_yr"] > 0


def test_halo_parameters():
    """Halo model values are reasonable."""
    assert 0.2 < c.RHO_DM_GEV_CM3 < 0.5
    assert 200e3 < c.V_0 < 250e3
    assert 500e3 < c.V_ESC < 600e3
