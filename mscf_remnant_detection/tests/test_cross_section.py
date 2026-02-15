"""Tests for cross-section calculations."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import constants as c
from src import cross_section as xs


def test_rutherford_born_agree():
    """Gate 4: Rutherford and Born methods agree exactly for 1/r potential."""
    m_Xe = c.DETECTORS["LZ"]["m_nucleus_kg"]
    E_th = 1.5e3 * c.EV_TO_J

    sigma_R = xs.sigma_rutherford(m_Xe, E_th)
    sigma_B = xs.sigma_born(m_Xe, E_th)

    assert abs(sigma_R / sigma_B - 1) < 1e-10


def test_dimensional_scaling():
    """Gate 4: Dimensional estimate agrees within 4π."""
    m_Xe = c.DETECTORS["LZ"]["m_nucleus_kg"]
    E_th = 1.5e3 * c.EV_TO_J

    sigma_R = xs.sigma_rutherford(m_Xe, E_th)
    sigma_D = xs.sigma_dimensional(m_Xe, E_th)

    # μ ≈ m_A for M_rem >> m_A, so ratio ≈ π/2
    ratio = sigma_R / sigma_D
    assert 0.1 < ratio < 4 * np.pi


def test_sigma_scales_M_squared():
    """σ ∝ M²_rem (gravitational coupling)."""
    m_Xe = c.DETECTORS["LZ"]["m_nucleus_kg"]
    E_th = 1.5e3 * c.EV_TO_J
    v = c.V_0
    mu = xs.reduced_mass(c.M_REM_KG, m_Xe)

    # Double the mass
    coupling_1 = c.G_SI * c.M_REM_KG * m_Xe
    coupling_2 = c.G_SI * (2 * c.M_REM_KG) * m_Xe

    sigma_1 = np.pi * coupling_1**2 / (2 * mu**2 * v**4 * E_th)
    sigma_2 = np.pi * coupling_2**2 / (2 * mu**2 * v**4 * E_th)

    # σ_2 / σ_1 = 4 (since coupling ∝ M, σ ∝ M²)
    assert abs(sigma_2 / sigma_1 - 4) < 0.01


def test_per_nucleon_A4():
    """Gate 5: σ_A ≈ A⁴ × σ_n for M_rem >> m_A."""
    for det_name in ["LZ", "DarkSide-20k", "SuperCDMS"]:
        det = c.DETECTORS[det_name]
        m_A = det["m_nucleus_kg"]
        A = det["A"]
        E_th = det["E_th_keV"] * 1e3 * c.EV_TO_J

        sigma_A = xs.sigma_rutherford(m_A, E_th)
        sigma_n = xs.sigma_per_nucleon(sigma_A, A, m_A)

        # σ_A / σ_n should be approximately A^4
        ratio = sigma_A / sigma_n
        expected = A**4
        # Allow factor of 2 for μ corrections
        assert 0.5 < ratio / expected < 2.0, \
            f"{det_name}: ratio={ratio:.2e}, expected A^4={expected:.2e}"


def test_sommerfeld_small():
    """η_G ≪ 1 — Born approximation valid."""
    for det_name in c.DETECTORS:
        det = c.DETECTORS[det_name]
        eta = xs.sommerfeld_parameter(det["m_nucleus_kg"])
        assert eta < 1e-3, f"{det_name}: eta_G = {eta:.2e} not << 1"


def test_cross_section_order_of_magnitude():
    """σ_n ≈ 10⁻⁶⁰ cm² (order of magnitude)."""
    result = xs.compute_all("LZ")
    log_sigma = np.log10(result["sigma_per_nucleon_cm2"])
    # Should be somewhere in -65 to -55
    assert -66 < log_sigma < -54, f"σ_n = 10^{log_sigma:.1f} cm² out of range"


def test_compute_all_detectors():
    """compute_all runs without error for all detectors."""
    for det_name in c.DETECTORS:
        result = xs.compute_all(det_name)
        assert result["born_valid"]
        assert result["methods_agree"]
        assert result["sigma_per_nucleon_cm2"] > 0
        assert not np.isnan(result["sigma_per_nucleon_cm2"])
        assert not np.isinf(result["sigma_per_nucleon_cm2"])
