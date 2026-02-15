"""Consistency and cross-check tests."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import constants as c
from src import cross_section as xs
from src import remnant
from src import limits
from src import alternatives as alt


def test_gate0_teff_vanishes():
    """Gate 0: T_eff(M_P/2) = 0."""
    result = remnant.verify_eq43()
    assert result["gate_0_pass"]
    assert abs(result["correction_factor"]) < 1e-15


def test_gate1_constants():
    """Gate 1: Constants match CODATA to 0.1%."""
    assert abs(c.M_PLANCK_KG - 2.176e-8) / 2.176e-8 < 1e-3
    assert abs(c.M_REM_KG - 1.088e-8) / 1.088e-8 < 1e-3
    assert abs(c.L_PLANCK - 1.616e-35) / 1.616e-35 < 1e-3


def test_derivation_chain():
    """Full derivation chain is self-consistent."""
    chain = remnant.derivation_chain()
    assert chain["M_min_agreement"]
    assert abs(chain["T_eff_at_Mmin_K"]) < 1e-10


def test_gate7_sigma_ratio():
    """Gate 7: σ_grav / σ_LZ < 10⁻²⁰."""
    result = xs.compute_all("LZ")
    sigma_n = result["sigma_per_nucleon_cm2"]
    M_rem = c.M_REM_GEV

    sigma_lz = limits.lz_limit(np.array([M_rem]))[0]
    ratio = sigma_n / sigma_lz
    assert ratio < 1e-20, f"σ/σ_LZ = {ratio:.2e} (should be < 1e-20)"


def test_reduced_mass_limits():
    """μ ≈ m_A when M_rem >> m_A."""
    for det_name in c.DETECTORS:
        det = c.DETECTORS[det_name]
        m_A = det["m_nucleus_kg"]
        mu = xs.reduced_mass(c.M_REM_KG, m_A)
        # M_rem ~ 10^-8 kg, m_A ~ 10^-25 kg, so μ ≈ m_A
        assert abs(mu / m_A - 1) < 1e-10, \
            f"{det_name}: μ/m_A = {mu/m_A:.15f}"


def test_eta_G_small():
    """η_G ≪ 1 for all targets."""
    for det_name in c.DETECTORS:
        det = c.DETECTORS[det_name]
        eta = xs.sommerfeld_parameter(det["m_nucleus_kg"])
        assert eta < 1e-10, f"{det_name}: η_G = {eta:.2e}"


def test_overclosure():
    """Remnants don't overclose the universe."""
    oc = alt.overclosure_check()
    assert oc["consistent"]


def test_femtolensing_null():
    """Femtolensing undetectable."""
    femto = alt.femtolensing_check()
    assert not femto["detectable"]


def test_no_nan_inf():
    """No NaN or Inf in any computation."""
    for det_name in c.DETECTORS:
        result = xs.compute_all(det_name)
        for key, val in result.items():
            if isinstance(val, float):
                assert not np.isnan(val), f"{det_name}.{key} is NaN"
                assert not np.isinf(val), f"{det_name}.{key} is Inf"

    props = remnant.remnant_properties()
    for key, val in props.items():
        if isinstance(val, float):
            assert not np.isnan(val), f"remnant.{key} is NaN"
            assert not np.isinf(val), f"remnant.{key} is Inf"
