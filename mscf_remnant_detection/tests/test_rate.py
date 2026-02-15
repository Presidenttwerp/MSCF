"""Tests for event rate calculations."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import constants as c
from src import cross_section as xs
from src import rate


def test_rate_tiny():
    """Gate 6: Rate < 10⁻²⁰ events/kg/year for all detectors."""
    for det_name in c.DETECTORS:
        r = rate.total_rate(det_name)
        r_per_year = r * c.YR_TO_S
        assert r_per_year < 1e-20, \
            f"{det_name}: rate = {r_per_year:.2e} events/kg/yr (too high!)"


def test_expected_events_tiny():
    """Expected events ≪ 1 for all experiments."""
    for det_name in c.DETECTORS:
        N = rate.expected_events(det_name)
        assert N < 1e-15, \
            f"{det_name}: N = {N:.2e} events (too many!)"


def test_E_R_max_above_threshold():
    """Maximum recoil energy exceeds threshold for all detectors."""
    for det_name in c.DETECTORS:
        det = c.DETECTORS[det_name]
        m_A = det["m_nucleus_kg"]
        mu = xs.reduced_mass(c.M_REM_KG, m_A)
        v_max = c.V_ESC + c.V_E
        E_R_max = 2 * mu**2 * v_max**2 / m_A
        E_th = det["E_th_keV"] * 1e3 * c.EV_TO_J
        assert E_R_max > E_th, \
            f"{det_name}: E_R_max = {E_R_max:.2e} J < E_th = {E_th:.2e} J"


def test_energy_deposit():
    """Energy deposition returns finite positive values."""
    for det_name in ["LZ", "DarkSide-20k", "SuperCDMS"]:
        info = rate.energy_deposit_per_transit(det_name)
        assert info["dE_dx_eV_m"] > 0
        assert not np.isnan(info["dE_dx_eV_m"])
        assert not np.isinf(info["dE_dx_eV_m"])
        assert info["dE_total_eV"] > 0


def test_v_min_physical():
    """v_min is sub-luminal for all relevant recoil energies."""
    for det_name in c.DETECTORS:
        det = c.DETECTORS[det_name]
        m_A = det["m_nucleus_kg"]
        E_th = det["E_th_keV"] * 1e3 * c.EV_TO_J
        v = rate.v_min(E_th, m_A)
        assert v < c.C_SI, \
            f"{det_name}: v_min = {v:.2e} m/s >= c"
