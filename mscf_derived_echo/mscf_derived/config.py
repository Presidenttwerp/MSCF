"""
Physical constants and event catalog for MSCF derived-parameter echo search.

Reuses constants from mscf_echo_interference where possible.
"""

import math
import os
import sys

# ---------------------------------------------------------------------------
# Add sibling packages to path so we can import from them
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _pkg in ('mscf_echo_interference', 'mscf_grav_atom'):
    _path = os.path.join(_PROJECT_ROOT, _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Physical Constants (SI units)
# ---------------------------------------------------------------------------

G_SI = 6.67430e-11
C_SI = 299792458.0
MSUN_KG = 1.98847e30
MSUN_S = G_SI * MSUN_KG / C_SI**3   # ~4.926e-6 s

# ---------------------------------------------------------------------------
# QNM Coefficients (Berti-Cardoso-Will, l=m=2, n=0)
# ---------------------------------------------------------------------------

QNM_220_COEFFS = {
    'f': (1.5251, -1.1568, 0.1292),
    'Q': (0.7000, 1.4187, -0.4990),
}

# Schwarzschild l=2 fundamental (Leaver 1985): Momega = 0.37367 - 0.08896i
SCHW_QNM_220_MOMEGA = 0.37367

# ---------------------------------------------------------------------------
# Greybody Solver Parameters
# ---------------------------------------------------------------------------

RSTAR_HORIZON = -200.0     # Near-horizon starting point for scattering
RSTAR_INFINITY = 80.0      # Far-field extraction point
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12
GREYBODY_N_GRID = 500      # Default number of Momega grid points for caching
GREYBODY_MOMEGA_MIN = 0.01
GREYBODY_MOMEGA_MAX = 5.0

# ---------------------------------------------------------------------------
# Analysis Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_ECHOES = 6
DEFAULT_FMIN = 20.0         # Broader band than old pipeline (was 150)
DEFAULT_FMAX = 2048.0       # Up to Nyquist/2 for kHz ripple search
DEFAULT_SAMPLE_RATE = 4096
DEFAULT_TUKEY_ALPHA = 0.1

# PSD estimation
DEFAULT_PSD_SEGMENT_S = 32.0
DEFAULT_PSD_OFFSET_S = 2.0

# Background
DEFAULT_N_BACKGROUND = 200

# Detection thresholds
DISCOVERY_SIGMA = 5.0
EVIDENCE_SIGMA = 3.0

# ---------------------------------------------------------------------------
# Event Catalog
# ---------------------------------------------------------------------------

EVENTS = {
    'GW150914': {
        'gps_merger': 1126259462.4,
        'Mf_msun': 62.0,
        'chi_f': 0.67,
        'detectors': ['H1', 'L1'],
        'network_snr': 24.4,
    },
    'GW170104': {
        'gps_merger': 1167559936.6,
        'Mf_msun': 47.5,
        'chi_f': 0.66,
        'detectors': ['H1', 'L1'],
        'network_snr': 13.0,
    },
    'GW170814': {
        'gps_merger': 1186741861.5,
        'Mf_msun': 53.2,
        'chi_f': 0.72,
        'detectors': ['H1', 'L1', 'V1'],
        'network_snr': 18.0,
    },
    'GW190521': {
        'gps_merger': 1242442967.4,
        'Mf_msun': 142.0,
        'chi_f': 0.72,
        'detectors': ['H1', 'L1'],
        'network_snr': 14.7,
    },
}

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
CACHE_DIR = os.path.join(RESULTS_DIR, 'cache')
