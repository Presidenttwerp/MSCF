# mscf/echo_waveform.py
"""
MSCF echo waveform factor.

Applies MSCF-constrained echo modulation to a ringdown waveform.
The echo delay is deterministic from Mf and chi - no free spectral
fitting parameters (f_cut, roll) that caused overfitting.
"""
import numpy as np
from .echo_delay import mscf_echo_delay_seconds


def apply_mscf_echoes(H0_f: np.ndarray,
                      freqs: np.ndarray,
                      Mf_solar: float,
                      chi: float,
                      R0: float,
                      phi0: float,
                      necho: int = 2) -> np.ndarray:
    """
    Apply MSCF echo modulation to a ringdown waveform.

    H1(f) = H0(f) * [1 + sum_{k=1..necho} (R0^k) exp(i k (2π f Δt + phi0))]

    Parameters
    ----------
    H0_f : np.ndarray
        Frequency-domain ringdown waveform (complex)
    freqs : np.ndarray
        Frequency array (Hz)
    Mf_solar : float
        Final mass in solar masses
    chi : float
        Dimensionless spin parameter
    R0 : float
        Echo amplitude (reflectivity), 0 <= R0 <= 1
    phi0 : float
        Echo phase offset (radians)
    necho : int
        Number of echo copies to include (default 2)

    Returns
    -------
    np.ndarray
        H1(f) = H0(f) * echo_factor (complex)
    """
    # Get deterministic echo delay from MSCF physics
    dt = mscf_echo_delay_seconds(Mf_solar, chi)

    # Phase accumulated per echo: 2πf*Δt + φ0
    phase = 2.0 * np.pi * freqs * dt + phi0

    # Build echo transfer function:
    # T(f) = 1 + R0*exp(i*phase) + R0^2*exp(2i*phase) + ...
    factor = np.ones_like(H0_f, dtype=np.complex128)
    rk = complex(R0, 0.0)

    for k in range(1, necho + 1):
        factor += (rk**k) * np.exp(1j * k * phase)

    return H0_f * factor
