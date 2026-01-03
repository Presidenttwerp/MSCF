# mscf/mscf_echo_train.py
"""
MSCF two-surface echo train construction.

This module builds echo signals with:
- Delays (t_echo, delta_t) FIXED by MSCF timing relations
- Amplitudes FIXED by the R1 hierarchy
- Only ONE global reflection phase parameter (shared across detectors)

This kills the "generic smooth spectral fitter" failure mode that caused
the old H1_echo model to overfit pure noise with ln BF ~ 1272.
"""
import numpy as np
from .mscf_timing import t_echo_seconds, delta_t_seconds


def echo_amplitudes(R1: float, N: int) -> np.ndarray:
    """
    MSCF two-surface amplitude hierarchy.

    Physical model:
    - First reflection from outer surface: A1 = R1
    - Signal that passes through enters inner region
    - First transmission then reflection from inner surface: A2 = (1-R1)^2
    - Subsequent echoes: An = (1-R1)^2 * R1^(n-2) for n >= 2

    These amplitudes are relative to the incident amplitude A0.

    Parameters
    ----------
    R1 : float
        Outer surface reflectivity, must be in [0, 1)
    N : int
        Number of echoes to compute

    Returns
    -------
    np.ndarray
        Array of N amplitude coefficients [A1, A2, ..., AN]
    """
    if not (0.0 <= R1 < 1.0):
        raise ValueError(f"R1 must be in [0,1), got {R1}")

    amps = []
    if N >= 1:
        amps.append(R1)
    if N >= 2:
        base = (1.0 - R1) ** 2
        amps.append(base)
        for n in range(3, N + 1):
            amps.append(base * (R1 ** (n - 2)))

    return np.array(amps, dtype=float)


def mscf_echo_signal_td(ringdown_td_fn, params: dict, t: np.ndarray,
                        N_echo: int = 6, R1: float = 0.05,
                        phi_reflect: float = 0.0) -> np.ndarray:
    """
    Build H1(t) = ringdown(t) + sum_k A_k * ringdown(t - t_k) in time domain.

    Timings are DETERMINISTIC from Mf:
      t_1 = t_echo(Mf)
      t_k = t_echo + (k-1) * delta_t   for k >= 2

    Parameters
    ----------
    ringdown_td_fn : callable
        Function that takes (params, t) and returns ringdown waveform h(t)
    params : dict
        Dictionary containing at least "Mf" (final mass in solar masses)
        and any other parameters needed by ringdown_td_fn
    t : np.ndarray
        Time array (seconds)
    N_echo : int
        Number of echo copies to include (default 6)
    R1 : float
        Outer surface reflectivity (default 0.05)
    phi_reflect : float
        Global reflection phase (radians), shared across all IFOs

    Returns
    -------
    np.ndarray
        Time-domain waveform h(t) = h0(t) + echoes
    """
    M_solar = params["Mf"]  # Adapt to your parameter naming
    t1 = t_echo_seconds(M_solar)
    dt = delta_t_seconds(M_solar)

    # Base ringdown
    h0 = ringdown_td_fn(params, t)

    # Echo amplitudes from R1 hierarchy
    amps = echo_amplitudes(R1, N_echo)

    # Build echo train
    h = h0.copy()

    # Global reflection phase (single parameter for all echoes, shared across IFOs)
    phase_factor = np.cos(phi_reflect)  # Real part for amplitude modulation

    for k, Ak in enumerate(amps, start=1):
        tk = t1 + (k - 1) * dt
        # Shift the ringdown by tk
        h_echo_k = ringdown_td_fn(params, t - tk)
        h += Ak * phase_factor * h_echo_k

    return h


def mscf_echo_transfer_fd(freqs: np.ndarray, M_solar: float, R1: float,
                           phi_reflect: float, N_echo: int = 6) -> np.ndarray:
    """
    Compute MSCF echo transfer function in frequency domain.

    T(f) = 1 + sum_k A_k * exp(i * (2*pi*f*t_k + phi_reflect))

    where t_k are the DETERMINISTIC echo times from MSCF physics.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (Hz)
    M_solar : float
        Final mass in solar masses
    R1 : float
        Outer surface reflectivity
    phi_reflect : float
        Global reflection phase (radians)
    N_echo : int
        Number of echo copies

    Returns
    -------
    np.ndarray
        Complex transfer function T(f)
    """
    t1 = t_echo_seconds(M_solar)
    dt = delta_t_seconds(M_solar)
    amps = echo_amplitudes(R1, N_echo)

    # Start with 1 (original ringdown passes through)
    T = np.ones_like(freqs, dtype=np.complex128)

    for k, Ak in enumerate(amps, start=1):
        tk = t1 + (k - 1) * dt
        # Phase shift: 2*pi*f*t_k + global phase
        phase = 2.0 * np.pi * freqs * tk + phi_reflect
        T += Ak * np.exp(1j * phase)

    return T


def apply_mscf_echo_train_fd(H0_f: np.ndarray, freqs: np.ndarray,
                              M_solar: float, R1: float, phi_reflect: float,
                              N_echo: int = 6) -> np.ndarray:
    """
    Apply MSCF echo train to a ringdown waveform in frequency domain.

    H1(f) = H0(f) * T(f)

    where T(f) is the MSCF echo transfer function.

    Parameters
    ----------
    H0_f : np.ndarray
        Frequency-domain ringdown waveform (complex)
    freqs : np.ndarray
        Frequency array (Hz)
    M_solar : float
        Final mass in solar masses
    R1 : float
        Outer surface reflectivity
    phi_reflect : float
        Global reflection phase (radians)
    N_echo : int
        Number of echo copies

    Returns
    -------
    np.ndarray
        H1(f) = H0(f) * T(f) (complex)
    """
    T = mscf_echo_transfer_fd(freqs, M_solar, R1, phi_reflect, N_echo)
    return H0_f * T
