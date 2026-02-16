"""
Derived echo amplitudes from greybody factors.

The angular momentum barrier at r ~ 3M has frequency-dependent
reflectivity R_b(ω) computed from the Regge-Wheeler potential.
This is a zero-free-parameter prediction.

Echo amplitude hierarchy:
    A_1(ω) = T_b²(ω)         — first echo: transmit through barrier twice
    A_n(ω) = T_b²(ω) × R_b^{n-1}(ω)  — n-th echo: 2 transmissions + (n-1) reflections

Note: T_b² here means |T_b|² (power, not amplitude squared of complex T_b).
Each traversal of the barrier transmits amplitude |T_b|; two traversals
give power factor |T_b|⁴? No — the echo sees the barrier from inside
the cavity. The first echo enters the cavity (transmission T_b from outside),
bounces off the MSCF wall, and exits (transmission T_b from inside).
So A_1 = |T_b|² in power. Each subsequent echo picks up one R_b reflection
inside the cavity.

Actually, more precisely:
    A_n(ω) = |T_b(ω)|² × |R_b(ω)|^{2(n-1)}  in power
But in amplitude (for the template, which is linear in h):
    a_n(ω) = |T_b(ω)| × |R_b(ω)|^{n-1}

The template uses amplitude coefficients.
"""

import numpy as np

from .greybody import get_greybody_interpolator
from .config import MSUN_S


def derived_echo_amplitudes_at_Momega(Momega, N_echo=6, l=2):
    """
    Compute derived echo amplitude coefficients at given Mω values.

    Parameters
    ----------
    Momega : float or np.ndarray
        Dimensionless frequency Mω.
    N_echo : int
        Number of echo copies.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        Shape (N_echo, N_freq) array of amplitude coefficients a_n(ω).
        If Momega is scalar, shape is (N_echo,).
    """
    interp = get_greybody_interpolator(l=l)
    Momega = np.atleast_1d(np.asarray(Momega, dtype=float))

    Rb2 = interp['Rb2_interp'](Momega)
    Tb2 = interp['Tb2_interp'](Momega)

    # Amplitude (not power) coefficients
    Rb = np.sqrt(np.clip(Rb2, 0, 1))
    Tb = np.sqrt(np.clip(Tb2, 0, 1))

    # a_n(ω) = Tb(ω) × Rb(ω)^{n-1}
    amps = np.zeros((N_echo, len(Momega)))
    for n in range(N_echo):
        amps[n] = Tb * Rb**n

    if len(Momega) == 1:
        return amps[:, 0]
    return amps


def derived_echo_amplitudes_at_freq(f_Hz, Mf_msun, N_echo=6, l=2):
    """
    Compute derived echo amplitude coefficients at physical frequencies.

    Parameters
    ----------
    f_Hz : float or np.ndarray
        Frequency in Hz.
    Mf_msun : float
        Remnant mass in solar masses.
    N_echo : int
        Number of echo copies.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        Shape (N_echo, N_freq) or (N_echo,) amplitude coefficients.
    """
    M_sec = Mf_msun * MSUN_S
    f_Hz = np.atleast_1d(np.asarray(f_Hz, dtype=float))
    Momega = 2.0 * np.pi * f_Hz * M_sec

    return derived_echo_amplitudes_at_Momega(Momega, N_echo=N_echo, l=l)
