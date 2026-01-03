# mscf/mscf_timing.py
"""
MSCF timing relations for echo delays.

Primary delay: t_echo = (r_s/c) * ln(r_s / l_P)
Secondary offset: delta_t = r_s/c

These are derived from the MSCF two-surface model - timing is NOT free,
it's determined by the same mass scale the paper claims.
"""
import math

# Fundamental constants (SI units)
G = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
c = 299792458.0          # Speed of light [m/s]
hbar = 1.054571817e-34   # Reduced Planck constant [J s]
M_sun = 1.98847e30       # Solar mass [kg]


def planck_length() -> float:
    """Planck length: l_P = sqrt(hbar * G / c^3)"""
    return math.sqrt(hbar * G / c**3)


def schwarzschild_radius(M_kg: float) -> float:
    """Schwarzschild radius: r_s = 2GM/c^2"""
    return 2 * G * M_kg / c**2


def t_echo_seconds(M_solar: float) -> float:
    """
    MSCF primary echo delay:
      t_echo = (r_s/c) * ln(r_s / l_P)

    This is the fundamental delay between the original ringdown
    and the first echo, determined by the log enhancement from
    Planck-scale physics near the horizon.

    Parameters
    ----------
    M_solar : float
        Final mass in solar masses

    Returns
    -------
    float
        Echo delay in seconds
    """
    M = M_solar * M_sun
    rs = schwarzschild_radius(M)
    lp = planck_length()
    return (rs / c) * math.log(rs / lp)


def delta_t_seconds(M_solar: float) -> float:
    """
    MSCF secondary echo offset:
      delta_t = t_echo / ln(r_s/l_P) = r_s/c

    This is the spacing between consecutive echoes after the first,
    which is ~1% of the primary delay (the log factor is O(100) for
    stellar-mass black holes).

    Parameters
    ----------
    M_solar : float
        Final mass in solar masses

    Returns
    -------
    float
        Secondary echo spacing in seconds
    """
    M = M_solar * M_sun
    rs = schwarzschild_radius(M)
    return rs / c


def echo_time_k(M_solar: float, k: int) -> float:
    """
    Time of the k-th echo:
      t_1 = t_echo
      t_k = t_echo + (k-1) * delta_t   for k >= 2

    Parameters
    ----------
    M_solar : float
        Final mass in solar masses
    k : int
        Echo number (1-indexed)

    Returns
    -------
    float
        Time of k-th echo in seconds
    """
    if k < 1:
        raise ValueError("Echo number k must be >= 1")
    t1 = t_echo_seconds(M_solar)
    dt = delta_t_seconds(M_solar)
    return t1 + (k - 1) * dt
