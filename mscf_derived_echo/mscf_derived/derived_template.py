"""
Zero-free-parameter echo interference template.

Constructs:
    t(f) = h_QNM(f) × Σ_n a_n(f) exp(i(2πf·n·dt + φ))

where:
    h_QNM(f) = Lorentzian ringdown from Berti fits
    a_n(f) = |T_b(f)| × |R_b(f)|^{n-1}  (derived from greybody)
    dt = mscf_echo_delay_seconds(Mf, chi)  (Kerr tortoise)
    φ = global reflection phase (only free parameter, marginalized via |Z|)
"""

import numpy as np

from .derived_amplitudes import derived_echo_amplitudes_at_freq
from .config import MSUN_S, QNM_220_COEFFS, DEFAULT_N_ECHOES


# ---- QNM properties (from Berti fits) ----

def qnm_220_freq_tau(Mf_msun, chi):
    """QNM frequency and damping time for l=m=2, n=0 mode."""
    f1, f2, f3 = QNM_220_COEFFS['f']
    q1, q2, q3 = QNM_220_COEFFS['Q']

    chi = float(np.clip(chi, 0.0, 0.9999))
    M_omega_R = f1 + f2 * (1.0 - chi)**f3
    Q = q1 + q2 * (1.0 - chi)**q3

    M_sec = float(Mf_msun) * MSUN_S
    f0 = M_omega_R / (2.0 * np.pi * M_sec)
    tau = Q / (np.pi * f0)
    return float(f0), float(tau)


def lorentzian_ringdown(freqs, f0, tau):
    """QNM Lorentzian spectral profile: H(f) = tau / (1 + i·2π(f-f0)·tau)."""
    return tau / (1.0 + 1j * 2.0 * np.pi * (freqs - f0) * tau)


# ---- Echo timing (self-contained, no cross-package import) ----

def _light_ring_radius_M(chi):
    """Co-rotating equatorial light-ring radius in units of M."""
    chi = float(np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15))
    if chi >= 0.0:
        return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-chi)))
    return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(abs(chi))))


def _kerr_rstar_M(r_M, chi):
    """Kerr tortoise coordinate r*(r) in units of M."""
    r = float(r_M)
    chi = float(np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15))
    s = np.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + s
    r_minus = 1.0 - s
    denom = r_plus - r_minus
    if abs(denom) < 1e-14:
        return r + 2.0 * np.log(abs(r / 2.0 - 1.0))
    A_plus = (2.0 * r_plus) / denom
    A_minus = (2.0 * r_minus) / denom
    return r + A_plus * np.log(abs((r - r_plus) / 2.0)) - A_minus * np.log(abs((r - r_minus) / 2.0))


def mscf_echo_delay_seconds(Mf_msun, chi):
    """MSCF echo delay dt_echo(Mf, chi) in seconds."""
    from .config import G_SI, C_SI, MSUN_KG
    chi = float(np.clip(chi, -1.0 + 1e-12, 1.0 - 1e-12))
    M_si = Mf_msun * MSUN_KG
    tM = G_SI * M_si / C_SI**3

    r_lr = _light_ring_radius_M(chi)
    r_b = 1.0  # MSCF barrier at r = M
    L = abs(_kerr_rstar_M(r_lr, chi) - _kerr_rstar_M(r_b, chi))
    return 2.0 * L * tM


# ---- Template ----

def derived_interference_template(
    freqs,
    Mf_msun,
    chi,
    phi=0.0,
    N_echo=DEFAULT_N_ECHOES,
    l=2,
):
    """
    Zero-free-parameter echo interference template.

    t(f) = h_QNM(f) × Σ_n a_n(f) exp(i(2πf·n·dt + φ))

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array [Hz].
    Mf_msun : float
        Remnant mass in solar masses.
    chi : float
        Dimensionless spin (0 <= chi < 1).
    phi : float
        Global reflection phase [radians].
    N_echo : int
        Number of echo copies.
    l : int
        Angular momentum quantum number for greybody.

    Returns
    -------
    np.ndarray
        Complex template in frequency domain (echo excess only).
    """
    freqs = np.asarray(freqs, dtype=float)

    # QNM Lorentzian envelope
    f0, tau = qnm_220_freq_tau(Mf_msun, chi)
    h_qnm = lorentzian_ringdown(freqs, f0, tau)

    # MSCF cavity timing
    dt = mscf_echo_delay_seconds(Mf_msun, chi)

    # Derived amplitude hierarchy: a_n(f) from greybody factors
    # Shape: (N_echo, N_freq)
    amps = derived_echo_amplitudes_at_freq(freqs, Mf_msun, N_echo=N_echo, l=l)

    # Build echo sum
    echo_sum = np.zeros(len(freqs), dtype=np.complex128)
    for n in range(N_echo):
        k = n + 1  # echo number (1-indexed)
        tk = k * dt
        a_n = amps[n] if amps.ndim > 1 else amps  # handle scalar case
        echo_sum += a_n * np.exp(1j * (2.0 * np.pi * freqs * tk + phi))

    return h_qnm * echo_sum


def derived_template_at_delay(
    freqs,
    dt,
    f0,
    tau,
    Mf_msun,
    phi=0.0,
    N_echo=DEFAULT_N_ECHOES,
    l=2,
):
    """
    Derived template at an arbitrary delay (for Z(dt) scanning).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array [Hz].
    dt : float
        Echo delay [s].
    f0 : float
        QNM frequency [Hz].
    tau : float
        QNM damping time [s].
    Mf_msun : float
        Remnant mass (needed for greybody frequency conversion).
    phi : float
        Global reflection phase [radians].
    N_echo : int
        Number of echoes.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        Complex template in frequency domain.
    """
    freqs = np.asarray(freqs, dtype=float)
    h_qnm = lorentzian_ringdown(freqs, f0, tau)

    amps = derived_echo_amplitudes_at_freq(freqs, Mf_msun, N_echo=N_echo, l=l)

    echo_sum = np.zeros(len(freqs), dtype=np.complex128)
    for n in range(N_echo):
        k = n + 1
        tk = k * dt
        a_n = amps[n] if amps.ndim > 1 else amps
        echo_sum += a_n * np.exp(1j * (2.0 * np.pi * freqs * tk + phi))

    return h_qnm * echo_sum
