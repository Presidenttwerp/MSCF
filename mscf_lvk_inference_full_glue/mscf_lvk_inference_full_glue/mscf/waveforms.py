import numpy as np
from .echo_geometry import delta_t_echo_seconds
from .reflectivity import Rw_of_f

# Physical constants
MSUN_SEC = 4.925491025543576e-06  # M_sun in seconds (G*M_sun/c^3)

def qnm_220_freq_tau(Mf, chi):
    """
    Compute QNM frequency and damping time for l=m=2, n=0 mode.

    Uses Berti et al. fitting formulae (gr-qc/0512160, Table VIII):
        M * omega_R = f1 + f2*(1-chi)^f3   (dimensionless)
        Q = q1 + q2*(1-chi)^q3             (quality factor)

    Physical quantities:
        f = (c^3 / 2*pi*G*M) * (M*omega_R)   [Hz]
        tau = Q / (pi * f)                    [seconds]

    Parameters
    ----------
    Mf : float
        Remnant mass in solar masses
    chi : float
        Dimensionless spin (0 <= chi < 1)

    Returns
    -------
    f0 : float
        QNM frequency in Hz
    tau : float
        Damping time in seconds
    """
    # Berti et al. coefficients for l=m=2, n=0 mode
    f1, f2, f3 = 1.5251, -1.1568, 0.1292
    q1, q2, q3 = 0.7000, 1.4187, -0.4990

    # Ensure chi is bounded
    chi = np.clip(chi, 0.0, 0.9999)

    # Dimensionless frequency and quality factor from fits
    M_omega_R = f1 + f2 * (1.0 - chi)**f3  # dimensionless M*omega
    Q = q1 + q2 * (1.0 - chi)**q3          # quality factor (dimensionless)

    # Convert to physical units
    # M_sec = G*M/c^3, so c^3/(G*M) = 1/M_sec
    # f = (1/2pi) * (1/M_sec) * M_omega_R
    M_sec = float(Mf) * MSUN_SEC  # remnant mass in seconds

    f0 = M_omega_R / (2.0 * np.pi * M_sec)  # Hz
    tau = Q / (np.pi * f0)                   # seconds: tau = Q/(pi*f)

    return float(f0), float(tau)


def ringdown_time_series(t, A, f0, tau, phi, t0):
    """Single damped sinusoid ringdown in strain units."""
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    m = t >= float(t0)
    tt = t[m] - float(t0)
    y[m] = float(A) * np.exp(-tt/float(tau)) * np.cos(2*np.pi*float(f0)*tt + float(phi))
    return y

def tukey_window(N, alpha=0.1):
    """Tukey window: flat in middle, tapered at edges. alpha=fraction tapered."""
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    n = np.arange(N)
    w = np.ones(N)
    # Left taper
    left = n < alpha * N / 2
    w[left] = 0.5 * (1 - np.cos(2 * np.pi * n[left] / (alpha * N)))
    # Right taper
    right = n >= N * (1 - alpha / 2)
    w[right] = 0.5 * (1 - np.cos(2 * np.pi * (N - 1 - n[right]) / (alpha * N)))
    return w


def planck_taper(N, epsilon_start=0.01, epsilon_end=0.1):
    """
    Planck-taper window.

    The Planck taper is C^infinity smooth (infinitely differentiable) and
    provides better spectral leakage suppression than Tukey.

    Parameters
    ----------
    N : int
        Window length
    epsilon_start : float
        Fraction of window for left ramp-up (0 < epsilon < 0.5)
    epsilon_end : float
        Fraction of window for right ramp-down (0 < epsilon < 0.5)

    Returns
    -------
    w : ndarray
        Window values in [0, 1]
    """
    w = np.ones(N)

    # Left taper (rising edge)
    n_left = int(epsilon_start * N)
    if n_left > 0:
        for i in range(1, n_left):
            x = epsilon_start * (1.0 / (i / N) + 1.0 / (i / N - epsilon_start))
            w[i] = 1.0 / (1.0 + np.exp(x))
        w[0] = 0.0

    # Right taper (falling edge)
    n_right = int(epsilon_end * N)
    if n_right > 0:
        for i in range(N - n_right, N - 1):
            x = epsilon_end * (1.0 / (1.0 - i / N) + 1.0 / (1.0 - i / N - epsilon_end))
            w[i] = 1.0 / (1.0 + np.exp(x))
        w[N - 1] = 0.0

    return w

def ringdown_fd(t, A, f0, tau, phi, t0,
                taper_alpha=0.1,
                window=None,
                planck_eps_start=0.01,
                planck_eps_end=0.1,
                N_window=None):
    """
    Return frequency grid f and complex FFT of ringdown time series.

    Parameters
    ----------
    t : array
        Time array (may be zero-padded)
    A, f0, tau, phi, t0 : float
        Ringdown parameters
    taper_alpha : float
        Tukey window alpha (fraction tapered). Only used if window is None or "tukey".
    window : str or None
        Window type: None (default Tukey), "tukey", "planck", or "none"
    planck_eps_start, planck_eps_end : float
        Planck taper fractions (only used if window="planck")
    N_window : int or None
        If set, apply window only to first N_window samples (for zero-padded data).
        Remaining samples are set to zero. If None, window all samples.

    Returns
    -------
    f : array
        Frequency grid
    Y : array
        Complex FFT of windowed ringdown
    """
    y = ringdown_time_series(t, A, f0, tau, phi, t0)
    dt = t[1] - t[0]
    N = len(y)

    # Determine window length
    N_win = N_window if N_window is not None else N

    # Select window (applied to N_win samples)
    if window is None:
        # Default: Tukey for backward compatibility
        w = tukey_window(N_win, alpha=taper_alpha) if (taper_alpha and taper_alpha > 0) else None
    elif window == "tukey":
        w = tukey_window(N_win, alpha=taper_alpha)
    elif window == "planck":
        w = planck_taper(N_win, epsilon_start=planck_eps_start, epsilon_end=planck_eps_end)
    elif window == "none":
        w = None
    else:
        raise ValueError(f"Unknown window={window}")

    if w is not None:
        if N_window is not None and N_window < N:
            # Apply window to first N_window samples, zero the rest
            y_windowed = np.zeros(N)
            y_windowed[:N_win] = y[:N_win] * w
            y = y_windowed
        else:
            y = y * w

    Y = np.fft.rfft(y) * dt  # approx continuous FT convention
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, Y


def ringdown_fd_qnm(t, A, Mf, chi, phi, t0,
                    window=None, planck_eps_start=0.01, planck_eps_end=0.1,
                    N_window=None):
    """
    GR-consistent ringdown in frequency domain.

    Derives f0 and tau from (Mf, chi) using Berti et al. QNM fits.

    Parameters
    ----------
    window : str or None
        Window type: None (default Tukey), "tukey", "planck", or "none"
    planck_eps_start, planck_eps_end : float
        Planck taper fractions (only used if window="planck")
    N_window : int or None
        If set, apply window only to first N_window samples (for zero-padded data).
    """
    f0, tau = qnm_220_freq_tau(Mf, chi)
    return ringdown_fd(t, A, f0, tau, phi, t0,
                       window=window,
                       planck_eps_start=planck_eps_start,
                       planck_eps_end=planck_eps_end,
                       N_window=N_window)

def msfc_echo_transfer_function(f, Mf, chi, R0, f_cut, roll, phi0, Rb0=0.5, T0=1.0, denom_floor=1e-6):
    """
    Frequency-domain cavity echo transfer function:

      EchoTF(f) = T0*Rw(f) / (1 - Rb0*Rw(f)*exp(i 2π f Δt_echo))

    Δt_echo is derived from (Mf,chi). Rb0 and T0 are fixed to avoid degeneracy.

    A floor is applied to |denom| to prevent numerical poles when the cavity
    approaches perfect resonance. This caps the maximum gain of the transfer
    function to ~1/denom_floor.
    """
    dt_echo = delta_t_echo_seconds(Mf, chi)
    Rw = Rw_of_f(f, R0=R0, f_cut=f_cut, roll=roll, phi0=phi0)
    phase = np.exp(1j * 2*np.pi * f * dt_echo)
    denom = (1.0 - float(Rb0) * Rw * phase)

    # Apply floor to prevent division by near-zero (numerical poles)
    # This caps the maximum transfer function gain
    abs_denom = np.abs(denom)
    too_small = abs_denom < denom_floor
    if np.any(too_small):
        # Preserve phase, but floor the magnitude
        denom = np.where(too_small, denom / abs_denom * denom_floor, denom)

    return float(T0) * Rw / denom

def ringdown_plus_echo_fd(t, params, Rb0=0.5, T0=1.0,
                          window=None, planck_eps_start=0.01, planck_eps_end=0.1,
                          N_window=None):
    """
    Build H_total(f) = H_ring(f) + H_ring(f)*EchoTF(f).

    Uses GR-consistent QNM: f0 and tau derived from (Mf, chi).

    params dict must include:
      Ringdown: A, Mf, chi, phi, t0
      Echo:     R0, f_cut, roll, phi0

    Parameters
    ----------
    window : str or None
        Window type: None (default Tukey), "tukey", "planck", or "none"
    planck_eps_start, planck_eps_end : float
        Planck taper fractions (only used if window="planck")
    N_window : int or None
        If set, apply window only to first N_window samples (for zero-padded data).
    """
    # GR-consistent ringdown from (Mf, chi)
    f, Hring = ringdown_fd_qnm(
        t,
        A=params["A"], Mf=params["Mf"], chi=params["chi"],
        phi=params["phi"], t0=params["t0"],
        window=window, planck_eps_start=planck_eps_start, planck_eps_end=planck_eps_end,
        N_window=N_window
    )
    EchoTF = msfc_echo_transfer_function(
        f, Mf=params["Mf"], chi=params["chi"],
        R0=params["R0"], f_cut=params["f_cut"], roll=params["roll"], phi0=params["phi0"],
        Rb0=Rb0, T0=T0
    )
    Htot = Hring * (1.0 + EchoTF)
    return f, Htot
