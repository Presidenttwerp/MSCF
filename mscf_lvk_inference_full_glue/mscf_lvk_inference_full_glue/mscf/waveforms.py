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

def ringdown_fd(t, A, f0, tau, phi, t0, taper_alpha=0.1):
    """Return frequency grid f and complex FFT of ringdown time series (with Tukey taper)."""
    y = ringdown_time_series(t, A, f0, tau, phi, t0)
    dt = t[1] - t[0]
    w = tukey_window(len(y), alpha=taper_alpha)
    Y = np.fft.rfft(y * w) * dt  # approx continuous FT convention
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, Y


def ringdown_fd_qnm(t, A, Mf, chi, phi, t0):
    """
    GR-consistent ringdown in frequency domain.

    Derives f0 and tau from (Mf, chi) using Berti et al. QNM fits.
    """
    f0, tau = qnm_220_freq_tau(Mf, chi)
    return ringdown_fd(t, A, f0, tau, phi, t0)

def msfc_echo_transfer_function(f, Mf, chi, R0, f_cut, roll, phi0, Rb0=0.5, T0=1.0):
    """
    Frequency-domain cavity echo transfer function:

      EchoTF(f) = T0*Rw(f) / (1 - Rb0*Rw(f)*exp(i 2π f Δt_echo))

    Δt_echo is derived from (Mf,chi). Rb0 and T0 are fixed to avoid degeneracy.
    """
    dt_echo = delta_t_echo_seconds(Mf, chi)
    Rw = Rw_of_f(f, R0=R0, f_cut=f_cut, roll=roll, phi0=phi0)
    phase = np.exp(1j * 2*np.pi * f * dt_echo)
    denom = (1.0 - float(Rb0) * Rw * phase)
    return float(T0) * Rw / denom

def ringdown_plus_echo_fd(t, params, Rb0=0.5, T0=1.0):
    """
    Build H_total(f) = H_ring(f) + H_ring(f)*EchoTF(f).

    Uses GR-consistent QNM: f0 and tau derived from (Mf, chi).

    params dict must include:
      Ringdown: A, Mf, chi, phi, t0
      Echo:     R0, f_cut, roll, phi0
    """
    # GR-consistent ringdown from (Mf, chi)
    f, Hring = ringdown_fd_qnm(
        t,
        A=params["A"], Mf=params["Mf"], chi=params["chi"],
        phi=params["phi"], t0=params["t0"]
    )
    EchoTF = msfc_echo_transfer_function(
        f, Mf=params["Mf"], chi=params["chi"],
        R0=params["R0"], f_cut=params["f_cut"], roll=params["roll"], phi0=params["phi0"],
        Rb0=Rb0, T0=T0
    )
    Htot = Hring * (1.0 + EchoTF)
    return f, Htot
