import numpy as np
from .echo_geometry import delta_t_echo_seconds
from .reflectivity import Rw_of_f

def ringdown_time_series(t, A, f0, tau, phi, t0):
    """Single damped sinusoid ringdown in strain units."""
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    m = t >= float(t0)
    tt = t[m] - float(t0)
    y[m] = float(A) * np.exp(-tt/float(tau)) * np.cos(2*np.pi*float(f0)*tt + float(phi))
    return y

def ringdown_fd(t, A, f0, tau, phi, t0):
    """Return frequency grid f and complex FFT of ringdown time series."""
    y = ringdown_time_series(t, A, f0, tau, phi, t0)
    dt = t[1] - t[0]
    Y = np.fft.rfft(y) * dt  # approx continuous FT convention
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, Y

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

    params dict must include:
      Ringdown: A, f0, tau, phi, t0
      Remnant:  Mf, chi
      Echo:     R0, f_cut, roll, phi0
    """
    f, Hring = ringdown_fd(
        t,
        A=params["A"], f0=params["f0"], tau=params["tau"],
        phi=params["phi"], t0=params["t0"]
    )
    EchoTF = msfc_echo_transfer_function(
        f, Mf=params["Mf"], chi=params["chi"],
        R0=params["R0"], f_cut=params["f_cut"], roll=params["roll"], phi0=params["phi0"],
        Rb0=Rb0, T0=T0
    )
    Htot = Hring * (1.0 + EchoTF)
    return f, Htot
