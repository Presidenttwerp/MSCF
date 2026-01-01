import numpy as np

def Rw_of_f(f, R0=0.6, f_cut=512.0, roll=4.0, phi0=np.pi):
    """
    Minimal inner reflectivity model for the MSCF barrier.

    |Rw(f)| = R0 / (1 + (f/f_cut)^roll)
    Rw(f) = |Rw(f)| * exp(i*phi0)

    Parameters:
      R0   : low-frequency reflectivity magnitude (0..1)
      f_cut: roll-off frequency (Hz)
      roll : roll-off exponent (>0)
      phi0 : constant phase shift (radians)
    """
    f = np.asarray(f, dtype=float)
    mag = R0 / (1.0 + (f / f_cut)**roll)
    return mag * np.exp(1j * float(phi0))
