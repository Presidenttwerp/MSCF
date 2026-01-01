import numpy as np

def Rw_of_f(f_hz, R0=0.6, f_cut=512.0, roll=4.0, phi0=np.pi):
    """
    Minimal inner reflectivity model:
      |Rw(f)| = R0 / (1 + (f/f_cut)^roll)
      Rw(f) = |Rw(f)| * exp(i*phi0)
    """
    f = np.asarray(f_hz, dtype=float)
    mag = float(R0) / (1.0 + (f / float(f_cut))**float(roll))
    return mag * np.exp(1j * float(phi0))
