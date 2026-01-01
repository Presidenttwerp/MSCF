import numpy as np

def light_ring_radius_M(chi: float) -> float:
    """Co-rotating equatorial light-ring radius in units of M (geometric units)."""
    chi = float(chi)
    chi = np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15)
    if chi >= 0.0:
        return 2.0 * (1.0 + np.cos((2.0/3.0) * np.arccos(-chi)))
    return 2.0 * (1.0 + np.cos((2.0/3.0) * np.arccos(abs(chi))))

def kerr_rstar_M(r_M: float, chi: float) -> float:
    """
    Kerr tortoise-like primitive r_*(r) in units of M (M=1). Additive constants cancel in differences.
    dr*/dr = (r^2 + a^2)/Delta, Delta=(r-r+)(r-r-), a=chi*M.
    """
    r = float(r_M)
    chi = float(chi)
    chi = np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15)
    s = np.sqrt(1.0 - chi*chi)
    r_plus = 1.0 + s
    r_minus = 1.0 - s
    denom = (r_plus - r_minus)
    A_plus = (2.0 * r_plus) / denom
    A_minus = (2.0 * r_minus) / denom
    return r + A_plus*np.log(abs((r - r_plus)/2.0)) - A_minus*np.log(abs((r - r_minus)/2.0))

def delta_t_echo_seconds(M_solar: float, chi: float) -> float:
    """
    MSCF echo delay derived from geometry:
      Δt = 2 * |r_*(r_lr) - r_*(r_b)| * (GM/c^3)
    with r_b=M (x_max=2 inversion barrier), r_lr light ring.
    """
    G = 6.67430e-11
    c = 299792458.0
    M_sun = 1.98847e30

    M = float(M_solar) * M_sun
    tM = G * M / c**3  # seconds per geometric M

    r_lr = light_ring_radius_M(chi)
    r_b = 1.0
    L = abs(kerr_rstar_M(r_lr, chi) - kerr_rstar_M(r_b, chi))
    return 2.0 * L * tM
