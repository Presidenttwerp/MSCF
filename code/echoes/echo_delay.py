"""
MSCF echo delay from Kerr tortoise coordinates.

Computes the deterministic echo delay dt_echo(Mf, chi) from:
  - Outer barrier: co-rotating equatorial light ring r_lr(chi)
  - Inner reflector: MSCF barrier at r_b = M
  - Travel coordinate: Kerr tortoise r*(r)

References:
    MSCF v2.1.7, Section XI.D, Table II.

Key results (dt_echo / M):
    chi = 0:     4.000
    chi = 0.5:   0.127
    chi = 0.9:   9.409
    chi = 0.99:  36.207
    chi = 0.999: 117.145

Output:
    results/echo_predictions/echo_delay_table.json
"""

import json
from pathlib import Path

import numpy as np

# ================================================================
# Physical constants (SI units)
# ================================================================
G_SI: float = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
C_SI: float = 299792458.0          # Speed of light [m/s]
MSUN_KG: float = 1.98847e30        # Solar mass [kg]
MSUN_S: float = G_SI * MSUN_KG / C_SI**3  # Solar mass in seconds (~4.93e-6 s)

# Verification table: {chi: dt/M reference value}
ECHO_DELAY_VERIFICATION: dict[float, float] = {
    0.0:   4.000,
    0.5:   0.1274,
    0.9:   9.4086,
    0.99:  36.2068,
    0.999: 117.1445,
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "echo_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Kerr geometry helpers ----

def light_ring_radius_M(chi: float) -> float:
    """Co-rotating equatorial light-ring radius in units of M (geometric units)."""
    chi = float(np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15))
    if chi >= 0.0:
        return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(-chi)))
    return 2.0 * (1.0 + np.cos((2.0 / 3.0) * np.arccos(abs(chi))))


def kerr_rstar_M(r_M: float, chi: float) -> float:
    """
    Kerr tortoise-like primitive r_*(r) in units of M (M=1).
    Additive constants cancel in differences.
    """
    r = float(r_M)
    chi = float(np.clip(chi, -1.0 + 1e-15, 1.0 - 1e-15))
    s = np.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + s
    r_minus = 1.0 - s
    denom = r_plus - r_minus
    if abs(denom) < 1e-14:
        # Schwarzschild limit
        return r + 2.0 * np.log(abs(r / 2.0 - 1.0))
    A_plus = (2.0 * r_plus) / denom
    A_minus = (2.0 * r_minus) / denom
    return r + A_plus * np.log(abs((r - r_plus) / 2.0)) - A_minus * np.log(abs((r - r_minus) / 2.0))


# ---- Echo delay ----

def mscf_echo_delay_seconds(Mf_solar: float, chi: float) -> float:
    """
    MSCF echo delay dt_echo(M, chi) using Kerr tortoise coordinates.

    Parameters
    ----------
    Mf_solar : float
        Final mass in solar masses.
    chi : float
        Dimensionless spin parameter, -1 < chi < 1.

    Returns
    -------
    float
        Echo delay in seconds.
    """
    chi = float(np.clip(chi, -1.0 + 1e-12, 1.0 - 1e-12))

    M_si = Mf_solar * MSUN_KG
    tM = G_SI * M_si / C_SI**3  # seconds per geometric M

    r_lr = light_ring_radius_M(chi)
    r_b = 1.0  # MSCF barrier at r_b = M

    L = abs(kerr_rstar_M(r_lr, chi) - kerr_rstar_M(r_b, chi))
    dt_echo = 2.0 * L * tM

    # Guardrail: stellar-mass BHs should give < 20 ms
    if Mf_solar < 200.0 and dt_echo > 0.020:
        raise ValueError(
            f"MSCF timing guardrail failed: dt_echo = {dt_echo*1000:.1f} ms > 20 ms "
            f"for Mf = {Mf_solar:.1f} Msun. Expected sub-ms to few-ms range."
        )

    return dt_echo


def mscf_echo_delay_M(chi: float) -> float:
    """
    Echo delay in units of M: dt/M = 2 |r*(r_lr) - r*(r_b)|.

    Parameters
    ----------
    chi : float
        Dimensionless spin.

    Returns
    -------
    float
        dt_echo / M (dimensionless).
    """
    chi = float(np.clip(chi, -1.0 + 1e-12, 1.0 - 1e-12))
    r_lr = light_ring_radius_M(chi)
    r_b = 1.0
    return 2.0 * abs(kerr_rstar_M(r_lr, chi) - kerr_rstar_M(r_b, chi))


def verify_mscf_timing() -> bool:
    """
    Verify MSCF timing against analytically known reference values.

    Uses ECHO_DELAY_VERIFICATION table: {chi: dt/M_ref}.
    Returns True if all pass (< 1% relative error on dt/M).
    """
    all_pass = True
    for chi, dt_over_M_ref in ECHO_DELAY_VERIFICATION.items():
        dt_over_M = mscf_echo_delay_M(chi)
        rel_err = abs(dt_over_M - dt_over_M_ref) / dt_over_M_ref

        if rel_err > 0.01:
            print(f"  FAIL: chi={chi} dt/M={dt_over_M:.4f} vs ref={dt_over_M_ref:.4f} (err={rel_err:.3%})")
            all_pass = False
        else:
            print(f"  PASS: chi={chi} dt/M={dt_over_M:.4f} (ref={dt_over_M_ref:.4f}, err={rel_err:.3%})")

    # Absolute check for GW150914-like params
    dt = mscf_echo_delay_seconds(67.8, 0.68)
    dt_ms = dt * 1000
    if 0.1 < dt_ms < 10.0:
        print(f"  PASS: GW150914-like dt={dt_ms:.4f} ms (in range)")
    else:
        print(f"  FAIL: GW150914-like dt={dt_ms:.4f} ms (out of range)")
        all_pass = False

    return all_pass


def main() -> None:
    print("=" * 60)
    print("MSCF ECHO DELAY TABLE")
    print("=" * 60)

    # Verification
    print("\nVerification against reference values:")
    ok = verify_mscf_timing()
    print(f"\nAll checks passed: {ok}")

    # Full table
    chi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.95, 0.99, 0.999]
    Mf_solar = 67.8  # GW150914 remnant

    print(f"\nEcho delay table for Mf = {Mf_solar} Msun:")
    print(f"{'chi':>8} {'dt/M':>10} {'dt [ms]':>10} {'r_lr/M':>8}")
    print("-" * 40)

    table = []
    for chi in chi_values:
        dt_M = mscf_echo_delay_M(chi)
        dt_s = mscf_echo_delay_seconds(Mf_solar, chi)
        r_lr = light_ring_radius_M(chi)
        print(f"{chi:8.3f} {dt_M:10.4f} {dt_s*1000:10.4f} {r_lr:8.4f}")
        table.append({
            'chi': chi,
            'dt_over_M': dt_M,
            'dt_seconds': dt_s,
            'dt_ms': dt_s * 1000,
            'r_lr_over_M': r_lr,
        })

    # Save
    output = {
        'Mf_solar': Mf_solar,
        'verification': {str(k): v for k, v in ECHO_DELAY_VERIFICATION.items()},
        'all_verified': ok,
        'table': table,
    }

    json_path = OUTPUT_DIR / "echo_delay_table.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == '__main__':
    main()
