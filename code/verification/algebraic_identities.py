#!/usr/bin/env python3
"""
Algebraic identity verification for MSCF v2.1.7.

Reproduces two results from the paper:

1. Self-Consistency Identity (Section VI E, Theorem 6.4):
   xi / eta^2 = 1 for any value of l_* / l_P.

2. Substrate-Matter Unity (Section IX C, Theorem 9.8, Eq. 24):
   F_s^total / M^2 = epsilon_m - 1, where epsilon_m = 3(1+w)/2.
   Stable (positive) for w > -1/3.

Reference: Roland, I. (2026). Matter-Space Coupling Framework v2.1.7.
"""
import numpy as np

# Fundamental constants (CODATA 2018)
G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 2.99792458e8       # m/s
hbar = 1.054571817e-34  # J s

# Planck length
l_P = np.sqrt(hbar * G / c**3)


def verify_self_consistency():
    """
    Theorem 6.4: xi / eta^2 = 1 identically.

    From Theorem 6.2: eta = c^3 l_*^2 / (hbar G) = (l_* / l_P)^2
    From Theorem 6.3: xi = rho_crit l_*^4 / (hbar c) = (l_* / l_P)^4

    Therefore xi / eta^2 = (l_*/l_P)^4 / ((l_*/l_P)^2)^2 = 1.

    Tested at l_*/l_P = {0.37, 0.5, 1.0, sqrt(2), 2.0} per paper.
    """
    print("=" * 60)
    print("Theorem 6.4: Self-Consistency Identity")
    print("  xi / eta^2 = 1 for all l_*")
    print("=" * 60)

    rho_crit = c**7 / (hbar * G**2)  # Axiom 5

    test_ratios = [0.37, 0.5, 1.0, np.sqrt(2), 2.0]
    all_pass = True

    print(f"\n  {'l_*/l_P':>10}  {'eta':>14}  {'xi':>14}  {'xi/eta^2':>16}  {'Pass':>6}")
    print("  " + "-" * 66)

    for r in test_ratios:
        l_star = r * l_P

        # Theorem 6.2: entropy per link
        eta = c**3 * l_star**2 / (hbar * G)

        # Theorem 6.3: max links per 4-cell
        xi = rho_crit * l_star**4 / (hbar * c)

        ratio = xi / eta**2
        ok = abs(ratio - 1.0) < 1e-10
        all_pass &= ok

        print(f"  {r:10.4f}  {eta:14.6e}  {xi:14.6e}  {ratio:16.10f}  {'PASS' if ok else 'FAIL':>6}")

    print(f"\n  All values: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def verify_stability_gradient():
    """
    Theorem 9.8: Substrate-Matter Unity.

    F_s^total = M^2 (epsilon_m - 1), where epsilon_m = 3(1+w)/2.
    Stable when F_s^total > 0, i.e., w > -1/3.

    Table from paper (Eq. 24):
      Dust       w=0    epsilon_m=3/2  F_s/M^2=+1/2
      Radiation  w=1/3  epsilon_m=2    F_s/M^2=+1
      Stiff      w=1    epsilon_m=3    F_s/M^2=+2
      Dark energy w=-1  epsilon_m=0    F_s/M^2=-1 (unstable)
    """
    print("\n" + "=" * 60)
    print("Theorem 9.8: Substrate-Matter Unity")
    print("  F_s^total / M^2 = epsilon_m - 1 = 3(1+w)/2 - 1")
    print("=" * 60)

    cases = [
        ("Dust",        0.0,   3/2, +0.5),
        ("Radiation",   1/3,   2.0, +1.0),
        ("Stiff matter", 1.0,  3.0, +2.0),
        ("Dark energy", -1.0,  0.0, -1.0),
    ]

    all_pass = True

    print(f"\n  {'Matter':>14}  {'w':>6}  {'eps_m':>8}  {'F_s/M^2':>10}  {'Stable':>8}  {'Pass':>6}")
    print("  " + "-" * 62)

    for name, w, eps_expected, fs_expected in cases:
        eps_m = 3.0 * (1.0 + w) / 2.0
        fs_ratio = eps_m - 1.0
        stable = fs_ratio > 0

        ok_eps = abs(eps_m - eps_expected) < 1e-14
        ok_fs = abs(fs_ratio - fs_expected) < 1e-14
        ok = ok_eps and ok_fs
        all_pass &= ok

        stability_str = "yes" if stable else "no"
        print(f"  {name:>14}  {w:>6.2f}  {eps_m:>8.3f}  {fs_ratio:>+10.3f}  {stability_str:>8}  {'PASS' if ok else 'FAIL':>6}")

    # Verify critical threshold
    w_crit = -1/3
    eps_crit = 3.0 * (1.0 + w_crit) / 2.0
    fs_crit = eps_crit - 1.0
    threshold_ok = abs(fs_crit) < 1e-14
    all_pass &= threshold_ok

    print(f"\n  Critical threshold: w = -1/3 -> F_s/M^2 = {fs_crit:.1f} (marginal)")
    print(f"  Stability requires w > -1/3: {'PASS' if threshold_ok else 'FAIL'}")
    print(f"\n  All values: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    p1 = verify_self_consistency()
    p2 = verify_stability_gradient()

    print("\n" + "=" * 60)
    if p1 and p2:
        print("ALL ALGEBRAIC IDENTITIES VERIFIED.")
    else:
        print("SOME CHECKS FAILED.")
    print("=" * 60)
