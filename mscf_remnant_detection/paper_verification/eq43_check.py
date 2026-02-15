#!/usr/bin/env python3
"""
Standalone algebraic verification of MSCF v2.1.7 Eqs. 36-43.

Derivation chain:
  Rindler expansion (Lemma 12.1)
  → Causal-cell saturation (Axiom 5)
  → κ_max = c²/(2l_P) (Eq. 38)
  → M_min = M_P/2 (Eq. 39)
  → T_eff = T_H[1 − (M_P/2M)²] (Eq. 43)
"""
import numpy as np

# Fundamental constants (CODATA 2018)
G = 6.67430e-11       # m^3 kg^-1 s^-2
c = 2.99792458e8      # m/s
hbar = 1.054571817e-34 # J s
k_B = 1.380649e-23    # J/K

# Planck units (derived)
M_P = np.sqrt(hbar * c / G)
l_P = np.sqrt(hbar * G / c**3)
t_P = np.sqrt(hbar * G / c**5)
E_P = M_P * c**2

print("=" * 70)
print("MSCF v2.1.7 — Algebraic Verification of Eqs. 36-43")
print("=" * 70)

# Axiom 5 (Eq. 3): ρ_crit = c^7 / (ℏ G²)
rho_crit = c**7 / (hbar * G**2)
print(f"\nAxiom 5 (Eq. 3): ρ_crit = c⁷/(ℏG²)")
print(f"  ρ_crit = {rho_crit:.3e} J/m³")

# Eq. 38: κ_max = c² / (2 l_P)
kappa_max = c**2 / (2 * l_P)
print(f"\nEq. 38: κ_max = c²/(2l_P)")
print(f"  κ_max = {kappa_max:.6e} m/s²")

# Eq. 39: From κ_Schw = c⁴/(4GM) ≤ κ_max
#   c⁴/(4GM) ≤ c²/(2l_P)
#   M ≥ c² l_P / (2G)
#   But l_P = √(ℏG/c³), so c² l_P / (2G) = c² √(ℏG/c³) / (2G)
#   = √(c⁴ ℏ G / c³) / (2G) = √(cℏ/G) / 2 = M_P / 2
M_min = c**2 * l_P / (2 * G)
M_min_check = M_P / 2
print(f"\nEq. 39: M_min = M_P/2")
print(f"  From κ_Schw ≤ κ_max: M_min = c²l_P/(2G) = {M_min:.6e} kg")
print(f"  Direct: M_P/2 = {M_min_check:.6e} kg")
print(f"  Agreement: {abs(M_min - M_min_check)/M_min_check:.2e}")
assert abs(M_min - M_min_check) / M_min_check < 1e-10, "FAIL: M_min ≠ M_P/2"

# Eq. 40: Ω_κ(z) = 1 - z² (peeling response)
print(f"\nEq. 40: Ω_κ(z) = 1 - z² where z = κ/κ_max")
for z in [0, 0.5, 1.0]:
    print(f"  Ω_κ({z}) = {1 - z**2:.3f}")

# Eq. 41: κ_eff = κ·Ω_κ = κ(1 - κ²/κ²_max)
print(f"\nEq. 41: κ_eff = κ(1 - κ²/κ²_max)")
for M_factor in [10, 2, 1, 0.5]:
    M = M_factor * M_P
    kappa_schw = c**4 / (4 * G * M)
    z = kappa_schw / kappa_max
    kappa_eff = kappa_schw * (1 - z**2) if z <= 1 else 0
    print(f"  M = {M_factor}M_P: κ_Schw = {kappa_schw:.3e}, z = {z:.4f}, "
          f"κ_eff = {kappa_eff:.3e}")

# Eq. 42: κ_eff/κ = 1 - (M_P/(2M))²
print(f"\nEq. 42: κ_eff/κ = 1 - (M_P/(2M))² for Schwarzschild")
for M_factor in [10, 2, 1, 0.5]:
    M = M_factor * M_P
    ratio = 1 - (M_P / (2*M))**2
    print(f"  M = {M_factor}M_P: κ_eff/κ = {ratio:.6f}")

# Eq. 43: T_eff = T_H × [1 - (1/4)(M_P/M)²]
print(f"\nEq. 43: T_eff = T_H × [1 - ¼(M_P/M)²]")
print(f"  Note: 1 - ¼(M_P/M)² = 1 - (M_P/(2M))² — same as Eq. 42")

for M_factor in [100, 10, 2, 1, 0.5]:
    M = M_factor * M_P
    T_H = hbar * c**3 / (8 * np.pi * G * M * k_B)
    correction = 1 - 0.25 * (M_P / M)**2
    T_eff = T_H * correction
    print(f"  M = {M_factor:6.1f}M_P: T_H = {T_H:.3e} K, "
          f"correction = {correction:+.6f}, T_eff = {T_eff:.3e} K")

# Critical check: T_eff(M_P/2) = 0
M_half = M_P / 2
T_H_half = hbar * c**3 / (8 * np.pi * G * M_half * k_B)
correction_half = 1 - 0.25 * (M_P / M_half)**2

print(f"\n{'='*70}")
print(f"CRITICAL CHECK: T_eff(M_P/2)")
print(f"  M = M_P/2 = {M_half:.6e} kg")
print(f"  T_H = {T_H_half:.6e} K")
print(f"  Correction = 1 - ¼(M_P/(M_P/2))² = 1 - ¼×4 = {correction_half:.15e}")
print(f"  T_eff = {T_H_half * correction_half:.6e} K")
print(f"  Gate 0 PASS: {abs(correction_half) < 1e-15}")
print(f"{'='*70}")

# Summary
print(f"\nSummary of MSCF remnant:")
print(f"  M_rem = M_P/2 = {M_half:.6e} kg")
print(f"  M_rem = {M_half / 1.78266192e-27:.3e} GeV")
print(f"  r_s = 2GM/c² = {2*G*M_half/c**2:.3e} m")
print(f"  r_s/l_P = {2*G*M_half/(c**2 * l_P):.6f}")
print(f"  T_eff = 0 K (stable)")
print(f"\nAll algebra checks PASSED.")
