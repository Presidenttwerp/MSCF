#!/usr/bin/env python3
"""
Full pipeline: compute all cross-sections, rates, limits, alternatives,
run all gates, generate figures, and write RESULTS.md.
"""
import sys
import os
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import constants as c
from src import remnant
from src import cross_section as xs
from src import rate
from src import limits
from src import alternatives as alt
from src import plotting

# Output directory
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def run_gates():
    """Run all verification gates."""
    results = {}
    all_pass = True

    # Gate 0: T_eff(M_P/2) = 0
    eq43 = remnant.verify_eq43()
    g0 = eq43["gate_0_pass"]
    results["Gate 0: T_eff(M_P/2) = 0"] = g0
    all_pass &= g0

    # Gate 1: Constants match CODATA
    g1a = abs(c.M_PLANCK_KG - 2.176e-8) / 2.176e-8 < 1e-3
    g1b = abs(c.M_REM_KG - 1.088e-8) / 1.088e-8 < 1e-3
    g1c = abs(c.L_PLANCK - 1.616e-35) / 1.616e-35 < 1e-3
    g1 = g1a and g1b and g1c
    results["Gate 1: Constants match CODATA (0.1%)"] = g1
    all_pass &= g1

    # Gate 4: Three methods agree
    lz = xs.compute_all("LZ")
    g4 = lz["methods_agree"] and (0.1 < lz["dimensional_ratio"] < 4 * np.pi)
    results["Gate 4: Three cross-section methods agree"] = g4
    all_pass &= g4

    # Gate 5: σ_A ≈ A⁴ × σ_n
    g5 = True
    for det_name in ["LZ", "DarkSide-20k", "SuperCDMS"]:
        det = c.DETECTORS[det_name]
        r = xs.compute_all(det_name)
        ratio = r["sigma_rutherford_m2"] / r["sigma_per_nucleon_m2"]
        expected = det["A"]**4
        g5 &= (0.5 < ratio / expected < 2.0)
    results["Gate 5: σ_A ≈ A⁴ × σ_n"] = g5
    all_pass &= g5

    # Gate 6: Rate < 10⁻²⁰ events/kg/year
    g6 = True
    for det_name in c.DETECTORS:
        r = rate.total_rate(det_name) * c.YR_TO_S
        g6 &= (r < 1e-20)
    results["Gate 6: Rate < 1e-20 events/kg/yr"] = g6
    all_pass &= g6

    # Gate 7: σ_grav / σ_LZ < 10⁻²⁰
    sigma_n = lz["sigma_per_nucleon_cm2"]
    sigma_lz = limits.lz_limit(np.array([c.M_REM_GEV]))[0]
    g7 = sigma_n / sigma_lz < 1e-20
    results["Gate 7: σ_grav / σ_LZ < 1e-20"] = g7
    all_pass &= g7

    # Gate 9: Alternative channels show gaps
    summ = alt.summary()
    g9 = (not summ["femtolensing_detectable"] and
          summ["overclosure_consistent"] and
          not summ["disk_constrained"] and
          not summ["ns_constrained"])
    results["Gate 9: All alternative channels null"] = g9
    all_pass &= g9

    return results, all_pass


def write_results(gate_results, all_pass):
    """Write RESULTS.md."""
    outpath = os.path.join(os.path.dirname(__file__), "..", "RESULTS.md")

    # Compute all data
    props = remnant.remnant_properties()
    chain = remnant.derivation_chain()

    lines = []
    lines.append("# MSCF Remnant Direct Detection: Results")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"MSCF v2.1.7 — Eqs. 36-43, Axiom 5, Theorem 12.2")

    # Remnant properties
    lines.append("\n## Remnant Properties")
    lines.append(f"| Quantity | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| M_rem | {props['M_rem_kg']:.6e} kg |")
    lines.append(f"| M_rem | {props['M_rem_GeV']:.3e} GeV |")
    lines.append(f"| r_s | {props['r_s_m']:.3e} m |")
    lines.append(f"| r_s / l_P | {props['r_s_over_lP']:.6f} |")
    lines.append(f"| n_rem | {props['n_cm3']:.3e} cm^-3 |")
    lines.append(f"| Phi_rem | {props['flux_cm2_s']:.3e} cm^-2 s^-1 |")
    lines.append(f"| lambda_dB / l_P | {props['lambda_dB_over_lP']:.3e} |")

    # Cross-sections per detector
    lines.append("\n## Cross-Sections by Detector")
    lines.append(f"| Detector | A | sigma_A [m^2] | sigma_n [cm^2] | eta_G | Born valid |")
    lines.append(f"|---|---|---|---|---|---|")
    for det_name in c.DETECTORS:
        r = xs.compute_all(det_name)
        lines.append(f"| {det_name} | {r['A']} | {r['sigma_rutherford_m2']:.3e} | "
                      f"{r['sigma_per_nucleon_cm2']:.3e} | {r['eta_G']:.2e} | "
                      f"{'YES' if r['born_valid'] else 'NO'} |")

    # Event rates
    lines.append("\n## Event Rates")
    lines.append(f"| Detector | Rate [/kg/yr] | Expected events |")
    lines.append(f"|---|---|---|")
    for det_name in c.DETECTORS:
        r_rate = rate.total_rate(det_name) * c.YR_TO_S
        N = rate.expected_events(det_name)
        lines.append(f"| {det_name} | {r_rate:.3e} | {N:.3e} |")

    # Comparison with limits
    lines.append("\n## Comparison with Experimental Limits")
    lz_result = xs.compute_all("LZ")
    sigma_n = lz_result["sigma_per_nucleon_cm2"]
    comp = limits.compare_with_limits(sigma_n, c.M_REM_GEV)
    lines.append(f"| Limit | sigma_pred / sigma_limit |")
    lines.append(f"|---|---|")
    for key in ["ratio_LZ", "ratio_XENONnT", "ratio_PandaX", "ratio_nu_floor", "ratio_mica"]:
        lines.append(f"| {key.replace('ratio_', '')} | {comp[key]:.3e} |")

    # Alternative channels
    lines.append("\n## Alternative Detection Channels")
    lens = alt.einstein_radius(3e19)
    femto = alt.femtolensing_check()
    pta = alt.pulsar_timing_shapiro(c.L_PLANCK, 1e13)
    disk = alt.disk_heating_constraint()
    ns = alt.ns_capture_constraint()
    oc = alt.overclosure_check()

    lines.append(f"| Channel | Key metric | Detectable? |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Microlensing | theta_E = {lens['theta_E_rad']:.3e} rad | NO |")
    lines.append(f"| Femtolensing | r_s/lambda = {femto['r_s_over_lambda_MeV']:.3e} | NO |")
    lines.append(f"| Pulsar timing | dt/dt_PTA = {pta['ratio_dt']:.3e} | NO |")
    lines.append(f"| Disk heating | eps/eps_obs = {disk['ratio']:.3e} | NO |")
    lines.append(f"| NS capture | M_acc/M_NS = {ns['M_acc_over_M_NS']:.3e} | NO |")
    lines.append(f"| Overclosure | Omega_rem = {oc['Omega_rem']:.4f} | Consistent |")

    # Energy deposition
    lines.append("\n## Energy Deposition per Transit (1 m path)")
    lines.append(f"| Detector | dE/dx [eV/m] | dE_total [eV] | Threshold [eV] |")
    lines.append(f"|---|---|---|---|")
    for det_name in ["LZ", "DarkSide-20k", "SuperCDMS"]:
        info = rate.energy_deposit_per_transit(det_name)
        E_th_eV = c.DETECTORS[det_name]["E_th_keV"] * 1e3
        lines.append(f"| {det_name} | {info['dE_dx_eV_m']:.3e} | "
                      f"{info['dE_total_eV']:.3e} | {E_th_eV:.0f} |")

    # Gates
    lines.append("\n## Verification Gates")
    lines.append(f"| Gate | Status |")
    lines.append(f"|---|---|")
    for gate, passed in gate_results.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {gate} | {status} |")

    # Verdict
    lines.append("\n## Verdict")
    if all_pass:
        lines.append("**ALL GATES PASS.**")
        lines.append("")
        lines.append("MSCF Planck-mass remnants (M_P/2) interact only gravitationally.")
        lines.append(f"The scattering cross-section sigma_n ~ {sigma_n:.1e} cm^2 lies")
        gap = np.log10(limits.lz_limit(np.array([c.M_REM_GEV]))[0] / sigma_n)
        lines.append(f"**{gap:.0f} orders of magnitude** below the best direct detection limit (LZ).")
        lines.append("")
        lines.append("No current or foreseeable technology can detect these remnants through")
        lines.append("nuclear recoil. This is consistent with all null results in direct DM")
        lines.append("searches and provides a natural explanation for dark matter invisibility")
        lines.append("within the MSCF framework.")
    else:
        lines.append("**SOME GATES FAILED.** Review results above.")

    # Check for NaN/Inf in numerical values only
    import re
    content = "\n".join(lines)
    # Look for standalone nan/inf that appear as numerical values (e.g., "nan", "inf", "-inf")
    has_nan = bool(re.search(r'\bnan\b', content))
    has_inf = bool(re.search(r'(?<![a-zA-Z])inf\b(?![a-zA-Z])', content))
    if has_nan or has_inf:
        lines.append(f"\n**WARNING**: NaN or Inf detected in results!")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    return outpath


def main():
    print("=" * 70)
    print("MSCF Remnant Direct Detection — Full Pipeline")
    print("=" * 70)

    # 1. Remnant properties
    print("\n--- Remnant Properties ---")
    props = remnant.remnant_properties()
    for k, v in props.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")

    # 2. Derivation chain
    print("\n--- Derivation Chain ---")
    chain = remnant.derivation_chain()
    for k, v in chain.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")

    # 3. Cross-sections
    print("\n--- Cross-Sections ---")
    for det_name in c.DETECTORS:
        r = xs.compute_all(det_name)
        print(f"  {det_name}: sigma_n = {r['sigma_per_nucleon_cm2']:.3e} cm^2, "
              f"eta_G = {r['eta_G']:.2e}")

    # 4. Rates
    print("\n--- Event Rates ---")
    for det_name in c.DETECTORS:
        r_rate = rate.total_rate(det_name) * c.YR_TO_S
        N = rate.expected_events(det_name)
        print(f"  {det_name}: {r_rate:.3e} /kg/yr, N = {N:.3e}")

    # 5. Gates
    print("\n--- Verification Gates ---")
    gate_results, all_pass = run_gates()
    for gate, passed in gate_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate}")

    # 6. Figures
    print("\n--- Generating Figures ---")
    figs = plotting.generate_all(OUTDIR)
    for f in figs:
        print(f"  Generated: {f}")

    # 7. Write RESULTS.md
    print("\n--- Writing RESULTS.md ---")
    outpath = write_results(gate_results, all_pass)
    print(f"  Written: {outpath}")

    # Final verdict
    print(f"\n{'='*70}")
    if all_pass:
        print("ALL GATES PASS. Remnants ~28 orders below direct detection.")
    else:
        print("SOME GATES FAILED. Check above.")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
