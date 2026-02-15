# MSCF Remnant Direct Detection Cross-Section

Computes the gravitational scattering cross-section of MSCF Planck-mass remnants against direct dark matter detection targets, and compares with experimental exclusion limits.

## Physics Summary

MSCF (Matter-Space Coupling Framework) v2.1.7 predicts stable Planck-mass remnants:

1. **Axiom 5** (Eq. 3): Maximum curvature density rho_crit = c^7/(hbar G^2)
2. **Eq. 38**: Surface gravity bound kappa_max = c^2/(2 l_P)
3. **Eq. 39**: Minimum black hole mass M_min = M_P/2
4. **Eq. 43**: Modified Hawking temperature T_eff = T_H[1 - (M_P/2M)^2], which vanishes at M_min

These remnants interact **only gravitationally**. The gravitational Rutherford cross-section against xenon nuclei is ~10^-63 cm^2, roughly **28 orders of magnitude below** the best direct detection limit (LZ 2024).

## Installation

```bash
pip install numpy scipy matplotlib pytest
```

## Usage

```bash
# Run full pipeline (all computations + figures + RESULTS.md)
python3 scripts/run_all.py

# Run tests
python3 -m pytest tests/ -v

# Verify paper equations
python3 paper_verification/eq43_check.py
```

## Output

- `figures/` — 6 publication-quality figures (300 dpi)
- `RESULTS.md` — Full numerical results and gate verification

## Verification Gates

| Gate | Check |
|---|---|
| 0 | T_eff(M_P/2) = 0 algebraically |
| 1 | Constants match CODATA 2018 to 0.1% |
| 4 | Three cross-section methods agree within 4pi |
| 5 | sigma_A ~ A^4 sigma_n |
| 6 | Rate < 10^-20 events/kg/year |
| 7 | sigma_grav / sigma_LZ < 10^-20 |
| 9 | All alternative channels show sensitivity gaps |

## Key Results

| Quantity | Value |
|---|---|
| M_rem | 1.088 x 10^-8 kg = 6.1 x 10^18 GeV |
| n_rem | ~5 x 10^-20 cm^-3 |
| sigma_n (per nucleon) | ~10^-59 cm^2 |
| sigma_grav / sigma_LZ | ~10^-28 |
| Expected events (LZ) | ~10^-35 |

**Verdict**: Remnants are ~28 orders below all direct detection. Consistent with null results. Gravitational-only coupling naturally explains dark matter invisibility.

## References

- MSCF v2.1.7: Eqs. 36-43, Axiom 5, Theorem 12.2
- LZ Collaboration (2024), Phys. Rev. Lett.
- XENON Collaboration (2024)
- Carr, Kohri, Sendouda, Yokoyama, Rep. Prog. Phys. 84, 116902 (2021)
- Lewin & Smith, Astropart. Phys. 6, 87 (1996) — Standard halo model

## License

MIT
