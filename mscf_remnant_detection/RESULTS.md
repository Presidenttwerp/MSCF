# MSCF Remnant Direct Detection: Results

Generated: 2026-02-12 14:38:26
MSCF v2.1.7 — Eqs. 36-43, Axiom 5, Theorem 12.2

## Remnant Properties
| Quantity | Value |
|---|---|
| M_rem | 1.088217e-08 kg |
| M_rem | 6.104e+18 GeV |
| r_s | 1.616e-35 m |
| r_s / l_P | 1.000000 |
| n_rem | 4.914e-20 cm^-3 |
| Phi_rem | 1.081e-12 cm^-2 s^-1 |
| lambda_dB / l_P | 2.725e+03 |

## Cross-Sections by Detector
| Detector | A | sigma_A [m^2] | sigma_n [cm^2] | eta_G | Born valid |
|---|---|---|---|---|---|
| LZ | 131 | 3.099e-56 | 1.068e-60 | 6.81e-15 | YES |
| XENONnT | 131 | 4.649e-56 | 1.602e-60 | 6.81e-15 | YES |
| PandaX-4T | 131 | 4.226e-56 | 1.456e-60 | 6.81e-15 | YES |
| DarkSide-20k | 40 | 2.028e-57 | 8.037e-60 | 2.08e-15 | YES |
| SuperCDMS | 73 | 6.477e-55 | 2.314e-58 | 3.79e-15 | YES |

## Event Rates
| Detector | Rate [/kg/yr] | Expected events |
|---|---|---|
| LZ | 4.857e-32 | 2.671e-28 |
| XENONnT | 7.287e-32 | 2.915e-28 |
| PandaX-4T | 6.624e-32 | 2.451e-28 |
| DarkSide-20k | 1.027e-32 | 2.054e-27 |
| SuperCDMS | 1.823e-30 | 8.021e-29 |

## Comparison with Experimental Limits
| Limit | sigma_pred / sigma_limit |
|---|---|
| LZ | 6.845e-31 |
| XENONnT | 1.884e-31 |
| PandaX | 1.841e-31 |
| nu_floor | 1.050e-29 |
| mica | 1.749e-36 |

## Alternative Detection Channels
| Channel | Key metric | Detectable? |
|---|---|---|
| Microlensing | theta_E = 7.340e-28 rad | NO |
| Femtolensing | r_s/lambda = 8.191e-23 | NO |
| Pulsar timing | dt/dt_PTA = 1.187e-34 | NO |
| Disk heating | eps/eps_obs = 2.246e-22 | NO |
| NS capture | M_acc/M_NS = 3.217e-15 | NO |
| Overclosure | Omega_rem = 0.2650 | Consistent |

## Energy Deposition per Transit (1 m path)
| Detector | dE/dx [eV/m] | dE_total [eV] | Threshold [eV] |
|---|---|---|---|
| LZ | 7.379e-23 | 7.379e-23 | 1500 |
| DarkSide-20k | 3.493e-23 | 3.493e-23 | 7000 |
| SuperCDMS | 1.322e-22 | 1.322e-22 | 40 |

## Verification Gates
| Gate | Status |
|---|---|
| Gate 0: T_eff(M_P/2) = 0 | PASS |
| Gate 1: Constants match CODATA (0.1%) | PASS |
| Gate 4: Three cross-section methods agree | PASS |
| Gate 5: σ_A ≈ A⁴ × σ_n | PASS |
| Gate 6: Rate < 1e-20 events/kg/yr | PASS |
| Gate 7: σ_grav / σ_LZ < 1e-20 | PASS |
| Gate 9: All alternative channels null | PASS |

## Verdict
**ALL GATES PASS.**

MSCF Planck-mass remnants (M_P/2) interact only gravitationally.
The scattering cross-section sigma_n ~ 1.1e-60 cm^2 lies
**30 orders of magnitude** below the best direct detection limit (LZ).

No current or foreseeable technology can detect these remnants through
nuclear recoil. This is consistent with all null results in direct DM
searches and provides a natural explanation for dark matter invisibility
within the MSCF framework.
