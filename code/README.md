# Computation Scripts

Each script is self-contained and independently verifiable. Scripts reference
the paper section, key equations, and output files they produce.

All references point to MSCF v2.1.7.

| Script | Paper section | Key result | Output |
|--------|---------------|------------|--------|
| `perturbations/coupled_mode_evolution.py` | IX.E.1, IX.E.3 | T^2(k) plateau ~6.4, min ~0.56; Landau-Zener 65% leakage | `results/transfer_functions/` |
| `ghost_analysis/ghost_window.py` | IX.E.2 | Ghost window 0.23 t_Pl; resonance threshold k > 1.44 | `results/ghost_analysis/` |
| `echoes/echo_delay.py` | XI.D | Echo delay table dt_echo(M, chi) from Kerr tortoise coordinates | `results/echo_predictions/` |
| `cmb/planck_commander_likelihood.py` | IX.E.4 | Commander Delta lnL = -3.26 to -10.68 (LCDM preferred) | `results/cmb_comparison/` |
| `cmb/phase_correlation_test.py` | IX.E.4 | Phase variation 2.9e-3 rad across CMB k-range (null) | `results/cmb_comparison/` |
| `plotting/fig1_summary_panel.py` | IX.E | Transfer function and Bogoliubov phase summary (Fig. 1) | `paper/figures/` |
| `plotting/fig2_two_surface.py` | X, XI | Two-surface interior structure (Fig. 2) | `paper/figures/` |
| `plotting/fig3_echo_inversion.py` | XI.C | Echo amplitude inversion A2 > A1 (Fig. 3) | `paper/figures/` |
| `verification/algebraic_identities.py` | VI.E, IX.C | Self-consistency xi/eta^2 = 1 (Thm 6.4); stability gradient table (Thm 9.8) | stdout |

## Dependencies

See `requirements.txt`. The Commander likelihood script additionally requires
Planck 2018 Commander TT data (see `data/README.md`).

## Code duplication

`phase_correlation_test.py` duplicates background and pump field routines from
`coupled_mode_evolution.py`. This is intentional: each script is independently
verifiable without importing from the other.
