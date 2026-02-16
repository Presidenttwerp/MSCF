# Matter-Space Coupling Framework: Computation Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18648268.svg)](https://doi.org/10.5281/zenodo.18648268)

**Author:** Ishay Roland

This repository contains the computation scripts, precomputed results, and
publication figures cited in:

> Roland, I. (2026). *Matter-Space Coupling Framework: Deriving General
> Relativity as the IR Limit of Causal Substrate Dynamics* (MSCF v2.2.0).

Every numerical result in the paper can be reproduced from the scripts in
`code/` and `mscf_derived_echo/`. Each script is self-contained and
independently verifiable.

## Quick start

To reproduce the cosmology and echo figures with inline results, install the
root dependencies and open the Jupyter notebook:

```
pip install -r code/requirements.txt
jupyter notebook notebooks/reproduce_results.ipynb
```

For the remnant cross-section analysis (Section XII), see
[`mscf_remnant_detection/README.md`](mscf_remnant_detection/README.md) and
the dedicated source in `mscf_remnant_detection/src/`.

For the derived-parameter echo search (Section XI), see
[`mscf_derived_echo/results/RESULTS.md`](mscf_derived_echo/results/RESULTS.md)
and the pipeline scripts in `mscf_derived_echo/scripts/`.

## Reproducing Figures from MSCF v2.2.0

| Script | Generates | Paper reference |
|--------|-----------|-----------------|
| `code/plotting/fig1_summary_panel.py` | Figure 1: Transfer function and Bogoliubov phase through the MSCF bounce | Section IX |
| `code/plotting/fig2_two_surface.py` | Figure 2: Two-surface interior structure with F(r) = (1-x_g)(2-x_g) | Section X |
| `code/plotting/fig3_echo_inversion.py` | Figure 3: Echo amplitude trains — generic ECO, classical MSCF, quantum MSCF | Section XI |

## Reproducing Tables from MSCF v2.2.0

| Script | Generates | Paper reference |
|--------|-----------|-----------------|
| `code/echoes/echo_delay.py` | Table II: Predicted echo delays dt_echo(M, chi) from Kerr tortoise coordinates | Section XI |

## Reproducing Numerical Results from MSCF v2.2.0

| Script | Result | Paper reference |
|--------|--------|-----------------|
| `code/perturbations/coupled_mode_evolution.py` | Transfer function T^2(k) with 2x2 coupled Landau-Zener evolution. Plateau ~6.4, minimum ~0.56, coupling leakage 65% | Section IX |
| `code/ghost_analysis/ghost_window.py` | Ghost window duration 0.23 t_Pl, resonance threshold k > 1.44, total kinetic coefficient remains positive | Section IX |
| `code/cmb/planck_commander_likelihood.py` | Planck Commander likelihood comparison: Delta lnL = -3.26 to -10.68 (LCDM preferred at all kappa values) | Section IX |
| `code/cmb/phase_correlation_test.py` | Bogoliubov phase variation across CMB k-range: 2.9e-3 rad (null feature, scale separation) | Section IX |
| `code/verification/algebraic_identities.py` | Self-consistency identity xi/eta^2 = 1 (Theorem 6.4); substrate-matter unity stability table (Theorem 9.8) | Sections VI, IX |

## Derived-Parameter Echo Search

Zero-free-parameter echo search using greybody-derived amplitudes. The barrier
reflectivity R_b(omega) is computed from the Regge-Wheeler scattering problem.

| Script | Result | Paper reference |
|--------|--------|-----------------|
| `mscf_derived_echo/scripts/00_validate_greybody.py` | Greybody factor validation: 6/6 gates PASS | Section XI |
| `mscf_derived_echo/scripts/01_derived_template_demo.py` | Derived template: \|R_b\|^2 = 0.008 at QNM, A_1/A_2 = 11.5 | Section XI |
| `mscf_derived_echo/scripts/02_gw150914_derived.py` | GW150914 matched filter: NULL (p = 0.100) | Section XI |
| `mscf_derived_echo/scripts/03_spectral_ratio_search.py` | Cepstral echo search: NULL | Section XI |
| `mscf_derived_echo/scripts/04_multi_event_search.py` | Multi-event search (4 events, 9 detectors): ALL NULL | Section XI |
| `mscf_derived_echo/scripts/05_stacked_analysis.py` | Stacked analysis: NULL | Section XI |
| `mscf_derived_echo/scripts/06_null_tests.py` | Null tests: 5/5 PASS | Section XI |

## Repository contents

| Directory | Description |
|-----------|-------------|
| `code/` | Self-contained Python scripts producing every cited numerical result and all three figures. |
| `mscf_derived_echo/` | Zero-parameter echo search pipeline: greybody solver, derived templates, matched filter, cepstrum analysis, and null tests for 4 BBH events. |
| `results/` | Precomputed output data (JSON and DAT files) for verification without re-running. |
| `paper/figures/` | The three publication figures in PNG format at 300 DPI. |
| `data/` | Instructions for obtaining required external data (Planck 2018 Commander TT). |
| `mscf_remnant_detection/` | Gravitational scattering cross-section of Planck-mass remnants against direct DM detection targets (LZ, XENONnT, PandaX-4T). |
| `notebooks/` | Interactive Jupyter notebook reproducing key results with inline figures. |

## Installation

```
pip install -r code/requirements.txt
```

The echo search pipeline requires GWOSC strain data, fetched automatically
via `gwpy` on first run. An internet connection is needed for the initial
data download.

## Running scripts

Run any script from the repository root:

```
python3 code/perturbations/coupled_mode_evolution.py
python3 code/ghost_analysis/ghost_window.py
python3 code/echoes/echo_delay.py
python3 code/cmb/planck_commander_likelihood.py
python3 code/cmb/phase_correlation_test.py
python3 code/plotting/fig1_summary_panel.py
python3 code/plotting/fig2_two_surface.py
python3 code/plotting/fig3_echo_inversion.py
python3 code/verification/algebraic_identities.py
```

Run the derived echo pipeline:

```
python3 mscf_derived_echo/scripts/00_validate_greybody.py
python3 mscf_derived_echo/scripts/02_gw150914_derived.py
python3 mscf_derived_echo/scripts/04_multi_event_search.py
python3 mscf_derived_echo/scripts/06_null_tests.py
```

The Commander likelihood script requires Planck 2018 data: see `data/README.md`
for installation instructions.

The summary panel script (`fig1_summary_panel.py`) requires precomputed results
from `coupled_mode_evolution.py` and `phase_correlation_test.py`.

## Citation

If you use this code, please cite the accompanying paper:

> Roland, I. (2026). Matter-Space Coupling Framework: Deriving General
> Relativity as the IR Limit of Causal Substrate Dynamics. Preprint.

See `CITATION.cff` for machine-readable citation metadata.

## License

MIT License. See [LICENSE](LICENSE).
