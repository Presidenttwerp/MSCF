# Matter-Space Coupling Framework: Computation Repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18648268.svg)](https://doi.org/10.5281/zenodo.18648268)

**Author:** Ishay Roland

This repository contains the computation scripts, precomputed results, and
publication figures cited in:

> Roland, I. (2026). *Matter-Space Coupling Framework: Deriving General
> Relativity as the IR Limit of Causal Substrate Dynamics* (MSCF v2.1.7).

Every numerical result in the paper can be reproduced from the scripts in
`code/`. Each script is self-contained and independently verifiable.

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

## Reproducing Figures from MSCF v2.1.7

| Script | Generates | Paper reference |
|--------|-----------|-----------------|
| `code/plotting/fig1_summary_panel.py` | Figure 1: Transfer function and Bogoliubov phase through the MSCF bounce | Section IX.E |
| `code/plotting/fig2_two_surface.py` | Figure 2: Two-surface interior structure of the MSCF black hole | Sections X, XI |
| `code/plotting/fig3_echo_inversion.py` | Figure 3: Echo amplitude inversion (A2 > A1), the MSCF discriminant | Section XI.C |

## Reproducing Tables from MSCF v2.1.7

| Script | Generates | Paper reference |
|--------|-----------|-----------------|
| `code/echoes/echo_delay.py` | Table II: Predicted echo delays dt_echo(M, chi) from Kerr tortoise coordinates | Section XI.D |

## Reproducing Numerical Results from MSCF v2.1.7

| Script | Result | Paper reference |
|--------|--------|-----------------|
| `code/perturbations/coupled_mode_evolution.py` | Transfer function T^2(k) with 2x2 coupled Landau-Zener evolution. Plateau ~6.4, minimum ~0.56, coupling leakage 65% | Section IX.E.1, IX.E.3 |
| `code/ghost_analysis/ghost_window.py` | Ghost window duration 0.23 t_Pl, resonance threshold k > 1.44, total kinetic coefficient remains positive | Section IX.E.2 |
| `code/cmb/planck_commander_likelihood.py` | Planck Commander likelihood comparison: Delta lnL = -3.26 to -10.68 (LCDM preferred at all kappa values) | Section IX.E.4 |
| `code/cmb/phase_correlation_test.py` | Bogoliubov phase variation across CMB k-range: 2.9e-3 rad (null feature, scale separation) | Section IX.E.4 |
| `code/verification/algebraic_identities.py` | Self-consistency identity xi/eta^2 = 1 (Theorem 6.4); substrate-matter unity stability table (Theorem 9.8, Eq. 24) | Sections VI.E, IX.C |

## Repository contents

| Directory | Description |
|-----------|-------------|
| `code/` | Nine self-contained Python scripts producing every cited numerical result and all three figures. |
| `results/` | Precomputed output data (JSON and DAT files) for verification without re-running. |
| `paper/figures/` | The three publication figures in PNG format at 300 DPI. |
| `data/` | Instructions for obtaining required external data (Planck 2018 Commander TT). |
| `mscf_remnant_detection/` | Gravitational scattering cross-section of Planck-mass remnants against direct DM detection targets (LZ, XENONnT, PandaX-4T). |
| `notebooks/` | Interactive Jupyter notebook reproducing key results with inline figures. |

## Installation

```
pip install -r code/requirements.txt
```

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
