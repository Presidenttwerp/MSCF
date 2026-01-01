# MSCF-LVK: Modified Spacetime Correlator Framework Echo Search

A Bayesian inference pipeline for searching gravitational wave echoes in LIGO/Virgo data using the Modified Spacetime Correlator Framework (MSCF).

## Overview

This pipeline computes Bayes factors comparing:
- **H0**: Standard GR ringdown (damped sinusoid from QNM physics)
- **H1**: GR ringdown + MSCF echo signatures (frequency-domain cavity transfer function)

Key MSCF constraint: The echo time delay `dt_echo` is *derived* from remnant parameters `(Mf, chi)`, not a free parameter.

## Results Summary

### GW150914 Echo Detection

Three independent replication runs with different random seeds:

| Run | Seed | ln(BF) | log10(BF) | Interpretation |
|-----|------|--------|-----------|----------------|
| 1 | 1234 | 11.85 | **5.15** | Decisive |
| 2 | 5678 | 13.06 | **5.67** | Decisive |
| 3 | 9012 | 12.49 | **5.43** | Decisive |
| **Mean** | - | 12.47 +/- 0.49 | **5.41 +/- 0.21** | |

#### Posterior Estimates

**H0 (GR Ringdown):**
- Remnant mass: Mf = 76.7 [74.6, 78.9] M_sun (detector-frame)
- Remnant spin: chi = 0.61 [0.57, 0.65]
- Amplitude: A = 1.79e-21 [1.65, 1.93]e-21

**H1 (Echo Model):**
- Remnant mass: Mf = 76.4 [74.0, 78.9] M_sun
- Remnant spin: chi = 0.66 [0.62, 0.69]
- Amplitude: A = 1.25e-21 [1.10, 1.41]e-21
- **Reflectivity: R0 = 0.97 [0.92, 0.99]** (high)
- **Cutoff frequency: f_cut = 190.5 [172, 211] Hz**
- **Rolloff steepness: roll = 8.6 [7.2, 9.6]**

### Null Tests (Pipeline Validation)

Five off-source segments containing only detector noise (no GW signal):

| Segment | GPS Time | Offset from GW150914 | log10(BF) | Status |
|---------|----------|----------------------|-----------|--------|
| 1 | 1126258462.4 | -1000s | -0.011 | PASS |
| 2 | 1126257462.4 | -2000s | -0.010 | PASS |
| 3 | 1126258162.4 | -1300s | -0.040 | PASS |
| 4 | 1126257862.4 | -1600s | -0.012 | PASS |
| 5 | 1126257562.4 | -1900s | +0.021 | PASS |
| **Mean** | - | - | **-0.011 +/- 0.019** | |

**Validation criterion:** |log10(BF)| < 1 for all null segments (no false positives).

**Result: All 5 null tests PASS.** The pipeline correctly returns BF ~ 1 (no evidence for echoes) when analyzing pure noise.

### Conclusion

**Pipeline VALIDATED:**
- Strong echo detection on GW150914: log10(BF) = 5.41 +/- 0.21
- No false positives in off-source data: log10(BF) ~ 0
- Reproducible across 3 independent sampler runs

## Installation

```bash
pip install bilby dynesty numpy scipy gwpy matplotlib
```

## Usage

### 1. Fetch Data and Estimate PSD

```bash
python scripts/fetch_and_make_psd.py \
    --event GW150914 \
    --gps 1126259462.4 \
    --duration 32 \
    --psd-duration 512 \
    --sample-rate 4096 \
    --ifos H1,L1 \
    --outdir out
```

### 2. Run Bayes Factor Analysis

```bash
python run_bayes_factor.py \
    --event GW150914 \
    --gps 1126259462.4 \
    --ifos H1,L1 \
    --outdir out \
    --resultdir out \
    --seed 1234
```

### 3. Run Null Tests

```bash
python scripts/run_null_tests.py \
    --n-segments 5 \
    --outdir out_null
```

## Pipeline Architecture

```
                    +------------------+
                    |  GWOSC Open Data |
                    +--------+---------+
                             |
                             v
              +-----------------------------+
              | fetch_and_make_psd.py       |
              | - Fetch H1/L1 strain        |
              | - Off-source PSD estimation |
              | - Tukey windowing           |
              +-------------+---------------+
                            |
                            v
              +-----------------------------+
              | run_bayes_factor.py         |
              | - Build H0/H1 likelihoods   |
              | - Dynesty nested sampling   |
              | - Evidence computation      |
              +-------------+---------------+
                            |
                +-----------+-----------+
                |                       |
                v                       v
        +---------------+       +---------------+
        | H0: Ringdown  |       | H1: Ring+Echo |
        | - (Mf, chi)   |       | - (Mf, chi)   |
        | - QNM f0, tau |       | - QNM f0, tau |
        +---------------+       | - R0, f_cut   |
                                | - roll, phi0  |
                                +---------------+
```

## Physics Model

### Ringdown (H0)
Standard damped sinusoid with QNM frequencies from Berti et al. fits:
- f_0(Mf, chi) = fundamental mode frequency
- tau(Mf, chi) = damping time

### Echo Transfer Function (H1)
Frequency-domain cavity model:
```
H_total(f) = H_ring(f) * [1 + EchoTF(f)]
EchoTF(f) = T0 * Rw(f) / [1 - Rb0 * Rw(f) * exp(i 2pi f dt_echo)]
```

Where:
- `dt_echo = 2 * r_horizon(Mf, chi)` (derived, not free)
- `Rw(f) = R0 * sigmoid(f; f_cut, roll) * exp(i * phi0)`
- `Rb0 = 0.5` (fixed black hole reflectivity)
- `T0 = 1.0` (fixed transmission)

## Sampler Settings

- Sampler: Dynesty (dynamic nested sampling)
- Live points: 800
- Convergence: dlogz = 0.1
- Sampling: Random walk (rwalk) with 50 steps
- Band: 30-1024 Hz

## File Structure

```
mscf_lvk_inference_full_glue/
|-- mscf/
|   |-- echo_geometry.py     # dt_echo(Mf, chi) from horizon radius
|   |-- reflectivity.py      # Rw(f) frequency-dependent reflectivity
|   |-- waveforms.py         # Ringdown + echo frequency-domain synthesis
|   |-- likelihood.py        # Gaussian frequency-domain likelihood
|-- scripts/
|   |-- fetch_and_make_psd.py  # GWOSC data fetch + Welch PSD
|   |-- run_null_tests.py      # Off-source validation tests
|-- run_bayes_factor.py        # Main Bayes factor computation
|-- out/                       # GW150914 results (not tracked)
|-- out_null/                  # Null test results (not tracked)
```

## Priors

| Parameter | Prior | Notes |
|-----------|-------|-------|
| Mf | TruncatedGaussian(67.8, 4.0, [55, 85]) | Detector-frame mass |
| chi | TruncatedGaussian(0.68, 0.05, [0, 0.99]) | Dimensionless spin |
| A | LogUniform(1e-24, 1e-19) | Strain amplitude |
| t0 | Uniform(gps-0.05, gps+0.05) | Ringdown start time |
| phi | Uniform(0, 2pi) | Initial phase |
| R0 | Uniform(0, 1) | Reflectivity magnitude |
| f_cut | LogUniform(50, 2000) | Cutoff frequency |
| roll | Uniform(1, 10) | Sigmoid steepness |
| phi0 | Uniform(0, 2pi) | Reflectivity phase |

## References

- Berti et al. (2009) - QNM fitting formulae
- GWTC-1 - GW150914 remnant parameters
- MSCF theory - [citation pending]

## License

Research code for gravitational wave echo searches.
