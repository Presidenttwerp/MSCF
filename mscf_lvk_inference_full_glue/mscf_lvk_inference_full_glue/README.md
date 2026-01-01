MSCF LVK-style Inference Glue (Bilby-ready minimal)
==================================================

What you get
------------
A minimal, end-to-end scaffold that you can run locally to compute a Bayes factor:

  H0: GR ringdown-only
  H1: GR ringdown + MSCF echoes

Key MSCF constraint:
  Δt_echo is *derived* from (Mf, chi_f), not a free parameter.

This code intentionally keeps the echo model parameter-light and transparent.
It is not "production LVK" (no calibration marginalization, etc.), but it is
the correct statistical skeleton: likelihood + waveform + sampling.

Install (local machine)
-----------------------
pip install bilby dynesty numpy scipy gwpy matplotlib

Quick start
-----------
1) Fetch open strain for GW150914 (H1 and L1) and estimate PSDs:
   python scripts/fetch_and_make_psd.py --event GW150914 --duration 32 --sample-rate 4096

2) Run nested sampling and compute Bayes factor:
   python run_bayes_factor.py --event GW150914 --duration 32 --sample-rate 4096

Outputs
-------
- out/GW150914_H0_result.json
- out/GW150914_H1_result.json
- Bayes factor printed as log10(BF_10)

Model notes (what is assumed)
-----------------------------
- Ringdown model is a single damped sinusoid (toy but standard as a first pass).
- Echoes are implemented via a frequency-domain cavity transfer function:
    H_total(f) = H_ring(f) + H_ring(f) * EchoTF(f)
  with EchoTF(f) = T0*Rw(f) / (1 - Rb0*Rw(f)*exp(i 2π f Δt_echo))

- Rb0 and T0 are fixed constants (defaults: Rb0=0.5, T0=1.0) to avoid
  degeneracy. You can later promote them to parameters if you want.

Files
-----
- mscf/echo_geometry.py      : Δt_echo(Mf, chi)
- mscf/reflectivity.py       : Rw(f) model (R0, f_cut, roll, phi0)
- mscf/waveforms.py          : ringdown + echoes frequency-domain synthesis
- mscf/likelihood.py         : Gaussian frequency-domain likelihood (Bilby Likelihood)
- scripts/fetch_and_make_psd.py : GWOSC fetch + PSD estimation via Welch
- run_bayes_factor.py        : runs H0 & H1 and prints Bayes factor

