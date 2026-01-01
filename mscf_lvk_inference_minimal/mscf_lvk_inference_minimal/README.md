MSCF LVK-style Echo Inference (Minimal Repo)
===========================================

What this is
------------
A minimal scaffold to run a *Bayesian model comparison* between:
  H0: GR ringdown-only model
  H1: GR ringdown + MSCF echo train

MSCF constraint (the "sharp" part)
----------------------------------
Δt_echo is NOT a free parameter. It is derived from (Mf, χf):
  Δt_echo(Mf, χf) = (2/c) * | r_*(r_lr(χf)) - r_*(r_b) |

where:
  - r_b = M (MSCF inversion barrier from x_max=2)
  - r_lr(χ) = co-rotating equatorial light-ring radius (analytic proxy)
  - r_*(r) = Kerr/Teukolsky travel coordinate primitive

What is still missing / you must choose
---------------------------------------
1) A reflectivity/impedance model Rw(f) for the inner boundary.
   This repo provides a minimal 3-parameter family:
     Rw(f) = [R0 / (1 + (f/f_cut)^roll)] * exp(i*phi0)
   with physically clear priors.

2) Data + inference engine:
   - Use GWOSC strain (e.g., GW150914) and run with Bilby or PyCBC inference.
   - This repo does not bundle heavy dependencies or datasets.

Suggested inference approach (LVK-style)
----------------------------------------
- Use open GWOSC strain around an event and an estimated PSD.
- Define likelihood p(d|θ,Hi) assuming Gaussian noise with PSD.
- Parameters:
    Shared: ringdown amplitude, phase, t0 (ringdown start), (optional) QNM params or (Mf,χf)
    Echo-only: R0, f_cut, roll, phi0  (Δt_echo derived from Mf,χf)
- Compare evidences Z1/Z0 -> Bayes factor.

References
----------
- GWOSC strain API: https://gwosc.org/api/  (strain-file listing endpoint)
- PyCBC GW150914 inference example: https://pycbc.org/pycbc/latest/html/inference/examples/gw150914.html
- Bilby: https://arxiv.org/abs/1811.02042
- Echo Bayes template search: Lo et al., PhysRevD 99, 084052 (2019)
