#!/usr/bin/env python
"""Test direct (f0, tau) parameterization to isolate bias source.

This test uses f0 and tau directly as parameters (bypassing the Mf, chi → f0, tau
QNM mapping) to determine if the ~+10 Hz f0 bias is due to:
1. Berti et al. QNM fitting formulas
2. Likelihood/waveform implementation
3. Sampler/prior issues

If direct (f0, tau) inference shows no bias, the problem is in the QNM formulas.
If bias persists, the problem is in the likelihood or sampler.
"""

import numpy as np
import bilby
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.waveforms import ringdown_fd, planck_taper


def generate_colored_noise(f, psd, rng=None):
    """Generate colored Gaussian noise from one-sided PSD."""
    if rng is None:
        rng = np.random.default_rng()

    df = f[1] - f[0] if len(f) > 1 else 1.0
    real_part = rng.standard_normal(len(f))
    imag_part = rng.standard_normal(len(f))
    sigma = np.sqrt(psd * df / 2)
    noise = sigma * (real_part + 1j * imag_part)

    if len(f) > 0:
        noise[0] = noise[0].real * np.sqrt(2)
    if len(f) > 1:
        noise[-1] = noise[-1].real * np.sqrt(2)

    return noise


class DirectF0TauLikelihood(bilby.core.likelihood.Likelihood):
    """Likelihood using f0 and tau directly as parameters."""

    def __init__(self, t, data_dict, psd_dict, fmin=150.0, fmax=400.0,
                 window="planck", planck_eps_start=0.01, planck_eps_end=0.1,
                 N_window=None):
        super().__init__(parameters={})
        self.t = np.asarray(t, dtype=float)
        self.data = data_dict
        self.psd = psd_dict
        self.fmin = fmin
        self.fmax = fmax
        self.window = window
        self.planck_eps_start = planck_eps_start
        self.planck_eps_end = planck_eps_end
        self.N_window = N_window

    def _waveform(self, f_grid):
        """Generate ringdown waveform using f0 and tau directly."""
        p = self.parameters
        # Use ringdown_fd which takes f0 and tau directly
        f, H = ringdown_fd(
            self.t,
            A=p["A"],
            f0=p["f0"],  # Direct f0, not derived from Mf/chi
            tau=p["tau"],  # Direct tau, not derived from Mf/chi
            phi=p["phi"],
            t0=p["t0"],
            window=self.window,
            planck_eps_start=self.planck_eps_start,
            planck_eps_end=self.planck_eps_end,
            N_window=self.N_window
        )
        return H

    def log_likelihood(self):
        logL = 0.0
        for ifo in self.data:
            f, d = self.data[ifo]
            _, Sn = self.psd[ifo]

            # Drop f=0 bin
            f = f[1:]
            d = d[1:]
            Sn = Sn[1:]

            # Band-limit
            band_mask = (f >= self.fmin) & (f <= self.fmax)
            df = f[1] - f[0]

            h_full = self._waveform(np.concatenate(([0.0], f)))
            h = h_full[1:]

            d_band = d[band_mask]
            Sn_band = Sn[band_mask]
            h_band = h[band_mask]

            if not np.all(np.isfinite(h_band)):
                return -np.inf

            resid = d_band - h_band
            logL += -2.0 * np.sum((np.abs(resid)**2) / (Sn_band * df))

        return float(logL)


def run_direct_f0_tau_test(args):
    """Run injection-recovery test with direct f0/tau parameterization."""

    # Setup
    sample_rate = args.sample_rate
    duration = args.duration
    pad_factor = args.pad_factor
    eps_start, eps_end = args.eps_start, args.eps_end

    dt = 1.0 / sample_rate
    N_orig = int(duration * sample_rate)
    N_pad = int(N_orig * pad_factor)

    t = np.arange(N_orig) * dt
    t_padded = np.arange(N_pad) * dt
    f = np.fft.rfftfreq(N_pad, d=dt)
    df = f[1] - f[0]

    # Simple analytical PSD
    S0 = 1e-47
    f0_psd = 100.0
    psd = np.zeros_like(f)
    mask = f > 10
    psd[mask] = S0 * ((f0_psd / f[mask])**4 + 2 + (f[mask] / f0_psd)**2)
    psd[~mask] = S0 * 1e6
    psd = np.maximum(psd, 1e-50)

    # Injection parameters - use DIRECT f0 and tau
    # These are typical values for a ~68 Msun, chi~0.68 remnant
    f0_inj = args.f0  # Hz
    tau_inj = args.tau * 1e-3  # Convert ms to seconds
    t0_inj = args.t0 * 1e-3  # Convert ms to seconds
    phi_inj = np.pi / 4

    # Compute required amplitude for target SNR
    rng = np.random.default_rng(args.seed)

    # Generate template at unit amplitude
    taper = planck_taper(N_orig, eps_start, eps_end)
    h_td_unit = np.zeros(N_orig)
    m = t >= t0_inj
    tt = t[m] - t0_inj
    h_td_unit[m] = np.exp(-tt / tau_inj) * np.cos(2 * np.pi * f0_inj * tt + phi_inj)
    h_td_unit *= taper

    h_padded = np.zeros(N_pad)
    h_padded[:N_orig] = h_td_unit
    h_fft_unit = np.fft.rfft(h_padded) * dt

    # Compute SNR normalization
    band = (f >= args.fmin) & (f <= args.fmax)
    snr_integrand = np.abs(h_fft_unit[band])**2 / (psd[band] * df)
    snr_unit_sq = 4.0 * np.sum(snr_integrand)
    snr_unit = np.sqrt(snr_unit_sq)

    A_inj = args.snr / snr_unit if snr_unit > 0 else 1e-21

    # Generate colored noise
    noise_fft = generate_colored_noise(f, psd, rng=rng)

    # Inject signal
    h_fft = A_inj * h_fft_unit
    d_fft = noise_fft + h_fft

    # Verify injected SNR
    snr_actual = A_inj * snr_unit

    print("=" * 60)
    print("DIRECT (f0, tau) PARAMETERIZATION TEST")
    print("=" * 60)
    print(f"Injected parameters:")
    print(f"  f0 = {f0_inj:.2f} Hz")
    print(f"  tau = {tau_inj*1000:.3f} ms")
    print(f"  t0 = {t0_inj*1000:.2f} ms")
    print(f"  A = {A_inj:.2e}")
    print(f"  SNR = {snr_actual:.1f}")
    print()

    # Setup bilby
    os.makedirs(args.outdir, exist_ok=True)

    data_dict = {"H1": (f, d_fft)}
    psd_dict = {"H1": (f, psd)}

    likelihood = DirectF0TauLikelihood(
        t_padded, data_dict, psd_dict,
        fmin=args.fmin, fmax=args.fmax,
        window="planck", planck_eps_start=eps_start, planck_eps_end=eps_end,
        N_window=N_orig
    )

    # UNIFORM priors on f0 and tau directly
    # f0: broad range around expected value
    # tau: 1-10 ms typical for BH ringdowns
    priors = bilby.core.prior.PriorDict()
    priors["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    priors["f0"] = bilby.core.prior.Uniform(150.0, 400.0, "f0")  # Hz
    priors["tau"] = bilby.core.prior.Uniform(0.001, 0.015, "tau")  # 1-15 ms in seconds
    priors["phi"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi")
    priors["t0"] = bilby.core.prior.Uniform(0.0, 0.010, "t0")  # 0-10 ms

    print("Priors (UNIFORM, not truth-centered):")
    for k, v in priors.items():
        print(f"  {k}: {v}")
    print()

    # Run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=args.nlive,
        sample="rslice",
        walks=100,
        outdir=args.outdir,
        label=f"direct_f0_tau_snr{args.snr}",
        resume=False,
    )

    # Analyze results
    post = result.posterior

    f0_median = float(np.median(post["f0"]))
    f0_std = float(np.std(post["f0"]))
    tau_median = float(np.median(post["tau"]))
    tau_std = float(np.std(post["tau"]))

    f0_bias = f0_median - f0_inj
    tau_bias = tau_median - tau_inj

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"f0:")
    print(f"  Injected: {f0_inj:.2f} Hz")
    print(f"  Recovered: {f0_median:.2f} +/- {f0_std:.2f} Hz")
    print(f"  Bias: {f0_bias:+.2f} Hz ({f0_bias/f0_inj*100:+.2f}%)")
    print()
    print(f"tau:")
    print(f"  Injected: {tau_inj*1000:.3f} ms")
    print(f"  Recovered: {tau_median*1000:.3f} +/- {tau_std*1000:.3f} ms")
    print(f"  Bias: {tau_bias*1000:+.3f} ms ({tau_bias/tau_inj*100:+.2f}%)")
    print()

    # Compute PIT for f0 and tau
    pit_f0 = float(np.mean(post["f0"] <= f0_inj))
    pit_tau = float(np.mean(post["tau"] <= tau_inj))

    print(f"PIT values:")
    print(f"  f0: {pit_f0:.3f}")
    print(f"  tau: {pit_tau:.3f}")
    print()

    # Check if injected value is within 90% CI
    f0_lower = np.percentile(post["f0"], 5)
    f0_upper = np.percentile(post["f0"], 95)
    tau_lower = np.percentile(post["tau"], 5)
    tau_upper = np.percentile(post["tau"], 95)

    f0_covered = f0_lower <= f0_inj <= f0_upper
    tau_covered = tau_lower <= tau_inj <= tau_upper

    print(f"90% CI coverage:")
    print(f"  f0: [{f0_lower:.2f}, {f0_upper:.2f}] Hz - {'PASS' if f0_covered else 'FAIL'}")
    print(f"  tau: [{tau_lower*1000:.3f}, {tau_upper*1000:.3f}] ms - {'PASS' if tau_covered else 'FAIL'}")
    print()

    # Diagnosis
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if abs(f0_bias) > 5:
        print("SIGNIFICANT f0 BIAS DETECTED with direct parameterization!")
        print("This suggests the bias is NOT in the QNM fitting formulas.")
        print("Possible causes:")
        print("  - Likelihood implementation issue")
        print("  - Window/tapering effects on frequency estimation")
        print("  - Sampler not converging properly")
    else:
        print("No significant f0 bias with direct parameterization.")
        print("If bias exists with (Mf, chi) parameterization,")
        print("then the issue is in the QNM fitting formulas.")

    # Save summary
    summary = {
        "injected": {
            "f0": f0_inj,
            "tau_ms": tau_inj * 1000,
            "t0_ms": t0_inj * 1000,
            "A": float(A_inj),
            "snr": snr_actual,
        },
        "recovered": {
            "f0_median": f0_median,
            "f0_std": f0_std,
            "tau_median_ms": tau_median * 1000,
            "tau_std_ms": tau_std * 1000,
        },
        "bias": {
            "f0_Hz": f0_bias,
            "f0_percent": f0_bias / f0_inj * 100,
            "tau_ms": tau_bias * 1000,
            "tau_percent": tau_bias / tau_inj * 100,
        },
        "pit": {
            "f0": pit_f0,
            "tau": pit_tau,
        },
        "coverage_90": {
            "f0": bool(f0_covered),
            "tau": bool(tau_covered),
        },
    }

    with open(os.path.join(args.outdir, "direct_f0_tau_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    print()
    print(f"Results saved to {args.outdir}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0", type=float, default=251.0, help="Injected f0 in Hz")
    parser.add_argument("--tau", type=float, default=4.0, help="Injected tau in ms")
    parser.add_argument("--t0", type=float, default=2.0, help="Injected t0 in ms")
    parser.add_argument("--snr", type=float, default=30.0, help="Target SNR")
    parser.add_argument("--duration", type=float, default=0.1, help="Segment duration (s)")
    parser.add_argument("--sample-rate", type=int, default=4096, help="Sample rate (Hz)")
    parser.add_argument("--fmin", type=float, default=150.0, help="Min frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=400.0, help="Max frequency (Hz)")
    parser.add_argument("--eps-start", type=float, default=0.01, help="Planck taper start")
    parser.add_argument("--eps-end", type=float, default=0.1, help="Planck taper end")
    parser.add_argument("--pad-factor", type=float, default=2.0, help="Zero-padding factor")
    parser.add_argument("--nlive", type=int, default=300, help="Number of live points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="out_direct_f0_tau", help="Output directory")

    args = parser.parse_args()
    run_direct_f0_tau_test(args)
