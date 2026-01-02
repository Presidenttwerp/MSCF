#!/usr/bin/env python3
"""
H0-only scan over early t_start values to find where ringdown is detectable.

This is a quick diagnostic - runs H0 only with fewer live points to check
if the posterior recovers sensible Mf, chi values.
"""
import argparse
import os
import sys
import json
import numpy as np
import bilby

# Add parent directory to path so we can import mscf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.likelihood import GaussianFDLikelihood


def load_npz(path):
    d = np.load(path)
    return d["t"], d["x"], d["f"], d["psd"]


def tukey_window(N, alpha=0.1):
    """Tukey window: flat in middle, tapered at edges."""
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    n = np.arange(N)
    w = np.ones(N)
    left = n < alpha * N / 2
    w[left] = 0.5 * (1 - np.cos(2 * np.pi * n[left] / (alpha * N)))
    right = n >= N * (1 - alpha / 2)
    w[right] = 0.5 * (1 - np.cos(2 * np.pi * (N - 1 - n[right]) / (alpha * N)))
    return w


def rfft_data(t, x, taper_alpha=0.1):
    """FFT with Tukey taper."""
    dt = t[1] - t[0]
    w = tukey_window(len(x), alpha=taper_alpha)
    X = np.fft.rfft(x * w) * dt
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, X


def build_likelihood(event, ifos, outdir, fmin=150.0, fmax=450.0):
    data_dict = {}
    psd_dict = {}
    t_ref = None

    for ifo in ifos:
        path = os.path.join(outdir, f"{event}_{ifo}_data_psd.npz")
        t, x, f_psd, psd = load_npz(path)

        f, d_f = rfft_data(t, x)
        psd_i = np.interp(f, f_psd, psd, left=psd[0], right=psd[-1])
        psd_floor = 1e-50
        psd_i = np.maximum(psd_i, psd_floor)

        data_dict[ifo] = (f, d_f)
        psd_dict[ifo] = (f, psd_i)

        if t_ref is None:
            t_ref = t
        else:
            if len(t_ref) != len(t) or np.max(np.abs(t_ref - t)) > 1e-9:
                raise ValueError("Time grids mismatch between IFOs")

    return GaussianFDLikelihood(t=t_ref, data_dict=data_dict, psd_dict=psd_dict,
                                model="H0_ringdown", fmin=fmin, fmax=fmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, required=True)
    p.add_argument("--gps", type=float, default=1126259462.4)
    p.add_argument("--t-start", type=float, required=True, help="t_start in seconds")
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--resultdir", type=str, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=450.0)
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    label = f"{args.event}_H0"

    print(f"H0-only scan for {args.event}")
    print(f"Frequency band: {args.fmin}-{args.fmax} Hz")

    like = build_likelihood(args.event, ifos, args.outdir, fmin=args.fmin, fmax=args.fmax)

    t_data = like.t
    segment_duration = t_data[-1] - t_data[0]
    print(f"Segment duration: {segment_duration*1000:.1f} ms")

    # Priors - detector-frame mass for GW150914
    z_gw150914 = 0.09
    Mf_source = 62.2
    Mf_source_err = 3.7
    Mf_detector = Mf_source * (1 + z_gw150914)
    Mf_detector_err = Mf_source_err * (1 + z_gw150914)

    priors = bilby.core.prior.PriorDict()
    priors["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    priors["Mf"] = bilby.core.prior.TruncatedGaussian(
        mu=Mf_detector, sigma=Mf_detector_err, minimum=55, maximum=85, name="Mf"
    )
    priors["chi"] = bilby.core.prior.TruncatedGaussian(
        mu=0.68, sigma=0.05, minimum=0.0, maximum=0.99, name="chi"
    )
    priors["phi"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi")
    # t0: ringdown starts at merger, allow ±10 ms
    t_merger = args.gps
    priors["t0"] = bilby.core.prior.Uniform(t_merger - 0.01, t_merger + 0.01, "t0")

    if args.seed is not None:
        np.random.seed(args.seed)

    sampler_kwargs = dict(
        sampler="dynesty",
        nlive=args.nlive,
        dlogz=0.5,  # Less strict for faster runs
        bound="multi",
        sample="rslice",
        slices=5,
        check_point_plot=False,
        resume=False,
    )
    if args.seed is not None:
        sampler_kwargs["seed"] = args.seed

    os.makedirs(args.resultdir, exist_ok=True)

    print(f"\nRunning H0 (ringdown only) with nlive={args.nlive}...")
    result = bilby.run_sampler(
        likelihood=like, priors=priors,
        outdir=args.resultdir, label=label,
        **sampler_kwargs
    )

    # Extract posteriors
    Mf_post = result.posterior["Mf"].values
    chi_post = result.posterior["chi"].values

    # Compute derived QNM frequency
    G = 6.67430e-11
    c = 299792458.0
    Msun = 1.98847e30
    f1, f2, f3 = 1.5251, -1.1568, 0.1292
    f0_post = (f1 + f2 * (1 - chi_post)**f3) / (2 * np.pi * Mf_post * Msun * G / c**3)

    summary = {
        "event": args.event,
        "t_start_ms": args.t_start * 1000,
        "segment_duration_ms": segment_duration * 1000,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "log_evidence": result.log_evidence,
        "log_evidence_err": result.log_evidence_err,
        "Mf_median": float(np.median(Mf_post)),
        "Mf_std": float(np.std(Mf_post)),
        "Mf_16": float(np.percentile(Mf_post, 16)),
        "Mf_84": float(np.percentile(Mf_post, 84)),
        "chi_median": float(np.median(chi_post)),
        "chi_std": float(np.std(chi_post)),
        "chi_16": float(np.percentile(chi_post, 16)),
        "chi_84": float(np.percentile(chi_post, 84)),
        "f0_median": float(np.median(f0_post)),
        "f0_std": float(np.std(f0_post)),
    }

    # Check for red flags
    summary["Mf_railing_upper"] = float(np.median(Mf_post)) > 82
    summary["Mf_railing_lower"] = float(np.median(Mf_post)) < 58
    summary["chi_railing_lower"] = float(np.median(chi_post)) < 0.1
    summary["chi_railing_upper"] = float(np.median(chi_post)) > 0.95

    # Check if recovered values are sensible
    Mf_expected = 67.8
    chi_expected = 0.68
    summary["Mf_within_2sigma"] = abs(np.median(Mf_post) - Mf_expected) < 2 * Mf_detector_err
    summary["chi_within_2sigma"] = abs(np.median(chi_post) - chi_expected) < 0.1

    with open(os.path.join(args.resultdir, f"{args.event}_h0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"H0 SCAN RESULT: t_start = {args.t_start*1000:.0f} ms")
    print(f"{'='*60}")
    print(f"Mf: {summary['Mf_median']:.1f} +/- {summary['Mf_std']:.1f} M_sun (expected ~67.8)")
    print(f"chi: {summary['chi_median']:.3f} +/- {summary['chi_std']:.3f} (expected ~0.68)")
    print(f"f0: {summary['f0_median']:.1f} +/- {summary['f0_std']:.1f} Hz (expected ~251)")
    print(f"log_evidence: {result.log_evidence:.1f}")
    if summary["Mf_railing_upper"]:
        print("*** RED FLAG: Mf railing at upper boundary! ***")
    if summary["Mf_railing_lower"]:
        print("*** RED FLAG: Mf railing at lower boundary! ***")
    if summary["Mf_within_2sigma"] and summary["chi_within_2sigma"]:
        print("*** GOOD: Parameters within expected range ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
