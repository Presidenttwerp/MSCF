#!/usr/bin/env python3
"""
Bayes factor analysis for RINGDOWN-GATED segments.

This version is designed for short post-merger segments (e.g., 200 ms)
that isolate the ringdown regime where the echo model applies.

Key differences from run_bayes_factor.py:
1. t0 prior is set relative to the segment start (first few ms of segment)
2. Lower nlive for faster runs on short segments
3. Handles metadata from fetch_ringdown_gated.py
"""
import argparse
import os
import json
import numpy as np
import bilby

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


def build_likelihood(event, ifos, outdir, model, fmin=30.0, fmax=1024.0):
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

    return GaussianFDLikelihood(t=t_ref, data_dict=data_dict, psd_dict=psd_dict, model=model, fmin=fmin, fmax=fmax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, required=True,
                   help="Event name (e.g., GW150914_gated_t10)")
    p.add_argument("--gps", type=float, required=True,
                   help="Merger GPS time")
    p.add_argument("--t-start", type=float, default=0.010,
                   help="Offset from merger where ringdown gate starts (seconds)")
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, default="out_gated",
                   help="Directory with gated data files")
    p.add_argument("--resultdir", type=str, default="out_gated")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--nlive", type=int, default=500,
                   help="Number of live points (default 500 for gated analysis)")
    p.add_argument("--fmin", type=float, default=150.0,
                   help="Minimum frequency for likelihood (default 150 Hz for gated to avoid leakage)")
    p.add_argument("--fmax", type=float, default=400.0,
                   help="Maximum frequency for likelihood (default 400 Hz for ringdown band)")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    label0 = f"{args.event}_H0"
    label1 = f"{args.event}_H1"

    print(f"Using frequency band: {args.fmin}-{args.fmax} Hz")

    # Build likelihoods with configurable frequency band
    like0 = build_likelihood(args.event, ifos, args.outdir, model="H0_ringdown", fmin=args.fmin, fmax=args.fmax)
    like1 = build_likelihood(args.event, ifos, args.outdir, model="H1_echo", fmin=args.fmin, fmax=args.fmax)

    # Get time grid from likelihood
    t_data = like0.t
    t_start_gps = t_data[0]
    t_end_gps = t_data[-1]
    segment_duration = t_end_gps - t_start_gps

    print(f"Gated segment: [{t_start_gps:.4f}, {t_end_gps:.4f}] ({segment_duration*1000:.1f} ms)")
    print(f"Merger GPS: {args.gps}")
    print(f"Gate starts at: merger + {args.t_start*1000:.1f} ms")

    # Priors
    # For gated data, t0 (ringdown start) should be near the start of the segment
    # The ringdown starts at t_merger, but our segment starts at t_merger + t_start
    # So t0 should be just before our segment starts
    t_merger = args.gps
    t0_center = t_merger  # ringdown starts at merger

    # Detector-frame mass (GW150914)
    z_gw150914 = 0.09
    Mf_source = 62.2
    Mf_source_err = 3.7
    Mf_detector = Mf_source * (1 + z_gw150914)
    Mf_detector_err = Mf_source_err * (1 + z_gw150914)

    pri0 = bilby.core.prior.PriorDict()
    pri0["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    pri0["Mf"] = bilby.core.prior.TruncatedGaussian(
        mu=Mf_detector, sigma=Mf_detector_err, minimum=55, maximum=85, name="Mf"
    )
    pri0["chi"] = bilby.core.prior.TruncatedGaussian(
        mu=0.68, sigma=0.05, minimum=0.0, maximum=0.99, name="chi"
    )
    pri0["phi"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi")
    # t0 prior: ringdown starts at merger, allow ±10 ms
    pri0["t0"] = bilby.core.prior.Uniform(t0_center - 0.01, t0_center + 0.01, "t0")

    # H1 adds echo params
    pri1 = pri0.copy()
    pri1["R0"] = bilby.core.prior.Uniform(0.0, 1.0, "R0")
    pri1["f_cut"] = bilby.core.prior.LogUniform(50, 2000, "f_cut")
    pri1["roll"] = bilby.core.prior.Uniform(1.0, 10.0, "roll")
    pri1["phi0"] = bilby.core.prior.Uniform(0.0, 2*np.pi, "phi0")

    if args.seed is not None:
        np.random.seed(args.seed)

    # Sampler settings - can use fewer live points for short segments
    sampler_kwargs = dict(
        sampler="dynesty",
        nlive=args.nlive,
        dlogz=0.1,
        bound="multi",
        sample="rslice",
        slices=10,
        check_point_plot=False,
        resume=False,
    )
    if args.seed is not None:
        sampler_kwargs["seed"] = args.seed

    print(f"\nRunning H0 (ringdown only)...")
    result0 = bilby.run_sampler(
        likelihood=like0, priors=pri0,
        outdir=args.resultdir, label=label0,
        **sampler_kwargs
    )
    print(f"H0 NaN/Inf: {like0.nan_inf_count}/{like0.eval_count}")

    print(f"\nRunning H1 (ringdown + echo)...")
    result1 = bilby.run_sampler(
        likelihood=like1, priors=pri1,
        outdir=args.resultdir, label=label1,
        **sampler_kwargs
    )
    print(f"H1 NaN/Inf: {like1.nan_inf_count}/{like1.eval_count}")

    logBF = result1.log_evidence - result0.log_evidence
    # Propagate evidence uncertainty: delta_BF = sqrt(err0^2 + err1^2)
    logBF_err = np.sqrt(result0.log_evidence_err**2 + result1.log_evidence_err**2)

    print(f"\n{'='*60}")
    print(f"GATED ANALYSIS RESULT")
    print(f"{'='*60}")
    print(f"Event: {args.event}")
    print(f"Gate: t_merger + {args.t_start*1000:.0f} ms, duration {segment_duration*1000:.0f} ms")
    print(f"ln BF_10 = {logBF:.3f} +/- {logBF_err:.3f}")
    print(f"log10 BF_10 = {logBF/np.log(10):.3f}")
    print(f"logZ_H0 = {result0.log_evidence:.2f} +/- {result0.log_evidence_err:.2f}")
    print(f"logZ_H1 = {result1.log_evidence:.2f} +/- {result1.log_evidence_err:.2f}")
    print(f"{'='*60}")

    # Save summary with evidence uncertainties and sampler diagnostics
    summary = {
        "event": args.event,
        "gps_merger": args.gps,
        "t_start_ms": args.t_start * 1000,
        "segment_duration_ms": segment_duration * 1000,
        "ln_BF": logBF,
        "ln_BF_err": logBF_err,
        "log10_BF": logBF / np.log(10),
        "log_evidence_H0": result0.log_evidence,
        "log_evidence_H1": result1.log_evidence,
        "logZerr_H0": result0.log_evidence_err,
        "logZerr_H1": result1.log_evidence_err,
        "ncall_H0": int(result0.num_likelihood_evaluations),
        "ncall_H1": int(result1.num_likelihood_evaluations),
        "sampling_efficiency_H0": float(result0.sampling_efficiency) if hasattr(result0, 'sampling_efficiency') else None,
        "sampling_efficiency_H1": float(result1.sampling_efficiency) if hasattr(result1, 'sampling_efficiency') else None,
    }
    with open(os.path.join(args.resultdir, f"{args.event}_bf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
