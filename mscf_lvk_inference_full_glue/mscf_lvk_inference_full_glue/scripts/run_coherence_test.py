#!/usr/bin/env python3
"""
Coherence discriminator test: H1_coh vs H1_incoh

This implements the key diagnostic that stops single-detector noise fitting
from becoming a fake network detection.

Computes:
    lnBF_coh/incoh = logZ(H1_coh) - logZ(H1_incoh)

Expected behavior:
- Noise / time-slides: incoherent wins or neutral → lnBF_coh/incoh ≤ 0
- Real astrophysical echoes: coherent wins → lnBF_coh/incoh > 0

Models:
- H1_coh (H1_mscf_train): shared R1, phi_reflect across IFOs
- H1_incoh (H1_mscf_train_incoh): separate R1, phi_reflect per IFO
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
from datetime import datetime
import bilby

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.likelihood import GaussianFDLikelihood


# Configuration
GW150914_GPS = 1126259462.4
FMIN = 150.0
FMAX = 400.0


def tukey_window(N, alpha=0.1):
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
    dt = t[1] - t[0]
    w = tukey_window(len(x), alpha=taper_alpha)
    X = np.fft.rfft(x * w) * dt
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, X


def apply_time_slide_fd(f, data_fd, time_slide):
    phase = np.exp(-2j * np.pi * f * time_slide)
    return data_fd * phase


def load_data(data_base_dir, gps_offset, apply_slide=0.0):
    """Load data for both IFOs."""
    offset_str = f"gps{gps_offset:+d}s"
    pattern = os.path.join(data_base_dir, f"*_{offset_str}_*")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No data directory found for GPS offset {gps_offset}")

    data_dir = matches[0]

    data_dict = {}
    psd_dict = {}
    t_ref = None
    gps_center = None

    for ifo in ["H1", "L1"]:
        pattern = os.path.join(data_dir, f"*_{ifo}_data_psd.npz")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No {ifo} data file found in {data_dir}")

        npz = np.load(files[0])
        t_arr = npz["t"]
        x = npz["x"]
        f_psd = npz["f"]
        psd = npz["psd"]

        f, d_f = rfft_data(t_arr, x)

        # Apply time-slide to L1
        if ifo == "L1" and apply_slide != 0.0:
            d_f = apply_time_slide_fd(f, d_f, apply_slide)

        data_dict[ifo] = (f, d_f)

        psd_i = np.interp(f, f_psd, psd, left=psd[0], right=psd[-1])
        psd_floor = 1e-50
        psd_i = np.maximum(psd_i, psd_floor)
        psd_dict[ifo] = (f, psd_i)

        if t_ref is None:
            t_ref = t_arr
            gps_center = np.median(t_arr)

    return t_ref, data_dict, psd_dict, gps_center


def run_coherence_test(t, data_dict, psd_dict, gps_center, nlive=500, seed=42,
                       fmin=FMIN, fmax=FMAX, outdir=None, label_prefix="test"):
    """
    Run coherence test: compare H1_coh vs H1_incoh.

    Returns:
        dict with logZ values and lnBF_coh/incoh
    """

    # Priors common to both models
    z_gw150914 = 0.09
    Mf_source = 62.2
    Mf_source_err = 3.7
    Mf_detector = Mf_source * (1 + z_gw150914)
    Mf_detector_err = Mf_source_err * (1 + z_gw150914)

    # Base priors (ringdown)
    base_priors = bilby.core.prior.PriorDict()
    base_priors["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    base_priors["Mf"] = bilby.core.prior.TruncatedGaussian(
        mu=Mf_detector, sigma=Mf_detector_err, minimum=55, maximum=85, name="Mf"
    )
    base_priors["chi"] = bilby.core.prior.TruncatedGaussian(
        mu=0.68, sigma=0.05, minimum=0.0, maximum=0.99, name="chi"
    )
    base_priors["phi"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi")
    base_priors["t0"] = bilby.core.prior.Uniform(gps_center - 0.01, gps_center + 0.01, "t0")

    # Coherent model priors: shared R1, phi_reflect
    pri_coh = base_priors.copy()
    pri_coh["R1"] = bilby.core.prior.Uniform(0.0, 0.5, "R1")
    pri_coh["phi_reflect"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi_reflect")

    # Incoherent model priors: separate R1, phi_reflect per IFO
    pri_incoh = base_priors.copy()
    for ifo in ["H1", "L1"]:
        pri_incoh[f"R1_{ifo}"] = bilby.core.prior.Uniform(0.0, 0.5, f"R1_{ifo}")
        pri_incoh[f"phi_reflect_{ifo}"] = bilby.core.prior.Uniform(0, 2*np.pi, f"phi_reflect_{ifo}")

    sampler_kwargs = dict(
        sampler="dynesty",
        nlive=nlive,
        dlogz=0.1,
        bound="multi",
        sample="rslice",
        slices=10,
        check_point_plot=False,
        resume=False,
        verbose=False,
        seed=seed,
    )

    # Build likelihoods
    like_coh = GaussianFDLikelihood(t, data_dict, psd_dict, model="H1_mscf_train",
                                     fmin=fmin, fmax=fmax)
    like_incoh = GaussianFDLikelihood(t, data_dict, psd_dict, model="H1_mscf_train_incoh",
                                       fmin=fmin, fmax=fmax)

    # Create output directory
    if outdir is None:
        import tempfile
        outdir = tempfile.mkdtemp()

    # Run coherent model
    print("  Running COHERENT model (shared R1, phi_reflect)...")
    result_coh = bilby.run_sampler(
        likelihood=like_coh, priors=pri_coh.copy(),
        outdir=outdir, label=f"{label_prefix}_coh",
        **sampler_kwargs
    )
    print(f"    logZ_coh = {result_coh.log_evidence:.2f} ± {result_coh.log_evidence_err:.2f}")

    # Run incoherent model
    print("  Running INCOHERENT model (per-IFO R1, phi_reflect)...")
    result_incoh = bilby.run_sampler(
        likelihood=like_incoh, priors=pri_incoh.copy(),
        outdir=outdir, label=f"{label_prefix}_incoh",
        **{**sampler_kwargs, "seed": seed + 1}
    )
    print(f"    logZ_incoh = {result_incoh.log_evidence:.2f} ± {result_incoh.log_evidence_err:.2f}")

    # Calculate coherence Bayes factor
    ln_BF_coh_incoh = result_coh.log_evidence - result_incoh.log_evidence
    ln_BF_err = np.sqrt(result_coh.log_evidence_err**2 + result_incoh.log_evidence_err**2)

    return {
        "logZ_coh": float(result_coh.log_evidence),
        "logZ_coh_err": float(result_coh.log_evidence_err),
        "logZ_incoh": float(result_incoh.log_evidence),
        "logZ_incoh_err": float(result_incoh.log_evidence_err),
        "ln_BF_coh_incoh": float(ln_BF_coh_incoh),
        "ln_BF_coh_incoh_err": float(ln_BF_err),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Coherence discriminator test: H1_coh vs H1_incoh"
    )
    parser.add_argument("--data-dir",
                       default="out_gw150914_offsource/data",
                       help="Base directory containing off-source data")
    parser.add_argument("--outdir",
                       default="out_coherence_test",
                       help="Output directory")
    parser.add_argument("--gps-offset", type=int, default=10,
                       help="GPS offset to analyze")
    parser.add_argument("--time-slide", type=float, default=0.0,
                       help="Time slide to apply to L1")
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--fmin", type=float, default=FMIN)
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-multiple", action="store_true",
                       help="Run on multiple time-slides")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("COHERENCE DISCRIMINATOR TEST")
    print("=" * 70)
    print(f"\nGPS offset: {args.gps_offset:+d}s")
    print(f"Time-slide: {args.time_slide}s")
    print(f"Frequency band: {args.fmin}-{args.fmax} Hz")
    print()
    print("Models:")
    print("  H1_coh: shared (R1, phi_reflect) across IFOs")
    print("  H1_incoh: separate (R1_H1, phi_H1), (R1_L1, phi_L1) per IFO")
    print()
    print("Expected:")
    print("  Noise/time-slides: lnBF_coh/incoh ≤ 0 (incoherent wins)")
    print("  Real signal: lnBF_coh/incoh > 0 (coherent wins)")
    print()

    if args.run_multiple:
        # Run on multiple time-slides
        time_slides = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
        results = []

        for slide in time_slides:
            print(f"\n{'='*60}")
            print(f"Time-slide: {slide}s")
            print(f"{'='*60}")

            t, data_dict, psd_dict, gps_center = load_data(
                args.data_dir, args.gps_offset, apply_slide=slide
            )

            result = run_coherence_test(
                t, data_dict, psd_dict, gps_center,
                nlive=args.nlive, seed=args.seed + int(slide * 10),
                fmin=args.fmin, fmax=args.fmax,
                outdir=args.outdir,
                label_prefix=f"offset{args.gps_offset:+d}_slide{slide}s"
            )

            result["time_slide"] = slide
            results.append(result)

            print(f"\n  lnBF_coh/incoh = {result['ln_BF_coh_incoh']:.2f} ± {result['ln_BF_coh_incoh_err']:.2f}")

            if result['ln_BF_coh_incoh'] <= 0:
                print("  => PASS: Incoherent wins (as expected for noise)")
            else:
                print("  => WARNING: Coherent wins on noise - investigate!")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Time-slide':<15} {'lnBF_coh/incoh':<20} {'Status'}")
        print("-" * 50)

        for r in results:
            bf = r['ln_BF_coh_incoh']
            status = "PASS" if bf <= 0 else "FAIL"
            print(f"{r['time_slide']:.1f}s{'':<10} {bf:+.2f} ± {r['ln_BF_coh_incoh_err']:.2f}{'':<5} {status}")

        # Save all results
        summary_path = os.path.join(args.outdir, f"coherence_summary_offset{args.gps_offset:+d}.json")
        with open(summary_path, "w") as f:
            json.dump({
                "gps_offset": args.gps_offset,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {summary_path}")

    else:
        # Single run
        t, data_dict, psd_dict, gps_center = load_data(
            args.data_dir, args.gps_offset, apply_slide=args.time_slide
        )

        result = run_coherence_test(
            t, data_dict, psd_dict, gps_center,
            nlive=args.nlive, seed=args.seed,
            fmin=args.fmin, fmax=args.fmax,
            outdir=args.outdir,
            label_prefix=f"offset{args.gps_offset:+d}_slide{args.time_slide}s"
        )

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"\nlogZ_coh   = {result['logZ_coh']:.2f} ± {result['logZ_coh_err']:.2f}")
        print(f"logZ_incoh = {result['logZ_incoh']:.2f} ± {result['logZ_incoh_err']:.2f}")
        print(f"\nlnBF_coh/incoh = {result['ln_BF_coh_incoh']:.2f} ± {result['ln_BF_coh_incoh_err']:.2f}")

        if result['ln_BF_coh_incoh'] <= 0:
            print("\n=> PASS: Incoherent model wins (expected for noise)")
        else:
            print("\n=> WARNING: Coherent model wins on noise - potential false positive!")

        # Save result
        result["gps_offset"] = args.gps_offset
        result["time_slide"] = args.time_slide
        result["timestamp"] = datetime.now().isoformat()

        summary_path = os.path.join(
            args.outdir,
            f"coherence_offset{args.gps_offset:+d}_slide{args.time_slide}s.json"
        )
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
