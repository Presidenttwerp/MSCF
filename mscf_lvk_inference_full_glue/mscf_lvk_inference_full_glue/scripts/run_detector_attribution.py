#!/usr/bin/env python3
"""
Per-detector attribution for false positive time-slide cases.

For each "bad" time-slide case (ln_BF > 10), runs:
1. H1-only analysis
2. L1-only analysis

This identifies which detector is driving the false positive.
If one detector dominates, it means the model can fit single-detector
noise features with the extra degrees of freedom.
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


def extract_gps_offset_from_path(data_dir):
    """Extract GPS offset from directory path like 'GW150914_offsource_gps-180s_joint'."""
    import re
    match = re.search(r'gps([+-]?\d+)s', data_dir)
    if match:
        return int(match.group(1))
    return None


def load_single_ifo_data(data_dir, ifo, apply_slide=0.0):
    """Load data for a single IFO from a direct data directory path."""
    # Find data file
    pattern = os.path.join(data_dir, f"*_{ifo}_data_psd.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No {ifo} data file found in {data_dir}")

    npz = np.load(files[0])
    t_arr = npz["t"]
    x = npz["x"]
    f_psd = npz["f"]
    psd = npz["psd"]

    # Compute FFT
    f, d_f = rfft_data(t_arr, x)

    # Apply time-slide if requested
    if apply_slide != 0.0:
        d_f = apply_time_slide_fd(f, d_f, apply_slide)

    # Interpolate PSD
    psd_i = np.interp(f, f_psd, psd, left=psd[0], right=psd[-1])
    psd_floor = 1e-50
    psd_i = np.maximum(psd_i, psd_floor)

    data_dict = {ifo: (f, d_f)}
    psd_dict = {ifo: (f, psd_i)}
    gps_center = np.median(t_arr)

    return t_arr, data_dict, psd_dict, gps_center


def run_single_ifo_test(t, data_dict, psd_dict, gps_center, nlive=500, seed=42,
                        fmin=FMIN, fmax=FMAX, label_prefix="test"):
    """Run H0 vs H1_mscf_train on a single IFO."""

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
    pri0["t0"] = bilby.core.prior.Uniform(gps_center - 0.01, gps_center + 0.01, "t0")

    pri1 = pri0.copy()
    pri1["R1"] = bilby.core.prior.Uniform(0.0, 0.5, "R1")
    pri1["phi_reflect"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi_reflect")

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

    like_h0 = GaussianFDLikelihood(t, data_dict, psd_dict, model="H0_ringdown",
                                    fmin=fmin, fmax=fmax)
    like_h1 = GaussianFDLikelihood(t, data_dict, psd_dict, model="H1_mscf_train",
                                    fmin=fmin, fmax=fmax)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result_h0 = bilby.run_sampler(
            likelihood=like_h0, priors=pri0.copy(),
            outdir=tmpdir, label=f"{label_prefix}_h0",
            **sampler_kwargs
        )

        result_h1 = bilby.run_sampler(
            likelihood=like_h1, priors=pri1.copy(),
            outdir=tmpdir, label=f"{label_prefix}_h1",
            **sampler_kwargs
        )

    ln_BF = result_h1.log_evidence - result_h0.log_evidence
    ln_BF_err = np.sqrt(result_h1.log_evidence_err**2 + result_h0.log_evidence_err**2)

    return {
        "logZ_H0": float(result_h0.log_evidence),
        "logZ_H0_err": float(result_h0.log_evidence_err),
        "logZ_H1": float(result_h1.log_evidence),
        "logZ_H1_err": float(result_h1.log_evidence_err),
        "ln_BF": float(ln_BF),
        "ln_BF_err": float(ln_BF_err),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-detector attribution for false positive cases"
    )
    parser.add_argument("--data-dir", required=True,
                       help="Direct path to data directory (e.g., out_gw150914_offsource/data/GW150914_offsource_gps-180s_joint)")
    parser.add_argument("--outdir",
                       default="out_detector_attribution",
                       help="Output directory")
    parser.add_argument("--time-slide", type=float, default=0.2,
                       help="Time slide that produced false positive")
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--fmin", type=float, default=FMIN)
    parser.add_argument("--fmax", type=float, default=FMAX)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Extract GPS offset from data directory path
    gps_offset = extract_gps_offset_from_path(args.data_dir)
    if gps_offset is None:
        raise ValueError(f"Could not extract GPS offset from path: {args.data_dir}")

    print("=" * 70)
    print("PER-DETECTOR ATTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"\nData directory: {args.data_dir}")
    print(f"GPS offset: {gps_offset:+d}s (auto-detected)")
    print(f"Time-slide: {args.time_slide}s")
    print(f"Frequency band: {args.fmin}-{args.fmax} Hz")
    print()

    results = {
        "data_dir": args.data_dir,
        "gps_offset": gps_offset,
        "time_slide": args.time_slide,
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "nlive": args.nlive,
            "fmin": args.fmin,
            "fmax": args.fmax,
        }
    }

    # First, run the joint analysis for reference
    print("Running JOINT (H1+L1) analysis with time-slide...")
    t_h1, data_h1, psd_h1, gps_h1 = load_single_ifo_data(
        args.data_dir, "H1", apply_slide=0.0
    )
    t_l1, data_l1, psd_l1, gps_l1 = load_single_ifo_data(
        args.data_dir, "L1", apply_slide=args.time_slide
    )

    # Combine for joint analysis
    data_joint = {**data_h1, **data_l1}
    psd_joint = {**psd_h1, **psd_l1}

    joint_result = run_single_ifo_test(
        t_h1, data_joint, psd_joint, gps_h1,
        nlive=args.nlive, seed=args.seed,
        fmin=args.fmin, fmax=args.fmax,
        label_prefix=f"joint_offset{gps_offset:+d}_slide{args.time_slide}s"
    )
    results["joint"] = joint_result
    print(f"  Joint ln_BF = {joint_result['ln_BF']:.2f} ± {joint_result['ln_BF_err']:.2f}")

    # H1-only analysis (no time-slide on H1)
    print("\nRunning H1-ONLY analysis...")
    t_h1_only, data_h1_only, psd_h1_only, gps_h1_only = load_single_ifo_data(
        args.data_dir, "H1", apply_slide=0.0
    )

    h1_result = run_single_ifo_test(
        t_h1_only, data_h1_only, psd_h1_only, gps_h1_only,
        nlive=args.nlive, seed=args.seed + 1,
        fmin=args.fmin, fmax=args.fmax,
        label_prefix=f"H1only_offset{gps_offset:+d}"
    )
    results["H1_only"] = h1_result
    print(f"  H1-only ln_BF = {h1_result['ln_BF']:.2f} ± {h1_result['ln_BF_err']:.2f}")

    # L1-only analysis (with time-slide)
    print("\nRunning L1-ONLY analysis (with time-slide)...")
    t_l1_only, data_l1_only, psd_l1_only, gps_l1_only = load_single_ifo_data(
        args.data_dir, "L1", apply_slide=args.time_slide
    )

    l1_result = run_single_ifo_test(
        t_l1_only, data_l1_only, psd_l1_only, gps_l1_only,
        nlive=args.nlive, seed=args.seed + 2,
        fmin=args.fmin, fmax=args.fmax,
        label_prefix=f"L1only_offset{gps_offset:+d}_slide{args.time_slide}s"
    )
    results["L1_only"] = l1_result
    print(f"  L1-only ln_BF = {l1_result['ln_BF']:.2f} ± {l1_result['ln_BF_err']:.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 70)

    joint_bf = joint_result['ln_BF']
    h1_bf = h1_result['ln_BF']
    l1_bf = l1_result['ln_BF']

    print(f"\nJoint ln_BF:   {joint_bf:.2f}")
    print(f"H1-only ln_BF: {h1_bf:.2f}")
    print(f"L1-only ln_BF: {l1_bf:.2f}")
    print(f"Sum of singles: {h1_bf + l1_bf:.2f}")

    # Determine which detector dominates
    if abs(h1_bf) > abs(l1_bf) * 2:
        dominant = "H1"
        print(f"\n=> H1 dominates the Bayes factor")
    elif abs(l1_bf) > abs(h1_bf) * 2:
        dominant = "L1"
        print(f"\n=> L1 dominates the Bayes factor")
    else:
        dominant = "Neither"
        print(f"\n=> Neither detector clearly dominates")

    results["analysis"] = {
        "dominant_detector": dominant,
        "sum_of_singles": float(h1_bf + l1_bf),
        "h1_contribution_frac": float(h1_bf / (abs(h1_bf) + abs(l1_bf))) if (abs(h1_bf) + abs(l1_bf)) > 0 else 0,
        "l1_contribution_frac": float(l1_bf / (abs(h1_bf) + abs(l1_bf))) if (abs(h1_bf) + abs(l1_bf)) > 0 else 0,
    }

    # Interpretation
    if joint_bf > 10:
        if h1_bf > 5 or l1_bf > 5:
            print("\nINTERPRETATION:")
            print("  The false positive is driven by single-detector noise fitting.")
            print("  This means the model can fit noise features with the echo parameters.")
            print("  => Need coherence discriminator to reject incoherent fits.")
        else:
            print("\nINTERPRETATION:")
            print("  Neither single detector shows strong preference for H1.")
            print("  The joint BF may be due to cross-detector coincidence.")

    # Save results
    summary_path = os.path.join(
        args.outdir,
        f"attribution_offset{gps_offset:+d}_slide{args.time_slide}s.json"
    )
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
