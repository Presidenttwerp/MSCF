#!/usr/bin/env python3
"""
Simulation-Based Calibration (SBC) for ringdown H0 inference.

Proper SBC protocol:
1. For each run i = 1..N:
   a) Draw θ_i from the PRIOR (same prior used in inference)
   b) Simulate data d_i from θ_i
   c) Run inference to get posterior p(θ|d_i)
   d) Compute PIT: u_i = Pr(θ <= θ_true | d_i)
2. If inference is calibrated, u ~ Uniform(0, 1)

Key requirement: priors must be FIXED (not centered on injected values).
"""
import argparse
import os
import sys
import json
import numpy as np
import bilby

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.waveforms import qnm_220_freq_tau, planck_taper
from mscf.likelihood import GaussianFDLikelihood


# Fixed priors for SBC - MUST match inference priors exactly
PRIOR_BOUNDS = {
    "A": (1e-24, 1e-19),      # log-uniform
    "Mf": (50.0, 85.0),       # uniform, solar masses
    "chi": (0.0, 0.99),       # uniform, dimensionless spin
    "phi": (0.0, 2 * np.pi),  # uniform, phase
    "t0": (0.0, 0.010),       # uniform, 0-10 ms
}


def generate_colored_noise(f, psd, rng):
    """Generate colored Gaussian noise from one-sided PSD."""
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


def ringdown_td_template(t, A, f0, tau, phi, t0):
    """Generate ringdown time series."""
    y = np.zeros_like(t)
    mask = t >= t0
    tt = t[mask] - t0
    y[mask] = A * np.exp(-tt / tau) * np.cos(2 * np.pi * f0 * tt + phi)
    return y


def draw_from_prior(rng):
    """Draw injection parameters from the prior."""
    # Draw from uniform/log-uniform priors
    log_A = rng.uniform(np.log10(PRIOR_BOUNDS["A"][0]), np.log10(PRIOR_BOUNDS["A"][1]))
    A = 10**log_A
    Mf = rng.uniform(*PRIOR_BOUNDS["Mf"])
    chi = rng.uniform(*PRIOR_BOUNDS["chi"])
    phi = rng.uniform(*PRIOR_BOUNDS["phi"])
    t0 = rng.uniform(*PRIOR_BOUNDS["t0"])

    return {"A": A, "Mf": Mf, "chi": chi, "phi": phi, "t0": t0}


def run_single_sbc_injection(
    run_id, base_seed, outdir,
    duration_sec=0.1, sample_rate=4096, fmin=150.0, fmax=400.0,
    epsilon_start=0.01, epsilon_end=0.1, pad_factor=2.0,
    nlive=300
):
    """Run a single SBC injection."""

    # Seed management: each run gets unique seeds for:
    # - Parameter draw (seed=base_seed + run_id)
    # - Noise generation (seed=base_seed + run_id + 10000)
    # - Sampler (seed=base_seed + run_id + 20000)
    param_rng = np.random.default_rng(base_seed + run_id)
    noise_rng = np.random.default_rng(base_seed + run_id + 10000)
    sampler_seed = base_seed + run_id + 20000

    run_outdir = os.path.join(outdir, f"run_{run_id:04d}")
    os.makedirs(run_outdir, exist_ok=True)

    # Step 1: Draw θ from prior
    params = draw_from_prior(param_rng)
    Mf_inj = params["Mf"]
    chi_inj = params["chi"]
    A_template = params["A"]
    phi_inj = params["phi"]
    t0_inj = params["t0"]

    # Compute QNM parameters
    f0_inj, tau_inj = qnm_220_freq_tau(Mf_inj, chi_inj)

    print(f"Run {run_id}: Mf={Mf_inj:.2f}, chi={chi_inj:.3f}, f0={f0_inj:.1f} Hz, t0={t0_inj*1000:.2f} ms")

    # Setup time/frequency grids
    fs = sample_rate
    dt = 1.0 / fs
    N_orig = int(duration_sec * fs)
    N_pad = int(N_orig * pad_factor)
    t = np.arange(N_orig) * dt
    t_pad = np.arange(N_pad) * dt
    f = np.fft.rfftfreq(N_pad, d=dt)
    df = f[1] - f[0]

    # Analytical PSD
    S0 = 1e-47
    f0_psd = 100.0
    psd = np.zeros_like(f)
    mask = f > 10
    psd[mask] = S0 * ((f0_psd / f[mask])**4 + 2 + (f[mask] / f0_psd)**2)
    psd[~mask] = S0 * 1e6
    psd = np.maximum(psd, 1e-50)

    # Generate signal at prior-drawn amplitude (NO rescaling for proper SBC!)
    # The amplitude from the prior is used directly - this is required for SBC validity
    A_inj = A_template  # Use the prior-drawn amplitude directly

    h_td = ringdown_td_template(t, A_inj, f0_inj, tau_inj, phi_inj, t0_inj)
    taper = planck_taper(N_orig, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
    h_tapered = h_td * taper
    h_padded = np.zeros(N_pad)
    h_padded[:N_orig] = h_tapered
    h_fft = np.fft.rfft(h_padded) * dt

    # Compute actual SNR (for diagnostics only - NOT used to rescale)
    band = (f >= fmin) & (f <= fmax)
    snr_sq = 4 * np.sum(np.abs(h_fft[band])**2 / psd[band]) * df
    snr_inj = np.sqrt(snr_sq) if snr_sq > 0 else 0.0

    # Generate colored noise
    noise_fft = generate_colored_noise(f, psd, noise_rng)

    # Data = signal + noise
    d_fft = h_fft + noise_fft

    # Save injection data
    np.savez(
        os.path.join(run_outdir, "injection_data.npz"),
        t=t_pad, f=f, d_fft=d_fft, h_fft=h_fft, psd=psd,
        Mf_inj=Mf_inj, chi_inj=chi_inj, f0_inj=f0_inj, tau_inj=tau_inj,
        A_inj=A_inj, t0_inj=t0_inj, phi_inj=phi_inj, snr_inj=snr_inj
    )

    # Build likelihood
    data_dict = {"H1": (f, d_fft)}
    psd_dict = {"H1": (f, psd)}

    like = GaussianFDLikelihood(
        t=t_pad, data_dict=data_dict, psd_dict=psd_dict,
        model="H0_ringdown", fmin=fmin, fmax=fmax,
        window="planck",
        planck_eps_start=epsilon_start,
        planck_eps_end=epsilon_end,
        N_window=N_orig
    )

    # Priors - MUST match PRIOR_BOUNDS exactly
    priors = bilby.core.prior.PriorDict()
    priors["A"] = bilby.core.prior.LogUniform(*PRIOR_BOUNDS["A"], "A")
    priors["Mf"] = bilby.core.prior.Uniform(*PRIOR_BOUNDS["Mf"], "Mf")
    priors["chi"] = bilby.core.prior.Uniform(*PRIOR_BOUNDS["chi"], "chi")
    priors["phi"] = bilby.core.prior.Uniform(*PRIOR_BOUNDS["phi"], "phi")
    priors["t0"] = bilby.core.prior.Uniform(*PRIOR_BOUNDS["t0"], "t0")

    # Run sampler
    np.random.seed(sampler_seed)

    sampler_kwargs = dict(
        sampler="dynesty",
        nlive=nlive,
        dlogz=0.5,
        bound="multi",
        sample="rslice",
        slices=5,
        check_point_plot=False,
        resume=False,
    )

    result = bilby.run_sampler(
        likelihood=like, priors=priors,
        outdir=run_outdir, label=f"sbc_run_{run_id:04d}",
        **sampler_kwargs
    )

    # Extract posteriors (these are equal-weight samples from bilby)
    Mf_post = result.posterior["Mf"].values
    chi_post = result.posterior["chi"].values
    A_post = result.posterior["A"].values
    t0_post = result.posterior["t0"].values

    # Compute f0 posterior
    f0_post = np.array([qnm_220_freq_tau(m, c)[0] for m, c in zip(Mf_post, chi_post)])

    # Compute PIT values (for equal-weight samples: PIT = fraction <= true)
    pit = {
        "Mf": float(np.mean(Mf_post <= Mf_inj)),
        "chi": float(np.mean(chi_post <= chi_inj)),
        "f0": float(np.mean(f0_post <= f0_inj)),
        "A": float(np.mean(A_post <= A_inj)),
        "t0": float(np.mean(t0_post <= t0_inj)),
    }

    # Compute coverage at multiple CI levels
    ci_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    coverage = {}
    for param, samples, true_val in [
        ("Mf", Mf_post, Mf_inj),
        ("chi", chi_post, chi_inj),
        ("f0", f0_post, f0_inj),
    ]:
        coverage[param] = {}
        for ci in ci_levels:
            alpha = 1 - ci
            lo = np.percentile(samples, alpha/2 * 100)
            hi = np.percentile(samples, (1 - alpha/2) * 100)
            coverage[param][str(ci)] = bool(lo <= true_val <= hi)

    # Summary
    summary = {
        "run_id": run_id,
        "injected": {
            "Mf": Mf_inj,
            "chi": chi_inj,
            "f0": f0_inj,
            "tau_ms": tau_inj * 1000,
            "A": A_inj,
            "t0_ms": t0_inj * 1000,
            "phi": phi_inj,
            "snr": snr_inj,
        },
        "recovered": {
            "Mf_median": float(np.median(Mf_post)),
            "Mf_std": float(np.std(Mf_post)),
            "chi_median": float(np.median(chi_post)),
            "chi_std": float(np.std(chi_post)),
            "f0_median": float(np.median(f0_post)),
            "f0_std": float(np.std(f0_post)),
        },
        "pit": pit,
        "coverage": coverage,
        "diagnostics": {
            "log_evidence": float(result.log_evidence),
            "log_evidence_err": float(result.log_evidence_err),
            "n_samples": len(Mf_post),
        }
    }

    with open(os.path.join(run_outdir, "sbc_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def aggregate_sbc_results(outdir, n_runs):
    """Aggregate SBC results and compute calibration diagnostics."""

    pit_values = {"Mf": [], "chi": [], "f0": [], "A": [], "t0": []}
    coverage_counts = {}
    ci_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    for ci in ci_levels:
        coverage_counts[str(ci)] = {"Mf": 0, "chi": 0, "f0": 0}

    n_valid = 0
    for run_id in range(n_runs):
        summary_path = os.path.join(outdir, f"run_{run_id:04d}", "sbc_summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as fp:
            summary = json.load(fp)

        for param in pit_values:
            if param in summary["pit"]:
                pit_values[param].append(summary["pit"][param])

        for param in ["Mf", "chi", "f0"]:
            for ci in ci_levels:
                if summary["coverage"][param].get(str(ci), False):
                    coverage_counts[str(ci)][param] += 1

        n_valid += 1

    if n_valid == 0:
        print("No valid runs found!")
        return None

    # Compute PIT statistics
    pit_stats = {}
    for param in pit_values:
        if len(pit_values[param]) > 0:
            arr = np.array(pit_values[param])
            pit_stats[param] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "expected_mean": 0.5,
                "expected_std": 1.0 / np.sqrt(12),  # std of Uniform(0,1)
            }

    # Compute coverage fractions
    coverage_fractions = {}
    for ci in ci_levels:
        coverage_fractions[str(ci)] = {}
        for param in ["Mf", "chi", "f0"]:
            coverage_fractions[str(ci)][param] = coverage_counts[str(ci)][param] / n_valid

    # Aggregate summary
    aggregate = {
        "n_runs": n_valid,
        "pit_stats": pit_stats,
        "coverage_fractions": coverage_fractions,
        "expected_coverage": {str(ci): ci for ci in ci_levels},
    }

    with open(os.path.join(outdir, "sbc_aggregate.json"), "w") as fp:
        json.dump(aggregate, fp, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print(f"SBC RESULTS ({n_valid} runs)")
    print("=" * 60)
    print()
    print("PIT Statistics (expected: mean=0.5, std=0.289):")
    print("-" * 40)
    for param in ["Mf", "chi", "f0"]:
        if param in pit_stats:
            ps = pit_stats[param]
            print(f"  {param:5s}: mean={ps['mean']:.3f}, std={ps['std']:.3f}")
    print()
    print("Coverage Fractions (expected = CI level):")
    print("-" * 40)
    for ci in [0.5, 0.9, 0.95]:
        print(f"  {ci*100:.0f}% CI:")
        for param in ["Mf", "chi", "f0"]:
            frac = coverage_fractions[str(ci)][param]
            print(f"    {param:5s}: {frac:.2f} (expected {ci:.2f})")

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Run SBC for ringdown inference")
    parser.add_argument("--n-runs", type=int, default=200, help="Number of SBC runs")
    parser.add_argument("--start-run", type=int, default=0, help="Starting run index")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="out_sbc", help="Output directory")
    parser.add_argument("--nlive", type=int, default=300, help="Number of live points")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results")
    parser.add_argument("--fmin", type=float, default=150.0)
    parser.add_argument("--fmax", type=float, default=400.0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.aggregate_only:
        aggregate_sbc_results(args.outdir, args.n_runs)
        return

    print(f"Running SBC with {args.n_runs} injections (amplitude drawn from prior)")
    print(f"Output directory: {args.outdir}")
    print(f"Priors: A~LogU(1e-24,1e-19), Mf~U(50,85), chi~U(0,0.99)")
    print()

    for run_id in range(args.start_run, args.start_run + args.n_runs):
        try:
            summary = run_single_sbc_injection(
                run_id=run_id,
                base_seed=args.base_seed,
                outdir=args.outdir,
                nlive=args.nlive,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            print(f"  PIT: Mf={summary['pit']['Mf']:.3f}, chi={summary['pit']['chi']:.3f}, f0={summary['pit']['f0']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Aggregate results
    aggregate_sbc_results(args.outdir, args.start_run + args.n_runs)


if __name__ == "__main__":
    main()
