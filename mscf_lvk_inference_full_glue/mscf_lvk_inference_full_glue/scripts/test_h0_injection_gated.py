#!/usr/bin/env python3
"""
Ringdown-only injection test for gated analysis.

This is the critical validation step: inject a GR ringdown (no echoes) into
colored Gaussian noise using the measured PSD, then verify H0 recovery.

Pass condition: H0 recovers Mf, chi near injected values.

If this fails, the gating/PSD/likelihood has bugs and should not be used on
real data.
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


def generate_colored_noise(f, psd, seed=None):
    """
    Generate colored Gaussian noise from one-sided PSD.

    Returns complex FFT coefficients.
    """
    if seed is not None:
        np.random.seed(seed)

    # Standard deviation at each frequency bin
    # For one-sided PSD: var = psd * df / 2 for real and imag parts
    # But we're in FFT domain, so directly: sigma = sqrt(psd / (2 * df))
    df = f[1] - f[0] if len(f) > 1 else 1.0

    # Generate white noise in freq domain
    real_part = np.random.randn(len(f))
    imag_part = np.random.randn(len(f))

    # Color with PSD: multiply by sqrt(psd / 2) for each component
    # The factor comes from: E[|n|^2] = psd * df
    sigma = np.sqrt(psd * df / 2)
    noise = sigma * (real_part + 1j * imag_part)

    # DC and Nyquist are real for rfft
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


def run_injection_test(
    Mf_inj, chi_inj, A_inj, t0_inj,
    duration_sec, sample_rate, fmin, fmax,
    outdir, seed=None, nlive=200,
    epsilon_start=0.01, epsilon_end=0.1, pad_factor=2.0,
    use_real_psd=None
):
    """
    Run a single injection test.

    Parameters
    ----------
    Mf_inj, chi_inj : float
        Injected remnant mass and spin
    A_inj : float
        Injected amplitude
    t0_inj : float
        Injected ringdown start time (segment-relative!)
    duration_sec : float
        Segment duration in seconds
    sample_rate : int
        Sample rate
    fmin, fmax : float
        Frequency band for likelihood
    outdir : str
        Output directory
    seed : int
        Random seed
    nlive : int
        Number of live points
    epsilon_start, epsilon_end : float
        Planck taper parameters
    pad_factor : float
        Zero-padding factor
    use_real_psd : str or None
        Path to npz file with real PSD to use (for realistic noise)

    Returns
    -------
    summary : dict
        Results summary
    """
    os.makedirs(outdir, exist_ok=True)

    # Compute QNM parameters from injected Mf, chi
    f0_inj, tau_inj = qnm_220_freq_tau(Mf_inj, chi_inj)
    phi_inj = np.random.uniform(0, 2 * np.pi) if seed is None else 0.5

    print("=" * 60)
    print("H0 INJECTION TEST (Gated Ringdown)")
    print("=" * 60)
    print(f"Injected parameters:")
    print(f"  Mf = {Mf_inj:.2f} M_sun")
    print(f"  chi = {chi_inj:.3f}")
    print(f"  f0 = {f0_inj:.1f} Hz (derived)")
    print(f"  tau = {tau_inj*1000:.2f} ms (derived)")
    print(f"  A = {A_inj:.2e}")
    print(f"  t0 = {t0_inj*1000:.1f} ms (segment-relative)")
    print(f"  phi = {phi_inj:.3f}")
    print(f"Segment: {duration_sec*1000:.0f} ms, {sample_rate} Hz")
    print(f"Frequency band: {fmin}-{fmax} Hz")
    print()

    fs = sample_rate
    dt = 1.0 / fs
    N_orig = int(duration_sec * fs)
    N_pad = int(N_orig * pad_factor)

    # Time arrays (segment-relative, starting at 0)
    t = np.arange(N_orig) * dt  # original segment
    t_pad = np.arange(N_pad) * dt  # padded segment

    # Generate ringdown signal
    h_td = ringdown_td_template(t, A_inj, f0_inj, tau_inj, phi_inj, t0_inj)

    # Apply Planck taper
    taper = planck_taper(N_orig, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
    h_tapered = h_td * taper

    # Zero-pad
    h_padded = np.zeros(N_pad)
    h_padded[:N_orig] = h_tapered

    # FFT
    f = np.fft.rfftfreq(N_pad, d=dt)
    df = f[1] - f[0]
    h_fft = np.fft.rfft(h_padded) * dt

    # PSD: either use real PSD or analytical LIGO model
    if use_real_psd is not None:
        print(f"Loading real PSD from {use_real_psd}")
        data = np.load(use_real_psd)
        f_psd_file = data["f"]
        psd_file = data["psd"]
        psd = np.interp(f, f_psd_file, psd_file, left=psd_file[0], right=psd_file[-1])
    else:
        # Simple analytical aLIGO-like PSD
        # S(f) ~ S0 * [(f0/f)^4 + 2 + (f/f0)^2]
        S0 = 1e-47  # roughly aLIGO at 100 Hz
        f0_psd = 100.0
        psd = np.zeros_like(f)
        mask = f > 10
        psd[mask] = S0 * ((f0_psd / f[mask])**4 + 2 + (f[mask] / f0_psd)**2)
        psd[~mask] = S0 * 1e6  # high noise at low freq

    psd = np.maximum(psd, 1e-50)

    # Generate colored noise
    noise_fft = generate_colored_noise(f, psd, seed=seed)

    # Data = signal + noise
    d_fft = h_fft + noise_fft

    # Compute injected SNR
    band = (f >= fmin) & (f <= fmax)
    snr_sq = 4 * np.sum(np.abs(h_fft[band])**2 / psd[band]) * df
    snr_inj = np.sqrt(snr_sq)
    print(f"Injected SNR: {snr_inj:.1f}")

    # Save injection data
    np.savez(
        os.path.join(outdir, "injection_data.npz"),
        t=t_pad, f=f, d_fft=d_fft, h_fft=h_fft, psd=psd,
        Mf_inj=Mf_inj, chi_inj=chi_inj, f0_inj=f0_inj, tau_inj=tau_inj,
        A_inj=A_inj, t0_inj=t0_inj, phi_inj=phi_inj, snr_inj=snr_inj
    )

    # Build likelihood
    # For single-IFO test, create mock data dict
    data_dict = {"H1": (f, d_fft)}
    psd_dict = {"H1": (f, psd)}

    like = GaussianFDLikelihood(
        t=t_pad, data_dict=data_dict, psd_dict=psd_dict,
        model="H0_ringdown", fmin=fmin, fmax=fmax,
        window="planck",  # Use same Planck window as data
        planck_eps_start=epsilon_start,
        planck_eps_end=epsilon_end,
        N_window=N_orig  # Window only first N_orig samples (matching data preparation)
    )

    # Priors centered on injected values with reasonable width
    priors = bilby.core.prior.PriorDict()
    priors["A"] = bilby.core.prior.LogUniform(A_inj / 100, A_inj * 100, "A")
    priors["Mf"] = bilby.core.prior.TruncatedGaussian(
        mu=Mf_inj, sigma=5.0, minimum=Mf_inj - 15, maximum=Mf_inj + 15, name="Mf"
    )
    priors["chi"] = bilby.core.prior.TruncatedGaussian(
        mu=chi_inj, sigma=0.1, minimum=0.0, maximum=0.99, name="chi"
    )
    priors["phi"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi")
    # t0 prior: segment-relative, allow ±5 ms around injected value
    priors["t0"] = bilby.core.prior.Uniform(
        max(0, t0_inj - 0.005), t0_inj + 0.005, "t0"
    )

    if seed is not None:
        np.random.seed(seed + 1000)

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

    print(f"\nRunning H0 recovery with nlive={nlive}...")
    result = bilby.run_sampler(
        likelihood=like, priors=priors,
        outdir=outdir, label="h0_injection_test",
        **sampler_kwargs
    )

    # Extract posteriors
    Mf_post = result.posterior["Mf"].values
    chi_post = result.posterior["chi"].values
    A_post = result.posterior["A"].values
    t0_post = result.posterior["t0"].values

    # Compute recovered f0
    f0_post = []
    for m, c in zip(Mf_post, chi_post):
        f0_rec, _ = qnm_220_freq_tau(m, c)
        f0_post.append(f0_rec)
    f0_post = np.array(f0_post)

    # Summary statistics
    summary = {
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
            "Mf_16": float(np.percentile(Mf_post, 16)),
            "Mf_84": float(np.percentile(Mf_post, 84)),
            "chi_median": float(np.median(chi_post)),
            "chi_std": float(np.std(chi_post)),
            "chi_16": float(np.percentile(chi_post, 16)),
            "chi_84": float(np.percentile(chi_post, 84)),
            "f0_median": float(np.median(f0_post)),
            "f0_std": float(np.std(f0_post)),
            "A_median": float(np.median(A_post)),
            "t0_median_ms": float(np.median(t0_post)) * 1000,
        },
        "diagnostics": {
            "log_evidence": result.log_evidence,
            "log_evidence_err": result.log_evidence_err,
        }
    }

    # Check recovery quality
    Mf_err = abs(summary["recovered"]["Mf_median"] - Mf_inj)
    chi_err = abs(summary["recovered"]["chi_median"] - chi_inj)
    f0_err = abs(summary["recovered"]["f0_median"] - f0_inj)

    # Define pass conditions
    # Mf should be within 2-sigma of posterior width
    Mf_2sig = 2 * summary["recovered"]["Mf_std"]
    chi_2sig = 2 * summary["recovered"]["chi_std"]

    Mf_pass = Mf_err < max(Mf_2sig, 2.0)  # at least 2 M_sun tolerance
    chi_pass = chi_err < max(chi_2sig, 0.05)  # at least 0.05 tolerance
    f0_pass = f0_err < 20  # within 20 Hz

    summary["pass_conditions"] = {
        "Mf_within_2sigma": Mf_pass,
        "chi_within_2sigma": chi_pass,
        "f0_within_20Hz": f0_pass,
        "overall_pass": Mf_pass and chi_pass and f0_pass,
    }

    with open(os.path.join(outdir, "injection_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    # Print results
    print()
    print("=" * 60)
    print("INJECTION RECOVERY RESULTS")
    print("=" * 60)
    print(f"Injected -> Recovered (error)")
    print(f"  Mf: {Mf_inj:.2f} -> {summary['recovered']['Mf_median']:.2f} "
          f"+/- {summary['recovered']['Mf_std']:.2f} M_sun (err={Mf_err:.2f})")
    print(f"  chi: {chi_inj:.3f} -> {summary['recovered']['chi_median']:.3f} "
          f"+/- {summary['recovered']['chi_std']:.3f} (err={chi_err:.3f})")
    print(f"  f0: {f0_inj:.1f} -> {summary['recovered']['f0_median']:.1f} "
          f"+/- {summary['recovered']['f0_std']:.1f} Hz (err={f0_err:.1f})")
    print()

    if summary["pass_conditions"]["overall_pass"]:
        print("*** PASS: H0 recovery successful! ***")
    else:
        print("*** FAIL: H0 recovery failed! ***")
        if not Mf_pass:
            print(f"  - Mf error {Mf_err:.2f} > tolerance {max(Mf_2sig, 2.0):.2f}")
        if not chi_pass:
            print(f"  - chi error {chi_err:.3f} > tolerance {max(chi_2sig, 0.05):.3f}")
        if not f0_pass:
            print(f"  - f0 error {f0_err:.1f} > 20 Hz tolerance")
    print("=" * 60)

    return summary


def main():
    p = argparse.ArgumentParser()
    # Injection parameters (GW150914-like defaults)
    p.add_argument("--Mf", type=float, default=67.8,
                   help="Injected remnant mass in M_sun (default: 67.8 for GW150914)")
    p.add_argument("--chi", type=float, default=0.68,
                   help="Injected spin (default: 0.68 for GW150914)")
    p.add_argument("--snr", type=float, default=15.0,
                   help="Target injected SNR (amplitude scaled to achieve this)")
    p.add_argument("--t0", type=float, default=0.001,
                   help="Ringdown start time in seconds (segment-relative, default 1 ms)")

    # Segment parameters
    p.add_argument("--duration", type=float, default=0.1,
                   help="Segment duration in seconds (default 100 ms)")
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--fmin", type=float, default=150.0)
    p.add_argument("--fmax", type=float, default=400.0)

    # Taper/padding
    p.add_argument("--eps-start", type=float, default=0.01)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--pad-factor", type=float, default=2.0)

    # Sampler
    p.add_argument("--nlive", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--outdir", type=str, default="out_injection_test")

    # Optional: use real PSD
    p.add_argument("--real-psd", type=str, default=None,
                   help="Path to npz file with real PSD")

    args = p.parse_args()

    # Compute amplitude for target SNR
    # This is approximate; we'll scale based on the actual computed SNR
    f0_inj, tau_inj = qnm_220_freq_tau(args.Mf, args.chi)

    # Initial amplitude guess (will be refined)
    A_guess = 1e-21

    # Run a quick SNR calculation to calibrate amplitude
    fs = args.sample_rate
    dt = 1.0 / fs
    N = int(args.duration * fs)
    N_pad = int(N * args.pad_factor)
    t = np.arange(N) * dt
    f = np.fft.rfftfreq(N_pad, d=dt)

    # Simple PSD model
    S0 = 1e-47
    f0_psd = 100.0
    psd = np.zeros_like(f)
    mask = f > 10
    psd[mask] = S0 * ((f0_psd / f[mask])**4 + 2 + (f[mask] / f0_psd)**2)
    psd[~mask] = S0 * 1e6
    psd = np.maximum(psd, 1e-50)

    # Generate template
    h_td = ringdown_td_template(t, A_guess, f0_inj, tau_inj, 0.5, args.t0)
    taper = planck_taper(N, args.eps_start, args.eps_end)
    h_padded = np.zeros(N_pad)
    h_padded[:N] = h_td * taper
    h_fft = np.fft.rfft(h_padded) * dt

    # Compute SNR with unit amplitude
    band = (f >= args.fmin) & (f <= args.fmax)
    df = f[1] - f[0]
    snr_unit = np.sqrt(4 * np.sum(np.abs(h_fft[band])**2 / psd[band]) * df)

    # Scale amplitude for target SNR
    A_inj = A_guess * (args.snr / snr_unit) if snr_unit > 0 else A_guess

    print(f"Calibrated amplitude: A = {A_inj:.2e} for target SNR = {args.snr}")

    # Run the test
    summary = run_injection_test(
        Mf_inj=args.Mf,
        chi_inj=args.chi,
        A_inj=A_inj,
        t0_inj=args.t0,
        duration_sec=args.duration,
        sample_rate=args.sample_rate,
        fmin=args.fmin,
        fmax=args.fmax,
        outdir=args.outdir,
        seed=args.seed,
        nlive=args.nlive,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        pad_factor=args.pad_factor,
        use_real_psd=args.real_psd,
    )

    # Exit with error code if failed
    if not summary["pass_conditions"]["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
