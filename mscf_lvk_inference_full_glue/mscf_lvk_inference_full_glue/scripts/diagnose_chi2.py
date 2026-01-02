#!/usr/bin/env python
"""Diagnose chi^2 normalization mismatch.

Per user's diagnostic guidance:
- Compute noise-only chi² after full preprocessing
- Compute empirical scale factor alpha such that noise-only statistic matches expected mean
- Confirm injected SNR matches likelihood's SNR computation

If chi²/dof >> 1, the PSD is underestimating the noise variance, making posteriors too narrow.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.waveforms import planck_taper


def generate_colored_noise(f, psd, rng=None):
    """Generate colored Gaussian noise from one-sided PSD (from injection script)."""
    if rng is None:
        rng = np.random.default_rng()

    df = f[1] - f[0] if len(f) > 1 else 1.0
    real_part = rng.standard_normal(len(f))
    imag_part = rng.standard_normal(len(f))
    sigma = np.sqrt(psd * df / 2)
    noise = sigma * (real_part + 1j * imag_part)

    # DC and Nyquist are real for rfft
    if len(f) > 0:
        noise[0] = noise[0].real * np.sqrt(2)
    if len(f) > 1:
        noise[-1] = noise[-1].real * np.sqrt(2)

    return noise


def compute_chi2_statistic(d_fft, psd, f, fmin, fmax, df):
    """
    Compute chi² statistic: sum(|d(f)|² / (S_n(f) * df)) * 2

    For the corrected likelihood:
      logL = -2 * sum(|d-h|² / (S_n * df))

    For noise-only (h=0), this is:
      logL = -2 * sum(|n|² / (S_n * df))

    The chi² is: 2 * sum(|n|² / (S_n * df))
    Expected value: 2 * N_bins (since Re and Im each contribute chi²_1)
    """
    band = (f >= fmin) & (f <= fmax)
    chi2 = 2.0 * np.sum(np.abs(d_fft[band])**2 / (psd[band] * df))
    n_bins = np.sum(band)
    # Each complex bin contributes 2 dof (Re and Im)
    expected_chi2 = 2 * n_bins
    return chi2, expected_chi2, n_bins


def run_chi2_diagnostic(fmin=150, fmax=400, n_trials=100, noise_seed=1234):
    """Run chi² diagnostic to find PSD scale factor."""

    # Setup (matching injection script)
    sample_rate = 4096
    duration = 0.1
    pad_factor = 2.0
    eps_start, eps_end = 0.01, 0.1

    dt = 1.0 / sample_rate
    N_orig = int(duration * sample_rate)
    N_pad = int(N_orig * pad_factor)

    t = np.arange(N_orig) * dt
    f = np.fft.rfftfreq(N_pad, d=dt)
    df = f[1] - f[0]

    # Simple analytical PSD (same as injection script)
    S0 = 1e-47
    f0_psd = 100.0
    psd = np.zeros_like(f)
    mask = f > 10
    psd[mask] = S0 * ((f0_psd / f[mask])**4 + 2 + (f[mask] / f0_psd)**2)
    psd[~mask] = S0 * 1e6
    psd = np.maximum(psd, 1e-50)

    print("=" * 60)
    print("CHI² DIAGNOSTIC FOR NOISE SCALING")
    print("=" * 60)
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration*1000:.0f} ms (zero-padded to {N_pad/sample_rate*1000:.0f} ms)")
    print(f"Frequency band: {fmin}-{fmax} Hz")
    print(f"df = {df:.3f} Hz")
    print()

    # Count bins in band
    band = (f >= fmin) & (f <= fmax)
    n_bins = np.sum(band)
    print(f"Number of frequency bins in band: {n_bins}")
    print(f"Expected chi²: 2 * {n_bins} = {2*n_bins}")
    print()

    # Monte Carlo over noise realizations
    rng = np.random.default_rng(noise_seed)
    chi2_vals = []

    print(f"Running {n_trials} noise-only realizations...")

    for trial in range(n_trials):
        # Generate noise in frequency domain (direct method - what injection does)
        noise_fft = generate_colored_noise(f, psd, rng=rng)

        # Compute chi²
        chi2, expected, _ = compute_chi2_statistic(noise_fft, psd, f, fmin, fmax, df)
        chi2_vals.append(chi2)

    chi2_vals = np.array(chi2_vals)
    mean_chi2 = np.mean(chi2_vals)
    std_chi2 = np.std(chi2_vals)
    expected_chi2 = 2 * n_bins

    print()
    print("RESULTS (direct noise generation):")
    print("-" * 40)
    print(f"Mean chi²:     {mean_chi2:.1f}")
    print(f"Expected chi²: {expected_chi2:.1f}")
    print(f"Ratio (mean/expected): {mean_chi2/expected_chi2:.3f}")
    print(f"Std chi²:      {std_chi2:.1f}")
    print(f"Expected std:  {np.sqrt(2 * 2 * n_bins):.1f} (for chi² with {2*n_bins} dof)")
    print()

    # Compute scale factor
    alpha = mean_chi2 / expected_chi2
    print(f"PSD SCALE FACTOR NEEDED: alpha = {alpha:.4f}")
    print()

    if abs(alpha - 1.0) > 0.1:
        print("*** WARNING: Significant mismatch detected! ***")
        print(f"The likelihood sees chi²/dof = {alpha:.2f}")
        print(f"This means posteriors are {np.sqrt(alpha):.2f}x too narrow.")
        print()
        print("POSSIBLE CAUSES:")
        print("1. PSD underestimated (divide by alpha to fix)")
        print("2. Noise generation overestimated (multiply by sqrt(alpha))")
        print("3. Factor of 2 mismatch in PSD convention")
    else:
        print("PASS: chi²/dof ~ 1.0 (no significant mismatch)")

    # Now test with windowing (like actual data processing)
    print()
    print("=" * 60)
    print("WINDOWED NOISE TEST (simulating actual preprocessing)")
    print("=" * 60)

    chi2_windowed = []
    taper = planck_taper(N_orig, eps_start, eps_end)

    for trial in range(n_trials):
        # Generate noise in TIME domain first
        noise_fft_raw = generate_colored_noise(f, psd, rng=rng)
        noise_td = np.fft.irfft(noise_fft_raw / dt, n=N_pad)[:N_orig]

        # Apply window
        noise_windowed = noise_td * taper

        # Zero-pad
        noise_padded = np.zeros(N_pad)
        noise_padded[:N_orig] = noise_windowed

        # FFT
        noise_fft = np.fft.rfft(noise_padded) * dt

        # Compute chi² using ORIGINAL PSD (problem: window changes noise stats!)
        chi2, _, _ = compute_chi2_statistic(noise_fft, psd, f, fmin, fmax, df)
        chi2_windowed.append(chi2)

    chi2_windowed = np.array(chi2_windowed)
    mean_chi2_win = np.mean(chi2_windowed)

    print(f"Mean chi² (windowed): {mean_chi2_win:.1f}")
    print(f"Expected chi²:        {expected_chi2:.1f}")
    print(f"Ratio: {mean_chi2_win/expected_chi2:.3f}")
    print()

    alpha_win = mean_chi2_win / expected_chi2
    print(f"WINDOWED PSD SCALE FACTOR: alpha = {alpha_win:.4f}")

    if abs(alpha_win - 1.0) > 0.1:
        print()
        print("*** WINDOWING CHANGES NOISE STATISTICS! ***")
        print(f"Window reduces effective noise power by factor {alpha_win:.3f}")
        print("This means the PSD should be SCALED by the window power:")
        window_power = np.mean(taper**2)
        print(f"  Window power factor: {window_power:.4f}")
        print(f"  Expected scale: ~{window_power:.3f}")
        print()
        print("FIX: Multiply PSD by window_power = sum(w²)/N before using in likelihood")

    return alpha, alpha_win


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fmin', type=float, default=150)
    parser.add_argument('--fmax', type=float, default=400)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    run_chi2_diagnostic(args.fmin, args.fmax, args.n_trials, args.seed)
