#!/usr/bin/env python
"""Detailed chi² diagnostic to find the exact normalization bug.

Trace through step by step:
1. What is E[|n_fft|²] from noise generation?
2. What does the likelihood assume E[|n|²] should be?
3. Where is the factor of ~51 coming from?
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Setup matching injection script
    sample_rate = 4096
    duration = 0.1  # 100 ms
    pad_factor = 2.0

    dt = 1.0 / sample_rate
    N_orig = int(duration * sample_rate)  # 410 samples
    N_pad = int(N_orig * pad_factor)       # 820 samples

    f = np.fft.rfftfreq(N_pad, d=dt)
    df = f[1] - f[0]  # = 1 / (N_pad * dt) = sample_rate / N_pad

    print("=" * 70)
    print("DETAILED CHI² NORMALIZATION ANALYSIS")
    print("=" * 70)
    print(f"N_orig = {N_orig}, N_pad = {N_pad}")
    print(f"dt = {dt:.6f} s, df = {df:.4f} Hz")
    print(f"T_orig = {N_orig * dt:.3f} s, T_pad = {N_pad * dt:.3f} s")
    print(f"Number of rfft bins: {len(f)}")
    print()

    # Simple flat PSD for clarity
    psd_value = 1e-46  # S_n(f) = constant
    psd = np.ones_like(f) * psd_value

    # =========================================================================
    # NOISE GENERATION (from test_h0_injection_gated.py lines 44-64)
    # =========================================================================
    print("=" * 70)
    print("STEP 1: NOISE GENERATION")
    print("=" * 70)

    # From generate_colored_noise():
    # sigma = np.sqrt(psd * df / 2)
    # noise = sigma * (real_part + 1j * imag_part)
    # where real_part, imag_part ~ N(0,1)

    sigma = np.sqrt(psd_value * df / 2)
    print(f"sigma = sqrt(psd * df / 2) = sqrt({psd_value:.2e} * {df:.4f} / 2)")
    print(f"      = {sigma:.6e}")
    print()

    # E[|noise|²] = E[sigma² * (Re² + Im²)] = sigma² * 2 = psd * df
    expected_noise_power = psd_value * df
    print(f"E[|noise_fft|²] = sigma² * 2 = psd * df = {expected_noise_power:.6e}")
    print()

    # Verify with Monte Carlo
    rng = np.random.default_rng(1234)
    n_trials = 10000
    noise_power_samples = []
    for _ in range(n_trials):
        real_part = rng.standard_normal()
        imag_part = rng.standard_normal()
        noise = sigma * (real_part + 1j * imag_part)
        noise_power_samples.append(np.abs(noise)**2)

    mc_mean = np.mean(noise_power_samples)
    print(f"Monte Carlo E[|noise|²] = {mc_mean:.6e} (expected {expected_noise_power:.6e})")
    print(f"Ratio: {mc_mean / expected_noise_power:.4f}")
    print()

    # =========================================================================
    # LIKELIHOOD FORMULA (from likelihood.py line 159)
    # =========================================================================
    print("=" * 70)
    print("STEP 2: LIKELIHOOD FORMULA")
    print("=" * 70)

    # logL = -4.0 * sum(|d-h|² / S_n) * df
    # For noise-only (d=n, h=0):
    # logL = -4.0 * sum(|n|² / S_n) * df

    # The chi² statistic used is:
    # chi² = 4 * sum(|n|² / S_n) * df

    print("Likelihood: logL = -4 * sum(|d-h|² / S_n) * df")
    print()
    print("For noise-only data (d=n, h=0):")
    print("  chi² = 4 * sum(|n|² / S_n) * df")
    print()

    # For a single bin:
    # chi²_bin = 4 * |n|² / S_n * df
    # E[chi²_bin] = 4 * E[|n|²] / S_n * df
    #             = 4 * (psd * df) / S_n * df
    #             = 4 * df²  (if psd = S_n)

    expected_chi2_per_bin = 4 * expected_noise_power / psd_value * df
    print(f"E[chi²] per bin = 4 * E[|n|²] / S_n * df")
    print(f"               = 4 * (psd * df) / S_n * df")
    print(f"               = 4 * df² = 4 * {df:.4f}² = {4 * df**2:.6f}")
    print()

    # Wait, this is weird. Let me reconsider.
    # The issue is that we're multiplying by df TWICE:
    # 1. Once in noise generation: E[|n|²] = psd * df
    # 2. Once in likelihood: chi² = 4 * sum(...) * df

    print("=" * 70)
    print("STEP 3: THE DOUBLE df PROBLEM")
    print("=" * 70)

    print("The chi² formula has:")
    print("  chi² = 4 * sum(|n|² / S_n) * df")
    print()
    print("With E[|n|²] = psd * df:")
    print("  E[chi²] = 4 * N_bins * (psd * df) / S_n * df")
    print("          = 4 * N_bins * df²  (if psd = S_n)")
    print()
    print(f"  = 4 * N_bins * {df:.4f}² = 4 * N_bins * {df**2:.6f}")
    print()

    # For 50 bins in band [150, 400] Hz:
    fmin, fmax = 150, 400
    band = (f >= fmin) & (f <= fmax)
    n_bins = np.sum(band)
    print(f"N_bins in [{fmin}, {fmax}] Hz: {n_bins}")
    print()

    expected_chi2_wrong = 4 * n_bins * df**2
    expected_chi2_correct = 2 * n_bins  # Standard: 2 dof per complex bin

    print(f"E[chi²] with current formula: 4 * {n_bins} * {df**2:.6f} = {expected_chi2_wrong:.1f}")
    print(f"E[chi²] standard (2 per bin): 2 * {n_bins} = {expected_chi2_correct}")
    print()
    print(f"Ratio: {expected_chi2_wrong / expected_chi2_correct:.1f}")
    print()

    # =========================================================================
    # ROOT CAUSE ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # The standard GW inner product for continuous FT is:
    # <a|b> = 4 Re ∫ a*(f) b(f) / S_n(f) df
    #
    # For discrete FT with rfft convention X(f) = rfft(x) * dt:
    # The integral becomes a sum: ∫ df → sum * df
    # So: <a|b> = 4 Re sum a*(f) b(f) / S_n(f) * df
    #
    # For the likelihood: ln L = -1/2 <d-h|d-h>
    #
    # The issue is the noise variance convention.
    # With X(f) = rfft(x) * dt, we have:
    # E[|X(f)|²] = E[|rfft(x)|²] * dt² = S_n(f) * df * dt²
    #            = S_n(f) / (N * df) * dt²   [since df = 1/(N*dt)]
    #            = S_n(f) * dt / N
    #
    # But generate_colored_noise uses:
    # E[|noise_fft|²] = psd * df
    #
    # The ratio is:
    # (psd * df) / (S_n * dt / N) = df * N / dt = N * df / dt
    #                             = N / (N * dt * dt) = 1/dt²

    print("The noise generation creates FFT coefficients with:")
    print(f"  E[|n_fft|²] = psd * df = {psd_value:.2e} * {df:.4f} = {psd_value * df:.6e}")
    print()
    print("But the standard continuous FT convention expects:")
    print("  X(f) = ∫ x(t) e^{-2πift} dt")
    print("  E[|X(f)|²] ~ S_n(f) * T  (for segment duration T)")
    print()

    T_pad = N_pad * dt
    expected_continuous = psd_value * T_pad
    print(f"For T = {T_pad:.3f} s:")
    print(f"  E[|X(f)|²]_continuous ~ S_n * T = {psd_value:.2e} * {T_pad:.3f} = {expected_continuous:.6e}")
    print()

    ratio = (psd_value * df) / expected_continuous
    print(f"Ratio (discrete/continuous) = {ratio:.6f}")
    print(f"                           = df / T = {df:.4f} / {T_pad:.3f} = {df/T_pad:.6f}")
    print(f"                           = 1/T² = {1/T_pad**2:.6f}")
    print()

    # The factor of ~51 comes from N_pad / 2 ≈ 50
    # because there are ~50 bins and df ≈ 5 Hz, so df² ≈ 25
    # and 4 * 50 * 25 / 100 ≈ 50

    print("=" * 70)
    print("THE FIX")
    print("=" * 70)

    print("The noise generation should NOT multiply by df.")
    print("Or equivalently, the likelihood should NOT multiply by df again.")
    print()
    print("Option 1: Fix noise generation")
    print("  sigma = sqrt(psd / 2)  # Remove df")
    print("  Then E[|n|²] = psd (per bin)")
    print("  And chi² = 4 * sum(|n|² / S_n) * df → E = 4 * N_bins * df")
    print()
    print("Option 2: Fix likelihood (remove the * df)")
    print("  logL = -4 * sum(|d-h|² / S_n)  # No * df at the end")
    print("  Then chi² = 4 * sum(|n|² / S_n)")
    print("  With E[|n|²] = psd * df:")
    print("  E[chi²] = 4 * N_bins * df  (still wrong)")
    print()
    print("Option 3: Match conventions properly")
    print("  The standard Whittle likelihood is:")
    print("  ln L = -sum( |d(f) - h(f)|² / (S_n(f) * df) )")
    print("  which gives chi² with E = 2 * N_bins")
    print()


if __name__ == "__main__":
    main()
