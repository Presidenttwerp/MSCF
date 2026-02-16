#!/usr/bin/env python3
"""
GW150914 derived-parameter echo search.

Searches for the zero-free-parameter MSCF echo signal using:
1. Matched filter with derived template
2. Z(dt) delay scan diagnostic
3. Off-source background estimation for p-value

All parameters (amplitudes, timing) are derived from first principles.
"""

import os
import sys
import time
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR, EVENTS, DEFAULT_N_BACKGROUND
from mscf_derived.derived_template import (
    derived_interference_template,
    derived_template_at_delay,
    mscf_echo_delay_seconds,
    qnm_220_freq_tau,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def _frequency_mask(freqs, fmin, fmax, psd):
    """Boolean mask for valid frequency bins."""
    return (freqs >= fmin) & (freqs <= fmax) & (psd > 0) & np.isfinite(psd)


def matched_filter_derived(freqs, fft_data, psd, Mf, chi,
                           fmin=20.0, fmax=2048.0, N_echo=6, phi=0.0):
    """
    Whitened matched filter with derived template.

    Returns (Z, sigma, snr, template).
    """
    mask = _frequency_mask(freqs, fmin, fmax, psd)
    f = freqs[mask]
    d = fft_data[mask]
    Sn = psd[mask]
    df = f[1] - f[0] if len(f) > 1 else 1.0

    template = derived_interference_template(f, Mf, chi, phi=phi, N_echo=N_echo)

    Z = np.sum(d * np.conj(template) / Sn) * df
    sigma_sq = np.sum(np.abs(template)**2 / Sn) * df
    sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1e-30

    return Z, sigma, abs(Z) / sigma, template


def delay_scan(freqs, fft_data, psd, Mf, chi,
               dt_min=None, dt_max=None, n_points=300,
               fmin=20.0, fmax=2048.0, N_echo=6):
    """Scan Z(dt) over delay range."""
    dt_mscf = mscf_echo_delay_seconds(Mf, chi)
    f0, tau = qnm_220_freq_tau(Mf, chi)

    if dt_min is None:
        dt_min = 0.2 * dt_mscf
    if dt_max is None:
        dt_max = 5.0 * dt_mscf

    dt_grid = np.linspace(dt_min, dt_max, n_points)

    mask = _frequency_mask(freqs, fmin, fmax, psd)
    f = freqs[mask]
    d = fft_data[mask]
    Sn = psd[mask]
    df = f[1] - f[0] if len(f) > 1 else 1.0

    snr_grid = np.zeros(n_points)
    Z_grid = np.zeros(n_points, dtype=complex)

    for i, dt_val in enumerate(dt_grid):
        t = derived_template_at_delay(f, dt_val, f0, tau, Mf, N_echo=N_echo)
        Z = np.sum(d * np.conj(t) / Sn) * df
        sigma_sq = np.sum(np.abs(t)**2 / Sn) * df
        sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1e-30
        Z_grid[i] = Z
        snr_grid[i] = abs(Z) / sigma

    return dt_grid, Z_grid, snr_grid


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ev = EVENTS['GW150914']
    Mf = ev['Mf_msun']
    chi = ev['chi_f']
    gps = ev['gps_merger']
    detectors = ev['detectors']

    dt_echo = mscf_echo_delay_seconds(Mf, chi)
    f0, tau = qnm_220_freq_tau(Mf, chi)

    print("=" * 70)
    print("GW150914 DERIVED-PARAMETER ECHO SEARCH")
    print("=" * 70)
    print(f"  Mf = {Mf} Msun, chi = {chi}")
    print(f"  f_QNM = {f0:.1f} Hz, tau = {tau*1000:.3f} ms")
    print(f"  dt_echo = {dt_echo*1000:.4f} ms")
    print(f"  Detectors: {detectors}")

    # Fetch data
    try:
        from gwpy.timeseries import TimeSeries
        from scipy.signal.windows import tukey
    except ImportError:
        print("\ngwpy not available. Skipping data analysis.")
        print("Install with: pip install gwpy")
        return 1

    results = {}

    for det in detectors:
        print(f"\n--- {det} ---")

        # Fetch pre-merger PSD segment
        t_psd_start = gps - 32.0 - 2.0
        t_psd_end = gps - 2.0
        log.info(f"  Fetching PSD segment [{t_psd_start:.0f}, {t_psd_end:.0f}]")
        psd_ts = TimeSeries.fetch_open_data(det, t_psd_start, t_psd_end, cache=True)
        psd_est = psd_ts.psd(fftlength=4.0, overlap=2.0, method='median')

        # Fetch ringdown segment
        ringdown_start = gps + 0.001  # 1 ms after merger
        ringdown_end = ringdown_start + 1.0  # 1 second
        log.info(f"  Fetching ringdown [{ringdown_start:.3f}, {ringdown_end:.3f}]")
        ring_ts = TimeSeries.fetch_open_data(
            det, ringdown_start - 0.1, ringdown_end + 0.1, cache=True)
        ring_ts = ring_ts.crop(ringdown_start, ringdown_end)

        sr = ring_ts.sample_rate.value
        n_samples = len(ring_ts)
        dt_samp = 1.0 / sr

        # Window and FFT
        window = tukey(n_samples, alpha=0.1)
        fft_data = np.fft.rfft(ring_ts.value * window) * dt_samp
        freqs = np.fft.rfftfreq(n_samples, d=dt_samp)

        # Interpolate PSD
        psd_f = psd_est.frequencies.value
        psd_v = psd_est.value
        psd_interp = np.interp(freqs, psd_f, psd_v,
                               left=psd_v[0], right=psd_v[-1])

        # --- On-source matched filter ---
        Z, sigma, snr, _ = matched_filter_derived(
            freqs, fft_data, psd_interp, Mf, chi)
        print(f"  On-source: SNR = {snr:.2f}, |Z| = {abs(Z):.4e}, sigma = {sigma:.4e}")

        # --- Z(dt) delay scan ---
        dt_grid, Z_scan, snr_scan = delay_scan(
            freqs, fft_data, psd_interp, Mf, chi)
        idx_mscf = np.argmin(np.abs(dt_grid - dt_echo))
        snr_at_mscf = snr_scan[idx_mscf]
        print(f"  SNR at MSCF delay: {snr_at_mscf:.2f}")
        print(f"  Peak SNR in scan: {np.max(snr_scan):.2f} at dt={dt_grid[np.argmax(snr_scan)]*1000:.3f} ms")

        # --- Off-source background ---
        off_snrs = []
        n_bg = DEFAULT_N_BACKGROUND
        bg_duration = 1.0
        bg_starts = np.linspace(gps - 12.0, gps - 3.0, n_bg)

        log.info(f"  Computing {n_bg} off-source trials...")
        for bg_start in bg_starts:
            try:
                bg_ts = TimeSeries.fetch_open_data(
                    det, bg_start - 0.1, bg_start + bg_duration + 0.1, cache=True)
                bg_ts = bg_ts.crop(bg_start, bg_start + bg_duration)

                if len(bg_ts) != n_samples:
                    continue

                bg_fft = np.fft.rfft(bg_ts.value * window) * dt_samp
                _, _, bg_snr, _ = matched_filter_derived(
                    freqs, bg_fft, psd_interp, Mf, chi)
                off_snrs.append(bg_snr)
            except Exception:
                continue

        off_snrs = np.array(off_snrs)
        if len(off_snrs) > 0:
            p_value = np.mean(off_snrs >= snr)
            bg_mean = np.mean(off_snrs)
            bg_std = np.std(off_snrs)
            sigma_excess = (snr - bg_mean) / bg_std if bg_std > 0 else 0.0
            print(f"  Background: mean={bg_mean:.2f}, std={bg_std:.2f}, N_trials={len(off_snrs)}")
            print(f"  Empirical p-value: {p_value:.4f} ({len(off_snrs)} trials)")
            print(f"  Sigma excess: {sigma_excess:.2f}")
        else:
            p_value = 1.0
            bg_mean = bg_std = sigma_excess = 0.0
            print("  No off-source trials succeeded")

        results[det] = {
            'snr': snr,
            'Z': Z,
            'sigma': sigma,
            'snr_at_mscf': snr_at_mscf,
            'p_value': p_value,
            'bg_mean': bg_mean,
            'bg_std': bg_std,
            'sigma_excess': sigma_excess,
            'off_snrs': off_snrs,
            'dt_grid': dt_grid,
            'snr_scan': snr_scan,
        }

    # --- Network SNR ---
    net_snr = np.sqrt(sum(r['snr']**2 for r in results.values()))
    print(f"\n  Network SNR: {net_snr:.2f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for det, r in results.items():
        print(f"  {det}: SNR={r['snr']:.2f}, p={r['p_value']:.4f}, sigma_excess={r['sigma_excess']:.2f}")
    print(f"  Network: SNR={net_snr:.2f}")

    # Save results
    np.savez(os.path.join(RESULTS_DIR, 'gw150914_derived.npz'),
             **{f'{det}_{k}': v for det, r in results.items()
                for k, v in r.items() if isinstance(v, (float, int, np.ndarray))},
             network_snr=net_snr,
             Mf=Mf, chi=chi, dt_echo=dt_echo)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(detectors), 1, figsize=(12, 5 * len(detectors)))
        if len(detectors) == 1:
            axes = [axes]

        for i, det in enumerate(detectors):
            r = results[det]
            ax = axes[i]

            ax.plot(r['dt_grid'] * 1000, r['snr_scan'], 'b-', linewidth=1)
            ax.axvline(dt_echo * 1000, color='r', linestyle='--',
                       label=f'MSCF dt={dt_echo*1000:.3f} ms')
            ax.axhline(r['bg_mean'] + 3 * r['bg_std'], color='gray',
                       linestyle=':', alpha=0.5, label='3σ background')
            ax.set_xlabel('Echo delay [ms]')
            ax.set_ylabel('SNR')
            ax.set_title(f'{det}: Z(dt) scan — GW150914 derived template')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(RESULTS_DIR, 'gw150914_derived_scan.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"\nPlot saved: {outfile}")

    except ImportError:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
