#!/usr/bin/env python3
"""
Spectral ratio / cepstral echo search for GW150914.

Complementary to matched filtering: looks for periodic spectral
modulation at the MSCF-predicted ripple spacing Δf = 1/dt_echo.

Method:
1. Compute ringdown power spectrum |d(f)|²
2. Whiten: R(f) = |d(f)|²/S_n(f)
3. Power cepstrum: C(τ) = |IFFT(log R(f))|²
4. Peak at τ = dt_echo? Compare with off-source background.
5. Spectral modulation fit: ε and φ from greybody model.
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR, EVENTS, DEFAULT_N_BACKGROUND
from mscf_derived.derived_template import mscf_echo_delay_seconds, qnm_220_freq_tau
from mscf_derived.cepstrum import (
    whitened_power_spectrum,
    power_cepstrum,
    cepstrum_snr,
    spectral_modulation_fit,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ev = EVENTS['GW150914']
    Mf = ev['Mf_msun']
    chi = ev['chi_f']
    gps = ev['gps_merger']
    detectors = ev['detectors']

    dt_echo = mscf_echo_delay_seconds(Mf, chi)
    f0, tau = qnm_220_freq_tau(Mf, chi)
    Delta_f = 1.0 / dt_echo

    print("=" * 70)
    print("SPECTRAL RATIO / CEPSTRAL ECHO SEARCH: GW150914")
    print("=" * 70)
    print(f"  Mf = {Mf} Msun, chi = {chi}")
    print(f"  dt_echo = {dt_echo*1000:.4f} ms")
    print(f"  Spectral ripple spacing = {Delta_f:.0f} Hz")
    print(f"  f_QNM = {f0:.1f} Hz")

    try:
        from gwpy.timeseries import TimeSeries
        from scipy.signal.windows import tukey
    except ImportError:
        print("\ngwpy not available. Skipping data analysis.")
        return 1

    results = {}

    for det in detectors:
        print(f"\n--- {det} ---")

        # Fetch data
        t_psd_start = gps - 34.0
        t_psd_end = gps - 2.0
        psd_ts = TimeSeries.fetch_open_data(det, t_psd_start, t_psd_end, cache=True)
        psd_est = psd_ts.psd(fftlength=4.0, overlap=2.0, method='median')

        ringdown_start = gps + 0.001
        ringdown_end = ringdown_start + 1.0
        ring_ts = TimeSeries.fetch_open_data(
            det, ringdown_start - 0.1, ringdown_end + 0.1, cache=True)
        ring_ts = ring_ts.crop(ringdown_start, ringdown_end)

        sr = ring_ts.sample_rate.value
        n_samples = len(ring_ts)
        dt_samp = 1.0 / sr

        window = tukey(n_samples, alpha=0.1)
        fft_data = np.fft.rfft(ring_ts.value * window) * dt_samp
        freqs = np.fft.rfftfreq(n_samples, d=dt_samp)

        psd_f = psd_est.frequencies.value
        psd_v = psd_est.value
        psd_interp = np.interp(freqs, psd_f, psd_v,
                               left=psd_v[0], right=psd_v[-1])

        # --- On-source cepstral analysis ---
        # Use wide band to capture kHz-scale ripples
        freqs_cut, R = whitened_power_spectrum(fft_data, psd_interp, freqs,
                                               fmin=50.0, fmax=2000.0)
        df = freqs_cut[1] - freqs_cut[0]
        quefrency, C = power_cepstrum(R, df)

        cep_result = cepstrum_snr(quefrency, C, dt_echo)
        print(f"  Cepstrum SNR at dt_echo: {cep_result['snr']:.2f}")
        print(f"  Peak quefrency: {cep_result['peak_quefrency']*1000:.4f} ms")
        print(f"  Background: median={cep_result['background_median']:.4e}, "
              f"std={cep_result['background_std']:.4e}")

        # --- Spectral modulation fit ---
        mod_result = spectral_modulation_fit(
            freqs_cut, R, dt_echo, Mf_msun=Mf)
        print(f"  Modulation fit: ε={mod_result['epsilon']:.4f}, "
              f"φ={np.degrees(mod_result['phi']):.1f}°, "
              f"SNR={mod_result['snr']:.2f}")

        # --- Off-source background for cepstrum ---
        off_cep_snrs = []
        off_mod_snrs = []
        n_bg = DEFAULT_N_BACKGROUND
        bg_starts = np.linspace(gps - 12.0, gps - 3.0, n_bg)

        log.info(f"  Computing {n_bg} off-source cepstrum trials...")
        for bg_start in bg_starts:
            try:
                bg_ts = TimeSeries.fetch_open_data(
                    det, bg_start - 0.1, bg_start + 1.1, cache=True)
                bg_ts = bg_ts.crop(bg_start, bg_start + 1.0)

                if len(bg_ts) != n_samples:
                    continue

                bg_fft = np.fft.rfft(bg_ts.value * window) * dt_samp

                # Cepstrum
                bg_freqs_cut, bg_R = whitened_power_spectrum(
                    bg_fft, psd_interp, freqs, fmin=50.0, fmax=2000.0)
                bg_q, bg_C = power_cepstrum(bg_R, df)
                bg_cep = cepstrum_snr(bg_q, bg_C, dt_echo)
                off_cep_snrs.append(bg_cep['snr'])

                # Modulation
                bg_mod = spectral_modulation_fit(bg_freqs_cut, bg_R, dt_echo, Mf_msun=Mf)
                off_mod_snrs.append(bg_mod['snr'])
            except Exception:
                continue

        off_cep_snrs = np.array(off_cep_snrs)
        off_mod_snrs = np.array(off_mod_snrs)

        if len(off_cep_snrs) > 0:
            p_cep = np.mean(off_cep_snrs >= cep_result['snr'])
            p_mod = np.mean(off_mod_snrs >= mod_result['snr'])
            print(f"  Cepstrum p-value: {p_cep:.4f} ({len(off_cep_snrs)} trials)")
            print(f"  Modulation p-value: {p_mod:.4f} ({len(off_mod_snrs)} trials)")
        else:
            p_cep = p_mod = 1.0
            print("  No off-source trials succeeded")

        results[det] = {
            'cepstrum_snr': cep_result['snr'],
            'cepstrum_p': p_cep,
            'modulation_snr': mod_result['snr'],
            'modulation_p': p_mod,
            'epsilon': mod_result['epsilon'],
            'phi': mod_result['phi'],
            'quefrency': quefrency,
            'cepstrum': C,
            'off_cep_snrs': off_cep_snrs,
            'off_mod_snrs': off_mod_snrs,
        }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CEPSTRAL ANALYSIS SUMMARY")
    print("=" * 70)
    for det, r in results.items():
        print(f"  {det}: Cepstrum SNR={r['cepstrum_snr']:.2f} (p={r['cepstrum_p']:.4f}), "
              f"Modulation SNR={r['modulation_snr']:.2f} (p={r['modulation_p']:.4f})")

    # Save
    np.savez(os.path.join(RESULTS_DIR, 'gw150914_cepstrum.npz'),
             **{f'{det}_{k}': v for det, r in results.items()
                for k, v in r.items() if isinstance(v, (float, int, np.ndarray))},
             Mf=Mf, chi=chi, dt_echo=dt_echo)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(detectors), 2, figsize=(14, 5 * len(detectors)))
        if len(detectors) == 1:
            axes = axes.reshape(1, -1)

        for i, det in enumerate(detectors):
            r = results[det]

            # Cepstrum
            ax = axes[i, 0]
            q_ms = r['quefrency'] * 1000
            mask_q = (q_ms > 0.05) & (q_ms < 5.0)
            ax.plot(q_ms[mask_q], r['cepstrum'][mask_q], 'b-', linewidth=0.5)
            ax.axvline(dt_echo * 1000, color='r', linestyle='--',
                       label=f'dt_echo={dt_echo*1000:.3f} ms')
            ax.set_xlabel('Quefrency [ms]')
            ax.set_ylabel('Power cepstrum')
            ax.set_title(f'{det}: Power cepstrum')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Background distribution
            ax = axes[i, 1]
            if len(r['off_cep_snrs']) > 0:
                ax.hist(r['off_cep_snrs'], bins=30, alpha=0.7, density=True,
                        label='Off-source')
                ax.axvline(r['cepstrum_snr'], color='r', linewidth=2,
                           label=f'On-source: {r["cepstrum_snr"]:.1f}')
            ax.set_xlabel('Cepstrum SNR')
            ax.set_ylabel('Density')
            ax.set_title(f'{det}: Cepstrum SNR distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(RESULTS_DIR, 'gw150914_cepstrum.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"\nPlot saved: {outfile}")

    except ImportError:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
