#!/usr/bin/env python3
"""
Stacked echo search across multiple events.

For each event:
1. Compute whitened spectral ratio R(f)
2. Phase-fold to align echo modulation pattern
3. Co-add (signal adds coherently, noise ~ sqrt(N))

With 4 events, gain factor ~ 2x in SNR.
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR, EVENTS, MSUN_S
from mscf_derived.derived_template import mscf_echo_delay_seconds, qnm_220_freq_tau
from mscf_derived.cepstrum import whitened_power_spectrum, power_cepstrum, cepstrum_snr

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        from gwpy.timeseries import TimeSeries
        from scipy.signal.windows import tukey
    except ImportError:
        print("gwpy not available.")
        return 1

    print("=" * 70)
    print("STACKED ECHO ANALYSIS")
    print("=" * 70)

    # Collect phase-folded spectral ratios
    stacked_cos = None
    stacked_sin = None
    n_stacked = 0
    n_phase_bins = 100
    phase_bins = np.linspace(0, 2 * np.pi, n_phase_bins + 1)[:-1]

    # Also collect individual cepstra aligned at dt_echo
    cepstra_aligned = []

    for name, ev in EVENTS.items():
        Mf = ev['Mf_msun']
        chi = ev['chi_f']
        gps = ev['gps_merger']

        dt_echo = mscf_echo_delay_seconds(Mf, chi)
        f0, tau = qnm_220_freq_tau(Mf, chi)

        print(f"\n  {name}: Mf={Mf}, chi={chi}, dt={dt_echo*1000:.3f} ms")

        for det in ev['detectors']:
            try:
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
                psd_interp = np.interp(freqs, psd_est.frequencies.value,
                                        psd_est.value,
                                        left=psd_est.value[0],
                                        right=psd_est.value[-1])

                # Whitened spectral ratio
                freqs_cut, R = whitened_power_spectrum(
                    fft_data, psd_interp, freqs, fmin=50.0, fmax=2000.0)

                # Phase-fold: map f → φ = 2πf·dt_echo mod 2π
                echo_phase = (2.0 * np.pi * freqs_cut * dt_echo) % (2.0 * np.pi)

                # Normalize R to zero mean, unit variance
                R_norm = (R - np.mean(R)) / np.std(R)

                # Bin by phase
                cos_binned = np.zeros(n_phase_bins)
                sin_binned = np.zeros(n_phase_bins)
                counts = np.zeros(n_phase_bins)

                dphi = 2.0 * np.pi / n_phase_bins
                bin_idx = np.clip((echo_phase / dphi).astype(int), 0, n_phase_bins - 1)

                for j in range(len(R_norm)):
                    b = bin_idx[j]
                    cos_binned[b] += R_norm[j] * np.cos(echo_phase[j])
                    sin_binned[b] += R_norm[j] * np.sin(echo_phase[j])
                    counts[b] += 1

                # Normalize by counts
                valid = counts > 0
                cos_binned[valid] /= counts[valid]
                sin_binned[valid] /= counts[valid]

                if stacked_cos is None:
                    stacked_cos = np.zeros(n_phase_bins)
                    stacked_sin = np.zeros(n_phase_bins)

                stacked_cos += cos_binned
                stacked_sin += sin_binned
                n_stacked += 1

                # Cepstrum for aligned stacking
                df_cut = freqs_cut[1] - freqs_cut[0]
                q, C = power_cepstrum(R, df_cut)
                cepstra_aligned.append((q, C, dt_echo, name, det))

                print(f"    {det}: OK (N_freq={len(freqs_cut)})")

            except Exception as e:
                print(f"    {det}: FAILED — {e}")
                continue

    if n_stacked == 0:
        print("\nNo data successfully processed.")
        return 1

    # Stacked result
    stacked_cos /= np.sqrt(n_stacked)
    stacked_sin /= np.sqrt(n_stacked)

    # Stacked modulation amplitude
    stacked_amp = np.sqrt(stacked_cos**2 + stacked_sin**2)
    mean_stacked_amp = np.mean(stacked_amp)

    # Cepstrum stacking: interpolate all cepstra onto common quefrency grid
    # Use the finest grid among all events
    q_min = max(c[0][1] for c in cepstra_aligned)  # Skip DC
    q_max = min(c[0][-1] for c in cepstra_aligned)
    n_q = 5000
    q_common = np.linspace(q_min, q_max, n_q)

    C_stacked = np.zeros(n_q)
    for q_arr, C_arr, dt, name, det in cepstra_aligned:
        C_interp = np.interp(q_common, q_arr, C_arr)
        C_stacked += C_interp / len(cepstra_aligned)

    # Look for peaks at each event's dt_echo (they differ!)
    # Instead, check if cepstrum at each event's own dt_echo is enhanced
    print(f"\n  Stacked {n_stacked} detector-event measurements")
    print(f"  Stacked modulation amplitude: {mean_stacked_amp:.4f}")

    # Per-event cepstrum check on stacked
    dt_unique = set(c[2] for c in cepstra_aligned)
    for dt in sorted(dt_unique):
        cep = cepstrum_snr(q_common, C_stacked, dt)
        print(f"  Stacked cepstrum at dt={dt*1000:.3f} ms: SNR={cep['snr']:.2f}")

    # Save
    np.savez(os.path.join(RESULTS_DIR, 'stacked_analysis.npz'),
             stacked_cos=stacked_cos,
             stacked_sin=stacked_sin,
             phase_bins=phase_bins,
             n_stacked=n_stacked,
             q_common=q_common,
             C_stacked=C_stacked)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # (a) Phase-folded stacked modulation
        ax = axes[0]
        ax.plot(np.degrees(phase_bins), stacked_amp, 'b-', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Echo phase [degrees]')
        ax.set_ylabel('Stacked modulation amplitude')
        ax.set_title(f'Phase-folded stack ({n_stacked} measurements)')
        ax.grid(True, alpha=0.3)

        # (b) Stacked cepstrum
        ax = axes[1]
        q_ms = q_common * 1000
        mask = (q_ms > 0.1) & (q_ms < 5.0)
        ax.plot(q_ms[mask], C_stacked[mask], 'b-', linewidth=0.5)
        for dt in sorted(dt_unique):
            ax.axvline(dt * 1000, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Quefrency [ms]')
        ax.set_ylabel('Stacked power cepstrum')
        ax.set_title('Stacked cepstrum (red = event dt_echo)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(RESULTS_DIR, 'stacked_analysis.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"\nPlot saved: {outfile}")

    except ImportError:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
