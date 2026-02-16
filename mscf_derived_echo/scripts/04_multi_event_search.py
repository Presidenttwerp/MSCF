#!/usr/bin/env python3
"""
Multi-event derived echo search.

Run derived matched filter and cepstral analysis on all catalog events.
Each event uses its own derived parameters (Mf, chi → dt, amplitudes).
"""

import os
import sys
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR, EVENTS, DEFAULT_N_BACKGROUND
from mscf_derived.derived_template import (
    derived_interference_template,
    mscf_echo_delay_seconds,
    qnm_220_freq_tau,
)
from mscf_derived.cepstrum import (
    whitened_power_spectrum,
    power_cepstrum,
    cepstrum_snr,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def _frequency_mask(freqs, fmin, fmax, psd):
    return (freqs >= fmin) & (freqs <= fmax) & (psd > 0) & np.isfinite(psd)


def analyze_event(event_name, ev_data, n_bg=50):
    """Run derived MF + cepstrum on one event."""
    from gwpy.timeseries import TimeSeries
    from scipy.signal.windows import tukey

    Mf = ev_data['Mf_msun']
    chi = ev_data['chi_f']
    gps = ev_data['gps_merger']
    detectors = ev_data['detectors']

    dt_echo = mscf_echo_delay_seconds(Mf, chi)
    f0, tau = qnm_220_freq_tau(Mf, chi)

    print(f"\n{'='*60}")
    print(f"  {event_name}: Mf={Mf} Msun, chi={chi}, dt={dt_echo*1000:.3f} ms")
    print(f"  f_QNM={f0:.0f} Hz, detectors={detectors}")
    print(f"{'='*60}")

    event_results = {}

    for det in detectors:
        try:
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
            psd_interp = np.interp(freqs, psd_est.frequencies.value,
                                    psd_est.value,
                                    left=psd_est.value[0],
                                    right=psd_est.value[-1])

            # Matched filter
            mask = _frequency_mask(freqs, 20.0, 2048.0, psd_interp)
            f = freqs[mask]
            d = fft_data[mask]
            Sn = psd_interp[mask]
            df = f[1] - f[0]

            template = derived_interference_template(f, Mf, chi)
            Z = np.sum(d * np.conj(template) / Sn) * df
            sigma_sq = np.sum(np.abs(template)**2 / Sn) * df
            sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1e-30
            snr = abs(Z) / sigma

            # Cepstrum
            freqs_cut, R = whitened_power_spectrum(
                fft_data, psd_interp, freqs, fmin=50.0, fmax=2000.0)
            df_cut = freqs_cut[1] - freqs_cut[0]
            q, C = power_cepstrum(R, df_cut)
            cep = cepstrum_snr(q, C, dt_echo)

            # Brief background (fewer trials for speed)
            off_snrs = []
            bg_starts = np.linspace(gps - 12.0, gps - 3.0, n_bg)
            for bg_start in bg_starts:
                try:
                    bg_ts = TimeSeries.fetch_open_data(
                        det, bg_start - 0.1, bg_start + 1.1, cache=True)
                    bg_ts = bg_ts.crop(bg_start, bg_start + 1.0)
                    if len(bg_ts) != n_samples:
                        continue
                    bg_fft = np.fft.rfft(bg_ts.value * window) * dt_samp
                    bg_t = derived_interference_template(f, Mf, chi)
                    bg_Z = np.sum(bg_fft[mask] * np.conj(bg_t) / Sn) * df
                    off_snrs.append(abs(bg_Z) / sigma)
                except Exception:
                    continue

            off_snrs = np.array(off_snrs)
            p_val = np.mean(off_snrs >= snr) if len(off_snrs) > 0 else 1.0

            print(f"  {det}: MF_SNR={snr:.2f}, p={p_val:.3f}, "
                  f"Cep_SNR={cep['snr']:.2f}")

            event_results[det] = {
                'mf_snr': snr,
                'p_value': p_val,
                'cepstrum_snr': cep['snr'],
            }

        except Exception as e:
            print(f"  {det}: FAILED — {e}")
            continue

    # Network SNR
    net_snr = np.sqrt(sum(r['mf_snr']**2 for r in event_results.values()))
    return {
        'per_detector': event_results,
        'network_snr': net_snr,
        'Mf': Mf,
        'chi': chi,
        'dt_echo': dt_echo,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        from gwpy.timeseries import TimeSeries
    except ImportError:
        print("gwpy not available.")
        return 1

    print("=" * 70)
    print("MULTI-EVENT DERIVED ECHO SEARCH")
    print("=" * 70)

    all_results = {}
    for name, ev in EVENTS.items():
        try:
            all_results[name] = analyze_event(name, ev, n_bg=50)
        except Exception as e:
            print(f"\n  {name}: FAILED — {e}")
            continue

    # Summary table
    print("\n" + "=" * 70)
    print("MULTI-EVENT SUMMARY")
    print("=" * 70)
    print(f"{'Event':<12s}  {'Net SNR':>8s}  {'p_best':>8s}  {'dt[ms]':>8s}")
    for name, r in all_results.items():
        p_best = min((d['p_value'] for d in r['per_detector'].values()), default=1.0)
        print(f"  {name:<12s}  {r['network_snr']:8.2f}  {p_best:8.4f}  "
              f"{r['dt_echo']*1000:8.3f}")

    # Save
    save_data = {}
    for name, r in all_results.items():
        save_data[f'{name}_net_snr'] = r['network_snr']
        save_data[f'{name}_dt_echo'] = r['dt_echo']
        for det, dr in r['per_detector'].items():
            save_data[f'{name}_{det}_snr'] = dr['mf_snr']
            save_data[f'{name}_{det}_p'] = dr['p_value']
            save_data[f'{name}_{det}_cep'] = dr['cepstrum_snr']

    np.savez(os.path.join(RESULTS_DIR, 'multi_event_results.npz'), **save_data)
    print(f"\nResults saved to {RESULTS_DIR}/multi_event_results.npz")

    return 0


if __name__ == '__main__':
    sys.exit(main())
