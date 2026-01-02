#!/usr/bin/env python3
"""
Improved gated ringdown analysis with proper windowing.

This version fixes three key issues from v1:
1. Uses Planck taper instead of Tukey (smoother, better for ringdown where
   signal is at the start)
2. Applies same window to PSD estimation for consistency
3. Uses segment-relative time for t0 prior (not global GPS time)

The Planck taper smoothly ramps from 0 to 1 over epsilon*N samples at the
start, stays at 1 in the middle, then ramps back to 0 at the end.
For ringdown, we want epsilon_start small (preserve signal) and epsilon_end
larger (suppress edge effects).
"""
import argparse
import numpy as np
from gwpy.timeseries import TimeSeries
import os
import json


def planck_taper(N, epsilon_start=0.01, epsilon_end=0.1):
    """
    Planck-taper window.

    The Planck taper is C^infinity smooth (infinitely differentiable) and
    provides better spectral leakage suppression than Tukey.

    Parameters
    ----------
    N : int
        Window length
    epsilon_start : float
        Fraction of window for left ramp-up (0 < epsilon < 0.5)
    epsilon_end : float
        Fraction of window for right ramp-down (0 < epsilon < 0.5)

    Returns
    -------
    w : ndarray
        Window values in [0, 1]
    """
    w = np.ones(N)

    # Left taper (rising edge)
    n_left = int(epsilon_start * N)
    if n_left > 0:
        for i in range(1, n_left):
            x = epsilon_start * (1.0 / (i / N) + 1.0 / (i / N - epsilon_start))
            w[i] = 1.0 / (1.0 + np.exp(x))
        w[0] = 0.0

    # Right taper (falling edge)
    n_right = int(epsilon_end * N)
    if n_right > 0:
        for i in range(N - n_right, N - 1):
            x = epsilon_end * (1.0 / (1.0 - i / N) + 1.0 / (1.0 - i / N - epsilon_end))
            w[i] = 1.0 / (1.0 + np.exp(x))
        w[N - 1] = 0.0

    return w


def welch_psd_windowed(x, fs, nperseg, noverlap, window_func):
    """
    Welch PSD with custom window function applied to each segment.

    This ensures PSD estimation is consistent with how we window the data.

    Parameters
    ----------
    x : ndarray
        Time series data
    fs : float
        Sample rate
    nperseg : int
        Samples per segment
    noverlap : int
        Overlap between segments
    window_func : callable
        Window function that takes N and returns window array

    Returns
    -------
    f : ndarray
        Frequency array
    P : ndarray
        One-sided PSD
    """
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")

    # Get window (use same window as data)
    w = window_func(nperseg)
    w_norm = np.sum(w**2)

    segs = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start + nperseg]
        segs.append(seg)

    if not segs:
        raise ValueError("Not enough data for PSD estimation")

    P = None
    for seg in segs:
        X = np.fft.rfft(seg * w)
        ps = (np.abs(X)**2) / (fs * w_norm)
        P = ps if P is None else (P + ps)
    P /= len(segs)
    P[1:-1] *= 2  # one-sided PSD
    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, P


def fetch_and_prepare_gated_data(
    event, t_merger, t_start_sec, gate_duration_sec,
    ifos, sample_rate, outdir, psd_duration=128.0,
    epsilon_start=0.01, epsilon_end=0.1, pad_factor=2.0
):
    """
    Fetch data, gate to ringdown segment, and prepare windowed data + PSD.

    Key improvements:
    1. Planck taper with asymmetric edges (small at start to preserve ringdown)
    2. Zero-pad to reduce frequency resolution and spread spectral leakage
    3. PSD computed with same window for consistency

    Parameters
    ----------
    event : str
        Event name
    t_merger : float
        GPS time of merger
    t_start_sec : float
        Offset from merger to start gate (seconds)
    gate_duration_sec : float
        Duration of gated segment (seconds)
    ifos : list
        List of IFO names
    sample_rate : int
        Sample rate in Hz
    outdir : str
        Output directory
    psd_duration : float
        Duration of pre-merger data for PSD (seconds)
    epsilon_start : float
        Planck taper start fraction (small to preserve ringdown start)
    epsilon_end : float
        Planck taper end fraction
    pad_factor : float
        Zero-pad to this multiple of original length
    """
    fs = sample_rate

    # Gated segment times
    gate_t0 = t_merger + t_start_sec
    gate_t1 = gate_t0 + gate_duration_sec

    # PSD segment: pre-merger with buffer
    buffer = 1.0
    psd_t0 = t_merger - psd_duration - buffer
    psd_t1 = t_merger - buffer

    # Fetch range
    fetch_t0 = psd_t0 - 10
    fetch_t1 = gate_t1 + 10

    os.makedirs(outdir, exist_ok=True)

    print(f"Gated ringdown v2 for {event}")
    print(f"  Merger GPS: {t_merger}")
    print(f"  Gate: [{gate_t0:.4f}, {gate_t1:.4f}] ({gate_duration_sec*1000:.1f} ms)")
    print(f"  PSD from: [{psd_t0:.4f}, {psd_t1:.4f}] ({psd_duration}s)")
    print(f"  Planck taper: eps_start={epsilon_start}, eps_end={epsilon_end}")
    print(f"  Pad factor: {pad_factor}x")
    print()

    results = {}

    for ifo in ifos:
        print(f"[{ifo}] Fetching data...")
        ts = TimeSeries.fetch_open_data(ifo, fetch_t0, fetch_t1, sample_rate=fs, cache=True)
        full_data = ts.value.astype(float)

        dt = 1.0 / fs
        times = fetch_t0 + np.arange(len(full_data)) * dt

        # Extract gated segment
        gate_mask = (times >= gate_t0) & (times < gate_t1)
        gated_data = full_data[gate_mask]
        gated_times = times[gate_mask]
        N_orig = len(gated_data)

        print(f"[{ifo}] Gated segment: {N_orig} samples ({N_orig/fs*1000:.1f} ms)")

        # Apply Planck taper
        taper = planck_taper(N_orig, epsilon_start=epsilon_start, epsilon_end=epsilon_end)
        gated_tapered = gated_data * taper

        # Zero-pad for better frequency resolution
        N_pad = int(N_orig * pad_factor)
        gated_padded = np.zeros(N_pad)
        gated_padded[:N_orig] = gated_tapered

        # Extend time array for padded data
        times_padded = gated_times[0] + np.arange(N_pad) * dt

        print(f"[{ifo}] After padding: {N_pad} samples ({N_pad/fs*1000:.1f} ms)")

        # Compute PSD from pre-merger data with same window
        psd_mask = (times >= psd_t0) & (times < psd_t1)
        psd_data = full_data[psd_mask]
        print(f"[{ifo}] PSD segment: {len(psd_data)} samples ({len(psd_data)/fs:.1f}s)")

        # Use same Planck taper for PSD segments for consistency
        nperseg = min(len(psd_data), 4 * fs)  # 4s segments
        noverlap = nperseg // 2

        def psd_window(N):
            return planck_taper(N, epsilon_start=0.05, epsilon_end=0.05)

        f_psd, P = welch_psd_windowed(psd_data, fs, nperseg=nperseg,
                                       noverlap=noverlap, window_func=psd_window)

        # FFT of padded data
        f_data = np.fft.rfftfreq(N_pad, d=dt)
        d_fft = np.fft.rfft(gated_padded) * dt  # continuous FT convention

        # Interpolate PSD to data frequency grid
        psd_interp = np.interp(f_data, f_psd, P, left=P[0], right=P[-1])
        psd_interp = np.maximum(psd_interp, 1e-50)  # floor

        # Save
        outpath = os.path.join(outdir, f"{event}_{ifo}_data_psd.npz")
        np.savez(
            outpath,
            # Segment-relative time (starts at 0)
            t=times_padded - times_padded[0],
            t_gps=times_padded,  # Also save GPS times if needed
            x=gated_padded,  # Already windowed and padded
            f=f_data,
            d_fft=d_fft,  # Pre-computed FFT
            psd=psd_interp,
            # Metadata
            t_merger=t_merger,
            t_segment_start=gated_times[0],
            t_start_from_merger=t_start_sec,
            gate_duration=gate_duration_sec,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            pad_factor=pad_factor,
            N_orig=N_orig,
        )
        print(f"[{ifo}] Saved {outpath}")

        results[ifo] = {
            "t": times_padded - times_padded[0],
            "f": f_data,
            "d_fft": d_fft,
            "psd": psd_interp,
            "t_segment_start": gated_times[0],
        }

    # Save metadata
    meta = {
        "event": event,
        "gps_merger": t_merger,
        "t_start_ms": t_start_sec * 1000,
        "gate_duration_ms": gate_duration_sec * 1000,
        "psd_duration_s": psd_duration,
        "sample_rate": fs,
        "ifos": ifos,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "pad_factor": pad_factor,
        "version": "v2_planck_taper"
    }
    with open(os.path.join(outdir, f"{event}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {outdir}/{event}_meta.json")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, default="GW150914_gated_v2")
    p.add_argument("--gps", type=float, required=True, help="Merger GPS time")
    p.add_argument("--t-start", type=float, default=0.003,
                   help="Offset from merger to start gate (seconds, default 3 ms)")
    p.add_argument("--gate-duration", type=float, default=0.1,
                   help="Duration of gated segment (seconds, default 100 ms)")
    p.add_argument("--psd-duration", type=float, default=128.0,
                   help="Duration of pre-merger data for PSD (seconds)")
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, default="out_gated_v2")
    p.add_argument("--eps-start", type=float, default=0.01,
                   help="Planck taper start fraction (default 0.01 = 1%)")
    p.add_argument("--eps-end", type=float, default=0.1,
                   help="Planck taper end fraction (default 0.1 = 10%)")
    p.add_argument("--pad-factor", type=float, default=2.0,
                   help="Zero-pad to this multiple of original length")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]

    fetch_and_prepare_gated_data(
        event=args.event,
        t_merger=args.gps,
        t_start_sec=args.t_start,
        gate_duration_sec=args.gate_duration,
        ifos=ifos,
        sample_rate=args.sample_rate,
        outdir=args.outdir,
        psd_duration=args.psd_duration,
        epsilon_start=args.eps_start,
        epsilon_end=args.eps_end,
        pad_factor=args.pad_factor,
    )


if __name__ == "__main__":
    main()
