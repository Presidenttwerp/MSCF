#!/usr/bin/env python3
"""
Fetch open strain data and estimate PSD from OFF-SOURCE segments.

Strategy:
- Fetch a long segment (default ±512s around event)
- Use off-source windows (before and after the on-source segment) for PSD estimation
- Save the on-source segment with the off-source PSD
"""
import argparse
import numpy as np
from gwpy.timeseries import TimeSeries


def welch_psd(x, fs, nperseg, noverlap):
    """Simple Welch PSD using numpy FFTs."""
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")
    w = np.hanning(nperseg)
    w_norm = np.sum(w**2)

    segs = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start+nperseg]
        segs.append(seg)
    if not segs:
        raise ValueError("Not enough data for PSD estimation. Increase duration or reduce nperseg.")

    P = None
    for seg in segs:
        X = np.fft.rfft(seg * w)
        # Periodogram: |X|^2 / (fs * w_norm) gives two-sided PSD
        ps = (np.abs(X)**2) / (fs * w_norm)
        P = ps if P is None else (P + ps)
    P /= len(segs)

    # Convert to one-sided PSD (factor of 2 for positive frequencies)
    # DC and Nyquist bins are not doubled
    P[1:-1] *= 2

    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, default="GW150914")
    p.add_argument("--duration", type=float, default=32.0,
                   help="On-source segment duration (seconds)")
    p.add_argument("--psd-duration", type=float, default=512.0,
                   help="Total duration to fetch for off-source PSD estimation (seconds on each side)")
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--gps", type=float, default=None,
                   help="Event GPS time (required). Example: 1126259462 for GW150914.")
    p.add_argument("--outdir", type=str, default="out")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    osr = args.sample_rate
    dur = args.duration
    psd_dur = args.psd_duration

    if args.gps is None:
        raise SystemExit("Please supply --gps (event GPS time). Example: 1126259462 for GW150914.")
    gps = float(args.gps)

    # On-source segment boundaries
    on_t0 = gps - dur / 2
    on_t1 = gps + dur / 2

    # Fetch wider segment for off-source PSD
    fetch_t0 = gps - psd_dur
    fetch_t1 = gps + psd_dur

    import os
    os.makedirs(args.outdir, exist_ok=True)

    for ifo in ifos:
        print(f"[{ifo}] Fetching {2*psd_dur}s of data around GPS {gps}...")
        ts = TimeSeries.fetch_open_data(ifo, fetch_t0, fetch_t1, sample_rate=osr, cache=True)
        full_data = ts.value.astype(float)
        fs = osr

        # Compute sample indices
        n_total = len(full_data)
        dt = 1.0 / fs
        times = fetch_t0 + np.arange(n_total) * dt

        # On-source indices
        on_mask = (times >= on_t0) & (times < on_t1)
        on_data = full_data[on_mask]
        on_times = times[on_mask]

        # Off-source: use data BEFORE and AFTER the on-source segment
        # Exclude a buffer around the event to avoid signal contamination
        # Buffer accounts for: on-source segment + taper regions + safety margin
        buffer = 8.0  # seconds buffer around on-source (conservative)
        off_mask = ((times < on_t0 - buffer) | (times >= on_t1 + buffer))
        off_data = full_data[off_mask]

        print(f"[{ifo}] On-source: {len(on_data)} samples ({dur}s)")
        print(f"[{ifo}] Off-source for PSD: {len(off_data)} samples ({len(off_data)/fs:.1f}s)")

        # PSD from off-source data
        nperseg = min(len(off_data), 4 * fs)  # 4s segments
        noverlap = nperseg // 2
        f, P = welch_psd(off_data, fs, nperseg=nperseg, noverlap=noverlap)

        # Save on-source time series with off-source PSD
        np.savez(
            os.path.join(args.outdir, f"{args.event}_{ifo}_data_psd.npz"),
            t=on_times,
            x=on_data,
            f=f,
            psd=P
        )
        print(f"[{ifo}] Saved {args.outdir}/{args.event}_{ifo}_data_psd.npz")


if __name__ == "__main__":
    main()
