#!/usr/bin/env python3
"""
Fetch strain data and extract RINGDOWN-ONLY gated segment.

For a proper ringdown echo search, we need ONLY the post-merger data
where the ringdown model applies. This script:

1. Fetches data around the event (for PSD from off-source)
2. Gates to a short post-merger segment (default: 0.2s starting at t_merger + t_start)
3. Uses off-source data (before merger) for PSD estimation
4. Saves the gated segment with the PSD

Key parameters:
  --t-start: Offset from merger to start ringdown window (default: 10 ms)
  --gate-duration: Length of ringdown window (default: 0.2 s = 200 ms)

The ringdown for GW150914 lasts ~25 ms (5*tau), so 200 ms captures:
  - The full ringdown
  - Several potential echo cycles (if echoes exist)
  - Not too much post-ringdown noise
"""
import argparse
import numpy as np
from gwpy.timeseries import TimeSeries
import os


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


def tukey_window(N, alpha=0.1):
    """Tukey window for tapering edges."""
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    n = np.arange(N)
    w = np.ones(N)
    left = n < alpha * N / 2
    w[left] = 0.5 * (1 - np.cos(2 * np.pi * n[left] / (alpha * N)))
    right = n >= N * (1 - alpha / 2)
    w[right] = 0.5 * (1 - np.cos(2 * np.pi * (N - 1 - n[right]) / (alpha * N)))
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, default="GW150914_gated")
    p.add_argument("--gps", type=float, required=True,
                   help="Merger GPS time")
    p.add_argument("--t-start", type=float, default=0.010,
                   help="Offset from merger to start ringdown gate (seconds, default 10 ms)")
    p.add_argument("--gate-duration", type=float, default=0.2,
                   help="Duration of gated ringdown segment (seconds, default 0.2)")
    p.add_argument("--psd-duration", type=float, default=128.0,
                   help="Duration of pre-merger data for PSD (seconds)")
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, default="out_gated")
    p.add_argument("--taper-alpha", type=float, default=0.1,
                   help="Tukey taper fraction (default 0.1)")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    fs = args.sample_rate
    t_merger = args.gps
    t_start = args.t_start
    gate_dur = args.gate_duration
    psd_dur = args.psd_duration

    # Gated segment: [t_merger + t_start, t_merger + t_start + gate_dur]
    gate_t0 = t_merger + t_start
    gate_t1 = gate_t0 + gate_dur

    # For PSD: use pre-merger data [t_merger - psd_dur - buffer, t_merger - buffer]
    # buffer to avoid any merger contamination
    buffer = 1.0  # 1s buffer before merger
    psd_t0 = t_merger - psd_dur - buffer
    psd_t1 = t_merger - buffer

    # Fetch range: need both PSD segment and gated segment
    fetch_t0 = psd_t0 - 10  # extra padding
    fetch_t1 = gate_t1 + 10

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Ringdown-gated data extraction for {args.event}")
    print(f"  Merger GPS: {t_merger}")
    print(f"  Gate start: t_merger + {t_start*1000:.1f} ms = {gate_t0}")
    print(f"  Gate duration: {gate_dur*1000:.1f} ms")
    print(f"  Gate end: {gate_t1}")
    print(f"  PSD from: [{psd_t0}, {psd_t1}] ({psd_dur}s pre-merger)")
    print()

    for ifo in ifos:
        print(f"[{ifo}] Fetching data...")
        ts = TimeSeries.fetch_open_data(ifo, fetch_t0, fetch_t1, sample_rate=fs, cache=True)
        full_data = ts.value.astype(float)

        n_total = len(full_data)
        dt = 1.0 / fs
        times = fetch_t0 + np.arange(n_total) * dt

        # Extract gated ringdown segment
        gate_mask = (times >= gate_t0) & (times < gate_t1)
        gated_data = full_data[gate_mask]
        gated_times = times[gate_mask]

        # Apply Tukey taper to gated segment
        taper = tukey_window(len(gated_data), alpha=args.taper_alpha)
        gated_data_tapered = gated_data * taper

        print(f"[{ifo}] Gated segment: {len(gated_data)} samples ({len(gated_data)/fs*1000:.1f} ms)")

        # Extract pre-merger PSD segment
        psd_mask = (times >= psd_t0) & (times < psd_t1)
        psd_data = full_data[psd_mask]
        print(f"[{ifo}] PSD segment: {len(psd_data)} samples ({len(psd_data)/fs:.1f}s)")

        # Compute PSD from pre-merger data
        nperseg = min(len(psd_data), 4 * fs)  # 4s segments
        noverlap = nperseg // 2
        f_psd, P = welch_psd(psd_data, fs, nperseg=nperseg, noverlap=noverlap)

        # Save gated segment with PSD
        # Note: we save the UN-tapered data, taper is applied at likelihood eval
        # This allows flexibility in changing taper later
        outpath = os.path.join(args.outdir, f"{args.event}_{ifo}_data_psd.npz")
        np.savez(
            outpath,
            t=gated_times,
            x=gated_data,  # raw gated data (taper applied in likelihood)
            f=f_psd,
            psd=P,
            t_merger=t_merger,
            t_start=t_start,
            gate_duration=gate_dur,
            taper_alpha=args.taper_alpha
        )
        print(f"[{ifo}] Saved {outpath}")

    # Save metadata
    import json
    meta = {
        "event": args.event,
        "gps_merger": t_merger,
        "t_start_ms": t_start * 1000,
        "gate_duration_ms": gate_dur * 1000,
        "psd_duration_s": psd_dur,
        "sample_rate": fs,
        "ifos": ifos,
        "taper_alpha": args.taper_alpha
    }
    with open(os.path.join(args.outdir, f"{args.event}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {args.outdir}/{args.event}_meta.json")


if __name__ == "__main__":
    main()
