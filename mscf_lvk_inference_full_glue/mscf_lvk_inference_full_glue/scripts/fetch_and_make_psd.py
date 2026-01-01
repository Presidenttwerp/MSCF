#!/usr/bin/env python3
import argparse
import numpy as np
from gwpy.timeseries import TimeSeries

def welch_psd(x, fs, nperseg, noverlap):
    # Simple Welch using numpy FFTs (keeps deps minimal). Use scipy.signal.welch if preferred.
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
        ps = (np.abs(X)**2) / (fs * w_norm)
        P = ps if P is None else (P + ps)
    P /= len(segs)
    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, P

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, default="GW150914")
    p.add_argument("--duration", type=float, default=32.0)
    p.add_argument("--sample-rate", type=int, default=4096)
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--gps", type=float, default=None, help="Override GPS time (otherwise use GWpy's open-data event mapping)")
    p.add_argument("--outdir", type=str, default="out")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    osr = args.sample_rate
    dur = args.duration

    # GWpy can fetch by event name for open data:
    # TimeSeries.fetch_open_data('H1', t0, t1, cache=True)
    # We'll use the event's GPS from GWpy open data helper via event_name=...
    # Simpler: ask GWpy to fetch around event central time if gps is given.
    if args.gps is None:
        # GWpy supports event name via `TimeSeries.fetch_open_data` with "event=..."
        # We will fetch the standard open segment around the event time by querying the event catalog implicitly.
        # If this fails on your machine, set --gps manually from GWOSC.
        raise SystemExit("Please supply --gps (event GPS time) for reproducible fetching. Example: 1126259462 for GW150914.")
    gps = float(args.gps)

    t0 = gps - dur/2
    t1 = gps + dur/2

    import os
    os.makedirs(args.outdir, exist_ok=True)

    for ifo in ifos:
        ts = TimeSeries.fetch_open_data(ifo, t0, t1, sample_rate=osr, cache=True)
        x = ts.value.astype(float)
        fs = osr

        # PSD from full segment (you may prefer off-source; this is minimal scaffold)
        nperseg = min(len(x), 4*fs)  # 4s segments
        noverlap = nperseg//2
        f, P = welch_psd(x, fs, nperseg=nperseg, noverlap=noverlap)

        # Save time series and PSD
        np.savez(
            os.path.join(args.outdir, f"{args.event}_{ifo}_data_psd.npz"),
            t=np.linspace(t0, t1, len(x), endpoint=False),
            x=x,
            f=f,
            psd=P
        )
        print(f"Saved {args.outdir}/{args.event}_{ifo}_data_psd.npz")

if __name__ == "__main__":
    main()
