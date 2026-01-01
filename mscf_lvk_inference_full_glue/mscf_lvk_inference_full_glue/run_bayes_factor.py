#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import bilby

from mscf.likelihood import GaussianFDLikelihood

def load_npz(path):
    d = np.load(path)
    return d["t"], d["x"], d["f"], d["psd"]

def tukey_window(N, alpha=0.1):
    """Tukey window: flat in middle, tapered at edges. alpha=fraction tapered."""
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    n = np.arange(N)
    w = np.ones(N)
    # Left taper
    left = n < alpha * N / 2
    w[left] = 0.5 * (1 - np.cos(2 * np.pi * n[left] / (alpha * N)))
    # Right taper
    right = n >= N * (1 - alpha / 2)
    w[right] = 0.5 * (1 - np.cos(2 * np.pi * (N - 1 - n[right]) / (alpha * N)))
    return w

def rfft_data(t, x, taper_alpha=0.1):
    """FFT with Tukey taper to prevent spectral leakage."""
    dt = t[1] - t[0]
    w = tukey_window(len(x), alpha=taper_alpha)
    X = np.fft.rfft(x * w) * dt
    f = np.fft.rfftfreq(len(t), d=dt)
    return f, X

def build_likelihood(event, ifos, outdir, model):
    data_dict = {}
    psd_dict = {}
    t_ref = None

    for ifo in ifos:
        path = os.path.join(outdir, f"{event}_{ifo}_data_psd.npz")
        t, x, f_psd, psd = load_npz(path)

        # time -> freq data
        f, d_f = rfft_data(t, x)

        # interpolate PSD onto the data freq grid
        psd_i = np.interp(f, f_psd, psd, left=psd[0], right=psd[-1])

        # Floor PSD to avoid numerical issues (1e-50 is far below any real noise)
        psd_floor = 1e-50
        psd_i = np.maximum(psd_i, psd_floor)

        data_dict[ifo] = (f, d_f)
        psd_dict[ifo] = (f, psd_i)

        if t_ref is None:
            t_ref = t
        else:
            if len(t_ref) != len(t) or np.max(np.abs(t_ref - t)) > 1e-9:
                raise ValueError("Time grids mismatch between IFOs; ensure same duration/sample rate.")

    return GaussianFDLikelihood(t=t_ref, data_dict=data_dict, psd_dict=psd_dict, model=model)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", type=str, default="GW150914")
    p.add_argument("--gps", type=float, default=1126259462.4,
                   help="Merger GPS time for t0 prior centering")
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--resultdir", type=str, default="out")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for sampler reproducibility")
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    label0 = args.label or f"{args.event}_H0"
    label1 = args.label or f"{args.event}_H1"

    # Build likelihoods
    like0 = build_likelihood(args.event, ifos, args.outdir, model="H0_ringdown")
    like1 = build_likelihood(args.event, ifos, args.outdir, model="H1_echo")

    # Priors - GR-consistent: parameterize by (Mf, chi), derive f0/tau from QNM fits
    # Use LVK-informed Gaussian priors for remnant parameters (GW150914 values from GWTC-1)
    #
    # IMPORTANT: GWTC-1 reports SOURCE-FRAME masses. Strain data is in DETECTOR FRAME.
    # Detector-frame mass = Source-frame mass * (1 + z)
    # For GW150914: z ≈ 0.09, so Mf_det = Mf_src * 1.09
    #
    # GWTC-1 source-frame: Mf = 62.2 ± 3.7 Msun
    # Detector-frame: Mf = 62.2 * 1.09 ≈ 67.8 ± 4.0 Msun
    z_gw150914 = 0.09
    Mf_source = 62.2
    Mf_source_err = 3.7
    Mf_detector = Mf_source * (1 + z_gw150914)
    Mf_detector_err = Mf_source_err * (1 + z_gw150914)

    pri0 = bilby.core.prior.PriorDict()
    pri0["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    pri0["Mf"] = bilby.core.prior.TruncatedGaussian(
        mu=Mf_detector, sigma=Mf_detector_err, minimum=55, maximum=85, name="Mf"
    )
    # chi is dimensionless, same in both frames
    pri0["chi"] = bilby.core.prior.TruncatedGaussian(
        mu=0.68, sigma=0.05, minimum=0.0, maximum=0.99, name="chi"
    )
    pri0["phi"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi")
    # Ringdown start time: tight prior around merger (±50 ms)
    # For GW150914, merger is at GPS 1126259462.4 (within on-source segment)
    t_merger = float(args.gps) if args.gps else float(np.median(like0.t))
    pri0["t0"] = bilby.core.prior.Uniform(t_merger - 0.05, t_merger + 0.05, "t0")

    # H1 shares same GR ringdown physics (Mf, chi) + adds echo reflectivity params
    pri1 = pri0.copy()
    pri1["R0"] = bilby.core.prior.Uniform(0.0, 1.0, "R0")
    pri1["f_cut"] = bilby.core.prior.LogUniform(50, 2000, "f_cut")
    pri1["roll"] = bilby.core.prior.Uniform(1.0, 10.0, "roll")
    pri1["phi0"] = bilby.core.prior.Uniform(0.0, 2*np.pi, "phi0")

    # Set random seed if provided (for reproducibility tests)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Run nested sampling with proper convergence settings
    sampler_kwargs = dict(
        sampler="dynesty",
        nlive=800,
        dlogz=0.1,           # Stop when remaining evidence < 0.1 (tight convergence)
        sample="rwalk",       # Random walk sampling (robust for multimodal)
        walks=50,             # Number of random walk steps
        check_point_plot=False,  # Disable plotting to avoid numpy 2.x compat issue
        resume=False,
    )
    if args.seed is not None:
        sampler_kwargs["seed"] = args.seed

    result0 = bilby.run_sampler(
        likelihood=like0, priors=pri0,
        outdir=args.resultdir, label=label0,
        **sampler_kwargs
    )
    result1 = bilby.run_sampler(
        likelihood=like1, priors=pri1,
        outdir=args.resultdir, label=label1,
        **sampler_kwargs
    )

    logBF = result1.log_evidence - result0.log_evidence
    print(f"ln BF_10 = {logBF:.3f} ; log10 BF_10 = {logBF/np.log(10):.3f}")

if __name__ == "__main__":
    main()
