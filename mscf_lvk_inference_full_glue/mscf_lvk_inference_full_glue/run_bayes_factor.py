#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import bilby

from mscf.likelihood import GaussianFDLikelihood

def load_npz(path):
    d = np.load(path)
    return d["t"], d["x"], d["f"], d["psd"]

def rfft_data(t, x):
    dt = t[1] - t[0]
    X = np.fft.rfft(x) * dt
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
    p.add_argument("--ifos", type=str, default="H1,L1")
    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--resultdir", type=str, default="out")
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    ifos = [s.strip() for s in args.ifos.split(",") if s.strip()]
    label0 = args.label or f"{args.event}_H0"
    label1 = args.label or f"{args.event}_H1"

    # Build likelihoods
    like0 = build_likelihood(args.event, ifos, args.outdir, model="H0_ringdown")
    like1 = build_likelihood(args.event, ifos, args.outdir, model="H1_echo")

    # Priors (minimal)
    pri0 = bilby.core.prior.PriorDict()
    pri0["A"] = bilby.core.prior.LogUniform(1e-24, 1e-19, "A")
    pri0["f0"] = bilby.core.prior.Uniform(50, 400, "f0")
    pri0["tau"] = bilby.core.prior.Uniform(0.001, 0.05, "tau")
    pri0["phi"] = bilby.core.prior.Uniform(0, 2*np.pi, "phi")
    pri0["t0"] = bilby.core.prior.Uniform(like0.t[0], like0.t[-1], "t0")

    # H1 adds Mf, chi and reflectivity params. Δt_echo derived inside model.
    pri1 = pri0.copy()
    pri1["Mf"] = bilby.core.prior.Uniform(10, 200, "Mf")           # Msun, widen later using LVK posteriors
    pri1["chi"] = bilby.core.prior.Uniform(0.0, 0.999, "chi")
    pri1["R0"] = bilby.core.prior.Uniform(0.0, 1.0, "R0")
    pri1["f_cut"] = bilby.core.prior.LogUniform(50, 2000, "f_cut")
    pri1["roll"] = bilby.core.prior.Uniform(1.0, 10.0, "roll")
    pri1["phi0"] = bilby.core.prior.Uniform(0.0, 2*np.pi, "phi0")

    # Run nested sampling
    result0 = bilby.run_sampler(
        likelihood=like0, priors=pri0,
        sampler="dynesty", nlive=800,
        outdir=args.resultdir, label=label0, resume=False
    )
    result1 = bilby.run_sampler(
        likelihood=like1, priors=pri1,
        sampler="dynesty", nlive=800,
        outdir=args.resultdir, label=label1, resume=False
    )

    logBF = result1.log_evidence - result0.log_evidence
    print(f"ln BF_10 = {logBF:.3f} ; log10 BF_10 = {logBF/np.log(10):.3f}")

if __name__ == "__main__":
    main()
