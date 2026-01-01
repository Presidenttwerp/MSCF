#!/usr/bin/env python3
import argparse
import numpy as np

from echo_geometry import delta_t_echo_seconds
from reflectivity import Rw_of_f

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Mf", type=float, default=62.0, help="Final mass (source frame) in Msun")
    p.add_argument("--chi", type=float, default=0.7, help="Final dimensionless spin")
    p.add_argument("--R0", type=float, default=0.6)
    p.add_argument("--f_cut", type=float, default=512.0)
    p.add_argument("--roll", type=float, default=4.0)
    p.add_argument("--phi0", type=float, default=float(np.pi))
    args = p.parse_args()

    dt = delta_t_echo_seconds(args.Mf, args.chi)
    print(f"Derived MSCF echo delay Δt_echo(Mf,chi) = {dt:.6f} s")

    f = np.linspace(20, 1024, 2000)
    Rw = Rw_of_f(f, args.R0, args.f_cut, args.roll, args.phi0)
    print(f"|Rw(50 Hz)|={abs(Rw[np.argmin(abs(f-50))]):.3f}  |Rw(500 Hz)|={abs(Rw[np.argmin(abs(f-500))]):.3f}")

    print("\nTo run LVK-style inference: plug Δt_echo into your echo transfer function phase Δ(ω)=ωΔt,")
    print("build likelihood with PSD, and compute evidences (nested sampling) for H0 vs H1.")

if __name__ == "__main__":
    main()
