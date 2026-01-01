#!/usr/bin/env python3
"""Quick BIC check (screening only; NOT a substitute for Bayesian evidence)."""
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logL0", type=float, required=True, help="Max log-likelihood for GR ringdown-only model")
    p.add_argument("--logL1", type=float, required=True, help="Max log-likelihood for GR+echo model")
    p.add_argument("--k0", type=int, required=True, help="# free params in H0")
    p.add_argument("--k1", type=int, required=True, help="# free params in H1")
    p.add_argument("--n", type=int, required=True, help="# data points used")
    a = p.parse_args()

    bic0 = -2*a.logL0 + a.k0*np.log(a.n)
    bic1 = -2*a.logL1 + a.k1*np.log(a.n)
    d = bic1 - bic0
    print(f"BIC0={bic0:.3f}  BIC1={bic1:.3f}  ΔBIC={d:.3f} (negative favors echoes)")
    print("Use this only as a quick sanity check. For LVK-grade results, compute evidences with nested sampling.")
if __name__ == "__main__":
    main()
