#!/usr/bin/env python3
"""
Injection/Recovery Tests for MSCF Echo Pipeline Validation

Two types of injections:
1. H0 injection: Pure GR ringdown into real detector noise
   - Should NOT prefer H1 (echo model)
   - Go/no-go: log10(BF) < 1 (no false echo detection)

2. H1 injection: GR ringdown + echoes into real detector noise
   - Should recover injected echo parameters
   - Should show positive Bayes factor for echoes
   - Go/no-go: log10(BF) > 2 and parameters recovered within uncertainties

Uses real LIGO noise from off-source segments (same as null tests).
"""
import argparse
import os
import sys
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscf.waveforms import ringdown_fd_qnm, ringdown_plus_echo_fd, tukey_window


def generate_colored_noise(f, psd, N, dt, seed=None):
    """
    Generate time-domain colored noise with given PSD.

    Parameters
    ----------
    f : array
        Frequency grid for PSD
    psd : array
        One-sided PSD values
    N : int
        Number of time samples
    dt : float
        Time step
    seed : int, optional
        Random seed

    Returns
    -------
    noise : array
        Time-domain noise realization
    """
    if seed is not None:
        np.random.seed(seed)

    # Frequency grid for rfft
    f_rfft = np.fft.rfftfreq(N, d=dt)

    # Interpolate PSD onto rfft grid
    psd_interp = np.interp(f_rfft, f, psd, left=psd[0], right=psd[-1])
    psd_interp = np.maximum(psd_interp, 1e-50)  # floor

    # Generate white noise in frequency domain
    # For rfft: need complex amplitudes with proper variance
    # Variance of each bin should be ~ PSD * df / 2 for one-sided
    df = 1.0 / (N * dt)

    # Standard deviation for each frequency bin
    # PSD is one-sided, so variance per bin is S(f) * df
    sigma = np.sqrt(psd_interp / (2 * dt))  # accounts for rfft normalization

    # Complex noise: real and imag parts each have variance sigma^2/2
    noise_fd = sigma * (np.random.randn(len(f_rfft)) + 1j * np.random.randn(len(f_rfft)))

    # DC and Nyquist bins should be real
    noise_fd[0] = noise_fd[0].real * np.sqrt(2)
    if N % 2 == 0:
        noise_fd[-1] = noise_fd[-1].real * np.sqrt(2)

    # Transform to time domain
    noise_td = np.fft.irfft(noise_fd, n=N)

    return noise_td


def create_injection_data(injection_type, inj_params, noise_data_path, outdir, event_name, seed=None):
    """
    Create injection data file by adding waveform to real noise.

    Parameters
    ----------
    injection_type : str
        "H0" for ringdown only, "H1" for ringdown + echo
    inj_params : dict
        Injection parameters
    noise_data_path : str
        Path to noise .npz file (from null test segment)
    outdir : str
        Output directory
    event_name : str
        Event label for output files
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Paths to created data files for each IFO
    """
    os.makedirs(outdir, exist_ok=True)

    # Load noise data
    noise = np.load(noise_data_path)
    t = noise["t"]
    noise_td = noise["x"]
    f_psd = noise["f"]
    psd = noise["psd"]

    N = len(t)
    dt = t[1] - t[0]

    # Generate the injection waveform
    if injection_type == "H0":
        # Pure ringdown
        f_wf, h_fd = ringdown_fd_qnm(
            t,
            A=inj_params["A"],
            Mf=inj_params["Mf"],
            chi=inj_params["chi"],
            phi=inj_params["phi"],
            t0=inj_params["t0"]
        )
    else:  # H1
        # Ringdown + echoes
        f_wf, h_fd = ringdown_plus_echo_fd(t, inj_params)

    # Convert waveform to time domain
    h_td = np.fft.irfft(h_fd / dt, n=N)  # undo the dt factor from rfft

    # Add waveform to noise
    data_td = noise_td + h_td

    # Apply Tukey window to injection data
    w = tukey_window(N, alpha=0.1)
    data_td_windowed = data_td * w

    # FFT the injected data
    data_fd = np.fft.rfft(data_td_windowed) * dt
    f_data = np.fft.rfftfreq(N, d=dt)

    # Save injection data
    # We need to save in the format expected by run_bayes_factor.py
    # Extract IFO name from noise path: null_seg1_H1_data_psd.npz -> H1
    basename = os.path.basename(noise_data_path)
    parts = basename.replace("_data_psd.npz", "").split("_")
    ifo = parts[-1]  # Last part before _data_psd.npz is the IFO (H1 or L1)

    out_path = os.path.join(outdir, f"{event_name}_{ifo}_data_psd.npz")
    np.savez(
        out_path,
        t=t,
        x=data_td,  # time-domain data with injection
        f=f_psd,
        psd=psd
    )

    return out_path


def main():
    p = argparse.ArgumentParser(description="Injection/Recovery Tests")
    p.add_argument("--type", choices=["H0", "H1", "both"], default="both",
                   help="Injection type: H0 (ringdown only), H1 (ringdown+echo), or both")
    p.add_argument("--n-injections", type=int, default=3,
                   help="Number of injections per type")
    p.add_argument("--outdir", type=str, default="out_injection",
                   help="Output directory")
    p.add_argument("--noise-dir", type=str, default="out_null",
                   help="Directory with null test noise data")
    p.add_argument("--seed-base", type=int, default=2000,
                   help="Base seed for reproducibility")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--snr", type=float, default=15.0,
                   help="Target optimal SNR for injections")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Reference GPS time (center of injection)
    gps_ref = 1126258462.4  # Same as null_seg1

    # Base injection parameters (GW150914-like)
    base_params = {
        "A": 1.5e-21,
        "Mf": 68.0,  # detector-frame mass
        "chi": 0.67,
        "phi": 1.5,
        "t0": gps_ref,
        # Echo params (only used for H1)
        "R0": 0.8,
        "f_cut": 200.0,
        "roll": 6.0,
        "phi0": 0.5
    }

    # Store all injection configs
    injection_configs = []

    # Define injection sets
    injection_types = []
    if args.type in ["H0", "both"]:
        injection_types.append("H0")
    if args.type in ["H1", "both"]:
        injection_types.append("H1")

    print("=" * 70)
    print("INJECTION/RECOVERY TESTS")
    print("=" * 70)

    for inj_type in injection_types:
        print(f"\n{'='*70}")
        print(f"Setting up {inj_type} injections ({args.n_injections} total)")
        print(f"{'='*70}")

        for i in range(args.n_injections):
            seed = args.seed_base + (0 if inj_type == "H0" else 100) + i
            event_name = f"inj_{inj_type}_{i+1}"
            inj_outdir = os.path.join(args.outdir, event_name)

            # Vary injection parameters slightly
            inj_params = base_params.copy()
            np.random.seed(seed)

            # Add small variations
            inj_params["A"] = base_params["A"] * (0.8 + 0.4 * np.random.random())
            inj_params["Mf"] = base_params["Mf"] + np.random.uniform(-3, 3)
            inj_params["chi"] = np.clip(base_params["chi"] + np.random.uniform(-0.05, 0.05), 0.1, 0.95)
            inj_params["phi"] = np.random.uniform(0, 2*np.pi)

            if inj_type == "H1":
                inj_params["R0"] = np.random.uniform(0.5, 0.95)
                inj_params["f_cut"] = np.random.uniform(100, 300)
                inj_params["roll"] = np.random.uniform(4, 9)
                inj_params["phi0"] = np.random.uniform(0, 2*np.pi)

            # Use different null segments for noise realization
            seg_idx = (i % 5) + 1
            noise_paths = {
                "H1": os.path.join(args.noise_dir, f"null_seg{seg_idx}", f"null_seg{seg_idx}_H1_data_psd.npz"),
                "L1": os.path.join(args.noise_dir, f"null_seg{seg_idx}", f"null_seg{seg_idx}_L1_data_psd.npz")
            }

            print(f"\n[{event_name}] Creating injection:")
            print(f"  Type: {inj_type}")
            print(f"  Using noise from: null_seg{seg_idx}")
            print(f"  Parameters:")
            print(f"    A = {inj_params['A']:.2e}")
            print(f"    Mf = {inj_params['Mf']:.1f} Msun")
            print(f"    chi = {inj_params['chi']:.3f}")
            if inj_type == "H1":
                print(f"    R0 = {inj_params['R0']:.3f}")
                print(f"    f_cut = {inj_params['f_cut']:.1f} Hz")
                print(f"    roll = {inj_params['roll']:.2f}")

            if not args.dry_run:
                # Create injection data for both IFOs
                os.makedirs(inj_outdir, exist_ok=True)

                for ifo, noise_path in noise_paths.items():
                    if os.path.exists(noise_path):
                        create_injection_data(
                            injection_type=inj_type,
                            inj_params=inj_params,
                            noise_data_path=noise_path,
                            outdir=inj_outdir,
                            event_name=event_name,
                            seed=seed + (1 if ifo == "L1" else 0)
                        )
                        print(f"  Created {ifo} injection data")
                    else:
                        print(f"  WARNING: Noise file not found: {noise_path}")

                # Save injection truth
                truth_path = os.path.join(inj_outdir, "injection_truth.json")
                with open(truth_path, "w") as f:
                    json.dump({
                        "type": inj_type,
                        "params": {k: float(v) for k, v in inj_params.items()},
                        "seed": seed,
                        "noise_segment": seg_idx
                    }, f, indent=2)

            injection_configs.append({
                "event_name": event_name,
                "outdir": inj_outdir,
                "type": inj_type,
                "params": inj_params,
                "gps": gps_ref
            })

    # Now run the Bayes factor analysis on each injection
    print(f"\n{'='*70}")
    print("Running Bayes factor analysis on injections")
    print(f"{'='*70}")

    import subprocess

    for config in injection_configs:
        event_name = config["event_name"]
        inj_outdir = config["outdir"]
        gps = config["gps"]
        seed = args.seed_base + (0 if config["type"] == "H0" else 100) + int(event_name.split("_")[-1])

        bf_cmd = [
            sys.executable, "run_bayes_factor.py",
            "--event", event_name,
            "--gps", str(gps),
            "--ifos", "H1,L1",
            "--outdir", inj_outdir,
            "--resultdir", inj_outdir,
            "--seed", str(seed)
        ]

        print(f"\n[{event_name}] Running: {' '.join(bf_cmd)}")

        if not args.dry_run:
            result = subprocess.run(bf_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            # Extract Bayes factor
            for line in result.stdout.split('\n'):
                if 'ln BF_10' in line or 'log10 BF_10' in line:
                    print(f"\n>>> RESULT: {line}")

    # Summary
    print(f"\n{'='*70}")
    print("INJECTION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\nInjection data created in: {args.outdir}")
    print(f"\nGo/no-go criteria:")
    print(f"  H0 injections: Should have log10(BF) < 1 (no false echo detection)")
    print(f"  H1 injections: Should have log10(BF) > 2 and recover parameters")
    print(f"\nRun with --dry-run=False to execute analysis")


if __name__ == "__main__":
    main()
