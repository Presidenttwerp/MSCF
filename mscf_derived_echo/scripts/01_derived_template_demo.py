#!/usr/bin/env python3
"""
Demonstrate the zero-free-parameter derived echo template.

Computes and plots the template for GW150914-like parameters,
showing how the greybody-derived amplitudes differ from the
old ad-hoc model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mscf_derived.config import RESULTS_DIR, MSUN_S
from mscf_derived.derived_template import (
    derived_interference_template,
    mscf_echo_delay_seconds,
    qnm_220_freq_tau,
)
from mscf_derived.derived_amplitudes import derived_echo_amplitudes_at_freq
from mscf_derived.greybody import get_greybody_interpolator


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # GW150914 parameters
    Mf = 62.0
    chi = 0.67
    N_echo = 6

    dt = mscf_echo_delay_seconds(Mf, chi)
    f0, tau = qnm_220_freq_tau(Mf, chi)
    Delta_f = 1.0 / dt  # Spectral ripple spacing

    print("=" * 60)
    print("DERIVED ECHO TEMPLATE: GW150914")
    print("=" * 60)
    print(f"  Mf = {Mf} Msun, chi = {chi}")
    print(f"  f_QNM = {f0:.1f} Hz, tau = {tau*1000:.3f} ms")
    print(f"  dt_echo = {dt*1000:.4f} ms")
    print(f"  Spectral ripple spacing = {Delta_f:.0f} Hz")
    print(f"  M omega at QNM = {2*np.pi*f0*Mf*MSUN_S:.4f}")

    # Compute derived amplitudes at QNM frequency
    amps_qnm = derived_echo_amplitudes_at_freq(np.array([f0]), Mf, N_echo=N_echo)
    print(f"\n  Echo amplitudes at f_QNM = {f0:.0f} Hz:")
    for n in range(N_echo):
        print(f"    A_{n+1} = {amps_qnm[n]:.6f}")

    # Compare with old ad-hoc model (R1=0.05)
    R1_old = 0.05
    amps_old = [R1_old]
    base = (1.0 - R1_old)**2
    for n in range(2, N_echo + 1):
        amps_old.append(base * R1_old**(n-2))
    print(f"\n  Old model (R1={R1_old}) amplitudes:")
    for n in range(N_echo):
        print(f"    A_{n+1} = {amps_old[n]:.6f}")

    # Key physics difference
    print(f"\n  Key physics:")
    print(f"    Derived A1/A2 ratio: {amps_qnm[0]/amps_qnm[1]:.2f}")
    print(f"    Old model A1/A2 ratio: {amps_old[0]/amps_old[1]:.2f}")
    print(f"    (Derived: first echo always strongest, no inversion)")

    # Generate template
    freqs = np.linspace(20, 2048, 10000)
    template = derived_interference_template(freqs, Mf, chi, N_echo=N_echo)

    # Spectral modulation depth at QNM
    interp = get_greybody_interpolator(l=2)
    Momega_qnm = 2 * np.pi * f0 * Mf * MSUN_S
    Rb2_qnm = float(interp['Rb2_interp'](Momega_qnm))
    Tb_qnm = float(np.sqrt(interp['Tb2_interp'](Momega_qnm)))
    print(f"\n  Greybody at QNM: |R_b|^2 = {Rb2_qnm:.4f}, |T_b| = {Tb_qnm:.4f}")
    print(f"  Spectral modulation depth ~ 2*A1 = {2*amps_qnm[0]:.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # (a) Template power spectrum around QNM
        ax = axes[0]
        mask = (freqs > f0 - 200) & (freqs < f0 + 200)
        ax.plot(freqs[mask], np.abs(template[mask])**2, 'b-', linewidth=0.8)
        ax.axvline(f0, color='r', linestyle='--', alpha=0.5, label=f'$f_{{QNM}}$={f0:.0f} Hz')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'$|t(f)|^2$')
        ax.set_title(f'Derived echo template — GW150914 (Mf={Mf}, $\\chi$={chi})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (b) Transfer function: 1 + echo_sum shows spectral ripples
        amps = derived_echo_amplitudes_at_freq(freqs, Mf, N_echo=N_echo)
        echo_sum = np.zeros(len(freqs), dtype=np.complex128)
        for n in range(N_echo):
            k = n + 1
            tk = k * dt
            echo_sum += amps[n] * np.exp(1j * 2.0 * np.pi * freqs * tk)
        transfer = 1.0 + echo_sum

        ax = axes[1]
        mask2 = (freqs > 100) & (freqs < 600)
        ax.plot(freqs[mask2], np.abs(transfer[mask2])**2, 'b-', linewidth=0.5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'$|1 + \Sigma_n a_n e^{i 2\pi f n \Delta t}|^2$')
        ax.set_title(f'Echo transfer function (ripple spacing $\\Delta f$ = {Delta_f:.0f} Hz)')
        ax.grid(True, alpha=0.3)

        # (c) Echo amplitudes vs frequency
        ax = axes[2]
        f_plot = np.linspace(50, 800, 500)
        amps_plot = derived_echo_amplitudes_at_freq(f_plot, Mf, N_echo=N_echo)
        for n in range(min(4, N_echo)):
            ax.plot(f_plot, amps_plot[n], linewidth=1.5, label=f'$a_{n+1}(f)$')
        ax.axvline(f0, color='gray', linestyle=':', alpha=0.5, label='$f_{QNM}$')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Echo amplitude')
        ax.set_title('Frequency-dependent echo amplitudes (derived from greybody)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(RESULTS_DIR, 'derived_template_demo.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"\nPlot saved: {outfile}")

    except ImportError:
        print("\nmatplotlib not available — skipping plots")

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
