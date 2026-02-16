"""
Cepstral / spectral-ratio analysis for echo detection.

Complementary to matched filtering: looks for periodic spectral modulation
at the echo repetition frequency Δf = 1/dt_echo.

Method:
1. Compute ringdown power spectrum |d(f)|²
2. Whiten: R(f) = |d(f)|² / S_n(f)
3. Compute power cepstrum: C(τ) = |IFFT(log R(f))|²
4. Look for peak at τ = dt_echo

The key advantage: this is a model-agnostic test for spectral ripples
at the MSCF-predicted spacing. No template needed (except the delay).
"""

import numpy as np


def whitened_power_spectrum(fft_data, psd, freqs, fmin=20.0, fmax=2048.0):
    """
    Compute whitened power spectrum R(f) = |d(f)|² / S_n(f).

    Parameters
    ----------
    fft_data : np.ndarray
        Complex FFT of ringdown data.
    psd : np.ndarray
        PSD estimate.
    freqs : np.ndarray
        Frequency array [Hz].
    fmin, fmax : float
        Frequency band.

    Returns
    -------
    freqs_cut : np.ndarray
        Frequencies in band.
    R : np.ndarray
        Whitened power spectrum.
    """
    mask = (freqs >= fmin) & (freqs <= fmax) & (psd > 0) & np.isfinite(psd)
    freqs_cut = freqs[mask]
    R = np.abs(fft_data[mask])**2 / psd[mask]
    return freqs_cut, R


def power_cepstrum(R, df):
    """
    Compute power cepstrum C(τ) = |IFFT(log R(f))|².

    Parameters
    ----------
    R : np.ndarray
        Whitened power spectrum (real, positive).
    df : float
        Frequency resolution [Hz].

    Returns
    -------
    quefrency : np.ndarray
        Quefrency axis [seconds].
    C : np.ndarray
        Power cepstrum values.
    """
    # Avoid log(0)
    R_safe = np.clip(R, 1e-30, None)
    log_R = np.log(R_safe)

    # Remove mean to suppress DC component
    log_R = log_R - np.mean(log_R)

    # IFFT
    c = np.fft.irfft(log_R)
    C = np.abs(c)**2

    # Quefrency axis
    n = len(log_R)
    quefrency = np.arange(len(C)) / (n * df)

    return quefrency, C


def cepstrum_snr(quefrency, C, dt_target, window_width=None):
    """
    Compute SNR of cepstral peak at target delay.

    Parameters
    ----------
    quefrency : np.ndarray
        Quefrency axis [seconds].
    C : np.ndarray
        Power cepstrum.
    dt_target : float
        Expected echo delay [seconds].
    window_width : float, optional
        Half-width of signal window [seconds]. Default: 3 quefrency bins.

    Returns
    -------
    dict with keys:
        'snr' : float — peak value / median
        'peak_value' : float — cepstrum at dt_target
        'peak_quefrency' : float — quefrency of maximum near dt_target
        'background_median' : float
        'background_std' : float
    """
    dq = quefrency[1] - quefrency[0] if len(quefrency) > 1 else 1e-6

    if window_width is None:
        window_width = 3 * dq

    # Signal window around dt_target
    sig_mask = np.abs(quefrency - dt_target) <= window_width
    if not np.any(sig_mask):
        # dt_target outside range
        return {
            'snr': 0.0,
            'peak_value': 0.0,
            'peak_quefrency': dt_target,
            'background_median': np.median(C),
            'background_std': np.std(C),
        }

    peak_idx = np.argmax(C * sig_mask.astype(float))
    peak_value = C[peak_idx]
    peak_q = quefrency[peak_idx]

    # Background: exclude signal region and its harmonics
    bg_mask = np.ones(len(C), dtype=bool)
    for harm in range(1, 4):
        bg_mask &= np.abs(quefrency - harm * dt_target) > 2 * window_width
    # Also exclude quefrency < 5*dq (DC region)
    bg_mask &= quefrency > 5 * dq

    if np.sum(bg_mask) < 10:
        bg_mask = quefrency > 5 * dq

    bg_median = np.median(C[bg_mask])
    bg_std = np.std(C[bg_mask])

    snr = (peak_value - bg_median) / bg_std if bg_std > 0 else 0.0

    return {
        'snr': snr,
        'peak_value': peak_value,
        'peak_quefrency': peak_q,
        'background_median': bg_median,
        'background_std': bg_std,
    }


def spectral_modulation_fit(freqs, R, dt_target, Mf_msun=None, l=2):
    """
    Fit the MSCF spectral modulation model to whitened data.

    Model: R(f) ≈ R_mean × [1 + 2ε(f)cos(2πf·dt + φ)]

    where ε(f) is the frequency-dependent modulation depth from greybody.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array [Hz].
    R : np.ndarray
        Whitened power spectrum.
    dt_target : float
        Echo delay [seconds].
    Mf_msun : float, optional
        If provided, uses greybody-derived ε(f). Otherwise uses flat ε.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    dict with keys:
        'epsilon' : float — best-fit modulation depth
        'phi' : float — best-fit phase
        'chi2' : float — reduced chi²
        'snr' : float — modulation significance
    """
    R_mean = np.mean(R)
    R_norm = R / R_mean - 1.0  # Fractional excess

    # Phase array: 2πf·dt
    phase = 2.0 * np.pi * freqs * dt_target

    # If greybody available, use frequency-dependent modulation
    if Mf_msun is not None:
        from .derived_amplitudes import derived_echo_amplitudes_at_freq
        amps = derived_echo_amplitudes_at_freq(freqs, Mf_msun, N_echo=1, l=l)
        eps_model = amps[0]  # First echo amplitude = T_b
    else:
        eps_model = np.ones_like(freqs)

    # Linear fit: R_norm ≈ A·cos(phase) + B·sin(phase)
    # where A = 2ε·cos(φ), B = -2ε·sin(φ)
    cos_term = eps_model * np.cos(phase)
    sin_term = eps_model * np.sin(phase)

    # Solve via normal equations
    X = np.column_stack([cos_term, sin_term])
    XtX = X.T @ X
    Xty = X.T @ R_norm

    try:
        coeffs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {'epsilon': 0.0, 'phi': 0.0, 'chi2': np.inf, 'snr': 0.0}

    A, B = coeffs
    epsilon = 0.5 * np.sqrt(A**2 + B**2)
    phi = np.arctan2(-B, A)

    # Residual
    R_model = A * cos_term + B * sin_term
    residual = R_norm - R_model
    chi2 = np.sum(residual**2) / max(1, len(R_norm) - 2)

    # Significance
    null_chi2 = np.sum(R_norm**2) / max(1, len(R_norm))
    if null_chi2 > 0:
        snr = np.sqrt(max(0, (null_chi2 - chi2) * len(R_norm)))
    else:
        snr = 0.0

    return {
        'epsilon': epsilon,
        'phi': phi,
        'chi2': chi2,
        'snr': snr,
    }
