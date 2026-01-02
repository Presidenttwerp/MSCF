import numpy as np
import bilby

from .waveforms import ringdown_fd, ringdown_fd_qnm, ringdown_plus_echo_fd

class GaussianFDLikelihood(bilby.core.likelihood.Likelihood):
    """
    Gaussian frequency-domain likelihood for complex rFFT data.

    Uses the Whittle likelihood for one-sided PSD:
      ln L = -2 * sum( |d(f) - h(f)|^2 / (S_n(f) * df) )

    This matches the noise generation convention where:
      noise_fft has E[|n(f)|^2] = S_n(f) * df

    Each complex frequency bin contributes 2 degrees of freedom (Re and Im),
    so chi^2 = 2 * sum(|d-h|^2 / (S_n * df)) has E[chi^2] = 2 * N_bins.

    FFT convention: X(f) = rfft(x) * dt (continuous FT approximation)
    PSD convention: one-sided, so S_n has units of 1/Hz
    """

    def __init__(self, t, data_dict, psd_dict, model="H0_ringdown", fmin=30.0, fmax=1024.0,
                 window=None, planck_eps_start=0.01, planck_eps_end=0.1, N_window=None):
        super().__init__(parameters={})
        self.t = np.asarray(t, dtype=float)
        self.data = data_dict  # dict: ifo -> (f, d_f)
        self.psd = psd_dict    # dict: ifo -> (f, Sn_f)
        self.model = model
        self.fmin = fmin  # Configurable frequency band limits
        self.fmax = fmax
        self.window = window  # Window type for waveform generation
        self.planck_eps_start = planck_eps_start
        self.planck_eps_end = planck_eps_end
        self.N_window = N_window  # Window length for zero-padded data
        self.nan_inf_count = 0  # Track how often we hit invalid values
        self.eval_count = 0     # Track total likelihood evaluations
        self.debug_max_prints = 10  # Limit debug output
        self.debug_prints_done = 0

        # Track which term caused the failure
        self.fail_counts = {"data": 0, "psd": 0, "waveform": 0, "resid": 0, "logL": 0}

        # sanity: ensure same freq grid between data and psd per ifo
        for ifo in self.data:
            f_d, d = self.data[ifo]
            f_p, Sn = self.psd[ifo]
            if len(f_d) != len(f_p) or np.max(np.abs(f_d - f_p)) > 1e-9:
                raise ValueError(f"Frequency grid mismatch for {ifo} between data and psd.")

        # Check data/PSD for issues at init time
        for ifo in self.data:
            f_d, d = self.data[ifo]
            f_p, Sn = self.psd[ifo]
            if not np.all(np.isfinite(d)):
                print(f"WARNING: {ifo} data contains NaN/Inf at init!")
            if not np.all(np.isfinite(Sn)):
                print(f"WARNING: {ifo} PSD contains NaN/Inf at init!")
            if np.any(Sn <= 0):
                print(f"WARNING: {ifo} PSD has non-positive values! min={np.min(Sn)}")

    def _waveform(self, f_grid):
        p = self.parameters
        if self.model == "H0_ringdown":
            # GR-consistent: derive f0, tau from (Mf, chi)
            f, H = ringdown_fd_qnm(
                self.t, p["A"], p["Mf"], p["chi"], p["phi"], p["t0"],
                window=self.window,
                planck_eps_start=self.planck_eps_start,
                planck_eps_end=self.planck_eps_end,
                N_window=self.N_window
            )
        elif self.model == "H1_echo":
            # H1 uses same GR ringdown + echo transfer function
            params = {k: p[k] for k in ["A","Mf","chi","phi","t0","R0","f_cut","roll","phi0"]}
            f, H = ringdown_plus_echo_fd(
                self.t, params,
                window=self.window,
                planck_eps_start=self.planck_eps_start,
                planck_eps_end=self.planck_eps_end,
                N_window=self.N_window
            )
        else:
            raise ValueError("Unknown model")
        # ensure f matches provided grid
        if len(f) != len(f_grid) or np.max(np.abs(f - f_grid)) > 1e-9:
            raise ValueError("Internal frequency grid mismatch (check dt and N).")
        return H

    def _debug_print(self, msg):
        """Print debug message if under limit."""
        if self.debug_prints_done < self.debug_max_prints:
            print(msg)
            self.debug_prints_done += 1

    def log_likelihood(self):
        self.eval_count += 1
        logL = 0.0
        for ifo in self.data:
            f, d = self.data[ifo]
            _, Sn = self.psd[ifo]

            # drop f=0 bin
            f = f[1:]
            d = d[1:]
            Sn = Sn[1:]

            # Band-limit: use configurable frequency bounds
            band_mask = (f >= self.fmin) & (f <= self.fmax)

            df = f[1] - f[0]
            h_full = self._waveform(np.concatenate(([0.0], f)))  # build on full grid, then slice
            h = h_full[1:]

            # Apply band mask to all arrays
            f_band = f[band_mask]
            d_band = d[band_mask]
            Sn_band = Sn[band_mask]
            h_band = h[band_mask]

            # Check data first (shouldn't fail if init checks passed)
            if not np.all(np.isfinite(d_band)):
                self.nan_inf_count += 1
                self.fail_counts["data"] += 1
                self._debug_print(f"FAIL[data] {ifo}: params={self.parameters}")
                return -np.inf

            # Guard against NaN/Inf in waveform
            if not np.all(np.isfinite(h_band)):
                self.nan_inf_count += 1
                self.fail_counts["waveform"] += 1
                n_bad = np.sum(~np.isfinite(h_band))
                self._debug_print(
                    f"FAIL[waveform] {ifo}: {n_bad}/{len(h_band)} non-finite, "
                    f"|h| range=[{np.nanmin(np.abs(h_band)):.2e}, {np.nanmax(np.abs(h_band)):.2e}], "
                    f"params={self.parameters}"
                )
                return -np.inf

            # Guard against bad PSD
            if not np.all(np.isfinite(Sn_band)) or np.any(Sn_band <= 0):
                self.nan_inf_count += 1
                self.fail_counts["psd"] += 1
                self._debug_print(
                    f"FAIL[psd] {ifo}: Sn range=[{np.nanmin(Sn_band):.2e}, {np.nanmax(Sn_band):.2e}], "
                    f"non-finite={np.sum(~np.isfinite(Sn_band))}, <=0={np.sum(Sn_band <= 0)}"
                )
                return -np.inf

            resid = d_band - h_band
            if not np.all(np.isfinite(resid)):
                self.nan_inf_count += 1
                self.fail_counts["resid"] += 1
                self._debug_print(f"FAIL[resid] {ifo}: params={self.parameters}")
                return -np.inf

            # Whittle likelihood: divide by (S_n * df) to match noise generation convention
            # where E[|n(f)|^2] = S_n * df. This gives E[chi^2] = 2 * N_bins.
            logL += -2.0 * np.sum((np.abs(resid)**2) / (Sn_band * df))

        # Final check on logL
        if not np.isfinite(logL):
            self.nan_inf_count += 1
            self.fail_counts["logL"] += 1
            self._debug_print(
                f"FAIL[logL] logL={logL}, params={self.parameters}"
            )
            return -np.inf

        return float(logL)
