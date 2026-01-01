import numpy as np
import bilby

from .waveforms import ringdown_fd, ringdown_fd_qnm, ringdown_plus_echo_fd

class GaussianFDLikelihood(bilby.core.likelihood.Likelihood):
    """
    Gaussian frequency-domain likelihood for complex rFFT data.

    Uses the Whittle likelihood for one-sided PSD:
      ln L = -4 * sum( |d(f) - h(f)|^2 / S_n(f) ) * df

    The factor of 4 comes from the standard GW inner product:
      <a|b> = 4 Re int_0^inf a*(f) b(f) / S_n(f) df
    and ln L = -1/2 <d-h|d-h> = -2 Re int |d-h|^2 / S_n df

    For complex data (rfft output), |d-h|^2 already includes both Re and Im,
    so we use factor 4 to match the standard normalization.

    FFT convention: X(f) = rfft(x) * dt (continuous FT approximation)
    PSD convention: one-sided, so S_n has units of 1/Hz
    """

    def __init__(self, t, data_dict, psd_dict, model="H0_ringdown"):
        super().__init__(parameters={})
        self.t = np.asarray(t, dtype=float)
        self.data = data_dict  # dict: ifo -> (f, d_f)
        self.psd = psd_dict    # dict: ifo -> (f, Sn_f)
        self.model = model

        # sanity: ensure same freq grid between data and psd per ifo
        for ifo in self.data:
            f_d, d = self.data[ifo]
            f_p, Sn = self.psd[ifo]
            if len(f_d) != len(f_p) or np.max(np.abs(f_d - f_p)) > 1e-9:
                raise ValueError(f"Frequency grid mismatch for {ifo} between data and psd.")

    def _waveform(self, f_grid):
        p = self.parameters
        if self.model == "H0_ringdown":
            # GR-consistent: derive f0, tau from (Mf, chi)
            f, H = ringdown_fd_qnm(self.t, p["A"], p["Mf"], p["chi"], p["phi"], p["t0"])
        elif self.model == "H1_echo":
            # H1 uses same GR ringdown + echo transfer function
            params = {k: p[k] for k in ["A","Mf","chi","phi","t0","R0","f_cut","roll","phi0"]}
            f, H = ringdown_plus_echo_fd(self.t, params)
        else:
            raise ValueError("Unknown model")
        # ensure f matches provided grid
        if len(f) != len(f_grid) or np.max(np.abs(f - f_grid)) > 1e-9:
            raise ValueError("Internal frequency grid mismatch (check dt and N).")
        return H

    def log_likelihood(self):
        logL = 0.0
        for ifo in self.data:
            f, d = self.data[ifo]
            _, Sn = self.psd[ifo]

            # drop f=0 bin
            f = f[1:]
            d = d[1:]
            Sn = Sn[1:]

            # Band-limit: only fit 30-1024 Hz
            fmin, fmax = 30.0, 1024.0
            band_mask = (f >= fmin) & (f <= fmax)

            df = f[1] - f[0]
            h_full = self._waveform(np.concatenate(([0.0], f)))  # build on full grid, then slice
            h = h_full[1:]

            # Apply band mask to all arrays
            f = f[band_mask]
            d = d[band_mask]
            Sn = Sn[band_mask]
            h = h[band_mask]

            resid = d - h
            # Factor of 4 for standard GW inner product with one-sided PSD
            logL += -4.0 * np.sum((np.abs(resid)**2) / Sn) * df
        return float(logL)
