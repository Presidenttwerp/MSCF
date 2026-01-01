import numpy as np
import bilby

from .waveforms import ringdown_fd, ringdown_plus_echo_fd

class GaussianFDLikelihood(bilby.core.likelihood.Likelihood):
    """
    Simple Gaussian frequency-domain likelihood for complex rFFT data:

      ln L = -2 * sum( |d(f) - h(f)|^2 / S_n(f) ) * df   (up to additive constant)

    Assumes one-sided PSD S_n(f) and excludes f=0 bin automatically.
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
            f, H = ringdown_fd(self.t, p["A"], p["f0"], p["tau"], p["phi"], p["t0"])
        elif self.model == "H1_echo":
            params = {k: p[k] for k in ["A","f0","tau","phi","t0","Mf","chi","R0","f_cut","roll","phi0"]}
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

            df = f[1] - f[0]
            h = self._waveform(np.concatenate(([0.0], f)))  # build on full grid, then slice
            h = h[1:]

            resid = d - h
            logL += -2.0 * np.sum((np.abs(resid)**2) / Sn) * df
        return float(logL)
