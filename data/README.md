# External Data

## Planck 2018 Commander TT

The script `code/cmb/planck_commander_likelihood.py` requires the Planck 2018
low-ell TT Commander likelihood data. Install via cobaya:

```
pip install cobaya
cobaya-install planck_2018_lowl.TT --packages-path ~/cobaya_packages
```

Expected path after installation:

```
~/cobaya_packages/data/planck_2018_lowT_native/
```

Required files within that directory:

- `cov.txt` (covariance matrix)
- `mu.txt` (Gaussianized means)
- `mu_sigma.txt` (offset spectrum)
- `cl2x_1.txt` (Gaussianization spline x-values)
- `cl2x_2.txt` (Gaussianization spline y-values)

An alternative path can be specified at runtime:

```
python3 code/cmb/planck_commander_likelihood.py --planck-data /path/to/planck_2018_lowT_native
```

All other scripts in this repository are self-contained and require no external
data beyond the Python packages listed in `code/requirements.txt`.
