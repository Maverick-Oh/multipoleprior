# multipoleprior

A physically-motivated prior for elliptical gravitational lenses with multipole perturbations, implemented as a pure-NumPy module and designed for use with [lenstronomy](https://github.com/lenstronomy/lenstronomy).

This package enables joint priors over the lens ellipticity and multipole terms (m=3, m=4), based on externally derived constraints from galaxy structure models. It is lightweight, portable, and easily pluggable into any Lenstronomy likelihood function.

This is a supplemantary code for "Joint Semi-Analytic Multipole Priors from Galaxy Isophotes and Their Constarints from Lensed Arcs" by Maverick S. H. Oh, Anna Nierenberg, Daniel Gilman, and Simon Birrer, submitted to Journal of Cosmology and Astroparticle Physics (JCAP). Please cite this work if you use this package.



## Features

- Drop-in support for [`EPL_MULTIPOLE_M3M4`](https://lenstronomy.readthedocs.io/en/latest/lens_model_profiles.html#epl-multipole) lens models
- Supports easy loading of multipole prior through custom log-priors
- Physically constrained distributions on:
  - Ellipticity (`e1`, `e2`)
  - Multipole amplitudes (`a3_a`, `a4_a`)
  - Multipole phases (`delta_phi_m3`, `delta_phi_m4`)
- No PyTorch or TensorFlow required — 100% NumPy/Scipy



## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/Maverick-Oh/multipoleprior.git
```

Or, you can download it and use it as a develop mode.

```bash
git clone https://github.com/Maverick-Oh/multipoleprior.git
cd multipoleprior
python setup.py develop
```



## How to Use

Once installed, simply import the multipole prior and add it to your `kwargs_likelihood`:

```python
from multipoleprior.multipole_prior import multipole_logL

kwargs_likelihood = {
    'custom_logL_addition': multipole_logL
}
```

This allows Lenstronomy to include the multipole prior in its log-likelihood evaluations.

For a complete working example, see:
 `example/example_inference_with_multipoles.ipynb`



## Development and Research Code

The `dev/` directory contains code used in the development and testing of this package. 

These files are **not required** to use the package, but are provided for reference, reproducibility, and transparency. 

This includes exploratory notebooks, model-building scripts, and prior visualizations.

Feel free to explore or adapt them for your own work!



The codes here do the followings:

1) Convert the circular multipoles of 840 S/E0 galaxies measured by Hao et al. 2006 ([link](https://academic.oup.com/mnras/article/370/3/1339/1156586)) data from amplitudes of cosine and sine $(a_m\!/\!a, b_m\!/\!a)$ basis to amplitude and phase $(a_m\!/\!a, \phi_m\!-\!\phi_0)$ basis for $m=3, 4$.
2) Fit a semi-analytic function to the joint distribution of observed $(a_3\!/\!a, \phi_3\!-\!\phi_0, a_4\!/\!a, \phi_4\!-\!\phi_0, q)$ to provide a semi-analytic prior of multipole parameters with axis ratio $q$.
