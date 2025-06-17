from numba import njit
from lenstronomy.Util.param_util import ellipticity2phi_q
import numpy as np
from multipoleprior import load_default_params
from multipoleprior.DistributionModel import DistributionModel_nontorch

params = load_default_params()
model = DistributionModel_nontorch(params)

@njit
def log_prior_q_phi(e1, e2):
    e = np.sqrt(e1 * e1 + e2 * e2)
    if e >= 1.0:
        return -np.inf
    eps = 1e-12
    return np.log(1.0 / np.pi / ((e + eps) * (1 + e)**2))

def multipole_logL(**kwargs):
    # fast part
    lp = log_prior_q_phi(kwargs['kwargs_lens'][0]['e1'],
                          kwargs['kwargs_lens'][0]['e2'])
    # slow part (model call remains in Python)
    L2 = model.prob_single_point(
        x3=kwargs['kwargs_lens'][0]['a3_a'],
        y3=kwargs['kwargs_lens'][0]['delta_phi_m3'],
        x4=kwargs['kwargs_lens'][0]['a4_a'],
        y4=kwargs['kwargs_lens'][0]['delta_phi_m4'],
        z=ellipticity2phi_q(kwargs['kwargs_lens'][0]['e1'],
                            kwargs['kwargs_lens'][0]['e2'])[1]
    )
    if L2 <= 0.0:
        return lp + -np.inf
    return lp + np.log(L2)