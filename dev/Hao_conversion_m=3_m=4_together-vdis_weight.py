import os
import numpy as np
import matplotlib.pyplot as plt

params = {}

vdis = np.loadtxt('./Data_Hao2006/veldis_corrected.dat')
params['vdis'] = vdis[:] #[inds_keep]

phot = np.loadtxt('./Data_Hao2006/photometry.dat')
param_names = ['Re', '_a3/a', '_delta_a3/a', '_a4/a', '_delta_a4/a', # 1-5
               'ellip', 'delta_ellip', '_b3/a', '_delta_b3/a', '_b4/a', # 6-10
               '_delta_b4/a',
              ]
# Full parameters
# Re	a3mean	a3mean_err	a4mean	a4mean_err	ellipmean	ellipmean_err	b3mean	b3mean_err	b4mean	b4mean_err	a3twist	a3twist_err	a3grad	a3grad_err	a4twist	a4twist_err	a4grad	a4grad_err	elliptwist	elliptwist_err	ellipgrad	ellipgrad_err	patwist	patwist_err	pagrad	pagrad_err	xtwist	xtwist_err	xgrad	xgrad_err	ytwist	ytwist_err	ygrad	ygrad_err	b3re	b3re_err	b4re	b4re_err	ellipre	ellipre_err	pare	pare_err	vdisp	vdisp_err	peR50	peR50_err	deVRad	deVRadErr	peMag_r	peMagErr_r	peMag_g	peMagErr_g	peMag_i	peMagErr_i	peR90	peR90_err	z	run	rerun	camcol	field	colci

for i, param_name in enumerate(param_names):
    params[param_name]=phot[:,i]
params['pare'] = phot[:,41] # Position angle at PeR50 (Petrosian Half-light Radius)
params['q'] = 1 - params['ellip']
params['delta_q'] = params['delta_ellip']

#%%

from util import params_multipole_field_update, params_nan_clear
# This adds the following keys and matching values to "params" dictionary
# phi3, phi4, a3/a, a4/a, delta_a3/a, delta_a4/a, delta_phi_3, delta_phi_4
# phi3 means phi_3-phi_0
# phi4 means phi_4-phi_0
params = params_multipole_field_update(params)
params = params_nan_clear(params)

#%%
from grid_info_module import grid_info
# grid_info class holds min, max, range, and delta values of (X,Y,Z)
# here X corredponds to a_m/a, Y corresponds to phi_m-phi_0

#%%
from util import observation_data
import pickle

m_list = [3,4]
for m in m_list:
    print(f"=== m={m} ===")
    hao_data = observation_data(params, m, vdis_weight=True)

    # x min and x max range is set to be min - 3*sigma, max + 3*sigma, where the sigma is the data point's.
    x_min_arg_num = np.argmin(hao_data.x_expanded); x_max_arg_num = np.argmax(hao_data.x_expanded)
    n_sigma = 3
    x_min = (hao_data.x_expanded - n_sigma * hao_data.dx_expanded).min()
    x_max = (hao_data.x_expanded + n_sigma * hao_data.dx_expanded).max()
    x_min = np.sign(x_min) * np.round(np.abs(x_min), decimals=2); x_max = np.sign(x_max) * np.round(x_max, decimals=2)
    print(f"x_min: {x_min}", f"x_max: {x_max}")

    y_min = - np.pi/2/m
    y_max = + np.pi/2/m
    print(f"y_min: {y_min}", f"y_max: {y_max}")

    z_min = (hao_data.z_expanded - n_sigma * hao_data.dz_expanded).min()
    z_min = np.sign(z_min) * np.round(np.abs(z_min), decimals=2)
    z_max = 1.
    print(f"z_min: {z_min}", f"z_max: {z_max}")

    Y_name = f'\Delta\phi_{m}'
    X_name = f'a_{m}/a'
    Z_name = 'q'

    n_grid = 100
    my_grid = grid_info(n_grid, x_min, x_max, y_min, y_max, z_min, z_max)

    from util import prob_dist_Hao_3D
    P_XYZ_Hao = prob_dist_Hao_3D(my_grid, hao_data, m)

    filename_for_torch = f'./Data_Hao2006_processed/data_for_torch_fitting_m={m}_vdis_4_weight.pkl'
    with open(filename_for_torch, "wb") as file:  # Use 'rb' mode for reading in binary
        pickle.dump([my_grid, P_XYZ_Hao], file)
    print("saved: ", filename_for_torch)

print("done!")

#%%
from scipy import stats
from sklearn.linear_model import LinearRegression

def partial_corr(x, y, z):
    """
    Compute the partial correlation between y and z given x.
    This tests whether Y and Z are conditionally independent given X.

    Arguments:
    x, y, z -- numpy arrays of shape (n_samples,)

    Returns:
    r_partial -- partial correlation coefficient
    p_value -- two-tailed p-value for testing non-zero correlation
    """
    x = x.reshape(-1, 1)

    # Regress y ~ x and z ~ x, get residuals
    reg_y = LinearRegression().fit(x, y)
    y_resid = y - reg_y.predict(x)

    reg_z = LinearRegression().fit(x, z)
    z_resid = z - reg_z.predict(x)

    plt.plot(y_resid, z_resid, '.')
    plt.xlabel('y residual')
    plt.ylabel('z residual')
    plt.show()

    # Correlate the residuals
    r_partial, p_value = stats.pearsonr(y_resid, z_resid)
    return r_partial, p_value

r_partial, p_val = partial_corr(params['a4/a'], params['phi4'], params['q'])
print(f"Partial correlation between phi4 and q given a4/a: {r_partial:.3f}, p = {p_val:.3g}")
