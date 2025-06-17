import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.special import gamma

#
def filename_slash_replacer(my_string):
    assert type(my_string) == str
    return my_string.replace('/', '_').replace('\\', '_')


class observation_data():
    def __init__(self, params, m, vdis_weight=False):
        # self.params = params
        self.m = m
        self.x, self.y, self.z, self.dx, self.dy, self.dz = params2xyzdxdydz(params, m=m)
        if vdis_weight:
            self.weight = params['vdis']**4
        else:
            self.weight = np.ones_like(self.x)
        assert len(self.x) == len(self.y) == len(self.z) == len(self.dx)
        self.num_datapoints = len(self.x)  # number of data points before expansion

        # expansion for y +/- 3*dy uncertainty going over (-pi/2m, pi/2m)
        cutoff_sigma = 3
        self.x_expanded, self.y_expanded, self.z_expanded, self.dx_expanded, self.dy_expanded, \
            self.dz_expanded, self.x_weight, self.y_weight, self.z_weight, self.weight_expanded = self.expanded_xyzdxdydz(cutoff_sigma)

    def expanded_xyzdxdydz(self, cutoff_sigma=3):
        # assert dx>0, dy>0
        # for each (x,y), grid_off = max((y + 3 * dy) / (pi/(2*m)), abs(y - 3 * dy) / (pi/(2*m)))
        # if grid_off is within (0, 1), x_weight=1, y_weight=1, pass
        # if grid_off is within (1,2), make two other data pair (-x, y + pi/2m), (-x, y - pi/2m).
        # y_weight = 1 for both old and new datapoints, but x_weight can be calculated by
        # dist = norm(loc=y, scale=dy); dist.cdf(pi/2m) - dist.cdf(-pi/2m)
        # if grid_off is within (2,3), make two other data pair (x, y + 2 * pi/2m), (x, y - 2 * pi/2m)
        # ...

        x_expanded = []
        dx_expanded = []
        x_weight = []
        #
        y_expanded = []
        dy_expanded = []
        y_weight = []
        #
        z_expanded = []
        dz_expanded = []
        z_weight = []
        #
        weight_expanded = []

        assert (self.dx > 0).all()
        assert (self.dy > 0).all()
        assert (self.dz > 0).all()

        y_lim = np.pi / 2 / self.m  # y should be within (-y_lim, +y_lim)
        for xi, dxi, yi, dyi, zi, dzi, wi in zip(self.x, self.dx, self.y, self.dy, self.z, self.dz, self.weight):
            grid_off = max((yi + cutoff_sigma * dyi) / y_lim, abs(yi - cutoff_sigma * dyi) / y_lim)
            grid_off_int = int(np.floor(grid_off))
            # if grid_off_int is 0, make [0]
            # if grid_off_int is 1, make [-1, 0, +1]
            # if grid_off_int is 2, make [-2, -1, 0, +1, +2]
            for i_off in np.arange(-grid_off_int, grid_off_int + 1):
                if i_off % 2 == 0:  # if i_off is an even number
                    x_expanded.append(xi)
                else:
                    x_expanded.append(-xi)
                y_expanded.append(yi + 2 * y_lim * i_off)
                z_expanded.append(zi)
                dx_expanded.append(dxi)
                dy_expanded.append(dyi)
                dz_expanded.append(dzi)
                #
                # if i_off = 0: dist.cdf(+pi/2m) - dist.cdf(-pi/2m)
                # if i_off = 1: dist.cdf(+pi/2m + pi/m) - dist.cdf(-pi/2m + pi/m)
                # if i_off = -1: dist.cdf(+pi/2m - pi/m) - dist.cdf(-pi/2m - pi/m)
                dist_y = norm(loc=yi, scale=dyi)
                x_weight.append(dist_y.cdf(y_lim + 2 * i_off * y_lim) - dist_y.cdf(-y_lim + 2 * i_off * y_lim))
                y_weight.append(1.0)  # such that when integrated from -pi/2m to pi/2m, it will be 1
                z_weight.append(dist_y.cdf(y_lim + 2 * i_off * y_lim) - dist_y.cdf(-y_lim + 2 * i_off * y_lim))
                weight_expanded.append(wi)
        x_expanded = np.array(x_expanded)
        y_expanded = np.array(y_expanded)
        z_expanded = np.array(z_expanded)
        dx_expanded = np.array(dx_expanded)
        dy_expanded = np.array(dy_expanded)
        dz_expanded = np.array(dz_expanded)
        x_weight = np.array(x_weight)
        y_weight = np.array(y_weight)
        z_weight = np.array(z_weight)
        weight_expanded = np.array(weight_expanded)
        return x_expanded, y_expanded, z_expanded, dx_expanded, dy_expanded, dz_expanded, x_weight, y_weight, z_weight, weight_expanded


#%% Uncertainty quantification
# For z = f(x,y),
# dz = ∂f/∂x dx + ∂f/∂y dy
# (dz)^2 = (∂f/∂x dx)^2 + (∂f/∂y dy)^2 + ∂f/∂x ∂f/∂y dx dy
# If Δx and Δy are independent, the last term can be ignored (because on average they will cancel out)
# and the uncertainty of z can be written as:
# Δz = sqrt( (∂f/∂x Δx)^2+(∂f/∂y Δy)^2 )
#
# In our case, where a4/a = sign(a4'/a) * sqrt(a4'/a**2 + params[b4'/a]**2),
# where a4'/a and b4'/a are from Hao's convention (cos and sin) and a4/a is our convention (amplitude and phase)
# and we know the uncertainties of Hao's measurements a4'/a and b4'/a.
# When _a4/a is big, _b4/a will be small because of the diskyness not aligning, so there is a covariance going on.
# So, we have to evaluate the following.
# Δz^2 = (∂f/∂x Δx)^2+(∂f/∂y Δy)^2+ 2 ∂f/∂x ∂f/∂y Cov(x,y)
# or equivalently
# Δz = sqrt( (∂f/∂x Δx)^2+(∂f/∂y Δy)^2+ 2 ∂f/∂x ∂f/∂y Cov(x,y) )

# calculate the covariance between two variables

def my_cov_two_variables(x,y):
    xy = np.stack((x,y), axis=0)
    my_cov_array = np.cov(xy)
    # double-check my_cov is symetric
    assert (my_cov_array.T == my_cov_array).all()
    cov_xy = my_cov_array[0,1] # covariance of x and y
    return cov_xy

# f(x,y) = sign(x) * (x^2 + y^2)^(1/2), where x=_a4/a and y=_b4/a
# ∂f/∂x = sign(x) * (x^2+y^2)^(-1/2) * x
# ∂f/∂y = sign(x) * (x^2+y^2)^(-1/2) * y
# Δz = sqrt( (∂f/∂x Δx)^2+(∂f/∂y Δy)^2+ 2 ∂f/∂x ∂f/∂y Cov(x,y) )

def df_dx_and_df_dy(x,y):
    # only for f(x,y) = sign(x) * (x^2 + y^2)^(1/2)
    df_dx = np.sign(x) * (x**2+y**2)**(-1/2) * x
    df_dy = np.sign(x) * (x**2+y**2)**(-1/2) * y
    return (df_dx, df_dy)

def r_function(x, y):
    # sign(x) * sqrt(x^2 + y^2)
    return np.sign(x) * np.sqrt(x**2 + y**2)

def first_derivatives_r(x, y):
    # df/dx, df/dy at (x,y)
    denom = np.sqrt(x**2 + y**2)
    if denom == 0:
        # handle x=0=y carefully or skip
        return (np.nan, np.nan)

    df_dx = np.sign(x) * x / denom
    df_dy = np.sign(x) * y / denom
    return (df_dx, df_dy)

def second_derivatives_r(x, y):
    # d^2f/dx^2, d^2f/dxdy, d^2f/dy^2 at (x,y)
    denom = (x**2 + y**2)**1.5
    if denom == 0:
        # handle x=0=y carefully or skip
        return (np.nan, np.nan, np.nan)

    signx = np.sign(x)
    d2f_dx2  = signx * (y**2 / denom)
    d2f_dxdy = -signx * (x * y / denom)
    d2f_dy2  = signx * (x**2 / denom)
    return (d2f_dx2, d2f_dxdy, d2f_dy2)

def variance_r_second_order(mu_x, mu_y, var_x, var_y, cov_xy):
    """
    Approximate Var[r] up to second order around (mu_x, mu_y).
    """
    # 1) Gradient at (mu_x, mu_y)
    dfdx, dfdy = first_derivatives_r(mu_x, mu_y)
    grad = np.array([dfdx, dfdy])

    # 2) Hessian at (mu_x, mu_y)
    d2fdx2, d2fdxdy, d2fdy2 = second_derivatives_r(mu_x, mu_y)
    H = np.array([
        [d2fdx2,    d2fdxdy ],
        [d2fdxdy,   d2fdy2  ]
    ])

    # 3) Covariance matrix
    Sigma = np.array([
        [var_x,   cov_xy],
        [cov_xy,  var_y ]
    ])

    # First-order term: g^T Σ g
    first_order = grad @ Sigma @ grad

    # Second-order term: 1/2 trace(H Σ^2)
    # where Σ^2 means Σ dot Σ
    Sigma_sq = Sigma @ Sigma
    second_order = 0.5 * np.trace(H @ Sigma_sq)

    # Approximate total
    var_r = first_order + second_order

    return var_r

def variance_phi_second_order(mu_x, mu_y, var_x, var_y, cov_xy):
    """
    Second-order approximation of Var[phi], where:
      phi = arctan(y/x).
    Args:
      mu_x, mu_y : float
        The 'central' values of x and y (e.g., single measurement).
      var_x, var_y : float
        The (measurement) variances of x and y, i.e. (delta_x^2, delta_y^2).
      cov_xy : float
        Covariance of x and y (measurement covariance).
    Returns:
      var_phi : float
        The approximate variance of phi at (mu_x, mu_y).
    """
    # 1) Gradient of phi at (mu_x, mu_y).
    denom = (mu_x**2 + mu_y**2)
    # Safeguard if denom == 0:
    if denom == 0.0:
        return np.nan  # or handle gracefully

    phi_x = -mu_y / denom
    phi_y =  mu_x / denom
    grad = np.array([phi_x, phi_y])

    # 2) Hessian of phi at (mu_x, mu_y).
    denom2 = denom**2
    phi_xx = (2 * mu_x * mu_y) / denom2
    phi_xy = (mu_y**2 - mu_x**2) / denom2
    phi_yy = -2 * mu_x * mu_y / denom2
    H = np.array([
        [phi_xx, phi_xy],
        [phi_xy, phi_yy]
    ])

    # 3) Covariance matrix
    Sigma = np.array([
        [var_x,   cov_xy],
        [cov_xy,  var_y]
    ])
    Sigma_sq = Sigma @ Sigma

    # 4) Terms of the second-order formula
    first_order = grad @ Sigma @ grad
    second_order = 0.5 * np.trace(H @ Sigma_sq)

    var_phi = first_order + second_order
    return var_phi

def r_uncertainty(x, y, delta_x, delta_y):
    # only for r = f(x,y) = sign(x) * (x^2 + y^2)^(1/2)
    df_dx, df_dy = df_dx_and_df_dy(x,y)
    cov_xy = my_cov_two_variables(x,y)
    # first order variance
    var_r = (df_dx * delta_x) ** 2 + (df_dy * delta_y) ** 2 + 2 * df_dx * df_dy * cov_xy
    delta_r = np.zeros_like(var_r)
    for i, var_ri in enumerate(var_r):
        if var_ri >= 0:
            delta_r[i] = np.sqrt(var_ri)
        else:
            print(f"i={i}")
            print("var_ri is negative, so trying the second order uncertainty")
            # second order variance
            var_ri_2nd_order = variance_r_second_order(mu_x=x[i], mu_y=y[i], var_x=delta_x[i], var_y=delta_y[i],
                                                 cov_xy=cov_xy)
            if var_ri_2nd_order > 0:
                delta_r[i] = np.sqrt(var_ri_2nd_order)
                print(f"var_ri_2nd_order = {var_ri_2nd_order}")
            else:
                raise ValueError(f"Negative Variance\nvar_ri_2nd_order = {var_ri_2nd_order}")
    if np.isnan(delta_r).any() or np.isinf(delta_r).any():
        print("Pausing for debugging...")
        raise ValueError("Invalid value encountered in sqrt")
    return delta_r

def dg_dx_and_dg_dy(x,y):
    # only for g(x,y) = arctan(y/x)
    # note: d/dx (arctan(x)) = 1/(1+x^2)
    dg_dx = 1/(1+(y/x)**2) * (y * (-x**(-2)))
    dg_dy = 1/(1+(y/x)**2) * (1/x)
    return (dg_dx, dg_dy)

# def phi_uncertainty(x, y, delta_x, delta_y):
#     cov_xy = my_cov_two_variables(x, y)
#     dg_dx, dg_dy = dg_dx_and_dg_dy(x, y)
#     delta_phi = np.sqrt((dg_dx * delta_x) ** 2 + (dg_dy * delta_y) ** 2 + 2 * dg_dx * dg_dy * cov_xy)
#     return delta_phi

def phi_uncertainty(x, y, delta_x, delta_y):
    """
    Compute uncertainty in phi = arctan(y/x)
    using first-order approach, and fallback to second-order if negative variance.

    x, y, delta_x, delta_y are arrays of the same length.
    """
    cov_xy = my_cov_two_variables(x, y)  # if x,y are arrays, this might return an array or a single value
    dg_dx, dg_dy = dg_dx_and_dg_dy(x, y) # first derivatives w.r.t x,y

    # First-order variance array
    var_phi_1st = ((dg_dx * delta_x)**2 +
                   (dg_dy * delta_y)**2 +
                   2 * dg_dx * dg_dy * cov_xy)

    delta_phi = np.zeros_like(var_phi_1st)

    for i, var_phi in enumerate(var_phi_1st):
        if var_phi >= 0:
            # Good: take sqrt of first-order
            delta_phi[i] = np.sqrt(var_phi)
        else:
            print(f"Index={i}: first-order var is negative => trying second-order.")
            # Second-order approach at the single point (x[i], y[i])
            var_phi_2nd = variance_phi_second_order(
                mu_x=x[i], mu_y=y[i],
                var_x=(delta_x[i])**2,  # or delta_x[i] if your function expects variance
                var_y=(delta_y[i])**2,
                cov_xy=cov_xy          # or the local covariance if you store as an array
            )
            if var_phi_2nd > 0:
                delta_phi[i] = np.sqrt(var_phi_2nd)
                print(f"  => second-order var = {var_phi_2nd}")
            else:
                # If even second-order gives negative => raise or handle
                delta_phi[i] = np.nan
                print(f"Index={i}, negative second-order variance: {var_phi_2nd}")

    # Final NaN/Inf check
    if np.isnan(delta_phi).any() or np.isinf(delta_phi).any():
        print("following indices have nan values:", np.where(np.isnan(delta_phi))[0].tolist())
        print("following indices have inf values:", np.where(np.isinf(delta_phi))[0].tolist())
        # raise ValueError("phi_uncertainty: Found NaN or Inf in the final results.")

    return delta_phi

def a_m_phi_m_resetter(m, a_m, phi_m, angle_unit='rad'):
    if angle_unit=='rad':
        angle_factor = 1.
    elif angle_unit=='degree':
        angle_factor = np.pi/180
    else:
        raise NotImplementedError()
    phi_m = phi_m * angle_factor
    if phi_m > np.pi/(m) or phi_m < -np.pi/(m): # for m=4, if phi_m > 45 degree or phi_m < -45 degree:
        phi_m = (phi_m + np.pi/(m)) % (2*np.pi/(m)) - np.pi/(m) # get the remainder after dividing by 90 degree and adjust the angle between -45 to 45 degree
    if phi_m > np.pi/(2*m) or phi_m < - np.pi/(2*m): # if the angle is not in between 22.5 and -22.5 degree, then
        a_m = -a_m # flip the sign of a_m
        phi_m = (phi_m + np.pi/(2*m)) % (np.pi/(m)) - np.pi/(2*m)  # get the remainder after dividing by 45 degree and adjust the angle between -22.5~+22.5
    phi_m = phi_m / angle_factor
    return a_m, phi_m

def a_m_phi_m_resetter_wrapper(m, a_m_phi_m, angle_unit='rad'):
    # To convert angles not in (-pi/2m, pi/2m) into (-pi/2m, pi/2m) with negated amplitude
    a_m, phi_m = a_m_phi_m
    if type(phi_m) in [float, np.float64, int]:
        a_m, phi_m = a_m_phi_m_resetter(m, a_m, phi_m, angle_unit)
    elif type(phi_m) in [list, np.ndarray]:
        assert len(a_m) == len(phi_m)
        for i in range(len(a_m)):
            a_m[i], phi_m[i] = a_m_phi_m_resetter(m, a_m[i], phi_m[i], angle_unit)
    else:
        raise NotImplementedError()
    return [a_m, phi_m]

def params_multipole_field_update(params):
    params['phi3'] = np.arctan(params['_b3/a'] / params['_a3/a']) / 3
    params['phi4'] = np.arctan(params['_b4/a'] / params['_a4/a']) / 4

    params['a3/a'] = np.sign(params['_a3/a']) * np.sqrt(params['_a3/a'] ** 2 + params['_b3/a'] ** 2)
    params['a3/a'] = params['a3/a'].reshape(-1)

    params['a4/a'] = np.sign(params['_a4/a']) * np.sqrt(params['_a4/a'] ** 2 + params['_b4/a'] ** 2)
    params['a4/a'] = params['a4/a'].reshape(-1)

    # uncertainty of a4/a
    print("Calculating uncertainty of a4/a...")
    delta_a4_a = r_uncertainty(params['_a4/a'], params['_b4/a'], params['_delta_a4/a'], params['_delta_b4/a'])
    # undertainty of phi_4 - phi_0
    print("Calculating uncertainty of phi_4-phi_0...")
    delta_phi_4 = phi_uncertainty(params['_a4/a'], params['_b4/a'], params['_delta_a4/a'], params['_delta_b4/a'])  #
    # uncertainty of a3/a
    print("Calculating uncertainty of a3/a...")
    delta_a3_a = r_uncertainty(params['_a3/a'], params['_b3/a'], params['_delta_a3/a'], params['_delta_b3/a'])
    # undertainty of phi_3 - phi_0
    print("Calculating uncertainty of phi_3-phi_0...")
    delta_phi_3 = phi_uncertainty(params['_a3/a'], params['_b3/a'], params['_delta_a3/a'], params['_delta_b3/a'])

    params['delta_a4/a'] = delta_a4_a
    params['delta_phi_4'] = delta_phi_4
    params['delta_a3/a'] = delta_a3_a
    params['delta_phi_3'] = delta_phi_3

    params['a3/a'], params['phi3'] = a_m_phi_m_resetter_wrapper(3, [params['a3/a'], params['phi3']], angle_unit='rad')
    params['a4/a'], params['phi4'] = a_m_phi_m_resetter_wrapper(4, [params['a4/a'], params['phi4']], angle_unit='rad')
    return params

def params2xyzdxdydz(params, m):
    if m==4:
        x = params['a4/a']
        y = params['phi4']
        dx = params['delta_a4/a']
        dy = params['delta_phi_4']
    elif m==3:
        x = params['a3/a']
        y = params['phi3']
        dx = params['delta_a3/a']
        dy = params['delta_phi_3']
    else:
        raise NotImplementedError()
    z = params['q']
    dz = params['delta_q']
    return x, y, z, dx, dy, dz

#

def splitter_vals(arr, n_groups, weight=None, method='number', verbose=False):
    # returns the values for splitting an array into a certain n_groups number of groups.
    n_output = n_groups + 1
    if method == 'linear':
        split_vals = np.linspace(arr.min(), arr.max(), n_output)
    elif method == 'number':
        if weight is None:
            n_per_group = len(arr) / n_groups
            indices = (n_per_group * np.arange(n_output)).round().astype(int)
            indices[-1] -= 1 # to adjust the final index
            arr_sorted = np.sort(arr)
            split_vals = arr_sorted[indices]
        else:
            #
            n_per_group = np.sum(weight) / n_groups
            sort_indices = np.argsort(arr)
            arr_sorted = np.sort(arr)
            weight_sorted = weight[sort_indices]
            # cumulative sum to decide where to cut
            weight_cumul = np.cumsum(weight_sorted)
            weight_cumul_residual = weight_cumul % n_per_group # This drops from [n_per_group-1, n_per_group) to [0,1),
            residual_increase = weight_cumul_residual[1:] - weight_cumul_residual[:-1]
            indices = (np.where(residual_increase<0)[0]+1).tolist()
            if len(indices) == n_groups:
                # usual behavior when weight_cumul_residual[-1] is 0
                pass
            elif len(indices) == n_groups - 1:
                # unusual behavior when weight_cumul_residual[-1] is nonzero.
                # it happens because the numerical instability & impreciseness.
                # in this case, weight_cumul_residual[-1] is expcted to be almost the same as n_per_group,
                # but very slightly less. So, we check and insert the last index in this case.
                if np.isclose(weight_cumul_residual[-1], n_per_group, rtol=1e-2):
                    indices.append(len(arr_sorted) - 1)
                else:
                    raise ValueError
            else:
                raise ValueError
            indices.insert(0, 0)
            assert len(indices) == n_groups + 1
            split_vals = arr_sorted[indices]
        # if debug:
        #     plt.figure()
        #     plt.plot(arr_sorted, weight_cumul_residual)
        #     for i in range(len(split_vals)):
        #         plt.axvline(x=split_vals[i], color='r', linestyle='--')
        #     plt.show()
    else:
        raise NotImplementedError()
    assert split_vals[0] == np.min(arr)
    assert split_vals[-1] == np.max(arr)
    return split_vals

def grouper(n_groups, arr_base, arr_target_list=[], weight=None, method='number', verbose=False):
    # arr_base: the array that will be the base of grouping. With the indices that satisfy
    # arr_base >= val1 and arr_base < val2, the arrays in arr_target_list will be masked and returned as
    # target_groups_list.
    # arr_target_list: the list of arrays that will be grouped based on arr_base.
    assert len(arr_target_list) >= 1
    split_vals = splitter_vals(arr_base, n_groups, weight=weight, method=method, verbose=verbose) # returns the values to evenly split
    # the arr_base into n_groups
    split_vals[-1] += 1e-6 # to include the last element in the final group
    base_groups = []
    target_groups_list = [[] for _ in arr_target_list]
    for i in range(len(split_vals) - 1):
        val1, val2 = split_vals[i], split_vals[i + 1]
        mask = np.logical_and(arr_base >= val1, arr_base < val2)
        # Append the base group
        base_groups.append(arr_base[mask])
        # Append the corresponding target groups
        for j, arr_target in enumerate(arr_target_list):
            target_groups_list[j].append(arr_target[mask])

    return base_groups, target_groups_list, split_vals

#

def order2poly(x, a, x1, x2):
    return a * (x-x1) * (x-x2)

def order3poly(x, a, x1, x2, x3):
    return a * (x-x1) * (x-x2) * (x-x3)

def order4poly(x, a, x1, x2, x3, x4):
    return a * (x-x1) * (x-x2) * (x-x3) * (x-x4)

def exponential(x, A, a):
    return A * np.exp(a*x)

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)*(x-x0)/(2*sigma*sigma))

# def gaussian_norm(x, x0, sigma):
#     # normalized gaussian
#     a = 1/(sigma * np.sqrt(2*np.pi))
#     return a * np.exp(-(x-x0)*(x-x0)/(2*sigma*sigma))

def gaussian_and_linear(x, a, x0, sigma, A, B):
    return gaussian(x, a, x0, sigma) + A * x + B

def double_gaussian(x, a1, x1, sigma1, a2, x2, sigma2):
    return gaussian(x, a1, x1, sigma1) + gaussian(x, a2, x2, sigma2)

def double_gaussian_and_constant(x, a1, x1, sigma1, a2, x2, sigma2, c):
    return gaussian(x, a1, x1, sigma1) + gaussian(x, a2, x2, sigma2) + c

def double_gaussian_and_constant_exp(x, a1, x1, sigma1, a2, x2, sigma2, c, A, k):
    return gaussian(x, a1, x1, sigma1) + gaussian(x, a2, x2, sigma2) + c + A * np.exp(k*x)

# def double_gaussian_norm(x, ratio, x1, sigma1, x2, sigma2):
#     return ratio * gaussian_norm(x, x1, sigma1) + (1 - ratio) * gaussian_norm(x, x2, sigma2)

def double_gaussian_and_linear(x, a1, x1, sigma1, a2, x2, sigma2, A, B):
    return gaussian(x, a1, x1, sigma1) + gaussian(x, a2, x2, sigma2) + A * x + B

def convert_to_2tuple_of_arrays(input_array):
    # Unpack the first and second elements of each tuple using zip
    first_elements, second_elements = zip(*input_array)
    # Return as a tuple of two lists
    return list(first_elements), list(second_elements)

#

def generalized_gaussian(x, alpha, beta, normalize=True, normalize_axis=None, verbose=False):
    if (alpha <=0).any():
        if verbose:
            warnings.warn('!!!!!!!!!! alpha must be positive, but nonpositive alpha detected... applying np.abs() !!!!!!!!!!')
        alpha = np.abs(alpha)
        # raise ValueError()
    if (beta <= 0).any():
        if verbose:
            warnings.warn('!!!!!!!!!! beta must be positive, but nonpositive beta decteded... applying np.abs() !!!!!!!!!!')
        beta = np.abs(beta)
        # raise ValueError()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # Convert RuntimeWarning to an exception
        try:
            result = (beta / (2 * alpha * gamma(1 / beta))) * np.exp(- (np.abs(x) / alpha) ** beta)
            if normalize:
                # normalization within the given range
                if len(x.shape) == 1:
                    integrated = np.trapz(result, x=x) # trapz makes erros for 3D "x"
                    result = result/integrated
                elif len(x.shape) > 1:
                    assert normalize_axis is not None
                    if normalize_axis == 0:
                        range_ = x[:,0,0]
                        assert (x[0, :, 0] == x[0, 0, 0]).all()
                        assert (x[0, 0, :] == x[0, 0, 0]).all()
                    elif normalize_axis==1:
                        range_ = x[0,:,0]
                        assert (x[:, 0, 0] == x[0, 0, 0]).all()
                        assert (x[0, 0, :] == x[0, 0, 0]).all()
                    elif normalize_axis==2:
                        range_ = x[0, 0, :]
                        assert (x[0, :, 0] == x[0, 0, 0]).all()
                        assert (x[:, 0, 0] == x[0, 0, 0]).all()
                    integrated = np.trapz(result, range_, axis=normalize_axis)
                    try:
                        result = result/np.expand_dims(integrated, axis=normalize_axis)
                    except RuntimeWarning as e:
                        print(f"RuntimeWarning encountered: {e}")
            return result
        except RuntimeWarning as e:
            print(f"RuntimeWarning encountered: {e}")
            raise  # Re-raise the exception if needed

# Fit a Generalized Gaussian to the data using curve fitting
def fit_generalized_gaussian(x, y):
    # Initial guesses for alpha and beta
    initial_guess_mean = 0
    initial_guess_sigma = np.sqrt(sum(x * x * y) / sum(y) - initial_guess_mean * initial_guess_mean)
    initial_guess = [initial_guess_sigma, 2.0]

    # Perform curve fitting
    params, params_covariance = curve_fit(generalized_gaussian, x, y, p0=initial_guess, bounds=([1e-2, 1e-2],
                                                                                                [5, np.inf]))

    # Extract fitted alpha and beta
    alpha_fitted, beta_fitted = params
    return alpha_fitted, beta_fitted

from scipy.stats import skewnorm
# Define the skew-normal PDF
def skew_normal_pdf(x, params):
    shape, loc, scale = params
    return skewnorm.pdf(x, shape, loc=loc, scale=scale)

# Function to fit a skew-normal distribution to the data
def fit_skew_normal(x, y, add_constant=False, verbose=2, test_plot=False):
    # y: P(X)
    # Initial guesses for loc (mean), scale (std), and shape (skewness)
    initial_guess_mean = sum(x * y) / sum(y)
    initial_guess_sigma = np.sqrt(sum(x * x * y) / sum(y) - initial_guess_mean * initial_guess_mean)
    initial_guess = [0., initial_guess_mean, initial_guess_sigma]  # loc, scale, shape
    if test_plot:
        plt.figure()
        plt.plot(x, y, label='given data')
        plt.plot(x, skew_normal_pdf(x, initial_guess), label='initial guess fit')
        plt.legend()
        plt.show()
    if add_constant:
        initial_guess.append(min(x)) # adding 0 as the initial constant baseline
    # initial_guess = [np.mean(x), np.std(x), 0]  # loc, scale, shape
    # Start the timer
    # start_time = time.time()
    # Use least_squares to find the best fit parameters with verbose output
    if add_constant:
        result = least_squares(residuals_w_constant, initial_guess, args=(x, y), verbose=verbose)
    else:
        result = least_squares(residuals, initial_guess, args=(x, y), verbose=verbose)
    # End the timer
    # end_time = time.time()
    # Print detailed fitting information
    # print(f"\nFitting process took {end_time - start_time:.4f} seconds")
    # print(f"Number of iterations: {result.nfev}")
    # print(f"Optimization status: {result.status}")
    # print(f"Message: {result.message}")

    # Extract the optimized parameters
    if add_constant:
        shape, loc, scale, constant = result.x
        # print(f"Fitted parameters: loc={loc}, scale={scale}, shape={shape}, constant={constant}")
        return shape, loc, scale, constant
    else:
        shape, loc, scale = result.x
        # print(f"Fitted parameters: loc={loc}, scale={scale}, shape={shape}")
        return shape, loc, scale


def filtered_xy(x,y,dx,dy,mask):
    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_dx = dx[mask]
    filtered_dy = dy[mask]
    return filtered_x, filtered_y, filtered_dx, filtered_dy

def plot_filtered_y(x1, x2, z1, z2, ax, canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name, dz=None):
    # Filter data within the range of x1, x2 and z1, z2
    mask = (x >= x1) & (x <= x2) & (z >= z1) & (z <= z2)
    filtered_x, filtered_y, filtered_dx, filtered_dy = filtered_xy(x, y, dx, dy, mask)

    if len(filtered_x) == 0:
        ax.set_title("No points in the given range")
        canvas.draw()
        return

    # Create a grid over which to evaluate the KDE
    y_grid = np.linspace(-np.pi / 2 / m, np.pi / 2 / m, 1000)

    # Initialize an array to store the combined KDE
    kde_total_y = np.zeros_like(y_grid)

    # Apply normal distribution for each y point using its corresponding dy (uncertainty in y)
    for i in range(len(filtered_y)):
        dist = norm(loc=filtered_y[i], scale=filtered_dy[i]) # A normal continuous random variable.
        if np.isnan(dist.pdf(y_grid)).any():
            raise ValueError(f"dist.pdf(y_grid) has NaN; i={i}")
        else:
            pass
        kde_total_y += dist.pdf(y_grid)

    # Normalize the total KDE
    kde_total_y /= len(filtered_y)

    # Plot the results
    ax.plot(y_grid, kde_total_y, label='P(y)', color='blue')
    # Fit the Generalized Gaussian to the noisy data
    alpha_fitted, beta_fitted = fit_generalized_gaussian(y_grid, kde_total_y)
    P_fitted = generalized_gaussian(y_grid, alpha_fitted, beta_fitted)
    ax.plot(y_grid, P_fitted, label='Fit (generalized Gaussian)', color='red', linestyle='--')

    # ax.scatter(filtered_y, np.zeros_like(filtered_y), marker='x', color='red', label='Filtered y points')
    ax.set_xlabel(Y_name)
    ax.set_ylabel(f'P({Y_name})')
    ax.set_title(
        f'Marginal Distribution of {Y_name} for\n{X_name} in [{x1:.3f}, {x2:.3f}] and {Z_name} in [{z1:.3f}, {z2:.3f}]')
    ax.legend()
    ax.text(0.25, 0.95, f'N={len(filtered_x)}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_filtered_x(x1, x2, z1, z2, ax, canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name, dz=None):
    # Filter data within the range of x1, x2 and z1, z2
    mask = (x >= x1) & (x <= x2) & (z >= z1) & (z <= z2)
    filtered_x, filtered_y, filtered_dx, filtered_dy = filtered_xy(x,y,dx,dy,mask)

    if len(filtered_x) == 0:
        ax.set_title("No points in the given range")
        canvas.draw()
        return

    # Create a grid over which to evaluate the KDE
    x_grid = np.linspace(x.min() - (x.max() - x.min()) / 10, x.max() + (x.max() - x.min()) / 10, 1000)

    kde_total_x = kde_with_error(filtered_x, filtered_dx, x_grid)

    # Fit the skew-normal to the noisy data
    loc_fit, scale_fit, shape_fit = fit_skew_normal(x_grid, kde_total_x, verbose=0)

    # Plot the results
    ax.plot(x_grid, kde_total_x, label='P(x)', color='blue')
    # Plot the fitted curve
    ax.plot(
        x_grid,
        skew_normal_pdf(x_grid, [shape_fit, loc_fit, scale_fit]),
        label='Fitted Skew Normal',
        color='red'
    )
    # ax.scatter(filtered_y, np.zeros_like(filtered_y), marker='x', color='red', label='Filtered y points')
    ax.set_xlabel(X_name)
    ax.set_ylabel(f'P({X_name})')
    ax.set_title(
        f'Marginal Distribution of {X_name} for\n{X_name} in [{x1:.3f}, {x2:.3f}] and {Z_name} in [{z1:.3f}, {z2:.3f}]')
    ax.legend()
    ax.text(0.25, 0.95, f'N={len(filtered_x)}', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0,
    #                  box.width, box.height * 0.8])

def errorbar_xyz(ax, x, y, z, dx, dy, dz, **kwargs_errorbar):
    # Create arrays for error bars
    segments = []
    for xi, yi, zi in zip(x, y, z):
        # X direction error bars
        segments.append([(xi - dx, yi, zi), (xi + dx, yi, zi)])
        # Y direction error bars
        segments.append([(xi, yi - dy, zi), (xi, yi + dy, zi)])
        # Z direction error bars
        segments.append([(xi, yi, zi - dz), (xi, yi, zi + dz)])

    # Create a collection of segments with error bars
    lc = Line3DCollection(segments, **kwargs_errorbar)
    ax.add_collection(lc)

def scatter_w_errorbar(ax, x, y, z=None, dx=None, dy=None, dz=None, kwargs_scatter={}, kwargs_errorbar={}):
    if ax is None:
        raise ValueError("No axis specified...")
    else:
        pass
    if z is None:
        kwargs_errorbar['c'] = kwargs_errorbar['colors']
        del kwargs_errorbar['colors']
        ax.errorbar(x,y, xerr = dx, yerr = dy, **kwargs_errorbar)
        ax.scatter(x, y, **kwargs_scatter)
    else:
        errorbar_xyz(ax, x, y, z, dx, dy, dz, **kwargs_errorbar)
        del kwargs_scatter['s']
        ax.plot(x, y, z, ls='None', **kwargs_scatter)
    return ax


def prob_dist_surf_w_errors(ax, x, y, dx, dy, m, num_grid = 100, z_proj=False):
    # Create a meshgrid
    x_min, x_max = x.min() - (x.max() - x.min()) / 10, x.max() + (x.max() - x.min()) / 10
    y_min, y_max = -np.pi/2/m, np.pi/2/m #y.min() - (y.max() - y.min()) / 10, y.max() + (y.max() - y.min()) / 10
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Initialize the probability distribution
    prob_dist = np.zeros_like(X)

    # Convolve 2D Gaussian with each datapoint
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        gaussian = np.exp(-(((X - xi) ** 2) / (2 * dxi ** 2) + ((Y - yi) ** 2) / (2 * dyi ** 2)))
        prob_dist += gaussian

    # Normalize the probability distribution
    prob_dist /= (2 * np.pi * dx * dy).sum()

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Probability Density')
    # ax.set_title('Total Probability Distribution as 3D Surface')


    # Set the view for z-projection if requested
    if z_proj:
        ax.view_init(elev=90, azim=-90)
        # Add contour lines
        # ax.contourf(X, Y, , cmap='Spectral_r', offset=-1, alpha=0.75)
        surf = ax.contourf(X, Y, prob_dist, zdir='z', offset=0, cmap='jet')
    else:
        # Plotting the 3D surface
        surf = ax.plot_surface(X, Y, prob_dist, cmap='jet')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-np.pi/2/m, np.pi/2/m)

    return ax, surf

def plot_xy(x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z,X_name, Y_name, Z_name):
    #
    mask = (x >= x1) & (x <= x2) & (z >= z1) & (z <= z2)
    filtered_x = x[mask]
    filtered_dx = dx[mask]
    filtered_y = y[mask]
    filtered_dy = dy[mask]#
    filtered_z = z[mask]
    # filtered_dz = dz[mask]
    #
    mask_flipped = ~mask
    filtered_x_flipped = x[mask_flipped]
    filtered_dx_flipped = dx[mask_flipped]
    filtered_y_flipped = y[mask_flipped]
    filtered_dy_flipped = dy[mask_flipped]
    filtered_z_flipped = z[mask_flipped]
    # filtered_dz_flipped = dz[mask_flipped]
    #
    ax.set_xlabel(X_name)
    ax.set_ylabel(Y_name)
    print(f'')
    ax.set_title(f'Joint Distribution of {X_name} and {Y_name} for\n{X_name} in [{x1:.3f}, {x2:.3f}] and {Z_name} in [{z1:.3f}, {z2:.3f}]\n{Y_name} mean: {filtered_y.mean():.3f}, stdev: {filtered_y.std():.3f}')
    #
    alpha_errorbar = 0.2
    kwargs_errorbar0 = {'colors': 'k', 'alpha': 0.1, 'linewidth': 1.} # for non-selected points
    kwargs_errorbar1 = {'colors': 'r', 'alpha': alpha_errorbar, 'linewidth': 1.}
    kwargs_errorbar2 = {'colors': 'b', 'alpha': alpha_errorbar, 'linewidth': 1.}
    kwargs_scatter0 = {'c': 'k', 'alpha': 0.1, 'marker': 'o', 's': 2} # for non-selected points
    kwargs_scatter1 = {'c': 'r', 'alpha': 1.0, 'marker': 'o', 's': 2}
    kwargs_scatter2 = {'c': 'b', 'alpha': 1.0, 'marker': 'o', 's': 2}
    #
    ax = scatter_w_errorbar(ax, filtered_x_flipped, filtered_y_flipped,
                            dx=filtered_dx_flipped, dy=filtered_dy_flipped,
                            kwargs_scatter=kwargs_scatter0, kwargs_errorbar=kwargs_errorbar0)
    ax = scatter_w_errorbar(ax, filtered_x[filtered_x>0], filtered_y[filtered_x>0],
                            dx=filtered_dx[filtered_x>0], dy=filtered_dy[filtered_x>0],
                            kwargs_scatter=kwargs_scatter1, kwargs_errorbar=kwargs_errorbar1)
    ax = scatter_w_errorbar(ax, filtered_x[filtered_x<0], filtered_y[filtered_x<0],
                            dx=filtered_dx[filtered_x<0], dy=filtered_dy[filtered_x<0],
                            kwargs_scatter=kwargs_scatter2, kwargs_errorbar=kwargs_errorbar2)
    ax.set_xlim(x.min() - (x.max()-x.min())/10, x.max() + (x.max()-x.min())/10)
    ax.set_ylim(-np.pi/2/m, np.pi/2/m)

def plot_xy_3d(x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z,X_name, Y_name, Z_name, z_proj=False):
    #
    mask = (x >= x1) & (x <= x2) & (z >= z1) & (z <= z2)
    filtered_x = x[mask]
    filtered_dx = dx[mask]
    filtered_y = y[mask]
    filtered_dy = dy[mask]#
    filtered_z = z[mask]
    filtered_dz = dz[mask]
    #
    mask_flipped = ~mask
    filtered_x_flipped = x[mask_flipped]
    filtered_dx_flipped = dx[mask_flipped]
    filtered_y_flipped = y[mask_flipped]
    filtered_dy_flipped = dy[mask_flipped]
    filtered_z_flipped = z[mask_flipped]
    filtered_dz_flipped = dz[mask_flipped]
    #
    ax.set_xlabel(X_name)
    ax.set_ylabel(Y_name)
    print(f'')
    ax.set_title(f'Joint Distribution of {X_name} and {Y_name} for\n{X_name} in [{x1:.3f}, {x2:.3f}] and {Z_name} in [{z1:.3f}, {z2:.3f}]\n{Y_name} mean: {filtered_y.mean():.3f}, stdev: {filtered_y.std():.3f}')
    #
    alpha_errorbar = 0.2
    #
    ax, surf = prob_dist_surf_w_errors(ax, filtered_x, filtered_y,
                                filtered_dx, filtered_dy,
                                m,
                                num_grid=500, z_proj=z_proj)
    # ax.set_xlim(x.min() - (x.max()-x.min())/10, x.max() + (x.max()-x.min())/10)
    ax.set_ylim(-np.pi/2/m, np.pi/2/m)

    # plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    return ax

def plot_x_lines(x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z, X_name, Y_name, Z_name):
    ax.axvline(x=x1, color='g', linestyle=':')
    ax.axvline(x=x2, color='g', linestyle=':')

# Function to save the figure
def save_figure(x1, x2, z1, z2, fig, X_name, Y_name, Z_name, comment=''):
    filename = f"plot_{X_name}_{x1:.3f}_{x2:.3f}_{Z_name}_{z1:.3f}_{z2:.3f}"+comment+".pdf"
    filename = filename.replace('$','')
    filename = filename.replace('/','')
    fig.savefig(filename)
    print(f"Figure saved as {filename}")

def residuals(params, x, y):
    return skew_normal_pdf(x, params) - y

def residuals_w_constant(params, x, y):
    return skew_normal_pdf(x, params[:-1]) + params[-1] - y

def kde_with_error(x, dx, x_grid, amp_weight=None, norm_weight=None, verbose=False, symmetrize_over_zero=False,
                   return_each = False, test=False):
    # amp_weight: weight for amplitude
    # norm_weight: weight for normalization
    if amp_weight is None:
        amp_weight = np.ones_like(x)
        if verbose:
            print("amp_weight is not given; it is set to be np.ones_lie(x).")
    if norm_weight is None:
        norm_weight = amp_weight
        if verbose:
            print("norm_weight is not given; it is set to be the same as amp_weight.")

    # Initialize an array to store the combined KDE
    kde_total_x = np.zeros_like(x_grid)
    kde_each_x_list = []

    # Apply normal distribution for each y point using its corresponding dy (uncertainty in y)
    for i in range(len(x)):
        if symmetrize_over_zero:
            dist = norm(loc=x[i], scale=dx[i])
            kde_each = dist.pdf(x_grid) * amp_weight[i]/2 + dist.pdf(-x_grid) * amp_weight[i]/2
            kde_total_x += kde_each
            if return_each:
                kde_each_x_list.append(kde_each)
        else:
            dist = norm(loc=x[i], scale=dx[i])
            kde_each = dist.pdf(x_grid) * amp_weight[i]
            kde_total_x += kde_each
            if return_each:
                kde_each_x_list.append(kde_each)

    # Normalize the total KDE
    kde_total_x /= np.sum(norm_weight)

    integrated = np.trapz(kde_total_x, x_grid)
    if test:
        plt.figure(); plt.plot(x_grid, kde_total_x); plt.show()
    assert np.isclose(integrated, 1, rtol=1e-1)
    # assert np.isclose(kde_total_x.sum() * (x_grid[1] - x_grid[0]), 1, rtol=1e-5)
    if return_each:
        return kde_total_x, kde_each_x_list
    else:
        return kde_total_x


def transform_bounds(param_bounds):
    """
    param_bounds: list of tuples [(lb1, ub1), (lb2, ub2), ..., (lbN, ubN)]
    Returns a tuple of lists ([lb1, lb2, ..., lbN], [ub1, ub2, ..., ubN])
    suitable for many SciPy optimize functions.
    """
    lower_list = [b[0] for b in param_bounds]
    upper_list = [b[1] for b in param_bounds]
    return (lower_list, upper_list)

# def kde_with_error_alternate(y, dy, y_grid):
#     kde_total_x = np.zeros_like(y_grid)
#
#     # Apply normal distribution for each y point using its corresponding dy (uncertainty in y)
#     for i in range(len(y)):
#         dist = norm(loc=y[i], scale=dy[i])
#         kde_total_x += dist.pdf(y_grid)
#
#     # Normalize the total KDE
#     kde_total_x /= len(x)
#     return kde_total_x

#

def param_unpacker(fitted_params, return_as_1d_list=True, return_param_nums = True):
    # PZ_params: list
    # PX_Z_params: list of list (loc, scale, shape order) with array elements representing fitted params
    # PY_X_params: list of list (alpha, beta order) with array elements representing fitted params
    PZ_params, PX_Z_params, PY_X_params = fitted_params
    assert type(PZ_params) == list
    assert type(PX_Z_params) == list
    assert type(PY_X_params) == list
    param_return = []
    param_nums = []
    #
    if return_as_1d_list:
        for param in PZ_params:
            param_return.append(param)
        param_nums.append(len(PZ_params))
    else:
        raise NotImplementedError()
        # param_return.append(PZ_params)
    #
    for P_params in [PX_Z_params, PY_X_params]:
        if return_as_1d_list:
            param_nums_element = []
            for param in P_params:
                if type(param) == np.ndarray:
                    for element in param.tolist():
                        param_return.append(element)
                elif type(param) == list:
                    for element in param:
                        param_return.append(element)
                param_nums_element.append(len(param))
            param_nums.append(param_nums_element)
        else:
            raise NotImplementedError()
            # to_append = []
            # for key in param_list_element.keys():
            #     to_append.append(param_list_element[key].tolist())
            # param_return.append(to_append)
    #
    if return_param_nums:
        return param_return, param_nums
    else:
        return param_return

def my_recursive_appender(my_list, param_1d, nums, count=0, return_count=False):
    # my_list: the list to append
    if type(nums) == int:
        my_list.append(param_1d[count:count+nums])
        count += nums
    elif type(nums) == list:
        my_list_sub = []
        for num in nums:
            my_list_sub, count = my_recursive_appender(my_list_sub, param_1d, num, count=count, return_count=True)
        my_list.append(my_list_sub)
    if return_count:
        return my_list, count
    else:
        return my_list

def param_repacker(param_1d, param_nums):
    fitted_params_repacked = []
    count = 0
    for nums in param_nums:
        # The below command 'repacks' the parameters to the original form
        fitted_params_repacked, count = my_recursive_appender(fitted_params_repacked, param_1d, nums, count=count,
                                                              return_count=True)
    return fitted_params_repacked

import warnings

def full_prob_model(my_grid, param_1d, param_nums, fitted_functions, verbose=False):
    X, Y, Z = my_grid.XYZ
    n_i = 0
    n_f = param_nums[0]
    # P(Z)
    PZ = fitted_functions[0](Z, param_1d[n_i:n_f])
    #
    n_i = n_f
    n_f += param_nums[1][0]
    param1 = fitted_functions[1][0](Z, *param_1d[n_i:n_f])
    n_i = n_f
    n_f += param_nums[1][1]
    param2 = fitted_functions[1][1](Z, *param_1d[n_i:n_f])
    n_i = n_f
    n_f += param_nums[1][2]
    param3 = fitted_functions[1][2](Z, *param_1d[n_i:n_f])
    # P(X | Z)
    PX_Z = skew_normal_pdf(X, [param1, param2, param3])
    #
    if (np.isnan(PX_Z)).any():
        if verbose:
            warnings.warn("!!!!!!!!!! PX_Z has one or more nan values, and they will be replaced to 0 !!!!!!!!!!")
        PX_Z[np.isnan(PX_Z)] = 0.
    #
    n_i = n_f
    n_f += param_nums[2][0]
    param4 = fitted_functions[2][0](X, *param_1d[n_i:n_f])
    n_i = n_f
    n_f += param_nums[2][1]
    param5 = fitted_functions[2][1](X, *param_1d[n_i:n_f])
    # P(Y | X)
    PY_X = generalized_gaussian(Y, *[param4, param5], normalize_axis=1)
    if verbose:
        print(f"PZ: {PZ}, \nPX_Z: {PX_Z}, \nPY_X: {PY_X}")
    return PZ * PX_Z * PY_X

# TODO: update the function below.
def distribution_fit_result_plot(
    ax,
    data_values,
    data_errors,
    grid_values,
    static_params,           # e.g. [loc_fit0, scale_fit0, shape_fit0] for skew normal
    param_functions,         # e.g. fitted_functions[1] or fitted_functions[2]
    param_fits,              # e.g. fitted_params[1] or fitted_params[2]
    param_fits2=None,
    pdf_func=None,           # e.g. skew_normal_pdf or generalized_gaussian
    kde_func=None,           # e.g. kde_with_error
    group_xaxis_val=None,    # e.g. z_val or x_val
    distribution_name="",    # "Skew Normal" or "Generalized Gaussian"
    param_labels=None,       # e.g. ['α', 'ξ', 'ω'] for skew normal; ['α', 'β'] for gg
    amp_weight=None,
    norm_weight=None,
    symmetrize_over_zero=False,
    test=False,
):
    """
    A generic plotting function that:
      1) Computes a kernel density estimate (plus error).
      2) Plots the original distribution from `pdf_func` with static_params.
      3) Retrieves parameterized fit from `param_functions`, `param_fits`, and optionally `param_fits2`.
      4) Plots everything on the given `ax`.

    :param ax: Matplotlib Axes on which to plot
    :param data_values: data points for your variable of interest (e.g. x_group or y_group)
    :param data_errors: error or uncertainty in data (e.g. dx_group or dy_group)
    :param grid_values: range of values (e.g. x_grid or y_grid)
    :param static_params: list/tuple of “static” fitted parameters (the ones from `*_param_group`)
    :param param_functions: a list/tuple of sub-functions that compute each parameter given x_val
    :param param_fits: a list/tuple of parameter tuples for param_functions
    :param param_fits2: same structure as param_fits, but for a second set of parameterization (optional)
    :param pdf_func: function that takes (grid_values, [param1, param2, ...]) and returns PDF
    :param kde_func: function that estimates the kernel density with error, e.g. `kde_with_error`
    :param group_xaxis_val: the “input” to param_functions, e.g. z_val or x_val
    :param distribution_name: string to display in legend/title, e.g. "Skew Normal"
    :param param_labels: list of parameter symbols for display in the plot title
    :return: None
    """

    if param_labels is None:
        # fallback if no labels provided
        param_labels = [f"p{i}" for i in range(len(static_params))]

    # 1) Compute the kernel density estimate
    kde_values = kde_func(data_values, data_errors, grid_values, amp_weight=amp_weight,
                                     norm_weight=norm_weight, symmetrize_over_zero=symmetrize_over_zero, test=test)

    # 2) Plot the kernel density result
    ax.plot(grid_values, kde_values, color='blue', label=f'KDE (data + error)')

    # 3) Plot the “static” fitted curve (these come from static_params)
    ax.plot(
        grid_values,
        pdf_func(grid_values, static_params),
        label=f'Fitted {distribution_name}, without hyperparameterization.',
        color='red', linestyle='--'
    )

    # 4) Compute the parameterized fit from param_functions and param_fits
    #    Then plot that curve
    param_values = [
        func(group_xaxis_val, *fit_vals)
        for func, fit_vals in zip(param_functions, param_fits)
    ]
    ax.plot(
        grid_values,
        pdf_func(grid_values, param_values),
        label=f'Fitted {distribution_name}, param. Init.',
        color='orange', linestyle='--'
    )

    # 5) Optionally do the second set of parameters (param_fits2)
    if param_fits2 is not None:
        param_values2 = [
            func(group_xaxis_val, *fit_vals2)
            for func, fit_vals2 in zip(param_functions, param_fits2)
        ]
        ax.plot(
            grid_values,
            pdf_func(grid_values, param_values2),
            label=f'Fitted {distribution_name}, param. Final.',
            color='green', linestyle='--'
        )
        # Build the extra title line for param_fits2
        title_add = "\n" + ", ".join(
            rf'{label}_2={val:.3f}'
            for label, val in zip(param_labels, param_values2)
        )
    else:
        title_add = ''

    # 6) Build a dynamic title with both static_params and param_values
    title_static = ", ".join(
        rf'{label}_0={val:.3f}'
        for label, val in zip(param_labels, static_params)
    )
    title_param = ", ".join(
        rf'{label}_1={val:.3f}'
        for label, val in zip(param_labels, param_values)
    )

    ax.set_title(
        title_static + "\n" + title_param + title_add,
        fontsize=16
    )

    # 7) Adjust the axes position if desired
    pos = ax.get_position()
    new_pos = [pos.x0, pos.y0, pos.width * 0.8, pos.height * 0.6]
    ax.set_position(new_pos)

    # ax.legend()  # show the legend if you want

def skew_normal_fit_result_plot(ax, x_group, dx_group, z_val, x_grid, skew_normal_param_group,
                             fitted_functions, fitted_params, fitted_params2=None):
    # kde convolution with the error
    kde_total_x = kde_with_error(x_group, dx_group, x_grid)
    # fitting
    # loc_fit, scale_fit, shape_fit = fit_skew_normal(x_grid, kde_total_x, verbose=0)
    # loc_fit, scale_fit, shape_fit =
    loc_fit0, scale_fit0, shape_fit0 = skew_normal_param_group
    shape_fit1 = fitted_functions[1][0](z_val, *fitted_params[1][0])
    loc_fit1 = fitted_functions[1][1](z_val, *fitted_params[1][1])
    scale_fit1 = fitted_functions[1][2](z_val, *fitted_params[1][2])
    # Plot the results
    ax.plot(x_grid, kde_total_x, color='blue')
    # Plot the fitted curve
    ax.plot(x_grid,
            skew_normal_pdf(x_grid, [shape_fit0, loc_fit0, scale_fit0]),
            label='Fitted Skew Normal',
            color='red', linestyle='--')
    ax.plot(x_grid,
            skew_normal_pdf(x_grid, [shape_fit1, loc_fit1, scale_fit1]),
            label='Fitted Skew Normal, parameterized',
            color='orange', linestyle='--')
    if fitted_params2 is not None:
        shape_fit2 = fitted_functions[1][0](z_val, *fitted_params2[1][0])  # after fitting as a functional
        # form altogether
        loc_fit2 = fitted_functions[1][1](z_val, *fitted_params2[1][1])  # after fitting as a functional
        # form altogether
        scale_fit2 = fitted_functions[1][2](z_val, *fitted_params2[1][2])
        ax.plot(x_grid,
                skew_normal_pdf(x_grid, [shape_fit2, loc_fit2, scale_fit2]),
                label='Fitted Skew Normal, parameterized',
                color='green', linestyle='--')
        title_add = "\n" + rf'$\alpha_2={shape_fit2:.3f}, \xi_3={loc_fit2:.3f}, \omega_3={scale_fit2:.3f}$'
    else:
        title_add = ''
    my_title = rf'$\alpha_0={shape_fit0:.3f}, \xi_0={loc_fit0:.3f}, \omega_0={scale_fit0:.3f}$' + "\n" + \
                 rf'$\alpha_1={shape_fit1:.3f}, \xi_1={loc_fit1:.3f}, \omega_1={scale_fit1:.3f}$' + title_add
    ax.set_title(my_title, fontsize=16)
    # Get the current position of the axis
    pos = ax.get_position()
    new_pos = [pos.x0, pos.y0, pos.width*0.8, pos.height*0.6]
    ax.set_position(new_pos)

def generalized_gaussian_fit_result_plot(ax, y_group, dy_group, x_val, y_grid,
                                         generalized_gaussian_param_group, fitted_functions,
                                         fitted_params, fitted_params2=None):
    # kde convolution with the error
    kde_total_y = kde_with_error(y_group, dy_group, y_grid)
    # fitting
    # alpha_fit, beta_fit = fit_generalized_gaussian(y_grid, (kde_total_y + np.flip(kde_total_y))/2 )
    # generalized_gaussian_param_groups[0].append(alpha_fit)
    # generalized_gaussian_param_groups[1].append(beta_fit)
    alpha_fit, beta_fit = generalized_gaussian_param_group
    alpha_fit2 = fitted_functions[2][0](x_val, *fitted_params[2][0]) # after fitting as a functional form
    beta_fit2 = fitted_functions[2][1](x_val, *fitted_params[2][1]) # after fitting as a functional form
    # alpha_fit_list.append(alpha_fit)
    # beta_fit_list.append(beta_fit)
    # alpha_fit2_list.append(alpha_fit2)
    # beta_fit2_list.append(beta_fit2)
    # Plot the results
    ax.plot(y_grid, kde_total_y, color='blue')
    # Plot the fitted curve
    ax.plot(y_grid,
            generalized_gaussian(y_grid, alpha_fit, beta_fit),
            label='Fitted Generalized Gaussian',
            color='red', linestyle='--')
    ax.plot(y_grid,
            generalized_gaussian(y_grid, alpha_fit2, beta_fit2),
            label='Fitted Generalized Gaussian, parameterized',
            color='orange', linestyle='--')
    if fitted_params2 is not None:
        alpha_fit3 = fitted_functions[2][0](x_val, *fitted_params2[2][0])  # after fitting as a functional
        # form altogether
        beta_fit3 = fitted_functions[2][1](x_val, *fitted_params2[2][1])  # after fitting as a functional
        # form altogether
        ax.plot(y_grid,
                generalized_gaussian(y_grid, alpha_fit3, beta_fit3),
                label='Fitted Generalized Gaussian, parameterized',
                color='green', linestyle='--')
        title_add = "\n" + rf'$\alpha_3={alpha_fit3:.3f}, \beta_3={beta_fit3:.3f}$'
    else:
        title_add = ''
    ax.set_title(
        rf'$\alpha={alpha_fit:.3f}, \beta={beta_fit:.3f}$'+ "\n" +
        rf'$\alpha_2={alpha_fit2:.3f}, \beta_2={beta_fit2:.3f}$' + title_add,
        fontsize=16)
    # Get the current position of the axis
    pos = ax.get_position()
    new_pos = [pos.x0, pos.y0, pos.width * 0.8, pos.height * 0.6]
    ax.set_position(new_pos)

#

def probability_smoother(P, epsilon=1e-10):
    P = (P + epsilon)/(1+epsilon * len(P.reshape(-1)))
    return P

def JS_divergence(P, Q):
    M = (P+Q)/2
    val = (KL_divergence(P,M) + KL_divergence(Q, M))/2
    return val

def JS_divergence_array(P, Q):
    M = (P + Q) / 2
    val = (KL_divergence(P, M, element_wise = True) + KL_divergence(Q, M, element_wise = True)) / 2
    return val

def JS_normalized_divergence_array(P, Q):
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    M = (P + Q) / 2
    val = (KL_divergence(P, M, element_wise = True) + KL_divergence(Q, M, element_wise = True)) / 2
    return val

def KL_divergence(P, Q, element_wise=False):
    if element_wise:
        val = P * np.log(P / Q)
    else:
        val = np.sum(P * np.log(P / Q))
    if np.any(np.isinf(val)):
        raise ValueError("val is infinite!")
    else:
        return val

def L2_forward(P_data, P_model):
    dif = P_data - P_model
    val = np.sum(dif*dif)
    if np.isinf(val):
        raise ValueError("val is infinite!")
    else:
        return val

def to_be_minimized_JS_normalized(param_1d, smoothing=True):
    # x0 is not needed because we already defined X, Y, Z,
    if smoothing:
        P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
        P_data = probability_smoother(P_XYZ_Hao)
    else:
        P_model = full_prob_model(X, Y, Z, param_1d, verbose=False)
        P_data = P_XYZ_Hao
    P_model = P_model / np.sum(P_model)
    P_data = P_data / np.sum(P_data)
    return JS_divergence(P_data, P_model)

def to_be_minimized_JS(param_1d, my_grid, P_XYZ_Hao, param_nums, fitted_functions, smoothing=True):
    # x0 is not needed because we already defined X, Y, Z,
    # X, Y, Z = my_grid.XYZ
    if smoothing:
        P_model = full_prob_model(my_grid, param_1d, param_nums, fitted_functions, verbose=False)
        P_model = probability_smoother(P_model)
        P_data = probability_smoother(P_XYZ_Hao)
    else:
        P_model = full_prob_model(my_grid, param_1d, param_nums, fitted_functions, verbose=False)
        P_data = P_XYZ_Hao
    return JS_divergence(P_data, P_model)

def to_be_minimized(param_1d):
    # x0 is not needed because we already defined X, Y, Z,
    P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
    P_data = probability_smoother(P_XYZ_Hao)
    return KL_divergence(P_data, P_model)

def to_be_minimized_normalized(param_1d):
    # x0 is not needed because we already defined X, Y, Z,
    P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
    P_model = P_model/np.sum(P_model)
    P_data = probability_smoother(P_XYZ_Hao)
    P_data = P_data/np.sum(P_data)
    return KL_divergence(P_data, P_model)

def to_be_minimized_backward(param_1d):
    # x0 is not needed because we already defined X, Y, Z,
    P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
    P_data = probability_smoother(P_XYZ_Hao)
    return KL_divergence(P_model, P_data)

def to_be_minimized_backward_normalized(param_1d):
    # x0 is not needed because we already defined X, Y, Z,
    P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
    P_model = P_model/np.sum(P_model)
    P_data = probability_smoother(P_XYZ_Hao)
    P_data = P_data/np.sum(P_data)
    return KL_divergence(P_model, P_data)

def to_be_minimized_L2(param_1d):
    # x0 is not needed because we already defined X, Y, Z,
    P_model = probability_smoother(full_prob_model(X, Y, Z, param_1d, verbose=False))
    P_data = probability_smoother(P_XYZ_Hao)
    return L2_forward(P_data, P_model)

#

def bounds_checker(vals, bounds):
    if len(vals) == len(bounds):
        pass
    else:
        raise ValueError("len(params) != len(bounds)")

    check_result = []
    for i in range(len(vals)):
        bound1, bound2 = bounds[i]
        val = vals[i]
        if bound1 is None or val >= bound1:
            if bound2 is None or val <= bound2:
                check_result.append(True)
            else:
                check_result.append(False)
        else:
            check_result.append(False)
    return check_result

# test
def boundary_tester(z1, z2):
    if (z1==None):
        warnings.warn("z1 is not given; assumed as -np.inf.")
        z1 = -np.inf
    if (z2==None):
        warnings.warn("z2 is not given; assumed as +np.inf.")
        z2 = np.inf
    if z1 > z2:
        warnings.warn("z1 > z2. Flipping them now.")
        z1_temp = z1
        z1 = z2
        z2 = z1_temp
    if z1 == z2:
        raise ValueError("z1 == z2; they must be different.")
    return z1, z2

# class
# class ProbabilityModel():
#     def __init__(self, P_XYZ_Hao, P_XYZ_model, X, Y, Z):
#         if P_XYZ_Hao.shape != P_XYZ_model.shape:
#             raise ValueError(f"P_XYZ_Hao.shape != P_XYZ_model.shape. {P_XYZ_Hao.shape}!={P_XYZ_model.shape}")
#         self.P_XYZ_Hao = P_XYZ_Hao
#         self.P_XYZ_model = P_XYZ_model
#         self.X = X
#         self.Y = Y
#         self.Z = Z
#
#     def marginal_prob(self, P_XYZ, axis, lim1, lim2):
#
#     def P_YZ(self, P_XYZ, x1, x2):
#         x1, x2 = boundary_tester(x1, x2)
#         mask = (self.X >= x1) & (self.X <= x2)
#         if np.sum(mask) == 0:
#             warnings.warn("There is no X that satisfies x1 <= X <= x2. It may be because of x1 and x2 too close.")
#         return np.sum(P_XYZ(mask), axis=0)
#
#     def P_XZ(self, P_XYZ, y1, y2):
#         y1, y2 = boundary_tester(y1, y2)
#         mask = (self.Y >= y1) & (self.Y <= y2)
#         if np.sum(mask) == 0:
#             warnings.warn("There is no Z that satisfies z1 <= Z <= z2. It may be because of z1 and z2 too close.")
#         return np.sum(P_XYZ(mask), axis=1)
#
#     def P_XY(self, P_XYZ, z1=None, z2=None):
#         z1, z2 = boundary_tester(z1, z2)
#         mask = (self.Z >= z1) & (self.Z <= z2)
#         if np.sum(mask) == 0:
#             warnings.warn("There is no Z that satisfies z1 <= Z <= z2. It may be because of z1 and z2 too close.")
#         return np.sum(P_XYZ(mask), axis=2)

#

def prob_dist_Hao_3D(my_grid, hao_data, m, normalize=True, weight=None):
    x = hao_data.x_expanded
    y = hao_data.y_expanded
    z = hao_data.z_expanded
    dx = hao_data.dx_expanded
    dy = hao_data.dy_expanded
    dz = hao_data.dz_expanded
    weight = hao_data.weight_expanded

    X, Y, Z = my_grid.XYZ

    # Initialize the probability distribution
    prob_dist = np.zeros_like(X)

    # Convolve 2D Gaussian with each datapoint
    for i, (xi, yi, zi, dxi, dyi, dzi) in enumerate(zip(x, y, z, dx, dy, dz)):
        gaussian = np.exp(-(
                ((X - xi) ** 2) / (2 * dxi ** 2) + ((Y - yi) ** 2) / (2 * dyi ** 2) + ((Z - zi) ** 2) / (2 * dzi ** 2)
        ))
        # Normalize the probability distribution
        gaussian /= (2 * np.pi)**(3/2) * dxi * dyi * dzi
        if weight is None:
            pass
        else:
            gaussian *= weight[i]
        # if normalize:
        #     gaussian /= np.sum(gaussian) # not necessary for the experimental data because each data point has
        #     # significance of 1. The full normalization can be given at the end.
        prob_dist += gaussian
    if normalize:
        prob_dist = prob_dist / np.sum(prob_dist) / my_grid.delta_x / my_grid.delta_y / my_grid.delta_z
        # such that int prob_dist * delta_x * delta_y * delta_z is 1.
    else:
        prob_dist /= len(x)

    return prob_dist

# def prob_dist_surf_w_errors2(ax, x, y, dx, dy, m, n_grid = 100, **kwargs):
#     # no z projection; only 2d contourf.
#     # Create a meshgrid
#     # x_min, x_max = x.min() - (x.max() - x.min()) / 10, x.max() + (x.max() - x.min()) / 10
#     x_min, x_max = x.min(), x.max()
#     y_min, y_max = -np.pi/2/m, np.pi/2/m #y.min() - (y.max() - y.min()) / 10, y.max() + (y.max() - y.min()) / 10
#     X, Y = np.meshgrid(np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid))
#
#     # Initialize the probability distribution
#     prob_dist = np.zeros_like(X)
#
#     # Convolve 2D Gaussian with each datapoint
#     for xi, yi, dxi, dyi in zip(x, y, dx, dy):
#         gaussian = np.exp(-(((X - xi) ** 2) / (2 * dxi ** 2) + ((Y - yi) ** 2) / (2 * dyi ** 2)))
#         # Normalize the probability distribution
#         gaussian /= 2 * np.pi * dxi * dyi
#         prob_dist += gaussian
#     prob_dist /= len(x)
#
#     # ax.set_xlabel('X')
#     # ax.set_ylabel('Y')
#     # ax.set_zlabel('Probability Density')
#     # ax.set_title('Total Probability Distribution as 3D Surface')
#
#     # Set the view for z-projection if requested
#     # Add contour lines
#     # ax.contourf(X, Y, , cmap='Spectral_r', offset=-1, alpha=0.75)
#     surf = ax.contourf(X, Y, prob_dist, cmap='jet', **kwargs)
#
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(-np.pi/2/m, np.pi/2/m)
#
#     return ax, surf, prob_dist
#
# def plot_xy_2d(x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z,dz, X_name, Y_name, Z_name, plot_max_val, n_grid=100):
#     #
#     mask = (x >= x1) & (x <= x2) & (z >= z1) & (z <= z2)
#     filtered_x = x[mask]
#     filtered_dx = dx[mask]
#     filtered_y = y[mask]
#     filtered_dy = dy[mask]#
#     filtered_z = z[mask]
#     filtered_dz = dz[mask]
#     #
#     mask_flipped = ~mask
#     filtered_x_flipped = x[mask_flipped]
#     filtered_dx_flipped = dx[mask_flipped]
#     filtered_y_flipped = y[mask_flipped]
#     filtered_dy_flipped = dy[mask_flipped]
#     filtered_z_flipped = z[mask_flipped]
#     filtered_dz_flipped = dz[mask_flipped]
#     #
#     ax.set_xlabel(X_name)
#     ax.set_ylabel(Y_name)
#     print(f'')
#     ax.set_title(f"P({X_name},{Y_name}), from Hao")
#     #
#     ax, surf, prob_dist = prob_dist_surf_w_errors2(ax, filtered_x, filtered_y,
#                                 filtered_dx, filtered_dy,
#                                 m,
#                                 n_grid=n_grid,
#                                 vmin=0., vmax=plot_max_val, levels=np.linspace(0, 100, 101),)
#     # ax.set_xlim(x.min() - (x.max()-x.min())/10, x.max() + (x.max()-x.min())/10)
#     ax.set_ylim(-np.pi/2/m, np.pi/2/m)
#
#     # plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#
#     return ax, surf, prob_dist

def plot_xy_prob_dist(x, y, dx, dy, weight,
                      x_min, x_max, y_min, y_max, n_grid, x_name='x', y_name='y'):
    """
    Summation of 2D Gaussian 'footprints' over a grid, then plot via contourf.

    Parameters
    ----------
    x, y, dx, dy, weight : array-like
        Arrays specifying the center (x, y), the standard deviations (dx, dy),
        and the 'weight' (amplitude factor) for each 2D Gaussian.
    x_min, x_max, y_min, y_max : float
        Boundaries for the 2D grid on which we'll evaluate the Gaussians.
    n_grid : int
        Number of points along each dimension for the meshgrid.

    Raises
    ------
    ValueError
        If input arrays have mismatched length or contain NaN.
    """

    # 1) Convert inputs to numpy arrays (at least 1D) and sanity-check
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    dx = np.atleast_1d(dx)
    dy = np.atleast_1d(dy)
    weight = np.atleast_1d(weight)

    # Check they all have the same shape
    n_data = len(x)
    if not (len(y) == len(dx) == len(dy) == len(weight) == n_data):
        raise ValueError(
            "x, y, dx, dy, and weight must all have the same length."
        )

    # Check for NaN
    if (np.isnan(x).any() or np.isnan(y).any() or
            np.isnan(dx).any() or np.isnan(dy).any() or
            np.isnan(weight).any()):
        raise ValueError("One or more input arrays contain NaN.")

    # 2) Create meshgrid
    X_lin = np.linspace(x_min, x_max, n_grid)
    Y_lin = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(X_lin, Y_lin)  # shape = (n_grid, n_grid)

    # 3) Sum up 2D Gaussian distributions
    #    We'll accumulate in total_pdf an array of shape (n_grid, n_grid).
    total_pdf = np.zeros_like(X, dtype=float)

    for i in range(n_data):
        # Grab center & spread for the i-th Gaussian
        xi, yi = x[i], y[i]
        dxi, dyi = dx[i], dy[i]
        w_i = weight[i]

        # 2D Gaussian formula (normalized) at each (X, Y) point:
        #    Gauss_i(X, Y) = (1 / (2*pi*dxi*dyi)) * exp( -[ (X - xi)^2 / (2 dx_i^2) + (Y - yi)^2 / (2 dy_i^2) ] )
        # We'll multiply by w_i so that each data point’s Gaussian is scaled by 'weight[i]'.

        # Protect against zero or negative dx/dy:
        if dxi <= 0 or dyi <= 0:
            # You might choose to skip or raise an error. We'll raise here:
            raise ValueError(f"Found non-positive dx or dy at index {i} (dx={dxi}, dy={dyi}).")

        norm_factor = 1.0 / (2.0 * np.pi * dxi * dyi)
        exponent = (
                -((X - xi) ** 2 / (2.0 * dxi ** 2))
                - ((Y - yi) ** 2 / (2.0 * dyi ** 2))
        )
        gauss_ij = w_i * norm_factor * np.exp(exponent)

        total_pdf += gauss_ij

    # 4) Plot using contourf
    fig = plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, total_pdf, levels=101, cmap='jet')
    plt.colorbar(contour)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title('xy Probability Distribution')
    plt.tight_layout()
    plt.show()
    return fig

def params_nan_clear(params):
    for i, key in enumerate(params.keys()):
        vals = params[key]
        if np.isnan(vals).any():
            print(f"Folloing indices have nan value for {key}:")
            print(np.where(np.isnan(vals))[0].tolist())
        if i==0:
            nan_mask = np.isnan(vals)
        else:
            nan_mask += np.isnan(vals)

    print("Following indices will be removed from params:")
    print(np.where(nan_mask)[0].tolist())

    key_fin = key # final key

    print("Total number of data point changed from:")
    for i, key in enumerate(params.keys()):
        assert len(params[key]) == len(params[key_fin])
    print(len(params[key_fin]))

    for i, key in enumerate(params.keys()):
        params[key] = params[key][~nan_mask]

    print("to:")
    for i, key in enumerate(params.keys()):
        assert len(params[key]) == len(params[key_fin])
    print(len(params[key_fin]))

    return params

def plot_P_XYZ_cut(grid, P_XYZ, x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z,X_name, Y_name, Z_name, plot_max_val,
                   num_grid=100):
    x_rng = grid.x_rng
    y_rng = grid.y_rng
    z_rng = grid.z_rng
    x_ind1 = np.searchsorted(x_rng, x1, side='right')
    x_ind2 = np.searchsorted(x_rng, x2, side='right')
    z_ind1 = np.searchsorted(z_rng, z1, side='right')
    z_ind2 = np.searchsorted(z_rng, z2, side='right')
    x_rng_cut = x_rng[x_ind1:x_ind2]
    y_rng_cut = y_rng
    P_XYZ_cut = P_XYZ[x_ind1:x_ind2,:,z_ind1:z_ind2]
    P_XY_cut = np.sum(P_XYZ_cut, axis=2) * (z_rng[1] - z_rng[0])
    # ordered_vals = np.sort(P_XY_cut.reshape(-1))
    # plot_max_val = ordered_vals[int(plot_max_ratio * (len(ordered_vals)-1))]

    surf = ax.contourf(x_rng_cut, y_rng_cut, P_XY_cut.transpose(), cmap='jet', levels=np.linspace(0, 100, 101),
                       origin='lower', vmin=0., vmax=plot_max_val)
    ax.set_xlabel(f"{X_name}")
    ax.set_ylabel(f"{Y_name}")

    return ax, surf, P_XYZ_cut, P_XY_cut

def plot_P_XYZ_cut_diff(grid, P_XYZ_cut, P_XYZ_Hao_cut, P_XY_cut, P_XY_Hao_cut, x1, x2, z1, z2, ax, canvas, m, x,y,dx,dy,z,\
    X_name, Y_name, Z_name, vmin=None, vmax=None, diff_function=np.subtract):
    x_rng = grid.x_rng
    y_rng = grid.y_rng
    z_rng = grid.z_rng
    x_ind1 = np.searchsorted(x_rng, x1, side='right')
    x_ind2 = np.searchsorted(x_rng, x2, side='right')
    z_ind1 = np.searchsorted(z_rng, z1, side='right')
    z_ind2 = np.searchsorted(z_rng, z2, side='right')
    x_rng_cut = x_rng[x_ind1:x_ind2]
    y_rng_cut = y_rng
    P_XYZ_diff_cut = diff_function(probability_smoother(P_XYZ_cut), probability_smoother(P_XYZ_Hao_cut))
    P_XY_diff_cut = diff_function(probability_smoother(P_XY_cut), probability_smoother(P_XY_Hao_cut))
    if diff_function == np.subtract:
        my_levels = np.linspace(-40, 40, 101)
    else:
        my_levels = 101
    surf = ax.contourf(x_rng_cut, y_rng_cut, P_XY_diff_cut.transpose(), cmap='seismic',
                           origin='lower', levels=my_levels, vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"{X_name}")
    ax.set_ylabel(f"{Y_name}")

    return ax, surf, P_XYZ_diff_cut, P_XY_diff_cut

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the GUI using tkinter with real-time slider updatessave_figure
def create_gui(x, y, dx, dy, z, dz, m):
    Y_name = rf'$\phi_{m}$'
    X_name = rf'$a_{m}/a$'
    Z_name = r'$q$'

    # Initialize the main window
    root = tk.Tk()
    root.title("Joint Probability Plotter with Real-Time Range Updates and Save Button")

    # Define the range for x and z based on the data
    x_min = min(x) - 1e-3
    x_max = max(x) + 1e-3
    z_min = min(z) - 1e-3
    z_max = max(z) + 1e-3

    # Create the matplotlib figure and axis
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_position([0.15, 0.15, 0.8, 0.7])
    fig = plt.figure(figsize=(16, 12))
    ax0 = fig.add_axes([0.1, 0.55, 0.35, 0.35])
    ax1 = fig.add_axes([0.55, 0.55, 0.35, 0.35])
    ax2 = fig.add_axes([0.55, 0.05, 0.35, 0.35])
    ax3 = fig.add_axes([0.1, 0.05, 0.35, 0.35], projection='3d')
    axes = [ax0, ax1, ax2, ax3]
    # axes[0].set_position([0.15, 0.55, 0.4, 0.35])
    # axes[1].set_position([0.50, 0.55, 0.4, 0.35])
    # axes[2].set_position([0.50, 0.15, 0.4, 0.35])
    # axes[3].

    # Embed the matplotlib plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(column=0, row=2, columnspan=4)

    # Function to update the plot when the sliders are moved
    def update_plot(val=None):
        x1 = slider_x1.get()
        x2 = slider_x2.get()
        z1 = slider_z1.get()
        z2 = slider_z2.get()
        z_proj_state = z_proj.get()
        # z_proj = proj_check_button.get()
        if x1 > x2 or z1 > z2:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            print("Invalid ranges: x1 <= x2 and z1 <= z2 required.")
        else:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            axes[3].clear()
            plot_filtered_y(x1, x2, z1, z2, axes[0], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
            plot_filtered_x(x1, x2, z1, z2, axes[2], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
            plot_xy(x1, x2, z1, z2, axes[1], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
            plot_xy_3d(x1, x2, z1, z2, axes[3], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name, z_proj=z_proj_state)
            #
            #
            update_x_lines()
        fig.canvas.draw()

    def update_x_lines(val=None):
        for line in axes[1].lines:
            axes[1].lines.remove(line)
        x1 = slider_x1.get()
        x2 = slider_x2.get()
        z1 = slider_z1.get()
        z2 = slider_z2.get()
        if x1 > x2 or z1 > z2:
            axes[0].clear()
            axes[1].clear()
            print("Invalid ranges: x1 <= x2 and z1 <= z2 required.")
        else:
            plot_x_lines(x1, x2, z1, z2, axes[1], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
        fig.canvas.draw()

    resolution_number = 20
    # Create sliders for x1, x2, z1, and z2
    slider_x1 = tk.Scale(root, from_=x_min, to=x_max, resolution=(x_max - x_min) / resolution_number,
                         orient="horizontal", label=X_name + " min", command=update_x_lines)
    slider_x1.grid(column=0, row=0, padx=10, pady=10)

    slider_x2 = tk.Scale(root, from_=x_min, to=x_max, resolution=(x_max - x_min) / resolution_number,
                         orient="horizontal", label=X_name + " max", command=update_x_lines)
    slider_x2.grid(column=1, row=0, padx=10, pady=10)

    slider_z1 = tk.Scale(root, from_=z_min, to=z_max, resolution=(z_max - z_min) / resolution_number,
                         orient="horizontal", label=Z_name + " min")
    slider_z1.grid(column=2, row=0, padx=10, pady=10)

    slider_z2 = tk.Scale(root, from_=z_min, to=z_max, resolution=(z_max - z_min) / resolution_number,
                         orient="horizontal", label=Z_name + " max")
    slider_z2.grid(column=3, row=0, padx=10, pady=10)

    # Set default values for sliders
    slider_x1.set(x_min)
    slider_x2.set(x_max)
    slider_z1.set(z_min)
    slider_z2.set(z_max)

    # Create a button to update the plot
    plot_button = ttk.Button(root, text="Plot Now", command=update_plot)
    plot_button.grid(column=1, row=1, padx=5, pady=5)

    # Create a button to save the plot
    save_button = ttk.Button(root, text="Save Plot",
                             command=lambda: save_figure(slider_x1.get(), slider_x2.get(), slider_z1.get(),
                                                         slider_z2.get(), fig, X_name, Y_name, Z_name))
    save_button.grid(column=0, row=1, columnspan=4)

    # int_var = tk.IntVar(value=1)  # Represents True
    z_proj = tk.BooleanVar(value=True)
    proj_check_button = ttk.Checkbutton(root, text="z-Projection", variable=z_proj, onvalue=True, offvalue=False)
    proj_check_button.grid(column=2, row=1)
    # Initial plot with default values
    update_plot()

    # Run the main event loop
    root.mainloop()

# Create the GUI using tkinter with real-time slider updatessave_figure
def create_gui2(grid, hao_data, m, P_XYZ, P_XYZ_Hao, diff_function=np.subtract, num_grid=100):
    x = hao_data.x_expanded
    y = hao_data.y_expanded
    z = hao_data.z_expanded
    dx = hao_data.dx_expanded
    dy = hao_data.dy_expanded
    dz = hao_data.dz_expanded

    Y_name = rf'$\phi_{m}$'
    X_name = rf'$a_{m}/a$'
    Z_name = r'$q$'

    # Initialize the main window
    root = tk.Tk()
    root.title("Joint Probability Plotter with Real-Time Range Updates and Save Button")

    # Define the range for x and z based on the data
    x_min = min(x) - 1e-3
    x_max = max(x) + 1e-3
    z_min = min(z) - 1e-2
    z_max = max(z) + 1e-2

    # Create the matplotlib figure and axis
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.set_position([0.15, 0.15, 0.8, 0.7])
    size_ratio = 1.0
    fig = plt.figure(figsize=(16*size_ratio, 12*size_ratio))
    # ax0, ax1
    # ax2, ax3
    ax0 = fig.add_axes([0.075, 0.55, 0.35, 0.35]) # North West # plot_P_XYZ_cut, cbar needed
    ax1 = fig.add_axes([0.55, 0.55, 0.35, 0.35])  # North East # plot_xy
    ax2 = fig.add_axes([0.075, 0.05, 0.35, 0.35]) # South West # plot_xy_2d, cbar needed
    ax3 = fig.add_axes([0.55, 0.05, 0.35, 0.35]) # South East # different plot
    # ax3 = fig.add_axes([0.1, 0.05, 0.35, 0.35], projection='3d')
    axes = [ax0, ax1, ax2, ax3]
    # color axes
    # cax0, cax1
    # cax2, cax3
    cax0 = fig.add_axes([0.45, 0.55, 0.025, 0.35]) # North West
    cax2 = fig.add_axes([0.45, 0.05, 0.025, 0.35]) # South West
    cax3 = fig.add_axes([0.55+0.35+0.025, 0.05, 0.025, 0.35]) # South East
    caxes = [cax0, [], cax2, cax3]
    # axes[0].set_position([0.15, 0.55, 0.4, 0.35])
    # axes[1].set_position([0.50, 0.55, 0.4, 0.35])
    # axes[2].set_position([0.50, 0.15, 0.4, 0.35])
    # axes[3].

    # Embed the matplotlib plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(column=0, row=3, columnspan=4)

    return_values = {'P_XYZ_cut': None, 'P_XY_cut': None,
                     'P_XYZ_Hao_cut': None, 'P_XY_Hao_cut': None,
                     'P_XYZ_diff_cut': None, 'P_XY_diff_cut': None}

    # Function to update the plot when the sliders are moved
    def update_plot(val=None):
        x1 = slider_x1.get()
        x2 = slider_x2.get()
        z1 = slider_z1.get()
        z2 = slider_z2.get()
        slider_max = slider_max_contourf.get()
        slider_max2 = slider_max_contourf2.get() # for the difference colorbar
        # z_proj_state = z_proj.get()
        # z_proj = proj_check_button.get()
        if x1 > x2 or z1 > z2:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            print("Invalid ranges: x1 <= x2 and z1 <= z2 required.")
        else:
            axes[0].clear()
            axes[1].clear()
            axes[2].clear()
            axes[3].clear()
            caxes[0].clear()
            caxes[2].clear()
            caxes[3].clear()
            plot_xy(x1, x2, z1, z2, axes[1], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
            # plot_xy_3d(x1, x2, z1, z2, axes[3], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name, z_proj=z_proj_state)
            #
            _, surf2, P_XYZ_cut, P_XY_cut = plot_P_XYZ_cut(grid, P_XYZ, x1, x2, z1, z2, axes[2], canvas, m, x, y, dx,
                                                           dy, z,
                                                           X_name, Y_name, Z_name,
                                 slider_max, num_grid=num_grid)
            axes[2].set_title(f"P({X_name},{Y_name} | {z1}<{Z_name}<{z2}), parameterized")
            #
            cbar2 = fig.colorbar(surf2, cax=caxes[2])
            cbar2.set_ticks(np.linspace(0, 100, 11))  # Set ticks from 0 to 100 with an interval of 10
            cbar2.set_ticklabels([f"{int(tick)}" for tick in np.linspace(0, 100, 11)])
            #
            _, surf0, P_XYZ_Hao_cut, P_XY_Hao_cut = plot_P_XYZ_cut(grid, P_XYZ_Hao, x1, x2, z1, z2, axes[0], canvas, m, x, y,
                                                                dx,
                                                           dy, z, X_name, Y_name, Z_name,
                                                           slider_max, num_grid=num_grid)
            axes[0].set_title(f"P({X_name},{Y_name} | {z1}<{Z_name}<{z2}), Hao")
            # _, surf3, prob_dist_convolved = plot_xy_2d(x1, x2, z1, z2, axes[3], canvas, m, x, y, dx, dy, dz, X_name,
            #                                         Y_name, Z_name, slider_max, num_grid=num_grid)
            # surf3.set_clim(0,100)
            cbar0 = fig.colorbar(surf0, cax=caxes[0])
            cbar0.set_ticks(np.linspace(0, 100, 11))  # Set ticks from 0 to 100 with an interval of 10
            cbar0.set_ticklabels([f"{int(tick)}" for tick in np.linspace(0, 100, 11)])
            #
            _, surf3, P_XYZ_diff_cut, P_XY_diff_cut = plot_P_XYZ_cut_diff(grid, P_XYZ_cut, P_XYZ_Hao_cut, P_XY_cut,
                                                                         P_XY_Hao_cut, x1, x2, z1, z2, axes[3], canvas,
                                                                         m, x,y,dx,
                                                                            dy,z, X_name, Y_name, Z_name,
                                                                          #vmin=-slider_max2, vmax=slider_max2,
                                                                          diff_function=diff_function)
            #
            cbar3 = fig.colorbar(surf3, cax=caxes[3])
            # cbar3.set_ticks(np.linspace(-40, 40, 9))  # Set ticks from 0 to 100 with an interval of 10
            # cbar3.set_ticklabels([f"{int(tick)}" for tick in np.linspace(-40, 40, 9)])
            #
            return_values['P_XYZ_cut'] = P_XYZ_cut
            return_values['P_XY_cut'] = P_XY_cut
            return_values['P_XYZ_Hao_cut'] = P_XYZ_Hao_cut
            return_values['P_XY_Hao_cut'] = P_XY_Hao_cut
            return_values['P_XYZ_diff_cut'] = P_XYZ_diff_cut
            return_values['P_XY_diff_cut'] = P_XY_diff_cut
            update_x_lines()
        fig.canvas.draw()

    def update_x_lines(val=None):
        for line in axes[1].lines:
            axes[1].lines.remove(line)
        x1 = slider_x1.get()
        x2 = slider_x2.get()
        z1 = slider_z1.get()
        z2 = slider_z2.get()
        slider_max = slider_max_contourf.get()
        if x1 > x2 or z1 > z2:
            axes[0].clear()
            axes[1].clear()
            print("Invalid ranges: x1 <= x2 and z1 <= z2 required.")
        else:
            plot_x_lines(x1, x2, z1, z2, axes[1], canvas, m, x, y, dx, dy, z, X_name, Y_name, Z_name)
        fig.canvas.draw()

    resolution_number = 20
    # Create sliders for x1, x2, z1, and z2
    slider_x1 = tk.Scale(root, from_=x_min, to=x_max, resolution=(x_max - x_min) / resolution_number,
                         orient="horizontal", label=X_name + " min", command=update_x_lines)
    slider_x1.grid(column=0, row=0, padx=10, pady=10)

    slider_x2 = tk.Scale(root, from_=x_min, to=x_max, resolution=(x_max - x_min) / resolution_number,
                         orient="horizontal", label=X_name + " max", command=update_x_lines)
    slider_x2.grid(column=1, row=0, padx=10, pady=10)

    slider_z1 = tk.Scale(root, from_=z_min, to=z_max, resolution=(z_max - z_min) / resolution_number,
                         orient="horizontal", label=Z_name + " min")
    slider_z1.grid(column=2, row=0, padx=10, pady=10)

    slider_z2 = tk.Scale(root, from_=z_min, to=z_max, resolution=(z_max - z_min) / resolution_number,
                         orient="horizontal", label=Z_name + " max")
    slider_z2.grid(column=3, row=0, padx=10, pady=10)

    slider_max_contourf = tk.Scale(root, from_=0., to=100, resolution=1.,
                         orient="horizontal", label= "Contourf Max")
    slider_max_contourf.grid(column=0, row=1, padx=10, pady=10)

    # for the difference plot
    slider_max_contourf2 = tk.Scale(root, from_=0., to=40, resolution=1.,
                         orient="horizontal", label= "Difference Max")
    slider_max_contourf2.grid(column=1, row=1, padx=10, pady=10)

    # Set default values for sliders
    slider_x1.set(x_min)
    slider_x2.set(x_max)
    slider_z1.set(z_min)
    slider_z2.set(z_max)
    slider_max_contourf.set(100.)
    slider_max_contourf2.set(40.)

    # Create a button to update the plot
    plot_button = ttk.Button(root, text="Plot Now", command=update_plot)
    plot_button.grid(column=1, row=2, padx=5, pady=5)

    # Create a button to save the plot
    save_button = ttk.Button(root, text="Save Plot",
                             command=lambda: save_figure(slider_x1.get(), slider_x2.get(), slider_z1.get(),
                                                         slider_z2.get(), fig, X_name, Y_name, Z_name,
                                                         comment='_init_guess'))
    save_button.grid(column=0, row=2, columnspan=4)

    # int_var = tk.IntVar(value=1)  # Represents True
    # z_proj = tk.BooleanVar(value=True)
    # proj_check_button = ttk.Checkbutton(root, text="z-Projection", variable=z_proj, onvalue=True, offvalue=False)
    # proj_check_button.grid(column=2, row=1)
    # Initial plot with default values
    update_plot()

    # Run the main event loop
    root.mainloop()
    # After closing the GUI, return the values
    return return_values['P_XYZ_cut'], return_values['P_XY_cut'], return_values['P_XYZ_Hao_cut'], return_values[
        'P_XY_Hao_cut'], return_values['P_XYZ_diff_cut'], return_values['P_XY_diff_cut']
