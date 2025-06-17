import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

plot_projections_bool = False

result_path = "./torch_fit_result/run_result_20250609_022317"
assert os.path.exists(result_path)
save_path = "./torch_fit_result/run_result_20250609_022317_MCMC_validation"
if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)

best_result_json_file = os.path.join(result_path, "best_results.json")

n_interp_m3 = 2
n_interp_m4 = 3
load_key = f"n_interp_m3={n_interp_m3}__n_interp_m4={n_interp_m4}"

with open(best_result_json_file, "r") as json_file:
    loaded_best_results = json.load(json_file)

best_params = loaded_best_results[load_key]['params']
best_loss = loaded_best_results[load_key]['loss']

# Load from JSON file

def params_to_dict(params):
    # params: dictionary with all parameters
    z_parameters = {}
    m3_parameters = {}
    m4_parameters = {}
    for key in params.keys():
        if key[-2:] == '_Z':
            z_parameters[key] = params[key]
        elif key[-2:] == '_3':
            m3_parameters[key] = params[key]
        elif key[-2:] == '_4':
            m4_parameters[key] = params[key]
        else:
            raise ValueError(f"Unknown key {key}")
    return z_parameters, m3_parameters, m4_parameters

z_parameters, m3_parameters, m4_parameters = params_to_dict(best_params)

from grid_info_module import grid_info
from P_xyz_fit_w_torch_code import (load_model_state, load_best_result,
                                                                              best_result_analysis_plot,
                                                                              DistributionModel, grid_info,
                                                                              data_file_loader, plot_projections,js_divergence_loss)

my_grid3, P_X3Y3Z_Hao, my_grid4, P_X4Y4Z_Hao = data_file_loader()

model = DistributionModel(z_parameters, m3_parameters, m4_parameters, my_grid3, my_grid4,
                 n_interp_m3=2, n_interp_m4=3,
                 use_identical_domain_m3=False, use_identical_domain_m4=False)

Z_name = r"$q$"
X3_name = r"$a_3/a$"
Y3_name = r"$\phi_3-\phi_0$"
X4_name = r"$a_4/a$"
Y4_name = r"$\phi_4-\phi_0$"

n_slices = 4
if plot_projections_bool:
    plot_projections(model, P_X3Y3Z_Hao, P_X4Y4Z_Hao, my_grid3, my_grid4, n_slices=n_slices,save_path=save_path,
                     prefix='post_analysis',suffix=f'n_slices={n_slices}')
plt.show()
print("Plotting Done!")

# =================
# Test if probability sampling works. The output value here is probability density function's value
# =================
x3 = 0
y3 = 0
x4 = 0
y4 = 0
z = 0.5

prob = model.forward_from_nontorch_single_point(x3, y3, x4, y4, z)

print("Probability: ", prob)

print("Prob calculation Done!")

#%% MCMC test
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# MCMC Settings
# ----------------------
# Set n_points for the chain
ndim = 5          # number of parameters: [x3, y3, x4, y4, z]
nwalkers = 32     # number of walkers
total_samples = int(1e5)  # target total samples; better if it is int(1e6) or more.
nsteps = total_samples // nwalkers  # steps per walker

# Limits (from my_grid3 and my_grid4)
limits = {
    'x3_min': my_grid3.x_min, 'x3_max': my_grid3.x_max,
    'y3_min': my_grid3.y_min, 'y3_max': my_grid3.y_max,
    'x4_min': my_grid4.x_min, 'x4_max': my_grid4.x_max,
    'y4_min': my_grid4.y_min, 'y4_max': my_grid4.y_max,
    'z_min': my_grid3.z_min, 'z_max': my_grid3.z_max  # same for both grids
}

# Proposal standard deviations: sigma = (max - min)/10 for each variable.
# sigma_divider = 10
# proposal_sigmas = np.array([
#     (limits['x3_max'] - limits['x3_min']) / sigma_divider,
#     (limits['y3_max'] - limits['y3_min']) / sigma_divider,
#     (limits['x4_max'] - limits['x4_min']) / sigma_divider,
#     (limits['y4_max'] - limits['y4_min']) / sigma_divider,
#     (limits['z_max'] - limits['z_min']) / sigma_divider
# ])

# Initial state: Use the midpoint of each range.
init_state = np.array([
    0,
    0,
    0,
    0,
    (limits['z_min'] + limits['z_max']) / 2
])

# ----------------------
# Run MCMC
# ----------------------
# if n_points > 1e5:
#     chain, acc_rate = run_mcmc_memmap(model, n_points, init_state, proposal_sigmas, limits)
# else:
#     chain, acc_rate = run_mcmc(model, n_points, init_state, proposal_sigmas, limits)
# Initialize walkers in a small ball around the initial state.
# Here, we use a perturbation of 1e-4 times a random Gaussian.
pos = init_state + 1e-4 * np.random.randn(nwalkers, ndim)

import emcee

def log_prob(theta, model, limits):
    """
    Compute the log-probability for theta = [x3, y3, x4, y4, z].

    Parameters:
      theta: array-like, shape (5,)
      model: an object with a method forward_from_nontorch_single_point(x3, y3, x4, y4, z)
      limits: dictionary with keys: 'x3_min', 'x3_max', 'y3_min', 'y3_max',
              'x4_min', 'x4_max', 'y4_min', 'y4_max', 'z_min', 'z_max'

    Returns:
      log(probability) or -infinity if out of bounds.
    """
    x3, y3, x4, y4, z = theta
    # Enforce limits
    if (x3 < limits['x3_min'] or x3 > limits['x3_max'] or
            y3 < limits['y3_min'] or y3 > limits['y3_max'] or
            x4 < limits['x4_min'] or x4 > limits['x4_max'] or
            y4 < limits['y4_min'] or y4 > limits['y4_max'] or
            z < limits['z_min'] or z > limits['z_max']):
        return -np.inf

    # Evaluate the (unnormalized) probability density at the given point.
    prob = model.forward_from_nontorch_single_point(x3, y3, x4, y4, z)
    if prob <= 0:
        return -np.inf
    return np.log(prob)

# Set up the emcee EnsembleSampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(model, limits))

chain = None

# Run MCMC with progress output.
sampler.run_mcmc(pos, nsteps, progress=True)

#%%
# Get the flattened chain: shape (nwalkers * nsteps, ndim)
chain = sampler.get_chain(flat=True)

# Save the chain result to a file.
np.save("mcmc_chain.npy", chain)
print("MCMC chain saved to mcmc_chain.npy")

# --------------------------
# Plot the chain traces for each variable.
# --------------------------
labels = ["X3", "Y3", "X4", "Y4", "Z"]
fig, axs = plt.subplots(ndim, 1, figsize=(10, 2*ndim), sharex=True)
for i in range(ndim):
    axs[i].plot(chain[::100, i], lw=0.5)
    axs[i].set_ylabel(labels[i])
    axs[i].set_title(f"Chain trace for {labels[i]}")
axs[-1].set_xlabel("MCMC iteration")
plt.tight_layout()
plt.show()

# print(f"MCMC Acceptance Rate: {acc_rate:.3f}")

# ----------------------
# Plot the MCMC chain traces for all five variables
# ----------------------
labels = [X3_name, Y3_name, X4_name, Y4_name, Z_name]
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

for i in range(5):
    axs[i].plot(chain[:, i], lw=0.5)
    axs[i].set_ylabel(labels[i])
    axs[i].set_title(f"Chain for {labels[i]}")

axs[-1].set_xlabel("MCMC Iteration")
plt.tight_layout()
plt.show()


#%% MCMC Result histogram

def plot_mcmc_histograms(chain, n_hist=100, burn_in=0.1, save_path='.', prefix='', suffix=''):
    """
    Plots 1D histograms for each of the 5 variables from an MCMC chain.

    Parameters:
      chain: numpy array of shape (n_points, 5)
      n_hist: number of bins in each histogram (default 100)
      burn_in: fraction of initial samples to discard (default 0.1)

    This function creates one figure with 5 subplots (one per variable).
    """
    n_points = chain.shape[0]
    burn_in_index = int(n_points * burn_in)
    chain_used = chain[burn_in_index:, :]  # discard burn-in

    labels = ["X3", "Y3", "X4", "Y4", "Z"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].hist(chain_used[:, i], bins=n_hist, density=True, color='skyblue', edgecolor='k')
        axes[i].set_title(f"Histogram of {labels[i]}")
        axes[i].set_xlabel(labels[i])
        axes[i].set_ylabel("Density")
    plt.tight_layout()
    fig_savename = f"{prefix}_mcmc_result_histogram_{suffix}.pdf"
    fig.savefig(os.path.join(save_path, fig_savename))
    print(f"MCMC histogram: Saved figure to {fig_savename}")
    plt.show()

if chain is not None:
    plot_mcmc_histograms(chain, save_path=save_path, prefix='post_analysis', suffix='')
    # plot_mcmc_histograms(chain2, save_path=save_path, prefix='post_analysis', suffix='emcee_1e6')

plot_projections(model, P_X3Y3Z_Hao, P_X4Y4Z_Hao, my_grid3, my_grid4, n_slices=n_slices, save_path=save_path,
                 prefix='post_analysis', suffix=f'n_slices={n_slices}',
                 chain = chain, hist2d_bins=50, burn_in_ratio=0.2,
)

# plot_projections(model, P_X3Y3Z_Hao, P_X4Y4Z_Hao, my_grid3, my_grid4, n_slices=n_slices, save_path=save_path,
#                  prefix='post_analysis', suffix=f'n_slices={n_slices}_emcee_1e6',
#                  chain = chain2, hist2d_bins=50, burn_in_ratio=0.5,
# )

print("done!")