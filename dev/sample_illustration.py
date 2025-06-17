#!/usr/bin/env python
# ------------------------------------------------------------
# 11-axis illustration of the distribution design
# ------------------------------------------------------------
import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
# ------------- rc tweaks ------------
# mpl.rcParams.update({
#     "axes.labelsize": 11,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
# })

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 15
})

interpolation_color=(0.5,0.5,1)
markersize=4
markerfacecolor=(1,1,1)
markeredgecolor=(0,0,1)

blue_fill = "#b0c4de"
blue_edge = "#326ba8"
red_line  = "#d62728"

# ---------- helper to add arrows / cross-hair -------------
def sampling_arrow(ax, x, y, vertical=True, arrow=False):
    """Black arrow from PDF to axis at x (or y)."""
    lw=2
    ms=25
    if arrow:
        arrowprops=dict(arrowstyle="-|>", color="k", lw=lw, mutation_scale=ms)
        if vertical:
            ax.annotate("",
                xy=(x, 0), xycoords=("data", "axes fraction"),
                xytext=(x, y), textcoords=("data", "data"),
                arrowprops=arrowprops)
        else:
            ax.annotate("",
                xy=(0, y), xycoords=("axes fraction", "data"),
                xytext=(x, y), textcoords=("data", "data"),
                arrowprops=arrowprops)
    else:
        if vertical:
            ax.plot((x, x), (y, 0), 'k-')
        else:
            ax.plot((x, 0), (y, y), 'k-')

def param_crosshair(ax, x, y):
    lw=1.5
    # ax.plot(x, y, marker="+", ms=6, mec=red_line, mew=1.3)
    # tiny guidelines
    ax.hlines(y, ax.get_xlim()[0], x, colors=red_line, lw=lw)
    ax.vlines(x, ax.get_ylim()[0], y, colors=red_line, lw=lw)

#  -----------  CONFIG  -----------------
result_path   = "./torch_fit_result/run_result_20250609_022317"
n_interp_m3 = 2
n_interp_m4 = 3
load_key = f"n_interp_m3={n_interp_m3}__n_interp_m4={n_interp_m4}"
best_file     = os.path.join(result_path, "best_results.json")

# LaTeX axis names
Z_name  = r"$q$"
X3_name = r"$a_3/a$"
Y3_name = r"$\phi_3-\phi_0$"
X4_name = r"$a_4/a$"
Y4_name = r"$\phi_4-\phi_0$"

n_sigma = 2              # if you want tighter / broader x-ranges

# ----------  load params & model  -----------------
with open(best_file, "r") as jf:
    best_params = json.load(jf)[load_key]["params"]

def params_to_dict(params):
    z_par, m3_par, m4_par = {}, {}, {}
    for k, v in params.items():
        if k.endswith("_Z"):   z_par[k]  = v
        elif k.endswith("_3"): m3_par[k] = v
        elif k.endswith("_4"): m4_par[k] = v
        else: raise ValueError(f"Unknown key {k}")
    return z_par, m3_par, m4_par

z_par, m3_par, m4_par = params_to_dict(best_params)

# import your grid objects and DistributionModel
from grid_info_module import grid_info
from P_xyz_fit_w_torch_code import (load_model_state, load_best_result,
                                                                              best_result_analysis_plot,
                                                                              DistributionModel, grid_info,
                                                                              data_file_loader, plot_projections,js_divergence_loss)

my_grid3, P_X3Y3Z_Hao, my_grid4, P_X4Y4Z_Hao = data_file_loader()
# from run_result_20250323_001049_long_24_23.hao_p_xyz_fit_w_torch_code import DistributionModel, data_file_loader
# from grid_info_module import grid_info          # <- adjust import!
# my_grid3, P_X3Y3Z_Hao, my_grid4, P_X4Y4Z_Hao = data_file_loader()

model = DistributionModel(z_par, m3_par, m4_par,
                          my_grid3, my_grid4,
                          n_interp_m3=2, n_interp_m4=3,
                          use_identical_domain_m3=False,
                          use_identical_domain_m4=False)

device = torch.device("cpu")   # put tensors on GPU if you wish

# choose illustrative sampling points (mid-range values)
z0  = 0.6
x3  = 0.01
y3  = np.pi/12/2
x4  = 0.02
y4  = -np.pi/16/2

import matplotlib.gridspec as gridspec


def create_custom_layout(unit_width=2.5, unit_height=1.8, h_gap=1.5, v_gap=1.0):
    # Total width in inches: 4 units + 3 gaps + 2 side margins
    total_width = 4 * unit_width + 5 * h_gap
    # Total height in inches: 5 units + 4 gaps + 2 vertical margins
    total_height = 5 * unit_height + 6 * v_gap

    fig = plt.figure(figsize=(total_width, total_height))

    def add_ax(left_units, bottom_units, w_units, h_units):
        left = h_gap + left_units * (unit_width + h_gap)
        bottom = v_gap + bottom_units * (unit_height + v_gap)
        width = w_units * unit_width + (w_units - 1) * h_gap
        height = h_units * unit_height + (h_units - 1) * v_gap
        return fig.add_axes([
            left / total_width,
            bottom / total_height,
            width / total_width,
            height / total_height
        ])

    # # From bottom to top (matplotlib uses bottom-left origin)
    # ax0  = add_ax(0, 4, 3, 1)  # Top row, full width
    # ax1  = add_ax(0, 3, 1, 1)
    # ax4  = add_ax(1, 3, 1, 1)
    # ax5  = add_ax(2, 3, 1, 1)
    # ax6  = add_ax(3, 3, 1, 1)
    # ax2  = add_ax(0, 2, 1, 1)
    # ax7  = add_ax(1, 2, 3, 1)
    # ax8  = add_ax(1, 1, 1.5, 1)
    # ax9  = add_ax(2.5, 1, 1.5, 1)
    # ax3  = add_ax(0, 0, 1, 1)
    # ax10 = add_ax(1, 0, 3, 1)

    # From bottom to top (matplotlib uses bottom-left origin)
    ax0  = add_ax(0, 4, 2, 1)  # Top row, full width
    ax1  = add_ax(0, 3, 1, 1)
    ax4  = add_ax(1, 3, 1, 1)
    ax5  = add_ax(2, 3, 1, 1)
    ax6  = add_ax(3, 3, 1, 1)
    ax2  = add_ax(0, 2, 1, 1)
    ax7  = add_ax(1, 2, 3, 1)
    ax8  = add_ax(1, 1, 1.5, 1)
    ax9  = add_ax(2.5, 1, 1.5, 1)
    ax3  = add_ax(0, 0, 1, 1)
    ax10 = add_ax(1, 0, 3, 1)

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    return fig, axes

fig, axes = create_custom_layout(unit_width=1.5, unit_height=1.0, h_gap=0.75, v_gap=0.75)
# fig, axes = create_custom_layout(unit_width=2.5, unit_height=1.8, h_gap=1.5, v_gap=1)
# fig.savefig("text_ax_layout.pdf")
# plt.show()

# ---------- ax0:  P(q)  --------------------------
ax = axes[0]
z_grid = torch.linspace(my_grid3.z_min, my_grid3.z_max, 400)
pz = model.pz(z_grid).detach().numpy()
ax.fill_between(z_grid, pz, facecolor=blue_fill)
ax.set_xlim(my_grid3.z_min, my_grid3.z_max)
ax.set_ylim(bottom=0.)
ax.set_ylabel(r"$P(q)$")
ax.set_xlabel(Z_name)
# ax.axvline(z0, color="k", ls=":")
sampling_arrow(ax, z0, pz[np.abs(z_grid-z0).argmin()])
# param_crosshair(ax, z0, 0)
# ax.text(z0, ax.get_ylim()[1]*0.9, r"$q_0$", ha="center")

# ---------- ax1:  σ(q)  --------------------------
ax = axes[1]
sigma = model.sigma_function(z_grid, model.sigma_z_x_ctrl_3,
                             model.sigma_z_y_ctrl_3).detach().numpy()
ax.plot(z_grid, sigma, '-', color=interpolation_color)
ax.plot(model.sigma_z_x_ctrl_3.detach().numpy(), model.sigma_z_y_ctrl_3.detach().numpy(), 'o',
        markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
ax.set_xlim(my_grid3.z_min, my_grid3.z_max)
ax.set_ylim(bottom=0.003, top=0.01)
# ax.set_yticks([0, 0.005, 0.01])
ax.set_ylabel(r"$\sigma(q)$")
ax.set_xlabel(Z_name)
# ax.axvline(z0, color="k", ls=":")
param_crosshair(ax, z0, np.interp(z0, z_grid, sigma))
# ax.scatter([z0], [model.sigma_function(torch.tensor(z0), model.sigma_z_x_ctrl_3,
#                                        model.sigma_z_y_ctrl_3).item()], color="k")

# ---------- ax2:  P(X3|q0)  ----------------------
ax = axes[2]
x3_grid = torch.linspace(my_grid3.x_min, my_grid3.x_max, 400)
px3 = model.px3z(x3_grid, torch.tensor(z0)).detach().numpy()
ax.fill_between(x3_grid, px3, facecolor=blue_fill)
ax.set_xlim(my_grid3.x_min, my_grid3.x_max)
ax.set_ylim(bottom=0.)
ax.set_ylabel(r"$P(a_3/a\,|\,q)$")
ax.set_xlabel(X3_name)
# ax.axvline(x3, color="k", ls=":")
sampling_arrow(ax, x3, px3[np.abs(x3_grid-x3).argmin()])
# param_crosshair(ax, x3, 0)
# highlight
# ax.scatter([x3], [model.px3z(torch.tensor(x3), torch.tensor(z0),
#                              single_point=True).item()], color="k")

# ---------- ax3:  P(Y3|X3) -----------------------
ax = axes[3]
y3_grid = torch.linspace(my_grid3.y_min, my_grid3.y_max, 400)
py3 = model.py3x3(y3_grid, torch.tensor(x3)).detach().numpy()
ax.fill_between(y3_grid, py3, facecolor=blue_fill)
ax.set_xlim(my_grid3.y_min, my_grid3.y_max)
ax.set_ylim(bottom=0., top=1.5)
ax.set_xlabel(Y3_name)
ax.set_ylabel(r"$P(\phi_3-\phi_0\,|\,a_3/a)$")
ax.set_xticks([-np.pi/6, 0, np.pi/6])
ax.set_xticklabels([r"$-\pi/6$", r"$0$", r"$\pi/6$"])
# ax.axvline(y3, color="k", ls=":")
sampling_arrow(ax, y3, py3[np.abs(y3_grid-y3).argmin()])
# param_crosshair(ax, y3, 0)
# ax.scatter([y3], [model.py3x3(torch.tensor(y3), torch.tensor(x3),
#                               single_point=True).item()], color="k")

# Linear Spline Eval
def linear_spline_eval(x, x_ctrl, y_ctrl):
    """
    Evaluate a 1D linear spline with control points (x_ctrl, y_ctrl) at positions x.
    We now explicitly sort x_ctrl, y_ctrl each call to ensure ascending order.
    """
    # # Convert to tensors
    if type(x_ctrl) in [torch.Tensor, torch.nn.Parameter]:
        pass
    else:
        x_ctrl = torch.tensor(x_ctrl, dtype=x.dtype, device=x.device)
    if type(y_ctrl) in [torch.Tensor, torch.nn.Parameter]:
        pass
    else:
        y_ctrl = torch.tensor(y_ctrl, dtype=x.dtype, device=x.device)

    # Sort in ascending order
    x_ctrl_sorted, sorted_indices = torch.sort(x_ctrl)
    y_ctrl_sorted = y_ctrl[sorted_indices]

    # Clamp x to [x_ctrl_sorted[0], x_ctrl_sorted[-1]] so we can safely interpolate
    x_clamped = torch.clamp(x, min=x_ctrl_sorted[0], max=x_ctrl_sorted[-1])

    # Find sub-interval for each x
    idxs = torch.searchsorted(x_ctrl_sorted, x_clamped, right=True)
    idxs = torch.clamp(idxs, 1, len(x_ctrl_sorted)-1)

    # Linear interpolation
    x0 = x_ctrl_sorted[idxs-1]
    x1 = x_ctrl_sorted[idxs]
    y0 = y_ctrl_sorted[idxs-1]
    y1 = y_ctrl_sorted[idxs]

    w = (x_clamped - x0) / (x1 - x0 + 1e-12)
    y = y0 + w * (y1 - y0)
    return y

# ---------- ax4–6:  α(q), ξ(q), ω(q) ------------
z_tensor = z_grid
# alpha_q = torch.sigmoid(model.alpha_z_x_ctrl_4_unconstrained).detach().numpy()
alpha_q = linear_spline_eval(z_tensor, model.alpha_z_x_ctrl_4, model.alpha_z_y_ctrl_4)
# xi_q    = torch.sigmoid(model.xi_z_x_ctrl_4_unconstrained).detach().numpy()
xi_q = linear_spline_eval(z_tensor, model.xi_z_x_ctrl_4, model.xi_z_y_ctrl_4)
# omega_q = torch.sigmoid(model.omega_z_x_ctrl_4_unconstrained).detach().numpy()
omega_q = linear_spline_eval(z_tensor, model.omega_z_x_ctrl_4, model.omega_z_y_ctrl_4)

my_y_ranges = [(-1.5, 1.5), (-0.02, 0.06), (0, 0.03)]

for k, (arr, lab, x_ctrl, y_ctrl, y_rng) in enumerate(zip([alpha_q, xi_q, omega_q],
                                   [r"$\alpha(q)$", r"$\xi(q)$", r"$\omega(q)$"],
                                   [model.alpha_z_x_ctrl_4, model.xi_z_x_ctrl_4, model.omega_z_x_ctrl_4],
                                   [model.alpha_z_y_ctrl_4, model.xi_z_y_ctrl_4, model.omega_z_y_ctrl_4],
                                    my_y_ranges
                                   )):
    ax = axes[4+k]
    ax.plot(z_grid, arr.detach().numpy(), '-', color=interpolation_color)
    ax.plot(x_ctrl.detach().numpy(), y_ctrl.detach().numpy(), 'o',
            markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
    ax.set_xlim(my_grid4.z_min, my_grid4.z_max)
    ax.set_ylim(y_rng)
    ax.set_ylabel(lab)
    # ax.axvline(z0, color="k", ls=":")
    param_crosshair(ax, z0, np.interp(z0, z_grid, arr.detach().numpy()))
    ax.set_xlabel(Z_name)
    if k==2:
        ax.set_ylim(bottom=0.)

# ---------- ax7:  P(X4|q0)  ---------------------
ax = axes[7]
x4_grid = torch.linspace(my_grid4.x_min, my_grid4.x_max, 400)
px4 = model.px4z(x4_grid, torch.tensor(z0)).detach().numpy()
ax.fill_between(x4_grid, px4, facecolor=blue_fill)
ax.set_xlim(my_grid4.x_min, my_grid4.x_max)
ax.set_ylim(bottom=0.)
ax.set_ylabel(r"$P(a_4/a\,|\,q_)$")
ax.set_xlabel(X4_name)
# ax.axvline(x4, color="k", ls=":")
sampling_arrow(ax, x4, px4[np.abs(x4_grid-x4).argmin()])
# param_crosshair(ax, x4, 0)
# ax.scatter([x4], [model.px4z(torch.tensor(x4), torch.tensor(z0),
#                               single_point=True).item()], color="k")

# ---------- ax8–9:  α(X4), β(X4) ----------------
x4_tensor = x4_grid
# alpha_x4 = torch.sigmoid(model.alpha_x_ctrl_4_unconstrained).detach().numpy()
alpha_x4 = linear_spline_eval(x4_tensor, model.alpha_x_ctrl_4, model.alpha_y_ctrl_4)
# beta_x4  = torch.sigmoid(model.beta_x_ctrl_4_unconstrained).detach().numpy()
beta_x4 =  linear_spline_eval(x4_tensor, model.beta_x_ctrl_4, model.beta_y_ctrl_4)
my_y_ranges = [(0, 0.6), (1.0, 2.2)]
for k, (arr, lab, x_ctrl, y_ctrl, y_rng) in enumerate(zip([alpha_x4, beta_x4],
                                   [r"$\alpha(a_4/a)$", r"$\beta(a_4/a)$"],
                                   [model.alpha_x_ctrl_4, model.beta_x_ctrl_4],
                                   [model.alpha_y_ctrl_4, model.beta_y_ctrl_4],
                                   my_y_ranges)):
    ax = axes[8+k]
    ax.plot(x4_grid, arr.detach().numpy(), '-', color=interpolation_color)
    ax.plot(x_ctrl.detach().numpy(), y_ctrl.detach().numpy(), 'o',
            markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
    ax.set_xlim(my_grid4.x_min, my_grid4.x_max)
    ax.set_ylim(y_rng)
    ax.set_ylabel(lab)
    # ax.axvline(x4, color="k", ls=":")
    param_crosshair(ax, x4, np.interp(x4, x4_grid, arr.detach().numpy()))
    ax.set_xlabel(X4_name)

# ---------- ax10:  P(Y4|X4) ---------------------
ax = axes[10]
y4_grid = torch.linspace(my_grid4.y_min, my_grid4.y_max, 400)
py4 = model.py4x4(y4_grid, torch.tensor(x4)).detach().numpy()
ax.fill_between(y4_grid, py4, facecolor=blue_fill)
ax.set_xlim(my_grid4.y_min, my_grid4.y_max)
ax.set_ylim(bottom=0.)
ax.set_xlabel(Y4_name)
ax.set_ylabel(r"$P(\phi_4-\phi_0\,|\,a_4/a)$")
ax.set_xticks([-np.pi/8, -np.pi/16, 0, np.pi/16, np.pi/8])
ax.set_xticklabels([r"$-\pi/8$", r"$-\pi/16$", r"$0$", r"$\pi/16$", r"$\pi/8$"])
# ax.axvline(y4, color="k", ls=":")
# ax.scatter([y4], [model.py4x4(torch.tensor(y4), torch.tensor(x4),
#                               single_point=True).item()], color="k")
sampling_arrow(ax, y4, py4[np.abs(y4_grid-y4).argmin()])
# param_crosshair(ax, y4, 0)
plt.tight_layout()
fig.savefig("sampling_illustration.svg")
print("Saved sampling_illustration.svg")
# saving as SVG for editing in Adobe Illustrator
plt.show()

print("Done!")