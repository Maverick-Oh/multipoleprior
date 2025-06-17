#!/usr/bin/env python3
"""
DistributionModel_nontorch
--------------------------
A NumPy / SciPy replacement for the original torch-based `DistributionModel`.

* Reads **all parameters** and **domain limits** from a single JSON file that
  must contain two top-level keys: `"params"` and `"limits"`.
* Hard-codes `n_interp_m3 = 2` and `n_interp_m4 = 3`.
* Exposes a convenience method
      prob_single_point(x3, y3, x4, y4, z)
  that returns the fully-normalised joint probability density
  P(x3, y3, x4, y4, z).
  The symbols here correspond to physical variable as below.
  x3: a3/a          (strength of m=3 multipole, denoted as 'a3_a' in kwargs_lens)
  y3: phi3-\phi0    (angle of m=3 multipole, denoted as 'delta_phi_m3' in kwargs_lens)
  x4: a4/a          (strength of m=4 multipole, denoted as 'a4_a' in kwargs_lens)
  y4: phi4-\phi0    (angle of m=4 multipole, denoted as 'delta_phi_m4' in kwargs_lens)
  z : q             (axis ratio b/a, converted from 'e1' and 'e2' of kwargs_lens using q = ellipticity2phi_q(e1, e2)[1])
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def linear_spline_eval(x, x_ctrl, y_ctrl):
    """Simple piece-wise linear interpolation (1-D)."""
    return np.interp(x, x_ctrl, y_ctrl, left=y_ctrl[0], right=y_ctrl[-1])

def skew_normal_pdf(x, alpha, xi, omega):
    """
    Skew-normal PDF:
        f(x) = 2/ω · φ( (x-ξ)/ω ) · Φ( α·(x-ξ)/ω )
    where φ, Φ are the standard normal pdf/cdf.
    """
    t = (x - xi) / omega
    return 2.0 / omega * norm.pdf(t) * norm.cdf(alpha * t)

def generalized_gaussian_unnorm(x, alpha, beta):
    """
    Un-normalised Generalised-Gaussian:
        f(x) ∝ exp( - |x/α|^β )
    """
    return np.exp(-np.abs(x / alpha) ** beta)

def flat_unnorm(x):
    """Uniform (unnormalised) → 1 everywhere."""
    return np.ones_like(x, dtype=float)

# -----------------------------------------------------------------------------#
#                           The main (non-torch) class                          #
# -----------------------------------------------------------------------------#

class DistributionModel_nontorch:
    """NumPy / SciPy port of the original DistributionModel."""

    # --------------------------------------------------------------------- #
    #                                constructor                            #
    # --------------------------------------------------------------------- #
    def __init__(self, param_dict):

        p = param_dict["params"]
        lim = param_dict["limits"]

        # ------------------- basic limits ------------------- #
        self.x3_min, self.x3_max = lim["x3_min"], lim["x3_max"]
        self.y3_min, self.y3_max = lim["y3_min"], lim["y3_max"]
        self.x4_min, self.x4_max = lim["x4_min"], lim["x4_max"]
        self.y4_min, self.y4_max = lim["y4_min"], lim["y4_max"]
        self.z_min,  self.z_max  = lim["z_min"],  lim["z_max"]

        # ---------------------- P(Z) ------------------------- #
        self.alpha_Z = p["alpha_Z"]
        self.xi_Z    = p["xi_Z"]
        self.omega_Z = p["omega_Z"]

        # ------------------ controls (m = 3) ---------------- #
        self.sigma_z_x_ctrl_3 = np.asarray(p["sigma_z_x_ctrl_3"], dtype=float)
        self.sigma_z_y_ctrl_3 = np.asarray(p["sigma_z_y_ctrl_3"], dtype=float)

        # ------------------ controls (m = 4) ---------------- #
        self.alpha_z_x_ctrl_4  = np.asarray(p["alpha_z_x_ctrl_4"], dtype=float)
        self.alpha_z_y_ctrl_4  = np.asarray(p["alpha_z_y_ctrl_4"], dtype=float)
        self.xi_z_x_ctrl_4     = np.asarray(p["xi_z_x_ctrl_4"], dtype=float)
        self.xi_z_y_ctrl_4     = np.asarray(p["xi_z_y_ctrl_4"], dtype=float)
        self.omega_z_x_ctrl_4  = np.asarray(p["omega_z_x_ctrl_4"], dtype=float)
        self.omega_z_y_ctrl_4  = np.asarray(p["omega_z_y_ctrl_4"], dtype=float)

        self.alpha_x_ctrl_4    = np.asarray(p["alpha_x_ctrl_4"], dtype=float)
        self.alpha_y_ctrl_4    = np.asarray(p["alpha_y_ctrl_4"], dtype=float)
        self.beta_x_ctrl_4     = np.asarray(p["beta_x_ctrl_4"], dtype=float)
        self.beta_y_ctrl_4     = np.asarray(p["beta_y_ctrl_4"], dtype=float)

        # fixed – required by spec
        self.n_interp_m3 = 2
        self.n_interp_m4 = 3

    # --------------------------------------------------------------------- #
    #                          one-dimensional PDFs                         #
    # --------------------------------------------------------------------- #

    # ---------- P(Z) ----------
    def pz(self, z, single_point=True, n_grid=100):
        val = skew_normal_pdf(z, self.alpha_Z, self.xi_Z, self.omega_Z)


        z_grid = np.linspace(self.z_min, self.z_max, n_grid)
        normalize   = np.trapz(skew_normal_pdf(z_grid, self.alpha_Z,
                                          self.xi_Z,  self.omega_Z),
                          x=z_grid)
        return val / (normalize + 1e-12)

    # ---------- P(X3 | Z) ----------
    def px3z(self, x3, z, single_point=True, n_grid=100):
        sigma = linear_spline_eval(z, self.sigma_z_x_ctrl_3,
                                      self.sigma_z_y_ctrl_3)
        pdf   = norm.pdf(x3, loc=0.0, scale=sigma)


        x_grid = np.linspace(self.x3_min, self.x3_max, n_grid)
        normalize   = np.trapz(norm.pdf(x_grid, loc=0.0, scale=sigma), x=x_grid)
        return pdf / (normalize + 1e-12)

    # ---------- P(Y3 | X3) ----------
    def py3x3(self, y3, single_point=True):
        pdf = flat_unnorm(y3)
        return pdf / (self.y3_max - self.y3_min + 1e-12)

    # ---------- P(X4 | Z) ----------
    def px4z(self, x4, z, single_point=True, n_grid=100):
        alpha = linear_spline_eval(z, self.alpha_z_x_ctrl_4,
                                      self.alpha_z_y_ctrl_4)
        xi    = linear_spline_eval(z, self.xi_z_x_ctrl_4,
                                      self.xi_z_y_ctrl_4)
        omega = linear_spline_eval(z, self.omega_z_x_ctrl_4,
                                      self.omega_z_y_ctrl_4)

        pdf = skew_normal_pdf(x4, alpha, xi, omega)

        x_grid = np.linspace(self.x4_min, self.x4_max, n_grid)
        normalize   = np.trapz(skew_normal_pdf(x_grid, alpha, xi, omega),
                          x=x_grid)
        return pdf / (normalize + 1e-12)

    # ---------- P(Y4 | X4) ----------
    def py4x4(self, y4, x4, single_point=True, n_grid=100):
        alpha = linear_spline_eval(x4, self.alpha_x_ctrl_4,
                                      self.alpha_y_ctrl_4)
        beta  = linear_spline_eval(x4, self.beta_x_ctrl_4,
                                      self.beta_y_ctrl_4)

        pdf_unn = generalized_gaussian_unnorm(y4, alpha, beta)

        y_grid = np.linspace(self.y4_min, self.y4_max, n_grid)
        normalize   = np.trapz(generalized_gaussian_unnorm(y_grid, alpha, beta),
                          x=y_grid)
        return pdf_unn / (normalize + 1e-12)

    # --------------------------------------------------------------------- #
    #                     joint densities & public API                      #
    # --------------------------------------------------------------------- #

    def forward_m3(self, x3, y3, z, single_point=True):
        """P(Z) · P(X3|Z) · P(Y3|X3)."""
        return ( self.pz(z, single_point) *
                 self.px3z(x3, z, single_point) *
                 self.py3x3(y3, single_point) )

    def forward_m4(self, x4, y4, z, single_point=True):
        """P(Z) · P(X4|Z) · P(Y4|X4)."""
        return ( self.pz(z, single_point) *
                 self.px4z(x4, z, single_point) *
                 self.py4x4(y4, x4, single_point) )

    def forward(self, x3, y3, x4, y4, z, single_point=True):
        """
        Full joint assuming (X3,Y3) ⟂ (X4,Y4) | Z:
            P(X3,Y3,X4,Y4,Z) =
            P(Z)·P(X3|Z)·P(Y3|X3)·P(X4|Z)·P(Y4|X4) / P(Z)
        (The / P(Z) keeps the conditional-independence structure identical
        to the original torch implementation.)
        """
        return ( self.forward_m3(x3, y3, z, single_point) *
                 self.forward_m4(x4, y4, z, single_point) /
                 self.pz(z, single_point) )

    # --- convenience wrapper (matches old forward_from_nontorch_single_point) ---
    def prob_single_point(self, x3, y3, x4, y4, z):
        """Return the *fully* normalised joint PDF value at a single point."""
        return float(self.forward(x3, y3, x4, y4, z, single_point=True))


# -----------------------------------------------------------------------------#
#                                Quick visual test                             #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    json_file = Path(__file__).with_name("params_for_prior.json")
    model     = DistributionModel_nontorch(json_file)

    n_plots = 6

    # Equally spaced Z mid-points
    z_vals = np.linspace(model.z_min, model.z_max, n_plots)  # drop ends → 4 vals

    # --- m = 3 plots ------------------------------------------------------- #
    x3 = np.linspace(model.x3_min, model.x3_max, 100)
    y3 = np.linspace(model.y3_min, model.y3_max, 100)

    fig3, axes3 = plt.subplots(1, 5, figsize=(16, 3.5),
                               constrained_layout=True,
                               sharex=True, sharey=True)
    for ax, z in zip(axes3, z_vals):
        # outer product because px3z depends on x only, py3x3 on y only
        px = model.px3z(x3, z, single_point=True)
        py = model.py3x3(y3,          single_point=True)
        joint = np.outer(py, px)      # shape (len(y3), len(x3))

        im = ax.imshow(joint, origin="lower", aspect="auto",
                       extent=[model.x3_min, model.x3_max,
                               model.y3_min, model.y3_max])
        ax.set_title(f"m=3, z={z:.2f}")
        ax.set_xlabel("x3")
    axes3[0].set_ylabel("y3")
    fig3.colorbar(im, ax=axes3.ravel().tolist(), shrink=0.8)
    fig3.suptitle("P(x3, y3 | z) at four z-slices")

    # --- m = 4 plots ------------------------------------------------------- #
    x4 = np.linspace(model.x4_min, model.x4_max, 100)
    y4 = np.linspace(model.y4_min, model.y4_max, 100)

    fig4, axes4 = plt.subplots(1, n_plots, figsize=(16, 3.5),
                               constrained_layout=True,
                               sharex=True, sharey=True)

    for ax, z in zip(axes4, z_vals):
        # Initialise an empty 2-D array: rows → y4, columns → x4
        joint = np.empty((len(y4), len(x4)), dtype=float)

        # Loop over x4 (outer) and y4 (inner); alpha & beta stay scalar
        for i, x4_val in enumerate(x4):
            px_val = model.px4z(x4_val, z, single_point=True)  # scalar
            for j, y4_val in enumerate(y4):
                py_val = model.py4x4(y4_val, x4_val, single_point=True)  # scalar
                joint[j, i] = px_val * py_val  # build grid

        im = ax.imshow(joint, origin="lower", aspect="auto",
                       extent=[model.x4_min, model.x4_max,
                               model.y4_min, model.y4_max])
        ax.set_title(f"m=4, z={z:.2f}")
        ax.set_xlabel("x4")

    axes4[0].set_ylabel("y4")
    fig4.colorbar(im, ax=axes4.ravel().tolist(), shrink=0.8)
    fig4.suptitle("P(x4, y4 | z) at four z-slices")

    plt.show()

