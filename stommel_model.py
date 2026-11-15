"""
Stommel (1961) 2-Box Model: Analytical & Numerical Stability Analysis.

Direct Python translation of Section 2 of Mathematica notebook Work.nb:
    "The Stommel model"
"""

from typing import Tuple, Dict
import os
import sys
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class StommelModel:
    """
    Two-box model of thermohaline convection.
    
    Salinity ODEs:
        dS1/dt = (P1 + q * (S2 - S1)) / V1
        dS2/dt = (P2 + q * (S1 - S2)) / V2
    where:
        P1 = -S0 * F
        P2 =  S0 * F
        q = k * (beta * (S2 - S1) - alpha * Delta_T)
        
    At steady state:
        Delta_S = S2 - S1 = -S0 * F / q
        q^2 - sigma * q + gamma * F = 0
    with:
        sigma = k * alpha * Delta_T
        gamma = k * beta * S0
    """

    def __init__(
        self,
        alpha: float = 0.15,
        beta: float = 0.80,
        S0: float = 35.0,
        Delta_T: float = 20.0,
        k: float = 1.0,
        V1: float = 1.0,
        V2: float = 1.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.S0 = S0
        self.Delta_T = Delta_T
        self.k = k
        self.V1 = V1
        self.V2 = V2

        # Parameters for quadratic: q^2 - sigma * q + gamma * F = 0
        self.sigma = self.k * self.alpha * self.Delta_T
        self.gamma = self.k * self.beta * self.S0

    @property
    def critical_flux(self) -> float:
        """Saddle-node bifurcation threshold Fc (critical freshwater flux)."""
        return (self.sigma**2) / (4.0 * self.gamma)

    @property
    def critical_flow(self) -> float:
        """AMOC strength qc at the tipping point."""
        return self.sigma / 2.0

    def jacobian(self, q: float, F: float, V: float = 1.0) -> np.ndarray:
        """
        Jacobian matrix evaluated at steady state (Work.nb Cell M2).
        
        M = [[ -(q + gamma*F/q)/V,    (gamma*F/q)/V ],
             [    (gamma*F/q)/V,   -(q + gamma*F/q)/V ]]
        """
        g_term = self.gamma * F / q
        diag = -(q + g_term) / V
        off = g_term / V
        return np.array([[diag, off],
                         [off, diag]])

    def eigenvalues(self, q: float, F: float, V: float = 1.0) -> Tuple[float, float]:
        """
        Exact eigenvalues of the Jacobian matrix (Work.nb Cell EV & EV2):
            lambda_1 = -q / V                  (decay due to salt conservation)
            lambda_2 = -(q^2 - gamma*F) / (q*V) (dynamical stability eigenvalue)
        """
        lam1 = -q / V
        lam2 = -(q**2 - self.gamma * F) / (q * V)
        return lam1, lam2

    def plot_stability_curve(self, save_path: str = None):
        """Plot the equilibrium parabola and stability regions matching Work.nb."""
        if save_path is None:
            save_path = os.path.join(FIGURE_DIR, "stommel_stability.png")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        Fc = self.critical_flux
        qc = self.critical_flow

        F_vals = np.linspace(-0.5 * Fc, 1.0 * Fc, 400)
        disc = self.sigma**2 - 4.0 * self.gamma * F_vals
        valid = disc >= 0

        q_upper = 0.5 * (self.sigma + np.sqrt(np.maximum(disc, 0)))
        q_lower = 0.5 * (self.sigma - np.sqrt(np.maximum(disc, 0)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: Equilibrium Curve q vs F
        ax1.plot(F_vals[valid], q_upper[valid], "b-", linewidth=2.5, label=r"Stable branch ($\lambda_2 < 0$)")
        ax1.plot(F_vals[valid], q_lower[valid], "r--", linewidth=2.0, label=r"Unstable branch ($\lambda_2 > 0$)")
        ax1.plot(Fc, qc, "ro", markersize=8, label=f"Tipping Point ($F_c = {Fc:.3f}$, $q_c = {qc:.1f}$)")
        ax1.axvline(Fc, color="gray", linestyle=":", alpha=0.7)
        ax1.set_xlabel("Atmospheric Freshwater Flux $F$ (Sv)", fontsize=11)
        ax1.set_ylabel("Overturning Circulation $q$ (Sv)", fontsize=11)
        ax1.set_title("Stommel (1961) Equilibrium Branches", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=10)

        # Panel 2: Dynamical Eigenvalue lambda_2 along the branches
        q_range = np.linspace(0.1, self.sigma * 1.2, 300)
        # On equilibrium curve: gamma*F = sigma*q - q^2
        # lambda_2 = -(q^2 - (sigma*q - q^2)) / q = -(2*q - sigma)
        lam2_on_curve = -(2.0 * q_range - self.sigma)
        ax2.plot(q_range, lam2_on_curve, "k-", linewidth=2.0)
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.axvline(qc, color="red", linestyle=":", label=f"$q_c = {qc:.1f}$ Sv")
        ax2.fill_between(q_range, lam2_on_curve, 0, where=(q_range >= qc), color="blue", alpha=0.15, label="Stable ($\lambda_2 < 0$)")
        ax2.fill_between(q_range, lam2_on_curve, 0, where=(q_range < qc), color="red", alpha=0.15, label="Unstable ($\lambda_2 > 0$)")
        ax2.set_xlabel("AMOC Strength $q$ (Sv)", fontsize=11)
        ax2.set_ylabel("Dynamical Eigenvalue $\lambda_2$", fontsize=11)
        ax2.set_title(r"Linear Stability: $\lambda_2 = -(2q - \sigma)$", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=10)

        plt.suptitle("Stommel (1961) 2-Box Model Analysis (from Work.nb)", fontsize=13, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(save_path, dpi=200)
        print(f"Saved Stommel stability plot to: {save_path}")
        return fig


if __name__ == "__main__":
    m = StommelModel()
    print(f"Sigma (thermal driving): {m.sigma:.2f} Sv")
    print(f"Critical flux Fc:        {m.critical_flux:.4f} Sv")
    print(f"Critical flow qc:        {m.critical_flow:.2f} Sv")
    m.plot_stability_curve()
