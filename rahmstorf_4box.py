"""
Rahmstorf (1996) 4-Box Model: Versions I & II Stability Analysis.

Direct Python translation of Sections 3 & 4 of Mathematica notebook Work.nb:
    "Rahmstorfs 4 box model" and "Analyses of the Rahmstorf model version II"

References:
    Rahmstorf, S. (1996). On the freshwater forcing and transport of the Atlantic
    thermohaline circulation. Climate Dynamics, 12(12), 799-811.
"""

from typing import Tuple, Dict, List
import os
import sys
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class Rahmstorf4BoxModel:
    """
    Rahmstorf 4-Box Model of Atlantic Overturning Circulation.
    
    Boxes:
        - Box 1: South Atlantic / Southern Ocean (deep/mixed)
        - Box 2: North Atlantic deep water formation (deep)
        - Box 3: Subtropical North Atlantic thermocline (shallow)
        - Box 4: Deep ocean below thermocline (deep)
    """

    def __init__(
        self,
        version: int = 1,        # 1: Interhemispheric (S2 - S1), 2: Intra-basin (S2 - S3)
        Gamma: float = 420.0,    # k * beta * S0 (Sv^2 / Sv_freshwater)
        Sigma: float = 18.0,     # k * alpha * Theta (Sv)
        V: float = 1.0           # Box volume (equal volume assumption from Work.nb)
    ):
        self.version = version
        self.Gamma = Gamma
        self.Sigma = Sigma
        self.V = V

    def jacobian(self, q: float, F1: float, F2: float) -> np.ndarray:
        """
        Evaluate the 4x4 Jacobian matrix at steady state (Work.nb Cell M).
        """
        G = self.Gamma
        V = self.V

        # Flow derivatives: dq/dS1, dq/dS2, dq/dS3, dq/dS4
        if self.version == 1:
            # q = k * (beta * (S2 - S1) - alpha * Theta)
            dq_dS1 = -G
            dq_dS2 = +G
            dq_dS3 = 0.0
            dq_dS4 = 0.0
        else:
            # q = k * (beta * (S2 - S3) - alpha * Theta)
            dq_dS1 = 0.0
            dq_dS2 = +G
            dq_dS3 = -G
            dq_dS4 = 0.0

        # Steady state tracer differences (scaled by S0 / q):
        # S4 - S1 = S0 * F1 / q
        # S3 - S2 = S0 * F2 / q
        # S1 - S3 = -S0 * (F1 + F2) / q
        # S2 - S4 = 0
        diff_41 = F1 / q
        diff_32 = F2 / q
        diff_13 = -(F1 + F2) / q
        diff_24 = 0.0

        M = np.zeros((4, 4), dtype=float)

        # Row 1: dS1dot / dS_j
        M[0, 0] = (-q + dq_dS1 * diff_41) / V
        M[0, 1] = (dq_dS2 * diff_41) / V
        M[0, 2] = (dq_dS3 * diff_41) / V
        M[0, 3] = (+q + dq_dS4 * diff_41) / V

        # Row 2: dS2dot / dS_j
        M[1, 0] = (dq_dS1 * diff_32) / V
        M[1, 1] = (-q + dq_dS2 * diff_32) / V
        M[1, 2] = (+q + dq_dS3 * diff_32) / V
        M[1, 3] = (dq_dS4 * diff_32) / V

        # Row 3: dS3dot / dS_j
        M[2, 0] = (+q + dq_dS1 * diff_13) / V
        M[2, 1] = (dq_dS2 * diff_13) / V
        M[2, 2] = (-q + dq_dS3 * diff_13) / V
        M[2, 3] = (dq_dS4 * diff_13) / V

        # Row 4: dS4dot / dS_j
        M[3, 0] = (dq_dS1 * diff_24) / V
        M[3, 1] = (+q + dq_dS2 * diff_24) / V
        M[3, 2] = (dq_dS3 * diff_24) / V
        M[3, 3] = (-q + dq_dS4 * diff_24) / V

        return M

    def eigenvalues(self, q: float, F1: float, F2: float) -> np.ndarray:
        """Compute the 4 eigenvalues of the Jacobian matrix."""
        M = self.jacobian(q, F1, F2)
        evals = np.linalg.eigvals(M)
        # Sort by real part descending
        idx = np.argsort(evals.real)[::-1]
        return evals[idx]

    def stability_grid(
        self,
        q_range: Tuple[float, float] = (0.5, 15.0),
        F2_range: Tuple[float, float] = (0.0, 0.4),
        n_q: int = 60,
        n_F2: int = 60
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate the maximum real eigenvalue across (q, F2) on the equilibrium surface.
        In Version I: Gamma * F1 = q^2 + Sigma * q
        In Version II: Gamma * F2 = -(q^2 + Sigma * q)
        """
        q_vec = np.linspace(q_range[0], q_range[1], n_q)
        F2_vec = np.linspace(F2_range[0], F2_range[1], n_F2)
        Q, F2_grid = np.meshgrid(q_vec, F2_vec)

        max_re_eval = np.zeros_like(Q)

        for i in range(n_F2):
            for j in range(n_q):
                q_val = Q[i, j]
                f2_val = F2_grid[i, j]

                if self.version == 1:
                    # Equilibrium condition determines F1
                    f1_val = (q_val**2 + self.Sigma * q_val) / self.Gamma
                else:
                    # Version II equilibrium condition determines F2, vary F1
                    f1_val = f2_val
                    f2_val = max(0.01, (self.Sigma * q_val - q_val**2) / self.Gamma)

                evals = self.eigenvalues(q_val, f1_val, f2_val)
                # Ignore the zero eigenvalue (salt conservation)
                non_zero = evals[np.abs(evals) > 1e-10]
                max_re_eval[i, j] = np.max(non_zero.real) if len(non_zero) > 0 else 0.0

        return {"Q": Q, "F2": F2_grid, "max_re_eval": max_re_eval}

    def plot_stability_surface(self, save_path: str = None):
        """Plot the 3D surface and 2D contour stability diagrams matching Work.nb."""
        if save_path is None:
            save_path = os.path.join(
                FIGURE_DIR, f"rahmstorf_version{self.version}_stability.png"
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        grid = self.stability_grid()
        Q = grid["Q"]
        F2 = grid["F2"]
        max_re = grid["max_re_eval"]

        fig = plt.figure(figsize=(14, 6))

        # 3D surface plot
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        surf = ax1.plot_surface(F2, Q, max_re, cmap="coolwarm", edgecolor="none", alpha=0.85)
        ax1.set_xlabel("Southern Freshwater Flux $F_2$ (Sv)", fontsize=10)
        ax1.set_ylabel("AMOC Strength $q$ (Sv)", fontsize=10)
        ax1.set_zlabel(r"Max Re($\lambda$)", fontsize=10)
        ax1.set_title(f"Rahmstorf Version {self.version}: Max Eigenvalue Surface", fontsize=11)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

        # 2D contour plot (stability boundary: Re(lambda) = 0)
        ax2 = fig.add_subplot(1, 2, 2)
        cf = ax2.contourf(F2, Q, max_re, levels=np.linspace(np.min(max_re), np.max(max_re), 21), cmap="coolwarm")
        fig.colorbar(cf, ax=ax2, label=r"Max Re($\lambda$)")

        # Stability threshold line at Re(lambda) = 0
        cs = ax2.contour(F2, Q, max_re, levels=[0.0], colors="black", linewidths=2.5)
        ax2.clabel(cs, inline=True, fontsize=10, fmt="Stable / Unstable Boundary")

        ax2.set_xlabel("Southern Freshwater Flux $F_2$ (Sv)", fontsize=11)
        ax2.set_ylabel("AMOC Strength $q$ (Sv)", fontsize=11)
        ax2.set_title(f"Stability Map (Black Line: Re($\lambda$) = 0)", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(f"Rahmstorf (1996) 4-Box Model Analysis — Version {self.version} (Work.nb)", fontsize=13, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        fig.savefig(save_path, dpi=200)
        print(f"Saved Rahmstorf stability plot to: {save_path}")
        return fig


if __name__ == "__main__":
    # Version I: Interhemispheric
    r1 = Rahmstorf4BoxModel(version=1)
    r1.plot_stability_surface()

    # Version II: Intra-basin
    r2 = Rahmstorf4BoxModel(version=2)
    r2.plot_stability_surface()
