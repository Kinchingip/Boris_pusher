"""
plots/dB_field.py
-----------------
Three-panel figure showing the spatial structure of δB along the x-axis
(perpendicular to B₀) at z=0, t=0:
  Top    — raw wave field δB_y(x)
  Middle — lognormal envelope W(x)
  Bottom — envelope-modulated (intermittent) field W(x) · δB_y(x)

This makes the structure of intermittency directly visible: the envelope
concentrates wave energy into rare intense patches (large W) separated by
quiet regions (W ≈ 0).

Usage:
    python -m plots.dB_field
    python -m plots.dB_field --input results/sim_results_cluster.json --save dB_structure.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from simulation import load_results


def plot_dB_structure(data: dict, save_path: str | None = None) -> plt.Figure:
    """
    Plot raw δB_y, the lognormal envelope W(x), and the intermittent field W·δB.

    Parameters
    ----------
    data      : results dict from load_results (must contain k_vectors,
                amplitudes, phase_offsets, env_coords, env_W)
    save_path : output PNG path; None → show interactively
    """
    k_vectors     = data["k_vectors"]      # (n_modes, 3)
    amplitudes    = data["amplitudes"]     # (n_modes,)
    phase_offsets = data["phase_offsets"]  # (n_modes,)
    env_coords    = data["env_coords"]     # x-grid the envelope was built on
    env_W         = data["env_W"]          # W(x) lognormal envelope values

    L      = env_coords[-1]
    x_grid = np.linspace(0, L, 4000)      # evaluation grid spanning one full box

    # Vectorised sum: pos = (x, 0, 0) for each x in x_grid.
    # delta_B_y = sum_i A_i * cos(k_i . pos + phi_i)
    pos    = np.column_stack([x_grid, np.zeros(len(x_grid)), np.zeros(len(x_grid))])
    dB_raw = np.zeros(len(x_grid))
    for k_vec, amp, phi in zip(k_vectors, amplitudes, phase_offsets):
        dB_raw += amp * np.cos(pos @ k_vec + phi)

    # Evaluate W(x) on the same grid (interpolate + periodic wrap)
    W_grid = np.interp(x_grid % L, env_coords, env_W)
    dB_env = W_grid * dB_raw              # envelope-modulated field

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x_grid, dB_raw, lw=0.6, color="steelblue")
    axes[0].set_ylabel(r"$\delta B_y$ (raw)", fontsize=11)
    axes[0].set_title(r"Spatial structure of wave field along $x$  ($z=0,\;t=0$)",
                      fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_grid, W_grid, lw=1.2, color="darkorange")
    axes[1].set_ylabel(r"$W(x)$ (envelope)", fontsize=11)
    # W=1 everywhere is the Gaussian (non-intermittent) reference
    axes[1].axhline(1, color="k", lw=0.8, ls="--", alpha=0.5, label="W=1 (Gaussian)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x_grid, dB_env, lw=0.6, color="crimson")
    axes[2].set_ylabel(r"$W(x)\,\delta B_y$ (intermittent)", fontsize=11)
    axes[2].set_xlabel(r"$x$", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    else:
        plt.show()

    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot spatial delta_B structure")
    p.add_argument("--input", default="results/sim_results_cluster.json",
                   help="Path to JSON results file from simulation.py")
    p.add_argument("--save",  default="dB_structure.png",
                   help="Output PNG path (omit to display interactively)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run simulation.py first.")
    plot_dB_structure(load_results(path), save_path=args.save)
