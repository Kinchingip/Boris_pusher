"""
plot_results.py
---------------
Plotting routines for slab Alfvén wave particle simulation results.

Usage (standalone):
    python plot_results.py                          # loads results/sim_results.json
    python plot_results.py --input path/to/file.json
    python plot_results.py --save  figures/         # save PNGs instead of showing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simulation import load_results

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# ── Derived quantities ─────────────────────────────────────────────────────────

def compute_pitch_angles(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mu, pitch_angle_deg) from velocity array of shape (N, 3).
    mu  = v_z / |v|   (cosine of pitch angle w.r.t. B0 = z-axis)
    """
    v_mag       = np.linalg.norm(vel, axis=1)
    mu          = vel[:, 2] / v_mag
    pitch_angle = np.degrees(np.arccos(np.clip(mu, -1.0, 1.0)))
    return mu, pitch_angle


# ── Individual panel functions ─────────────────────────────────────────────────

def plot_trajectory(ax, pos: np.ndarray, box_size: float | None = None) -> None:
    # Unwrap each axis so periodic jumps don't appear as discontinuities
    if box_size is not None:
        pos = pos.copy()
        for dim in range(3):
            pos[:, dim] = np.unwrap(pos[:, dim], period=box_size)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=0.5)
    ax.set_title("Slab Wave Field Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_pitch_distribution(ax, pitch_angle: np.ndarray) -> None:
    ax.hist(pitch_angle, bins=50, density=True,
            alpha=0.7, color="steelblue", edgecolor="black")
    mu_fit, std_fit = norm.fit(pitch_angle)
    x_fit = np.linspace(0, 180, 200)
    ax.plot(x_fit, norm.pdf(x_fit, mu_fit, std_fit), "r-", lw=2,
            label=f"μ={mu_fit:.1f}°, σ={std_fit:.1f}°")
    ax.set_title("Pitch Angle Distribution")
    ax.set_xlabel("Pitch angle (°)")
    ax.legend()
    ax.grid(True)


def plot_bperp(ax, Bperp: np.ndarray) -> None:
    ax.plot(Bperp, lw=0.5, color="darkorange")
    ax.set_title(r"$|B_\perp|$ at Particle Position")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Record index")
    ax.grid(True)


def plot_pitch_evolution(ax, pitch_angle: np.ndarray) -> None:
    ax.plot(pitch_angle, lw=0.5)
    ax.axhline(y=90, color="r", linestyle="--")
    ax.set_title("Pitch Angle Evolution")
    ax.set_ylabel("Pitch angle (°)")
    ax.set_xlabel("Record index")
    ax.set_ylim([0, 180])
    ax.grid(True)


# ── Main figure builder ────────────────────────────────────────────────────────

def build_overview_figure(
    pos: np.ndarray,
    vel: np.ndarray,
    Bperp: np.ndarray,
    figsize: tuple = (12, 10),
    box_size: float | None = None,
) -> plt.Figure:
    """
    Build the four-panel overview figure.

    Parameters
    ----------
    pos      : (N, 3) position history
    vel      : (N, 3) velocity history
    Bperp    : (N,)   |B_perp| history
    figsize  : matplotlib figure size
    box_size : periodic box length (from SimConfig); used to unwrap trajectory

    Returns
    -------
    fig : the matplotlib Figure (not yet shown/saved)
    """
    _, pitch_angle = compute_pitch_angles(vel)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(221, projection="3d")
    plot_trajectory(ax1, pos, box_size=box_size)

    ax2 = fig.add_subplot(222)
    plot_pitch_distribution(ax2, pitch_angle)

    ax3 = fig.add_subplot(223)
    plot_bperp(ax3, Bperp)

    ax4 = fig.add_subplot(224)
    plot_pitch_evolution(ax4, pitch_angle)

    plt.tight_layout()
    return fig


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot simulation results")
    parser.add_argument(
        "--input", type=str,
        default="results/sim_results.json",
        help="Path to JSON results file produced by simulation.py",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Directory to save PNG figures (omit to display interactively)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    in_path = Path(args.input)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {in_path}\n"
            "Run `python simulation.py` first to generate it."
        )

    data     = load_results(in_path)
    pos      = data["pos"]
    vel      = data["vel"]
    Bperp    = data["Bperp"]
    box_size = data.get("box_size") 

    fig = build_overview_figure(pos, vel, Bperp, box_size=box_size)

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_fig  = save_dir / "overview.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Figure saved → {out_fig}")
    else:
        plt.show()