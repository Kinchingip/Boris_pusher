"""
analysis/diffusion.py
----------------------
Spatial diffusion coefficient D(t) = ⟨(Δz)²⟩ / (2t) analysis, normalised
by Bohm diffusion D_B = v² / (3 B₀).

Contains three sub-tools, each runnable as a subcommand:

  dmumu   — running D(t) for a single simulation run
            Style reproduces Caprioli & Spitkovsky 2014c Fig 4:
            gray curves = individual time windows (ergodic proxy for ensemble)
            red curve   = mean over all windows; plateau → diffusive regime

  scan    — 2D parameter scan D(σ_env, v) / D_B on a (sigma, v) grid.
            Velocity range set by resonance: v_min = 1/k_max, v_max = 1/k_min.
            Results cached to results/scan_D.json.

  scatter — Line plot of D(σ, v)/D_B from cached scan results (one line per σ).

Usage:
    python -m analysis.diffusion dmumu   [--input FILE] [--B0 2.0] [--save PATH]
    python -m analysis.diffusion scan    [--steps N]    [--skip]   [--save PATH]
    python -m analysis.diffusion scatter [--cache FILE] [--save PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from simulation import load_results, SimConfig, run_simulation


# ── Default scan grid ──────────────────────────────────────────────────────────

# σ_env: envelope intermittency strength (0 = Gaussian, 1.5 = strong)
SIGMA_VALS = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

# Resonant speed range: v ∈ [1/k_max, 1/k_min] = [0.001, 10]
# Skip v < 0.05 (impractically slow convergence near the resonance boundary)
V_VALS = np.array([0.1, 0.3, 1.0, 3.0, 9.0])

B0    = 2.0                          # background field strength
CACHE = Path("results/scan_D.json")  # default cache path for scan results


# ── Core computation ───────────────────────────────────────────────────────────

def unwrap_z(pos: np.ndarray, box_size: float | None) -> np.ndarray:
    """
    Return the z-coordinate time series suitable for MSD computation.

    With the current mirror-fold BC the particle position is already unbounded,
    so no unwrapping is needed and this function is effectively a passthrough.

    The np.unwrap call is kept for backward compatibility with older results that
    used a plain modular wrap (x % box_size): those results show large jumps in
    z that unwrap correctly removes. On fold-BC data the consecutive steps are
    always ≪ box_size/2, so unwrap is a no-op.
    """
    z = pos[:, 2].copy()
    if box_size is not None:
        z = np.unwrap(z, period=box_size)
    return z


def compute_running_D(
    z: np.ndarray,
    dt_rec: float,
    window_steps: int,
    stride_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute D(t, t₀) = (z(t₀+t) - z(t₀))² / (2t) for many starting points t₀.

    Many time windows act as an ergodic proxy for an ensemble of particles.

    Parameters
    ----------
    z            : (N,) unwrapped z-coordinate time series
    dt_rec       : time between records = dt * save_every
    window_steps : number of lag steps per window (usually N // 2)
    stride_steps : spacing between window starts (~N // 50 gives ~50 gray curves)

    Returns
    -------
    times     : (window_steps,) lag times
    D_windows : (n_windows, window_steps) D(t) for each starting point
    D_mean    : (window_steps,) mean over all windows
    """
    N      = len(z)
    times  = np.arange(1, window_steps + 1) * dt_rec
    starts = range(0, N - window_steps - 1, stride_steps)

    D_windows = np.array([
        (z[t0 + 1: t0 + window_steps + 1] - z[t0]) ** 2 / (2.0 * times)
        for t0 in starts
    ])
    return times, D_windows, D_windows.mean(axis=0)


def plateau_D(results: dict) -> float:
    """
    Extract the plateau value of D from a single simulation run.

    Uses a time-windowed ergodic estimator, then takes the median over the
    second half of the time series where D(t) has converged.

    Parameters
    ----------
    results : simulation results dict (must contain pos, dt, save_every, box_size)

    Returns
    -------
    D_plateau : float, the converged diffusion coefficient
    """
    pos      = results["pos"]
    dt_rec   = float(results["dt"]) * int(results["save_every"])
    box_size = results.get("box_size")

    z      = unwrap_z(pos, box_size)
    N      = len(z)
    window = N // 2
    stride = max(1, N // 40)
    times  = np.arange(1, window + 1) * dt_rec

    D_runs = [
        (( z[t0 + 1: t0 + window + 1] - z[t0]) ** 2 / (2.0 * times)).mean()
        for t0 in range(0, N - window - 1, stride)
    ]
    # Median is robust to outlier windows at the boundary
    return float(np.median(D_runs))


# ── Grid scan ─────────────────────────────────────────────────────────────────

def run_grid(
    sigma_vals: np.ndarray = SIGMA_VALS,
    v_vals: np.ndarray = V_VALS,
    steps: int = 50_000,
) -> np.ndarray:
    """
    Run simulations on the full (sigma, v) grid; return D[i,j] / D_B.

    Parameters
    ----------
    sigma_vals : 1D array of envelope_sigma values
    v_vals     : 1D array of particle speeds
    steps      : number of simulation steps per run

    Returns
    -------
    D_grid : (n_sigma, n_v) array of D/D_B ratios
    """
    D_grid = np.empty((len(sigma_vals), len(v_vals)))

    for i, sigma in enumerate(sigma_vals):
        for j, v in enumerate(v_vals):
            print(f"  sigma={sigma:.1f}  v={v:.2f} ...", end="  ", flush=True)
            cfg = SimConfig(
                v_total        = v,
                steps          = steps,
                envelope_sigma = sigma,
                theta_v        = 45.0,    # isotropic start: equal v_par and v_perp
            )
            res   = run_simulation(cfg)
            D     = plateau_D(res)
            D_B   = v**2 / (3.0 * B0)
            ratio = D / D_B if D_B > 0 else np.nan
            D_grid[i, j] = ratio
            print(f"D/D_B = {ratio:.3f}")

    return D_grid


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_running_D(
    data: dict,
    B0: float = 2.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot running D(t)/D_B: gray curves for individual time windows, red mean.

    Parameters
    ----------
    data      : results dict from load_results
    B0        : background field strength (for D_B = v²/3B₀)
    save_path : output PNG path; None → show interactively
    """
    vel      = data["vel"]
    pos      = data["pos"]
    dt_rec   = float(data["dt"]) * int(data["save_every"])
    box_size = data.get("box_size")

    v_total = float(np.linalg.norm(vel[0]))
    D_B     = v_total**2 / (3.0 * B0)
    omega_c = B0                        # cyclotron frequency (q = m = 1)

    z = unwrap_z(pos, box_size)
    N = len(z)

    # ~50 gray curves, each spanning half the trajectory
    window_steps = N // 2
    stride_steps = max(1, N // 50)
    times, D_windows, D_mean = compute_running_D(z, dt_rec, window_steps, stride_steps)

    t_cyc = times * omega_c             # convert to cyclotron periods

    fig, ax = plt.subplots(figsize=(8, 5))
    for D_win in D_windows:
        ax.semilogy(t_cyc, D_win / D_B, color="gray", lw=0.6, alpha=0.35)
    ax.semilogy(t_cyc, D_mean / D_B, color="red", lw=2,
                label=r"$\langle D(t)\rangle$")

    ax.set_xlabel(r"$t\;[\omega_c^{-1}]$", fontsize=13)
    ax.set_ylabel(r"$D(t)\,/\,D_B$",       fontsize=13)
    ax.set_title(r"Running diffusion coefficient  "
                 r"$D(t) = \langle(\Delta z_\parallel)^2\rangle / 2t$", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    # Convergence diagnostic: low CV in the second half → plateau reached
    late_mean = float(D_mean[len(D_mean) // 2:].mean() / D_B)
    print(f"v_total = {v_total:.2f},  B0 = {B0:.2f},  D_B = {D_B:.3f}")
    print(f"Late-time  D / D_B ≈ {late_mean:.3f}")
    converged = np.std(D_mean[len(D_mean) // 2:] / D_B) / late_mean < 0.2
    print("Mean curve:", "converged → diffusive" if converged else "not yet converged")

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    return fig


def plot_scatter(
    sigma_vals: np.ndarray,
    v_vals: np.ndarray,
    D_grid: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Line plot of D/D_B vs particle speed v, one coloured line per σ value.

    Parameters
    ----------
    sigma_vals : 1D array of envelope sigma values (one line each)
    v_vals     : 1D array of particle speeds (x-axis)
    D_grid     : 2D array (n_sigma, n_v) of D/D_B ratios
    save_path  : output PNG path; None → show interactively
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = plt.cm.viridis(np.linspace(0, 1, len(sigma_vals)))

    for sigma, color, row in zip(sigma_vals, colors, D_grid):
        ax.plot(v_vals, row, marker="o", lw=1.8, color=color,
                label=rf"$\sigma={sigma}$")

    ax.axhline(1, color="red", lw=1, ls="--", label=r"$D = D_B$ (Bohm)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p$ (momentum, $m=1$)", fontsize=13)
    ax.set_ylabel(r"$D\,/\,D_B$",           fontsize=13)
    ax.set_title(r"$D(\sigma_{\rm env},\,p)\,/\,D_B$", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    return fig


def plot_heatmap(
    sigma_vals: np.ndarray,
    v_vals: np.ndarray,
    D_grid: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of D(σ, v)/D_B on the (sigma, v) grid with log-normalised colour.

    Parameters
    ----------
    sigma_vals : 1D array of envelope sigma values (y-axis)
    v_vals     : 1D array of particle speeds (x-axis)
    D_grid     : 2D array (n_sigma, n_v) of D/D_B ratios
    save_path  : output PNG path; None → show interactively
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    finite_pos = D_grid[np.isfinite(D_grid) & (D_grid > 0)]
    norm = mcolors.LogNorm(vmin=max(finite_pos.min(), 0.1),
                           vmax=D_grid[np.isfinite(D_grid)].max())

    # imshow on a regular grid avoids pcolormesh log-axis artefacts
    im = ax.imshow(D_grid, norm=norm, cmap="plasma", aspect="auto", origin="lower",
                   extent=[0, len(v_vals), 0, len(sigma_vals)])
    fig.colorbar(im, ax=ax, label=r"$D / D_B$")

    # Manual tick labels for the discrete grid
    ax.set_xticks(np.arange(len(v_vals)) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in v_vals])
    ax.set_yticks(np.arange(len(sigma_vals)) + 0.5)
    ax.set_yticklabels([f"{s:g}" for s in sigma_vals])

    ax.set_xlabel(r"$v$ (particle speed)",          fontsize=13)
    ax.set_ylabel(r"$\sigma_{\rm env}$ (intermittency)", fontsize=13)
    ax.set_title(r"Running diffusion coefficient $D(\sigma,\,v)\,/\,D_B$", fontsize=12)

    # Annotate each cell with its value
    for i in range(len(sigma_vals)):
        for j in range(len(v_vals)):
            val = D_grid[i, j]
            ax.text(j + 0.5, i + 0.5, f"{val:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if val < 1000 else "black")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    return fig


# ── CLI (subcommands: dmumu | scan | scatter) ──────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diffusion coefficient analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m analysis.diffusion dmumu   --input results/sim_results_cluster.json\n"
            "  python -m analysis.diffusion scan    --steps 50000\n"
            "  python -m analysis.diffusion scatter --cache results/scan_D.json\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # dmumu: running D(t) for a single run
    p1 = sub.add_parser("dmumu", help="Plot running D(t) from a single simulation")
    p1.add_argument("--input", default="results/sim_results_cluster.json")
    p1.add_argument("--B0",   type=float, default=2.0)
    p1.add_argument("--save", default="dmumu.png")

    # scan: 2D sigma-v grid
    p2 = sub.add_parser("scan", help="Run 2D (sigma, v) scan and plot heatmap")
    p2.add_argument("--steps", type=int,       default=50_000)
    p2.add_argument("--skip",  action="store_true",
                    help="Skip simulation; load cached results/scan_D.json")
    p2.add_argument("--save",  default="scan_D.png")

    # scatter: line plot from cached scan
    p3 = sub.add_parser("scatter", help="Line plot of D(sigma, v)/D_B from cache")
    p3.add_argument("--cache", default=str(CACHE))
    p3.add_argument("--save",  default="dmumu_scatter.png")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.cmd == "dmumu":
        path = Path(args.input)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run simulation.py first.")
        plot_running_D(load_results(path), B0=args.B0, save_path=args.save)

    elif args.cmd == "scan":
        if args.skip and CACHE.exists():
            print(f"Loading cached results from {CACHE}")
            with open(CACHE) as f:
                raw = json.load(f)
            D_grid = np.array(raw["D_grid"])
            sigma_vals = np.array(raw["sigma_vals"])
            v_vals     = np.array(raw["v_vals"])
        else:
            print(f"Running {len(SIGMA_VALS) * len(V_VALS)} simulations "
                  f"({len(SIGMA_VALS)} sigma × {len(V_VALS)} v) ...")
            D_grid     = run_grid(SIGMA_VALS, V_VALS, steps=args.steps)
            sigma_vals = SIGMA_VALS
            v_vals     = V_VALS
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump({"sigma_vals": sigma_vals.tolist(),
                           "v_vals":    v_vals.tolist(),
                           "D_grid":    D_grid.tolist()}, f)
            print(f"Results cached -> {CACHE}")
        plot_heatmap(sigma_vals, v_vals, D_grid, save_path=args.save)

    elif args.cmd == "scatter":
        cache_path = Path(args.cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{cache_path} not found — run `python -m analysis.diffusion scan` first."
            )
        with open(cache_path) as f:
            raw = json.load(f)
        plot_scatter(
            np.array(raw["sigma_vals"]),
            np.array(raw["v_vals"]),
            np.array(raw["D_grid"]),
            save_path=args.save,
        )
