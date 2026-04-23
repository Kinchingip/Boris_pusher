"""
analysis/intermittency.py
--------------------------
Spatial intermittency analysis of the wave field:
  - Field evaluation along 1D transects (perpendicular and parallel to B₀)
  - Two-point correlation R(ℓ) and structure function S₂(ℓ)
  - Spatial kurtosis M₄(ℓ) = S₄/S₂²
  - Intermittency exponent μ (M₄ ~ ℓ^−μ)
  - Lagrangian temporal correlation C(τ) along the particle trajectory

This is the direct analogue of Maron & Goldreich (2001) Figures 22 & 23:
M₄ is computed as a function of spatial separation rather than lag time.

Usage:
    python -m analysis.intermittency
    python -m analysis.intermittency --input results/sim_results.json --save figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from physics import make_lognormal_envelope


# ── Field evaluation ───────────────────────────────────────────────────────────

def eval_field_1d(
    coords: np.ndarray,
    axis: int,
    k_vectors: np.ndarray,
    amplitudes: np.ndarray,
    phase_offsets: np.ndarray,
    t: float = 0.0,
    fixed: float = 0.0,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """
    Evaluate the total delta_By field along a 1D transect.

    Samples the superposition of all wave modes:
        delta_B(x) = W(x) * sum_i A_i * cos(k_i . x - omega_i * t + phi_i)

    where W(x) is an optional lognormal spatial envelope (see
    make_lognormal_envelope). W=1 everywhere recovers the standard plane-wave sum.

    Parameters
    ----------
    coords        : (N,) 1D coordinate values along the transect
    axis          : 0 for x (perpendicular), 2 for z (parallel)
    k_vectors     : (n_modes, 3) wave vectors
    amplitudes    : (n_modes,) wave amplitudes
    phase_offsets : (n_modes,) phase offset per mode
    t             : snapshot time (default 0)
    fixed         : value of the two fixed coordinates (default 0)
    envelope      : (N,) lognormal envelope W(x); None means W=1

    Returns
    -------
    dBy : (N,) perpendicular magnetic field at each point
    """
    N   = len(coords)
    dBy = np.zeros(N)
    pos = np.full((N, 3), fixed)
    pos[:, axis] = coords

    for k_vec, amp, phi in zip(k_vectors, amplitudes, phase_offsets):
        omega  = k_vec[2]                       # k_parallel for shear Alfvén
        phase  = pos @ k_vec - omega * t + phi  # (N,) dot product per grid point
        dBy   += amp * np.cos(phase)

    if envelope is not None:
        dBy *= envelope

    return dBy


# ── Temporal correlation along particle trajectory ─────────────────────────────

def temporal_correlation(
    Bperp: np.ndarray,
    dt_eff: float,
    n_lags: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the normalised Lagrangian correlation of B_perp along the particle
    trajectory:

        C(τ) = R(τ) / R(0) = ⟨B(t) B(t+τ)⟩ / ⟨B²⟩

    This is sampled along the particle's actual path, so it mixes spatial
    structure with wave propagation — it is the quantity directly related to
    pitch-angle scattering and the mean free path.

    C(0) = 1 by definition. C decorrelates to ~0 on the timescale τ_c, the
    field correlation time as seen by the particle.

    Parameters
    ----------
    Bperp  : (N,) time series of |B_perp| sampled every dt_eff
    dt_eff : time between samples = dt * save_every
    n_lags : number of lag values to compute (up to N//4)

    Returns
    -------
    tau : (n_lags,) time lags
    C   : (n_lags,) normalised correlation, C[0] = 1
    """
    N = len(Bperp)
    # Subtract mean so C measures fluctuation correlation, not DC offset
    B  = Bperp - Bperp.mean()
    R0 = np.mean(B**2)
    lags = np.unique(
        np.logspace(0, np.log10(min(n_lags, N // 4)), n_lags).astype(int)
    )
    tau = lags * dt_eff
    C   = np.array([np.mean(B[:-lag] * B[lag:]) for lag in lags]) / R0
    return tau, C


# ── Two-point correlation and structure functions ──────────────────────────────

def two_point_correlation(field: np.ndarray, seps: np.ndarray) -> np.ndarray:
    """
    Compute the 2-point correlation function R(ℓ) = ⟨B(x) B(x+ℓ)⟩.

    R(0) = ⟨B²⟩ is the field variance.
    The 2nd-order structure function follows directly:
        S₂(ℓ) = 2[R(0) - R(ℓ)]

    Note: R(ℓ) depends only on the power spectrum {Aᵢ²}, not on phases.
    Intermittency lives entirely in higher-order statistics (S₄, etc.).

    Parameters
    ----------
    field : (N,) 1D field on a uniform grid
    seps  : integer grid separations to evaluate

    Returns
    -------
    R : (len(seps),) correlation values, with R[0] ≈ ⟨B²⟩ when seps[0]=0
    """
    R0 = np.mean(field**2)
    R  = np.empty(len(seps))
    for i, sep in enumerate(seps):
        R[i] = R0 if sep == 0 else np.mean(field[:-sep] * field[sep:])
    return R


def s2_from_correlation(R: np.ndarray, R0: float) -> np.ndarray:
    """
    Derive S₂(ℓ) = 2[R(0) - R(ℓ)] from the 2-point correlation.

    This makes the link to R(ℓ) explicit: the structure function is not
    independently defined — it is a re-expression of the correlation function.
    """
    return 2 * (R0 - R)


def fit_intermittency_exponent(
    ell: np.ndarray,
    M4: np.ndarray,
    frac_small: float = 0.4,
) -> float:
    """
    Fit the intermittency exponent μ from M₄(ℓ) ~ ℓ^(−μ).

    Uses the smallest `frac_small` fraction of scales where intermittent
    corrections are largest. Returns μ > 0 for intermittent fields;
    μ ≈ 0 for Gaussian (non-intermittent) fields.

    The exponent connects structure-function scaling:
        μ = 2 ζ(2) − ζ(4)
    where Sₚ(ℓ) ~ ℓ^ζ(p).  Non-intermittent (self-similar): μ = 0.

    Parameters
    ----------
    ell       : (N,) physical separations (increasing)
    M4        : (N,) kurtosis at each separation
    frac_small: fraction of smallest scales to use for the fit

    Returns
    -------
    mu : intermittency exponent (negative slope of log M4 vs log ell)
    """
    mask = np.isfinite(M4) & (M4 > 0) & (ell > 0)
    ell_ok, M4_ok = ell[mask], M4[mask]
    n_fit   = max(3, int(frac_small * len(ell_ok)))
    slope, _ = np.polyfit(np.log(ell_ok[:n_fit]), np.log(M4_ok[:n_fit]), 1)
    return float(-slope)    # μ = −slope because M4 ~ ℓ^(−μ)


# ── Spatial increments and kurtosis ───────────────────────────────────────────

def spatial_increments(field: np.ndarray, sep: int) -> np.ndarray:
    """
    Return field increments at separation sep grid points:
        delta(ell) = field[i + sep] - field[i]  for all i

    Parameters
    ----------
    field : (N,) 1D field values
    sep   : integer grid separation

    Returns
    -------
    delta : (N - sep,) increments
    """
    return field[sep:] - field[:-sep]


def spatial_kurtosis(
    field: np.ndarray,
    seps: np.ndarray,
) -> np.ndarray:
    """
    Compute M₄(ℓ) = ⟨(δB(ℓ))⁴⟩ / ⟨(δB(ℓ))²⟩² at each spatial separation.

    M₄ = 3 everywhere → Gaussian (non-intermittent).
    M₄(ℓ) rising at small ℓ → intermittent turbulence.

    Parameters
    ----------
    field : (N,) 1D field on a uniform grid
    seps  : integer grid separations to evaluate

    Returns
    -------
    M4 : (len(seps),) kurtosis at each separation
    """
    M4 = np.empty(len(seps))
    for i, sep in enumerate(seps):
        delta = spatial_increments(field, sep)
        m2    = np.mean(delta**2)
        m4    = np.mean(delta**4)
        M4[i] = m4 / m2**2 if m2 > 0 else np.nan
    return M4


# ── Main analysis function ─────────────────────────────────────────────────────

def compute_spatial_kurtosis(
    k_vectors: np.ndarray,
    amplitudes: np.ndarray,
    phase_offsets: np.ndarray,
    n_points: int = 2000,
    n_seps: int = 40,
    t: float = 0.0,
    envelope_sigma: float = 0.0,
    envelope_corr_frac: float = 0.05,
    envelope_seed: int = 99,
    env_coords: np.ndarray | None = None,
    env_W: np.ndarray | None = None,
) -> dict:
    """
    Compute M₄(ℓ) along both perpendicular (x) and parallel (z) transects.

    The transect length covers the longest wavelength: L = 2π/k_min,
    sampled at n_points grid points.

    Parameters
    ----------
    k_vectors          : (n_modes, 3)
    amplitudes         : (n_modes,)
    phase_offsets      : (n_modes,)
    n_points           : number of grid points along each transect
    n_seps             : number of separation values (log-spaced)
    t                  : snapshot time
    envelope_sigma     : used only if env_coords/env_W are not provided
    envelope_corr_frac : used only if env_coords/env_W are not provided
    envelope_seed      : used only if env_coords/env_W are not provided
    env_coords         : (M,) saved envelope x-grid from simulation (preferred)
    env_W              : (M,) saved envelope W values from simulation (preferred)

    Returns
    -------
    dict with keys:
        ell_perp, M4_perp  : perpendicular transect results
        ell_para, M4_para  : parallel transect results
        R_perp, R_para     : 2-point correlations at the same separations
        S2_perp, S2_para   : S₂ = 2[R(0)−R(ℓ)]
        mu_perp, mu_para   : intermittency exponents
        L                  : transect length (longest wavelength)
    """
    k_mags = np.linalg.norm(k_vectors, axis=1)
    k_min  = k_mags.min()
    L      = 2 * np.pi / k_min    # transect length = longest wavelength

    coords = np.linspace(0, L, n_points)
    dx     = coords[1] - coords[0]

    # Log-spaced integer separations from 1 to N//4
    seps = np.unique(
        np.logspace(0, np.log10(n_points // 4), n_seps).astype(int)
    )
    ell = seps * dx    # physical separations

    # Use the envelope saved by the simulation if available (guarantees the
    # analysis sees the same field the particle moved through).
    # The ⊥ transect uses the saved envelope directly; the ∥ transect gets an
    # independent realization because the envelope varies in x, not z.
    if env_coords is not None and env_W is not None:
        env_perp = np.interp(coords % env_coords[-1], env_coords, env_W)
        corr_len = envelope_corr_frac * L
        env_para = make_lognormal_envelope(coords, envelope_sigma, corr_len,
                                           seed=envelope_seed + 1)
    else:
        corr_len = envelope_corr_frac * L
        env_perp = make_lognormal_envelope(coords, envelope_sigma, corr_len,
                                           seed=envelope_seed)
        env_para = make_lognormal_envelope(coords, envelope_sigma, corr_len,
                                           seed=envelope_seed + 1)

    # Perpendicular transect: vary x, fix y=z=0
    field_perp = eval_field_1d(coords, axis=0,
                               k_vectors=k_vectors, amplitudes=amplitudes,
                               phase_offsets=phase_offsets, t=t,
                               envelope=env_perp)
    M4_perp    = spatial_kurtosis(field_perp, seps)

    # Parallel transect: vary z, fix x=y=0
    field_para = eval_field_1d(coords, axis=2,
                               k_vectors=k_vectors, amplitudes=amplitudes,
                               phase_offsets=phase_offsets, t=t,
                               envelope=env_para)
    M4_para    = spatial_kurtosis(field_para, seps)

    # R(ℓ) depends only on {Aᵢ²} (power spectrum); S₂ = 2[R(0)−R(ℓ)].
    # Include lag=0 so R[0] gives the variance for normalisation, then strip it.
    seps_with_zero = np.concatenate([[0], seps])
    R_perp         = two_point_correlation(field_perp, seps_with_zero)
    R_para         = two_point_correlation(field_para, seps_with_zero)
    S2_perp = s2_from_correlation(R_perp[1:], R_perp[0])
    S2_para = s2_from_correlation(R_para[1:], R_para[0])

    mu_perp = fit_intermittency_exponent(ell, M4_perp)
    mu_para = fit_intermittency_exponent(ell, M4_para)

    return {
        "ell_perp": ell,   "M4_perp": M4_perp,
        "ell_para": ell,   "M4_para": M4_para,
        "R_perp":   R_perp[1:],    # R(ℓ) at the same separations as ell
        "R_para":   R_para[1:],
        "S2_perp":  S2_perp,       # S₂ = 2[R(0)−R(ℓ)]
        "S2_para":  S2_para,
        "mu_perp":  mu_perp,       # intermittency exponent ⊥
        "mu_para":  mu_para,       # intermittency exponent ∥
        "L":        L,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_correlation_and_s2(ax, results: dict, label: str = "") -> None:
    """
    Plot R(ℓ) for perpendicular and parallel transects.
    S₂ = 2[R(0)−R(ℓ)] is algebraically redundant given R, so it is omitted.
    """
    suffix = f" ({label})" if label else ""
    ell    = results["ell_perp"]

    ax.loglog(ell, results["R_perp"], "o-",  lw=1.5, markersize=3,
              label=f"R(ℓ) ⊥ B₀{suffix}")
    ax.loglog(ell, results["R_para"], "s--", lw=1.5, markersize=3,
              label=f"R(ℓ) ∥ B₀{suffix}")
    ax.set_xlabel(r"Spatial separation $\ell$")
    ax.set_ylabel(r"$R(\ell) = \langle B(x)\,B(x+\ell)\rangle$")
    ax.set_title(r"2-point correlation $R(\ell)$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_spatial_kurtosis(ax, results: dict, label: str = "") -> None:
    """
    Plot M₄(ℓ) = S₄/S₂² for both directions.

    M₄ = 3 everywhere → Gaussian (non-intermittent).
    M₄(ℓ) ~ ℓ^(−μ) rising at small ℓ → intermittent with exponent μ.
    Perpendicular is the direct analogue of Maron & Goldreich Figure 22.
    """
    suffix  = f" ({label})" if label else ""
    mu_perp = results.get("mu_perp", float("nan"))
    mu_para = results.get("mu_para", float("nan"))

    ax.semilogx(results["ell_perp"], results["M4_perp"],
                "o-", lw=1.5, markersize=3,
                label=f"⊥ B₀  μ={mu_perp:.3f}{suffix}")
    ax.semilogx(results["ell_para"], results["M4_para"],
                "s--", lw=1.5, markersize=3,
                label=f"∥ B₀  μ={mu_para:.3f}{suffix}")
    ax.axhline(y=3, color="k", linestyle=":", lw=1.2,
               label=r"Gaussian ($M_4=3$, $\mu=0$)")
    ax.set_xlabel(r"Spatial separation $\ell$")
    ax.set_ylabel(r"$M_4(\ell) = S_4(\ell)\,/\,S_2(\ell)^2$")
    ax.set_title(r"Kurtosis $M_4(\ell)$  —  intermittency exponent $\mu$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_temporal_correlation(
    ax,
    tau: np.ndarray,
    C: np.ndarray,
    label: str = "",
) -> None:
    """
    Plot the normalised Lagrangian correlation C(τ) along the particle trajectory.

    The 1/e level is marked as a reference for the decorrelation time τ_c.
    C(τ) ~ exp(−τ/τ_c) for a simple exponential model.
    """
    suffix = f" ({label})" if label else ""
    ax.semilogx(tau, C, "o-", lw=1.5, markersize=3,
                label=f"C(τ) along trajectory{suffix}")
    ax.axhline(y=0,       color="k",      linestyle="-",  lw=0.8, alpha=0.4)
    ax.axhline(y=1/np.e,  color="tomato", linestyle="--", lw=1.2,
               label=r"$1/e$  (decorrelation reference)")
    ax.set_xlabel(r"Time lag $\tau$")
    ax.set_ylabel(r"$C(\tau) = R(\tau)\,/\,R(0)$")
    ax.set_title(r"Lagrangian correlation $C(\tau)$  —  trajectory")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def build_analysis_figure(
    k_vectors: np.ndarray,
    amplitudes: np.ndarray,
    phase_offsets: np.ndarray,
    label: str = "",
    figsize: tuple = (21, 5),
    envelope_sigma: float = 0.0,
    envelope_corr_frac: float = 0.05,
    envelope_seed: int = 99,
    Bperp: np.ndarray | None = None,
    dt_eff: float | None = None,
    env_coords: np.ndarray | None = None,
    env_W: np.ndarray | None = None,
) -> plt.Figure:
    """
    Build a three-panel intermittency figure:
      Left   — R(ℓ): spatial 2-point correlation
      Middle — M₄(ℓ): spatial kurtosis with exponent μ
      Right  — C(τ): Lagrangian temporal correlation (omitted if Bperp/dt_eff absent)

    Parameters
    ----------
    k_vectors          : (n_modes, 3) from results dict
    amplitudes         : (n_modes,)   from results dict
    phase_offsets      : (n_modes,)   from results dict
    label              : legend label
    envelope_sigma     : lognormal envelope strength (0 = Gaussian baseline)
    envelope_corr_frac : envelope correlation length as fraction of box L
    envelope_seed      : RNG seed for envelope
    Bperp              : (N,) |B_perp| time series along particle trajectory
    dt_eff             : time between samples = dt * save_every
    env_coords         : saved envelope grid from simulation (preferred over sigma)
    env_W              : saved envelope W(x) values from simulation (preferred)
    """
    results = compute_spatial_kurtosis(
        k_vectors, amplitudes, phase_offsets,
        envelope_sigma=envelope_sigma,
        envelope_corr_frac=envelope_corr_frac,
        envelope_seed=envelope_seed,
        env_coords=env_coords,
        env_W=env_W,
    )
    print(f"Intermittency exponents:  μ_⊥ = {results['mu_perp']:.4f}"
          f"   μ_∥ = {results['mu_para']:.4f}")

    has_trajectory = (Bperp is not None) and (dt_eff is not None)
    n_panels = 3 if has_trajectory else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    plot_correlation_and_s2(axes[0], results, label=label)
    plot_spatial_kurtosis(axes[1], results, label=label)

    if has_trajectory:
        tau, C = temporal_correlation(Bperp, dt_eff)
        plot_temporal_correlation(axes[2], tau, C, label=label)

    plt.tight_layout()
    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial kurtosis and intermittency analysis"
    )
    parser.add_argument(
        "--input", default="results/sim_results_cluster.json",
        help="Path to JSON results file produced by simulation.py",
    )
    parser.add_argument(
        "--save", default=None,
        help="Directory to save PNG figure (omit to display interactively)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from simulation import load_results

    args    = _parse_args()
    in_path = Path(args.input)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {in_path}\n"
            "Run `python simulation.py` first to generate it.\n"
            "Note: results must include k_vectors, amplitudes, phase_offsets."
        )

    data          = load_results(in_path)
    k_vectors     = data["k_vectors"]
    amplitudes    = data["amplitudes"]
    phase_offsets = data["phase_offsets"]
    env_sigma     = float(data.get("envelope_sigma",     0.0))
    env_corr_frac = float(data.get("envelope_corr_frac", 0.05))
    env_seed      = int(data.get("envelope_seed",        99))
    env_coords    = data.get("env_coords")
    env_W         = data.get("env_W")
    label         = f"σ_env={env_sigma:.1f}" if env_sigma > 0 else "baseline"

    # Trajectory time series for Lagrangian correlation panel
    Bperp      = data.get("Bperp")
    dt_eff     = float(data.get("dt", 0.01)) * int(data.get("save_every", 10))

    fig = build_analysis_figure(
        k_vectors, amplitudes, phase_offsets,
        label=label,
        envelope_sigma=env_sigma,
        envelope_corr_frac=env_corr_frac,
        envelope_seed=env_seed,
        env_coords=env_coords,
        env_W=env_W,
        Bperp=Bperp,
        dt_eff=dt_eff,
    )

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_fig  = save_dir / "spatial_kurtosis.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Figure saved -> {out_fig}")
    else:
        plt.show()
