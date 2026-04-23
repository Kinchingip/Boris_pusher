"""
simulation.py
-------------
Simulation setup, run loop, and result persistence.

Usage (standalone):
    python simulation.py                  # runs with defaults, saves results
    python simulation.py --steps 100000   # override any SimConfig field via CLI
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from physics import boris_pusher, alfven_wave_fields, make_lognormal_envelope


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    # Integration
    dt: float       = 0.01
    steps: int      = 50_000
    save_every: int = 10          # record state every N steps

    # Background field
    B0: float = 2.0

    # Particle
    v_total: float = 9.0
    theta_v: float = 0.9          # pitch angle in degrees

    # Wave spectrum
    k_min: float          = 0.1
    k_max: float          = 1000
    n_k: int              = 30            # base log-spaced modes
    spectral_index: float = -5/6          # amplitude proportional to k^index
    amp_scale: float      = 1.5           # overall amplitude prefactor
    k_extra: list         = field(default_factory=lambda: [0.25])

    def __post_init__(self):
        # Guard against the dataclass field descriptor being passed instead of the value
        if not isinstance(self.k_extra, list):
            self.k_extra = [0.25]

    # Angular spread of wave vectors around B0 (quasi-parallel slab)
    angle_spread: float = np.pi / 6       # +/- this value in radians
    rng_seed: int       = 42

    # Intermittency control: phase coherence λ ∈ [0, 1].
    # Modes within a k-decade share one phase at λ=1, independent phases at λ=0.
    phase_coherence: float = 1.0

    # Intermittency control: lognormal spatial envelope.
    # The wave field is multiplied by W(x) = exp(σ_env · G(x)), where G(x) is a
    # smooth Gaussian random process with correlation length = envelope_corr_frac * L.
    # envelope_sigma = 0 → W = 1 everywhere → Gaussian (M4 ≈ 3, μ ≈ 0)
    # envelope_sigma = 0.5 → mild intermittency
    # envelope_sigma = 1.5 → strong intermittency (M4 >> 3 at small ℓ)
    # This is the physically correct mechanism: rare intense patches separated by
    # quiet regions, as seen in MHD current sheets.
    envelope_sigma: float      = 1.5
    envelope_corr_frac: float  = 0.05   # correlation length as fraction of box L
    envelope_seed: int         = 99

    # Mirror-fold boundary: primary box side length L = 2π/k_min.
    # The field is extended by reflecting across each face, so the particle
    # sees a smooth (C⁰) field everywhere — no discontinuity at the wall.
    # The particle position is tracked without wrapping (unbounded), so D(t)
    # can be read directly from pos[:, 2] without unwrapping.
    box_size: float = 2 * np.pi / k_min    # primary box side (fold period = 2L)

    # Output
    output_dir: str  = "results"
    output_file: str = "sim_results_cluster.json"


# ── Mirror-fold boundary condition ────────────────────────────────────────────

def _fold_pos(x: np.ndarray, L: float) -> np.ndarray:
    """
    Map position x into the primary box [0, L]³ via mirror folding.

    For each coordinate c:
        c_mod = c % (2L)
        c_eff = c_mod      if c_mod ≤ L   (inside primary box)
              = 2L - c_mod  if c_mod > L   (inside mirror image box)

    This creates a field that is C⁰ at every face — no jump when the particle
    crosses a boundary. Plain modular wrapping (c % L) creates a discontinuity
    whenever the wave modes are not integer multiples of k_min.

    The particle position x is kept unbounded; only field evaluation uses c_eff.
    """
    x_mod = x % (2 * L)
    return np.where(x_mod <= L, x_mod, 2 * L - x_mod)


# ── Wave spectrum builder ──────────────────────────────────────────────────────

def build_wave_spectrum(cfg: SimConfig):
    """Return (k_mags, amplitudes, k_vectors, phase_offsets) for the configured wave spectrum.

    Phase clustering (cfg.phase_coherence)
    ---------------------------------------
    λ = 1: modes sharing the same k-decade are assigned the same random phase,
           producing constructive interference → coherent structures in the field.
    λ = 0: each mode has an independent random phase → incoherent superposition.
    Intermediate λ interpolates between these extremes.
    The power spectrum (S₂) is unchanged by λ; only higher-order statistics
    (intermittency) are affected.
    """
    k_base = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.n_k)
    k_mags = np.sort(np.append(k_base, np.asarray(cfg.k_extra, dtype=float)))

    amplitudes  = np.array([cfg.amp_scale * (k / k_mags[0]) ** cfg.spectral_index
                            for k in k_mags])

    rng         = np.random.default_rng(cfg.rng_seed)

    wave_angles = rng.uniform(-cfg.angle_spread, cfg.angle_spread, len(k_mags))
    k_vectors   = [
        np.array([k * np.sin(ang), 0.0, k * np.cos(ang)])
        for k, ang in zip(k_mags, wave_angles)
    ]

    # Phase offsets: interpolate between independent (λ=0) and decade-shared (λ=1).
    # S_2(ℓ) = 2[R(0) - R(ℓ)] is set by the power spectrum alone; the phases
    # control only higher-order statistics (intermittency).
    k_decades      = np.floor(np.log10(k_mags)).astype(int)
    unique_decades = np.unique(k_decades)
    decade_phase   = {d: rng.uniform(0, 2 * np.pi) for d in unique_decades}
    clustered_phi  = np.array([decade_phase[d] for d in k_decades])
    random_phi     = rng.uniform(0, 2 * np.pi, len(k_mags))
    lam            = float(cfg.phase_coherence)
    phase_offsets  = (lam * clustered_phi + (1 - lam) * random_phi) % (2 * np.pi)
    print(f"Phase coherence λ={lam:.2f}: {len(unique_decades)} k-decades  |  "
          f"envelope σ={cfg.envelope_sigma:.2f}")

    return k_mags, amplitudes, k_vectors, phase_offsets


# ── Main run loop ──────────────────────────────────────────────────────────────

def run_simulation(cfg: SimConfig | None = None) -> dict:
    """
    Run the particle simulation and return a results dictionary.

    Returns
    -------
    dict with keys: pos, vel, Bperp, dt, steps, save_every
    """
    if cfg is None:
        cfg = SimConfig()

    k_mags, amplitudes, k_vectors, phase_offsets = build_wave_spectrum(cfg)

    # L is the primary box side length = longest wavelength in the spectrum.
    # The fold BC reflects the field across each face, creating a 4-box
    # (2L × 2L) symmetric structure that is C⁰ everywhere.
    L_env      = 2 * np.pi / k_mags.min()
    env_coords = np.linspace(0, L_env, 4000, endpoint=False)
    corr_len  = cfg.envelope_corr_frac * L_env
    env_W     = make_lognormal_envelope(env_coords, cfg.envelope_sigma,
                                        corr_len, seed=cfg.envelope_seed)

    # Initial conditions
    theta_rad = np.radians(cfg.theta_v)
    x = np.zeros(3)
    v = np.array([
        cfg.v_total * np.sin(theta_rad),
        0.0,
        cfg.v_total * np.cos(theta_rad),
    ])

    n_records     = cfg.steps // cfg.save_every
    pos_history   = np.empty((n_records, 3))
    vel_history   = np.empty((n_records, 3))
    Bperp_history = np.empty(n_records)
    record_idx    = 0

    print(f"Running {cfg.steps:,} steps (dt={cfg.dt})  ...")
    t0 = time.perf_counter()

    for i in range(cfg.steps):
        t       = i * cfg.dt
        E_total = np.zeros(3)
        B_total = np.array([0.0, 0.0, cfg.B0])

        # Fold x into the primary box for field evaluation.
        # x itself remains unbounded — the true physical trajectory for D(t).
        x_eff = _fold_pos(x, L_env)

        for k_vec, amp, phi in zip(k_vectors, amplitudes, phase_offsets):
            Ei, dBi  = alfven_wave_fields(x_eff, t, k_vec, amp, phi)
            E_total += Ei
            B_total += dBi

        # Envelope W(x_perp): x_eff[0] is already in [0, L_env], no extra wrap.
        W = np.interp(x_eff[0], env_coords, env_W)
        E_total *= W
        B_total[0] *= W   # only the wave perturbation, not B0
        B_total[1] *= W

        x, v = boris_pusher(x, v, E_total, B_total, cfg.dt)
        # No position wrapping — x is the true physical trajectory

        if i % cfg.save_every == 0:
            pos_history[record_idx]   = x
            vel_history[record_idx]   = v
            Bperp_history[record_idx] = np.sqrt(B_total[0]**2 + B_total[1]**2)
            record_idx += 1

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f} s  ({cfg.steps / elapsed:,.0f} steps/s)")

    return {
        "pos":        pos_history[:record_idx],
        "vel":        vel_history[:record_idx],
        "Bperp":      Bperp_history[:record_idx],
        "dt":         cfg.dt,
        "steps":      cfg.steps,
        "save_every": cfg.save_every,
        "box_size":            cfg.box_size,
        "phase_coherence":     cfg.phase_coherence,
        "envelope_sigma":      cfg.envelope_sigma,
        "envelope_corr_frac":  cfg.envelope_corr_frac,
        "envelope_seed":       cfg.envelope_seed,
        # Envelope grid — saved so analysis.py uses the identical realization
        "env_coords":   env_coords,   # (4000,) x-coordinates
        "env_W":        env_W,        # (4000,) W(x) values
        # Wave spectrum — needed for spatial field analysis
        "k_vectors":    np.array(k_vectors),     # (n_modes, 3)
        "amplitudes":   np.array(amplitudes),    # (n_modes,)
        "phase_offsets": phase_offsets,          # (n_modes,)
    }


# ── Persistence helpers ────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Allow json.dump to serialise numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()      # ndarray -> nested Python list
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def save_results(results: dict, cfg: SimConfig) -> Path:
    """
    Save results to a JSON file.

    The file is plain text: open it in any editor, load it in any language
    with a standard JSON parser, or inspect it with `python -m json.tool`.
    """
    out_dir  = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.output_file
    with open(out_path, "w") as f:
        json.dump(results, f, cls=_NumpyEncoder, indent=2)
    print(f"Results saved -> {out_path}")
    return out_path


def load_results(path: str | Path) -> dict:
    """
    Load a JSON results file produced by save_results.
    Lists are automatically converted back to numpy arrays.
    """
    with open(path) as f:
        raw = json.load(f)
    return {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in raw.items()
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> SimConfig:
    cfg = SimConfig()
    parser = argparse.ArgumentParser(description="Run slab Alfven wave particle simulation")
    parser.add_argument("--dt",      type=float, default=cfg.dt)
    parser.add_argument("--steps",   type=int,   default=cfg.steps)
    parser.add_argument("--B0",      type=float, default=cfg.B0)
    parser.add_argument("--v_total", type=float, default=cfg.v_total)
    parser.add_argument("--seed",    type=int,   default=cfg.rng_seed)
    parser.add_argument("--out_dir", type=str,   default=cfg.output_dir)
    args = parser.parse_args()

    cfg.dt         = args.dt
    cfg.steps      = args.steps
    cfg.B0         = args.B0
    cfg.v_total    = args.v_total
    cfg.rng_seed   = args.seed
    cfg.output_dir = args.out_dir
    return cfg


if __name__ == "__main__":
    cfg     = _parse_args()
    results = run_simulation(cfg)
    save_results(results, cfg)