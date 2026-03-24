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

from physics import boris_pusher, alfven_wave_fields


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
    k_max: float          = 50.0
    n_k: int              = 29            # base log-spaced modes
    spectral_index: float = -5/6          # amplitude proportional to k^index
    amp_scale: float      = 1.5           # overall amplitude prefactor
    k_extra: list         = field(default_factory=lambda: [0.25])

    # Angular spread of wave vectors around B0 (quasi-parallel slab)
    angle_spread: float = np.pi / 6       # +/- this value in radians
    rng_seed: int       = 42

    # Periodic boundary conditions
    box_size: float = None  # None means PBC disabled; otherwise, box size in all dimensions
    # Output
    output_dir: str  = "results"
    output_file: str = "sim_results.json"


# ── Wave spectrum builder ──────────────────────────────────────────────────────

def build_wave_spectrum(cfg: SimConfig):
    """Return (k_mags, amplitudes, k_vectors) for the configured wave spectrum."""
    k_base = np.logspace(np.log10(cfg.k_min), np.log10(cfg.k_max), cfg.n_k)
    k_mags = np.sort(np.append(k_base, np.asarray(cfg.k_extra, dtype=float)))

    amplitudes = [cfg.amp_scale * (k / k_mags[0]) ** cfg.spectral_index
                  for k in k_mags]

    rng         = np.random.default_rng(cfg.rng_seed)
    wave_angles = rng.uniform(-cfg.angle_spread, cfg.angle_spread, len(k_mags))
    k_vectors   = [
        np.array([k * np.sin(ang), 0.0, k * np.cos(ang)])
        for k, ang in zip(k_mags, wave_angles)
    ]
    return k_mags, amplitudes, k_vectors


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

    k_mags, amplitudes, k_vectors = build_wave_spectrum(cfg)

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

        for k_vec, amp in zip(k_vectors, amplitudes):
            Ei, dBi  = alfven_wave_fields(x, t, k_vec, amp)
            E_total += Ei
            B_total += dBi

        x, v = boris_pusher(x, v, E_total, B_total, cfg.dt)
        if cfg.box_size is not None:
            x = x % cfg.box_size   # periodic wrap: position only, never velocity

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
        "box_size":   cfg.box_size,   # None means PBC disabled
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
    Save results to a human-readable JSON file.

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