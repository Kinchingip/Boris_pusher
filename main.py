"""
main.py
-------
Entry point: run simulation then plot results in one command.

    python main.py                    # default config
    python main.py --steps 100000 --save figures/
"""

import argparse
from pathlib import Path

from simulation import SimConfig, run_simulation, save_results, load_results
from plot_results import build_overview_figure
import matplotlib.pyplot as plt


def parse_args() -> tuple[SimConfig, argparse.Namespace]:
    cfg = SimConfig()
    parser = argparse.ArgumentParser(description="Slab Alfvén particle simulation")
    parser.add_argument("--dt",       type=float, default=cfg.dt)
    parser.add_argument("--steps",    type=int,   default=cfg.steps)
    parser.add_argument("--B0",       type=float, default=cfg.B0)
    parser.add_argument("--v_total",  type=float, default=cfg.v_total)
    parser.add_argument("--seed",     type=int,   default=cfg.rng_seed)
    parser.add_argument("--out_dir",  type=str,   default=cfg.output_dir)
    parser.add_argument("--save_figs",type=str,   default=None,
                        help="Directory to save PNG figures (omit = show interactively)")
    parser.add_argument("--skip_sim", action="store_true",
                        help="Skip simulation; load existing results and just plot")
    args = parser.parse_args()

    cfg.dt         = args.dt
    cfg.steps      = args.steps
    cfg.B0         = args.B0
    cfg.v_total    = args.v_total
    cfg.rng_seed   = args.seed
    cfg.output_dir = args.out_dir
    return cfg, args


def main():
    cfg, args = parse_args()

    results_path = Path(cfg.output_dir) / cfg.output_file

    if args.skip_sim and results_path.exists():
        print(f"Loading existing results from {results_path}")
        results = load_results(results_path)
    else:
        results = run_simulation(cfg)
        save_results(results, cfg)

    fig = build_overview_figure(
        pos=results["pos"],
        vel=results["vel"],
        Bperp=results["Bperp"],
    )

    if args.save_figs:
        save_dir = Path(args.save_figs)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_fig  = save_dir / "overview.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Figure saved → {out_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()