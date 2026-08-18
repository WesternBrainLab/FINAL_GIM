"""Optional refined parameter optimization for the FINAL_GIM pipeline.

Refinement is OFF by default. Enable it with ``--refine``.
Run from the FINAL_GIM directory, for example:

    python Tests/Refine_Optimization.py --help
    python Tests/Refine_Optimization.py --refine --trials 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import CONFIG as cf
import GIM as I
import OPTIMIZE as optimization
import UTILS as utils


def build_training_dataset():
    """Return the local empirical FC matrix in the optimizer's expected format."""
    return {"0": {"matrix": np.asarray(cf.avg_FC, dtype=float)}}


def run(args):
    jij = np.asarray(cf.avg_Jij, dtype=float)
    jij = jij * utils.get_sign_matrix(cf.avg_FC, args.threshold)

    optimizer = optimization.optimize(
        steps=args.steps,
        therm=args.thermalization,
        Jij=jij,
        multiplier=cf.norm_ind_avg_Jij,
        ising=I.Jij_sorted_ising,
        partial=args.partial,
        save=False,
    )

    train_data = build_training_dataset()
    beta_range = tuple(args.beta_range)
    alpha_range = tuple(args.alpha_range)
    dataframes = []
    for round_index in range(args.refine_rounds if args.refine else 1):
        print(f"\nOptimization round {round_index + 1}")
        dataframe = optimizer.train(
            train_data,
            trials=args.trials,
            temp_range=beta_range,
            alpha_range=alpha_range,
        )
        dataframe["refinement_round"] = round_index
        dataframes.append(dataframe)

        if args.refine and round_index < args.refine_rounds - 1:
            best = optimizer.study.best_params
            beta_width = (beta_range[1] - beta_range[0]) * args.refine_factor / 2
            alpha_width = (alpha_range[1] - alpha_range[0]) * args.refine_factor / 2
            beta_range = (max(1e-12, best["t_glob"] - beta_width), best["t_glob"] + beta_width)
            alpha_range = (best["alpha"] - alpha_width, best["alpha"] + alpha_width)
            print(f"Refined beta bounds: {beta_range}")
            print(f"Refined alpha bounds: {alpha_range}")

    dataframe = dataframes[-1]

    best = optimizer.study.best_trial
    print("\nOptimization complete")
    print(f"Refinement enabled: {args.refine}")
    print(f"Best beta: {best.params['t_glob']:.6g}")
    print(f"Best global temperature: {1 / best.params['t_glob']:.6g}")
    print(f"Best alpha: {best.params['alpha']:.6g}")
    print(f"Best threshold: {best.params['thresh']:.6g}")
    print(f"Best objective: {best.value:.6g}")
    print(f"Trials recorded: {len(dataframe)}")
    return optimizer, dataframe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", action="store_true",
                        help="enable bound refinement; OFF by default")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--refine-factor", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--thermalization", type=int, default=100)
    parser.add_argument("--beta-range", nargs=2, type=float, default=(0.1, 10.0),
                        metavar=("LOW", "HIGH"),
                        help="beta search bounds; global T is 1 / beta")
    parser.add_argument("--alpha-range", nargs=2, type=float, default=(-3.0, 3.0),
                        metavar=("LOW", "HIGH"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--partial", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

    # Run the script from the FINAL_GIM directory with the following commands to test refinement:

#No refine # python Tests/Refine_Optimization.py 
#Yes refine # python Tests/Refine_Optimization.py --refine