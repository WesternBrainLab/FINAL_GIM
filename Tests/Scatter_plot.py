"""Real-data Pearson FC correlation versus structural-connectivity scrambling."""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import CONFIG as cf
import GIM as I
import UTILS as utils


def scramble_jij(jij, level, rng):
    """Shuffle a fraction of symmetric upper-triangular Jij edge weights."""
    matrix = np.asarray(jij, dtype=float).copy()
    rows, cols = np.triu_indices_from(matrix, k=1)
    count = int(round(np.clip(level, 0, 1) * len(rows)))
    if count > 1:
        selected = rng.choice(len(rows), count, replace=False)
        values = matrix[rows[selected], cols[selected]].copy()
        rng.shuffle(values)
        matrix[rows[selected], cols[selected]] = values
        matrix[cols[selected], rows[selected]] = values
    np.fill_diagonal(matrix, 0)
    return matrix


def simulate_correlation(jij, empirical_fc, temperature, alpha, steps, thermalization, rng):
    multiplier = utils.normalize_array(np.mean(np.abs(jij), axis=0))
    multiplier = np.nan_to_num(multiplier, nan=1.0) + 1e-12
    model = I.Jij_sorted_ising(
        temperature * multiplier ** alpha,
        Jij=jij,
        spin_ar=rng.choice([-1, 1], size=jij.shape[0]),
    )
    model.simulate(steps, thermalization)
    simulated_fc = np.asarray(model.generate_FC(partial=False)["matrix"])
    return float(utils.mat_corr(simulated_fc, empirical_fc))


def run(args):
    rng = np.random.default_rng(args.seed)
    empirical_fc = np.asarray(cf.avg_FC, dtype=float).copy()
    jij = np.asarray(cf.avg_Jij, dtype=float).copy()
    np.fill_diagonal(empirical_fc, 0)
    np.fill_diagonal(jij, 0)
    levels = np.linspace(0, 1, args.levels)
    all_values = []
    for level in levels:
        values = [simulate_correlation(
            scramble_jij(jij, level, rng), empirical_fc, args.temperature,
            args.alpha, args.steps, args.thermalization, rng
        ) for _ in range(args.repeats)]
        all_values.append(values)
        print(f"scrambling={level:.3f}: r={np.mean(values):.5f} +/- {np.std(values):.5f}")

    values = np.asarray(all_values, dtype=float)
    means = np.nanmean(values, axis=1)
    errors = np.nanstd(values, axis=1, ddof=1) if args.repeats > 1 else np.zeros_like(means)
    fit = linregress(levels, means)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "scrambling_vs_pearson_r.csv",
               np.column_stack((levels, means, errors)), delimiter=",")

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.errorbar(levels, means, yerr=errors, fmt="o", color="gray", capsize=3,
                label="simulations")
    ax.plot(levels, fit.slope * levels + fit.intercept, color="blue", linewidth=2,
            label=f"linear fit (slope={fit.slope:.3f}, r={fit.rvalue:.3f})")
    ax.set(xlabel="Level of scrambling of SC matrix",
           ylabel="Pearson correlation (simulated vs empirical FC)",
           title="Pearson FC correlation vs structural-connectivity scrambling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "scatter_scrambling_vs_pearson_r.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=2.13)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--thermalization", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(cf.PROJECT_ROOT + "RESULTS/Scatter_plot/"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
