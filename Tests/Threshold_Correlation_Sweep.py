"""Standalone FC-correlation sweep over Jij/FC thresholds."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import CONFIG as cf
import GIM as I
import UTILS 


def threshold_matrix(matrix, threshold):
    out = np.asarray(matrix, dtype=float).copy()
    out[np.abs(out) < threshold] = 0.0
    return out


def thresholded_correlation(sim_fc, emp_fc, threshold, include_diagonal):
    sim = threshold_matrix(sim_fc, threshold)
    emp = threshold_matrix(emp_fc, threshold)
    if include_diagonal:
        x, y = sim.ravel(), emp.ravel()
    else:
        keep = ~np.eye(sim.shape[0], dtype=bool)
        x, y = sim[keep], emp[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.nan_to_num(pearsonr(x, y)[0]))


def run_sweep(jij, empirical_fc, temperature, alpha, thresholds,
              restarts, steps, thermalization, partial=False, output_dir=None):
    multiplier = utils.normalize_array(np.mean(np.abs(jij), axis=0))
    multiplier = np.nan_to_num(multiplier, nan=1.0) + 1e-12
    temperatures = temperature * multiplier ** alpha
    simulations = []
    for index in range(restarts):
        model = I.Jij_sorted_ising(temperatures, Jij=jij, spin_ar=np.ones(jij.shape[0]))
        model.simulate(steps, thermalization)
        simulations.append(np.nan_to_num(model.generate_FC(partial)))
        print(f"simulation {index + 1}/{restarts}")

    rows = []
    for threshold in thresholds:
        excl = [thresholded_correlation(x, empirical_fc, threshold, False) for x in simulations]
        incl = [thresholded_correlation(x, empirical_fc, threshold, True) for x in simulations]
        rows.append((threshold, np.mean(excl), np.std(excl), np.mean(incl), np.std(incl)))
    results = np.asarray(rows)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(results[:, 0], results[:, 1], label="exclude diagonal", color="#2E86AB")
    ax.fill_between(results[:, 0], results[:, 1] - results[:, 2], results[:, 1] + results[:, 2], alpha=.2)
    ax.plot(results[:, 0], results[:, 3], label="include diagonal", color="#E84855")
    ax.fill_between(results[:, 0], results[:, 3] - results[:, 4], results[:, 3] + results[:, 4], alpha=.2)
    ax.set(xlabel="Threshold", ylabel="Pearson correlation coefficient r",
           title="Correlation vs threshold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    if output_dir is None:
        output_dir = PROJECT_DIR / "RESULTS"
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "threshold_corr_combined_diag_vs_no_diag.png"
    csv = png.with_suffix(".csv")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    np.savetxt(csv, results, delimiter=",", header="threshold,exclude_mean,exclude_std,include_mean,include_std", comments="")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=2.13)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold-start", type=float, default=0.1)
    parser.add_argument("--threshold-stop", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--thermalization", type=int, default=1000)
    args = parser.parse_args()

    jij = utils.get_matrix("Jij data_processed/avg_Jij_no_outliers_norm", directory=cf.DATA_DIR)
    np.fill_diagonal(jij, 0.0)
    run_sweep(
        jij, cf.avg_FC, args.temperature, args.alpha,
        np.arange(args.threshold_start, args.threshold_stop, args.threshold_step),
        args.restarts, args.steps, args.thermalization,
        output_dir=PROJECT_DIR / "RESULTS",
    )
