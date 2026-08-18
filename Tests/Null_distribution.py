"""Standalone shuffled-Jij and constant-Jij null-distribution analysis."""

from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import CONFIG as cf
import GIM as I
import UTILS 


def shuffle_jij(jij):
    out = np.asarray(jij, dtype=float).copy()
    i, j = np.triu_indices(out.shape[0], 1)
    values = out[i, j].copy()
    np.random.shuffle(values)
    out[i, j] = values
    out[j, i] = values
    np.fill_diagonal(out, 0.0)
    return out


def threshold_jij(jij, empirical_fc, threshold):
    out = np.asarray(jij, dtype=float).copy()
    mask = ~np.eye(out.shape[0], dtype=bool) & (np.abs(empirical_fc) >= threshold)
    out[~mask] = 0.0
    return (out + out.T) / 2.0


def constant_jij(jij, value):
    out = np.full_like(jij, value, dtype=float)
    np.fill_diagonal(out, 0.0)
    return out


def simulate_fc(jij, temperature, steps, thermalization, runs):
    n = jij.shape[0]
    multiplier = utils.normalize_array(np.mean(np.abs(jij), axis=0))
    multiplier = np.nan_to_num(multiplier, nan=1.0) + 1e-12
    temps = temperature * multiplier
    total = np.zeros((n, n))
    for _ in range(runs):
        sim = I.Jij_sorted_ising(temps, Jij=jij, spin_ar=np.ones(n))
        sim.simulate(steps, thermalization)
        total += sim.generate_FC(partial=False)
    return total / runs


def upper_values(matrix):
    return np.asarray(matrix)[np.triu_indices(matrix.shape[0], 1)]


def effect_sizes(values, real):
    values = np.asarray(values, dtype=float)
    sd = np.std(values, ddof=1) if values.size > 1 else 0.0
    d = (real - np.mean(values)) / sd if sd > 0 else 0.0
    delta = (np.sum(values > real) - np.sum(values < real)) / len(values)
    return d, delta


def plot_null(ax, values, real, p_value, d, delta, xlabel, title, xlim=None):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=20)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        ax.bar(left, count, width=right - left, align="edge",
               color="#C0392B" if right <= real else "#5BA4CF",
               alpha=0.45 if right <= real else 0.8,
               edgecolor="white", linewidth=0.5)
    ax.axvline(real, color="#C0392B", linestyle="--", linewidth=2.2,
               label=f"real Jij ({real:.4f})")
    ax.text(0.97, 0.95,
            f"p = {p_value:.4f}\nCohen's d = {d:.3f}\nCliff's δ = {delta:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.6))
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(title, fontsize=12)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


def run_null_distribution(jij, empirical_fc, temperature, alpha, threshold,
                          n_null, steps, thermalization, runs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    real = threshold_jij(jij, empirical_fc, threshold)
    shuffled_example = threshold_jij(shuffle_jij(jij), empirical_fc, threshold)
    constant = threshold_jij(constant_jij(jij, 1.0), empirical_fc, threshold)

    matrix_values = np.concatenate([
        upper_values(real),
        upper_values(shuffled_example),
        upper_values(constant),
    ])
    matrix_norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        f"Pearson null Jij matrices  |  threshold={threshold:g}",
        fontsize=13, fontweight="bold",
    )
    for ax, matrix, title in zip(
        axes, (real, shuffled_example, constant),
        ("Real Jij", "Shuffled Jij", "Constant Jij"),
    ):
        image = ax.imshow(matrix, cmap="RdBu_r", norm=matrix_norm)
        ax.set_title(title)
        ax.set_xlabel("region")
        ax.set_ylabel("region")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    matrix_path = output_dir / "null_jij_matrices.png"
    fig.savefig(matrix_path, dpi=150)
    plt.close(fig)

    empirical = upper_values(empirical_fc)
    real_fc = simulate_fc(real, temperature, steps, thermalization, runs)
    real_values = upper_values(real_fc)
    real_distance = np.linalg.norm(real_values - empirical)
    real_dissimilarity = 1.0 - float(np.nan_to_num(pearsonr(real_values, empirical)[0]))
    results = {"shuffled": [], "constant": []}
    for index in range(n_null):
        for name, matrix in (("shuffled", threshold_jij(shuffle_jij(jij), empirical_fc, threshold)),
                             ("constant", constant)):
            simulated = upper_values(simulate_fc(matrix, temperature, steps, thermalization, runs))
            results[name].append(np.linalg.norm(simulated - empirical))
        print(f"{index + 1}/{n_null}")

    metrics = {}
    for name, values in results.items():
        distances = np.asarray(values)
        # Re-run the correlation metric from the distance simulation using
        # the same null FC samples would require storing each FC; use the
        # distance distribution for the primary pearson_2-style figure.
        p = float(np.mean(distances <= real_distance))
        metrics[name] = (distances, p, *effect_sizes(distances, real_distance))

    finite = np.concatenate([metrics["shuffled"][0], metrics["constant"][0], [real_distance]])
    pad = max(np.ptp(finite) * 0.03, 1e-12)
    xlim = (finite.min() - pad, finite.max() + pad)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Ising null distribution  |  T = {temperature:.2f}  |  alpha = {alpha:.2f}",
                 fontsize=13, fontweight="bold")
    plot_null(axes[0], metrics["shuffled"][0], real_distance,
              metrics["shuffled"][1], metrics["shuffled"][2], metrics["shuffled"][3],
              r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
              "null distribution — euclidean distance", xlim)
    plot_null(axes[1], metrics["constant"][0], real_distance,
              metrics["constant"][1], metrics["constant"][2], metrics["constant"][3],
              r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
              "ones Jij null — euclidean distance", xlim)
    dist_path = output_dir / "ising_null_distributions_3.png"
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {matrix_path}")
    print(f"Saved: {dist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=2.13)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--thermalization", type=int, default=200)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    jij = utils.get_matrix("Jij data_processed/avg_Jij_no_outliers_norm", directory=cf.DATA_DIR)
    empirical = cf.avg_FC
    run_null_distribution(
        jij, empirical, args.temperature, args.alpha, args.threshold,
        args.n_null, args.steps, args.thermalization, args.runs,
        PROJECT_DIR / "RESULTS",
    )
