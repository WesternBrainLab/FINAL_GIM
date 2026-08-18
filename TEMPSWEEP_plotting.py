from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import CONFIG as cf
import GIM as I
import UTILS as utils


RESULTS_DIR = Path(cf.PROJECT_ROOT) / "RESULTS" / "TEMPSWEEP"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BLUE = "#2E86AB"
SD_BAND = "#2CA25F"
RED = "#E84855"
AMBER = "#F4A261"

## editing right now ...
def run_sweep(t_min=0.05, t_max=10.0, t_steps=100, alpha=2.0,
              steps=4000, thermalization=2000):
    jij = np.asarray(cf.avg_Jij, dtype=float) * utils.get_sign_matrix(cf.avg_FC)
    empirical = np.asarray(cf.avg_FC, dtype=float).copy()
    np.fill_diagonal(empirical, 0.0)
    multiplier = cf.norm_ind_avg_Jij
    temperatures = np.linspace(t_min, t_max, t_steps)

    models, correlations = [], []
    energy, magnetization, susceptibility, specific_heat = [], [], [], []

    for index, global_temp in enumerate(temperatures):
        model = I.Jij_sorted_ising(
            global_temp * multiplier ** alpha,
            Jij=jij,
            beta=1 / global_temp,
            spin_ar=np.random.choice([-1, 1], jij.shape[0]),
        )
        model.simulate(steps, thermalization, log=True)
        simulated = model.generate_FC(partial=False)
        models.append(model)
        energy.append(float(np.mean(model.energy_series)))
        magnetization.append(float(np.mean(np.abs(model.mag_series))))
        susceptibility.append(float(model.susceptibility(1 / global_temp)))
        specific_heat.append(float(model.specific_heat(1 / global_temp)))
        correlations.append(float(utils.mat_corr(np.asarray(simulated["matrix"]), empirical)))
        print(f"{index + 1}/{t_steps}: T={global_temp:.4f}")

    correlations = np.asarray(correlations)
    crit_index = int(np.nanargmax(specific_heat))
    best_index = int(np.nanargmax(correlations))
    return {
        "temperatures": temperatures,
        "models": models,
        "empirical": empirical,
        "jij": jij,
        "energy": np.asarray(energy),
        "magnetization": np.asarray(magnetization),
        "susceptibility": np.asarray(susceptibility),
        "specific_heat": np.asarray(specific_heat),
        "correlations": correlations,
        "crit_index": crit_index,
        "best_index": best_index,
        "alpha": alpha,
    }


def plot_temperature_sweep(result):
    t = result["temperatures"]
    crit_t = t[result["crit_index"]]
    best_t = t[result["best_index"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"Ising model — temperature sweep  |  alpha = {result['alpha']:.3f}",
                 fontsize=14, fontweight="bold")
    panels = [
        (axes[0, 0], result["energy"], "average energy", "Energy vs T", BLUE),
        (axes[0, 1], result["magnetization"], "average |magnetization|", "|Magnetization| vs T", BLUE),
        (axes[1, 0], result["specific_heat"], "specific heat C", "Specific Heat vs T", RED),
        (axes[1, 1], result["susceptibility"], "susceptibility S", "Susceptibility vs T", SD_BAND),
    ]
    for ax, values, ylabel, title, color in panels:
        ax.plot(t, values, color=color, linewidth=2.0)
        ax.axvline(crit_t, color=RED, linestyle="--", label=f"T_crit = {crit_t:.2f}")
        ax.axvline(best_t, color=AMBER, linestyle=":", label=f"T_best = {best_t:.2f}")
        ax.set_xlabel("global temperature T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS_DIR / "temperature_sweep_M_S_E_C.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(result):
    t = result["temperatures"]
    crit_t = t[result["crit_index"]]
    best_t = t[result["best_index"]]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(t, result["correlations"], color=BLUE, linewidth=2.0)
    ax.axvline(crit_t, color=RED, linestyle="--", label=f"T_crit = {crit_t:.2f}")
    ax.axvline(best_t, color=AMBER, linestyle=":", label=f"T_best = {best_t:.2f}")
    ax.set_xlabel("Global temperature T")
    ax.set_ylabel("Pearson r (simulated FC vs empirical FC)")
    ax.set_title("Correlation vs Temperature")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS_DIR / "correlation_r_vs_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_matrices(result):
    empirical, structural = result["empirical"], result["jij"]
    rows = [("Critical temperature", result["crit_index"]),
            ("Best Pearson r temperature", result["best_index"])]
    limit = max(np.max(np.abs(empirical)), np.max(np.abs(structural)))
    for _, index in rows:
        limit = max(limit, np.max(np.abs(result["models"][index].functional_connectivity["matrix"])))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row, (label, index) in enumerate(rows):
        temperature = result["temperatures"][index]
        correlation = result["correlations"][index]
        simulated = np.asarray(result["models"][index].functional_connectivity["matrix"])
        for col, (matrix, title) in enumerate([
            (simulated, f"Simulated Pearson FC\n{label}, T={temperature:.2f}, r={correlation:.4f}"),
            (empirical, "Empirical Pearson FC"),
            (structural, "Structural connectivity J"),
        ]):
            image = axes[row, col].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axes[row, col].set_title(title, fontsize=10, pad=10)
            axes[row, col].set_xlabel("region")
            axes[row, col].set_ylabel("region")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.suptitle("Functional connectivity and structural matrix comparisons", fontsize=14)
    fig.savefig(RESULTS_DIR / "FC_matrices_critical_and_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sweep_result = run_sweep()
    plot_temperature_sweep(sweep_result)
    plot_correlation(sweep_result)
    plot_matrices(sweep_result)
    print(f"Saved Pearson-style temperature-sweep results to {RESULTS_DIR}")
