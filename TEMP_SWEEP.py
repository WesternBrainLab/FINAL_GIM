"""Run an Ising temperature sweep and save Pearson-style diagnostic plots."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import CONFIG as cf
import FUNC_CON as fc
import GIM as I
import UTILS as utils


BLUE = "#2E86AB"
GREEN = "#2CA25F"
RED = "#E84855"
AMBER = "#F4A261"


class temp_sweep:
    """Temperature sweep with observables and empirical-FC comparisons."""

    def __init__(self, min_temp, max_temp, temp_step, alpha, multiplier, Jij,
                 ising=I.Jij_sorted_ising, save=False, name="temp_sweep",
                 path=cf.TEMP_SWEEP_DATA):
        self.T_global = np.linspace(min_temp, max_temp, temp_step)
        self.alpha = alpha
        self.multiplier = np.asarray(multiplier, dtype=float)
        self.Jij = np.asarray(Jij, dtype=float)
        self.ising = ising
        self.save = save
        self.path = Path(path)
        self.name = name

        self.sim_dataset = {}
        self.models = []
        self.ising_ar = []
        self.suscept_ar = []
        self.spec_heat_ar = []
        self.avg_energy_ar = []
        self.avg_mag_ar = []
        self.corr_ar = []

    def simulate(self, steps, thermalization, spin_array=None, partial=False,
                 empirical_fc=None, show=False):
        """Run the sweep and calculate observables for every temperature."""
        if spin_array is None:
            spin_array = np.random.choice([-1, 1], self.Jij.shape[0])
        else:
            spin_array = np.asarray(spin_array).copy()

        empirical_fc = cf.avg_FC if empirical_fc is None else np.asarray(empirical_fc)

        for index, temp in enumerate(self.T_global):
            temperatures = temp * self.multiplier ** self.alpha
            model = self.ising(
                temperatures,
                Jij=self.Jij,
                beta=1 / temp,
                spin_ar=spin_array.copy(),
            )
            log = model.simulate(steps, thermalization, log=True)
            simulated_fc = model.generate_FC(partial=partial)
            simulated_matrix = np.asarray(simulated_fc["matrix"])

            self.models.append(model)
            self.ising_ar.append(log)
            self.sim_dataset[index] = simulated_fc
            self.suscept_ar.append(float(model.susceptibility(1 / temp)))
            self.spec_heat_ar.append(float(model.specific_heat(1 / temp)))
            self.avg_energy_ar.append(float(np.mean(model.energy_series)))
            self.avg_mag_ar.append(float(np.mean(np.abs(model.mag_series))))
            self.corr_ar.append(float(utils.mat_corr(simulated_matrix, empirical_fc)))

            print(f"{index + 1}/{len(self.T_global)}: T={temp:.4f}")
            if show:
                model.graph_everything(show=True)

        self.crit_index = int(np.nanargmax(self.spec_heat_ar))
        self.best_index = int(np.nanargmax(self.corr_ar))
        self.crit_temp = float(self.T_global[self.crit_index])
        self.best_temp = float(self.T_global[self.best_index])
        self.crit_ising = self.models[self.crit_index]
        self.best_ising = self.models[self.best_index]

        self.log = {
            "alpha": float(self.alpha),
            "min temp": float(self.T_global[0]),
            "max temp": float(self.T_global[-1]),
            "temp steps": len(self.T_global),
            "critical temperature": self.crit_temp,
            "best temperature": self.best_temp,
            "best correlation": float(self.corr_ar[self.best_index]),
            "partial correlation": partial,
        }

        if self.save:
            self.path.mkdir(parents=True, exist_ok=True)
            fc.save_to_json(self.sim_dataset, str(self.path), "sim_dataset.json")
            with open(self.path / "log.json", "w") as file:
                json.dump(self.log, file, indent=2)
            with open(self.path / "crit_ising.pickle", "wb") as file:
                pickle.dump(self.crit_ising, file)

        return self

    def correlate(self, empirical_dataset):
        """Compare every simulated FC with every FC in an empirical dataset."""
        self.temp_corr = np.asarray([
            [utils.mat_corr(sim["matrix"], emp["matrix"])
             for emp in empirical_dataset.values()]
            for sim in self.sim_dataset.values()
        ])
        return self.temp_corr

    def graph_corr(self, output_path):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.plot(self.T_global, self.corr_ar, color=BLUE, linewidth=2)
        ax.axvline(self.crit_temp, color=RED, linestyle="--",
                   label=f"T_crit = {self.crit_temp:.2f}")
        ax.axvline(self.best_temp, color=AMBER, linestyle=":",
                   label=f"T_best = {self.best_temp:.2f}")
        ax.set_xlabel("Global temperature T")
        ax.set_ylabel("Pearson r (simulated FC vs empirical FC)")
        ax.set_title("Correlation vs Temperature")
        ax.legend(framealpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def graph_data(self, output_dir=None):
        """Save the requested temperature, matrix, and correlation figures."""
        output_dir = Path(output_dir or cf.PROJECT_ROOT + "RESULTS/TEMP_SWEEP/")
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        panels = [
            (axes[0, 0], self.avg_mag_ar, "M: average |magnetization|", BLUE),
            (axes[0, 1], self.suscept_ar, "S: susceptibility", GREEN),
            (axes[1, 0], self.avg_energy_ar, "E: average energy", BLUE),
            (axes[1, 1], self.spec_heat_ar, "C: specific heat", RED),
        ]
        for ax, values, ylabel, color in panels:
            ax.plot(self.T_global, values, color=color, linewidth=2)
            ax.axvline(self.crit_temp, color=RED, linestyle="--", label="T_crit")
            ax.set_xlabel("Global temperature T")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs temperature")
            ax.legend(framealpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output_dir / "temperature_sweep_3.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.graph_corr(output_dir / "correlation_r_vs_temperature.png")

        empirical = np.asarray(cf.avg_FC)
        structural = np.asarray(self.Jij)
        critical_fc = np.asarray(self.models[self.crit_index].functional_connectivity["matrix"])
        best_fc = np.asarray(self.models[self.best_index].functional_connectivity["matrix"])
        matrix_limit = max(np.max(np.abs(empirical)), np.max(np.abs(critical_fc)),
                           np.max(np.abs(best_fc)), np.max(np.abs(structural)))
        matrix_limit = max(float(matrix_limit), 1e-12)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        rows = [
            ("Critical temperature", self.crit_temp, critical_fc, self.corr_ar[self.crit_index]),
            ("Best Pearson r temperature", self.best_temp, best_fc, self.corr_ar[self.best_index]),
        ]
        for row, (label, temp, simulated, correlation) in enumerate(rows):
            matrices = [simulated, empirical, structural]
            titles = [
                f"Simulated FC\nT={temp:.3f}, r={correlation:.4f}",
                "Empirical FC",
                "Structural matrix J",
            ]
            for col, (matrix, title) in enumerate(zip(matrices, titles)):
                image = axes[row, col].imshow(
                    matrix, cmap="RdBu_r", vmin=-matrix_limit, vmax=matrix_limit
                )
                axes[row, col].set_title(title)
                axes[row, col].set_xlabel("region")
                axes[row, col].set_ylabel("region")
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)
        fig.suptitle("Functional connectivity and structural matrix comparisons", fontsize=14)
        fig.savefig(output_dir / "FC_matrices_critical_and_best.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    simulation = temp_sweep(
        0.05,
        10.0,
        100,
        2.0,
        multiplier=cf.norm_ind_avg_Jij,
        Jij=cf.avg_Jij * utils.get_sign_matrix(cf.avg_FC),
        ising=I.Jij_sorted_ising,
    )
    simulation.simulate(4000, 2000, partial=False, show=False)
    simulation.graph_data()
    print("Saved temperature_sweep_3.png, correlation_r_vs_temperature.png, and FC_matrices_critical_and_best.png")
