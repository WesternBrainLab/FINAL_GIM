"""Plot normalized structural hub strength (mu) against regional PET."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "RESULTS" / "Mu_vs_pet"
sys.path.insert(0, str(PROJECT_DIR))

import UTILS as utils
import CONFIG as cf

DATA_DIR = cf.DATA_DIR


def load_pet_values() -> np.ndarray:
    """Load the regional PET vector using the shared UTILS loader."""
    pet = utils.get_matrix("PET_data/PET_temp_no_outliers", directory=str(DATA_DIR))
    return np.asarray(pet, dtype=float).reshape(-1)


def run_mu_vs_pet_diagnostic(J_real: np.ndarray) -> Path:
    mu = utils.normalize_array(np.mean(np.abs(J_real), axis=0))
    pet_values = load_pet_values()

    if pet_values.size != mu.size:
        raise ValueError(
            f"PET has {pet_values.size} values but Jij has {mu.size} regions."
        )
    if not np.all(np.isfinite(pet_values)):
        raise ValueError("PET values contain NaN or infinite values.")

    r, p = pearsonr(mu, pet_values)
    print(f"mu vs PET: Pearson r={r:.6f}, p={p:.6g}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slope, intercept = np.polyfit(mu, pet_values, 1)
    order = np.argsort(mu)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(mu, pet_values, color="tab:blue", alpha=0.8, edgecolor="white")
    ax.plot(mu[order], slope * mu[order] + intercept, color="tab:red",
            linewidth=2, label=f"Pearson r = {r:.3f}, p = {p:.3g}")
    ax.set_xlabel(r"normalized structural hub strength $\mu_i$")
    ax.set_ylabel(r"PET value $PET_i$")
    ax.set_title(r"Structural hub strength $\mu$ vs PET")
    ax.legend(framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    output_path = RESULTS_DIR / "mu_vs_pet.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    run_mu_vs_pet_diagnostic(cf.avg_Jij)
