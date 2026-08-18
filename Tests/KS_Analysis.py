"""
ks_analysis.py

Kolmogorov-Smirnov based similarity metrics between simulated and empirical
brain time series data, extending your FC-correlation analysis pipeline
(ising.py / config.py / utils.py, and your `simulated_FC_vs_T_global` sweep
class - this file adds a `simulated_KS_vs_T_global` sibling built the same
way).

Three KS test variants, per spec:
  1. RAW   - KS test between simulated spin time series and raw empirical
             time series
  2. SIGN  - KS test between simulated spin time series and sign-binarized
             (+1/-1) empirical time series
  3. FLIP  - KS test between a simulated spin-flip indicator series
             (1 = spin changed between timesteps, 0 = no change) and a
             thresholded empirical change-indicator series (1 = |change|
             above threshold, 0 = below)

For each variant, per-node KS statistics are computed for every simulated
node against its matching empirical node (scipy.stats.ks_2samp - compares
the two value *distributions* directly, so no timestep alignment or equal
series length needed). These are then averaged:
  - across the subjects within each TS_Data subfolder -> TS_1, TS_2, TS_3
  - across all subjects in all three subfolders combined -> TS_avg
"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "RESULTS" / "KS_Analysis"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import UTILS as utils
import CONFIG as cf
import GIM as I
DATA_DIR = cf.DATA_DIR

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TS_DATA_DIR = DATA_DIR / "TS_Data"
TS_SUBFOLDERS = ["TS_1", "TS_2", "TS_3"]

FLIP_THRESHOLD = 0.5  # >>> tune this to your empirical signal's scale

JIJ_PATH = "Jij data_processed/avg_Jij_no_outliers_norm"

# Run both coupling models for the KS analysis.  This generates six plots:
# raw, sign, and flip KS sweeps for Pearson-Jij coupling and for
# functional-FC coupling.
RUN_FUNCTIONAL_COMPARISON = True

# KS analysis settings. Edit these values for this analysis only.
KS_USE_RANDOM_SPIN_FLIP = False
KS_ALPHA = 1.0
KS_T_MIN = 0.5
KS_T_MAX = 10.0
KS_T_STEPS = 100
KS_SWEEP_STEPS = 100
KS_SWEEP_THERMALIZATION = 200
KS_JIJ_THRESHOLD = 0.0


def load_jij(threshold=KS_JIJ_THRESHOLD):
    """Load and threshold the project Jij matrix using local KS settings."""
    jij = utils.get_matrix(JIJ_PATH, directory=str(DATA_DIR)).astype(float)
    np.fill_diagonal(jij, 0.0)
    jij[np.abs(jij) < threshold] = 0.0
    return jij


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_empirical_folder(subfolder):
    """
    Load all subject time series in one TS_Data subfolder as a list of
    (regions x timepoints) numpy arrays.

    Subject files do not all have the same number of timepoints, so keep
    them as a list instead of forcing them into one stacked numpy array.
    """
    folder_path = Path(
        utils.get_folder(f"TS_Data/{subfolder}", directory=str(DATA_DIR))
    )
    subject_matrices = []
    for file_path in sorted(folder_path.glob("time_series_*.csv")):
        subject_matrices.append(utils.get_matrix(file_path.name, directory=str(folder_path)).T)
    subjects = np.empty(len(subject_matrices), dtype=object)
    subjects[:] = subject_matrices
    return subjects


def load_all_empirical():
    """Return dict {subfolder_name: [subject_matrix, ...]} for all three folders."""
    return {sub: load_empirical_folder(sub) for sub in TS_SUBFOLDERS}


# ----------------------------------------------------------------------------
# Series transforms
# ----------------------------------------------------------------------------
def sign_binarize(ts_arr):
    """Binarize a (regions x timepoints) series to +1/-1 by sign."""
    ts_arr = np.asarray(ts_arr)
    out = np.ones_like(ts_arr)
    out[ts_arr < 0] = -1
    return out


def flip_series_discrete(spin_ts):
    """
    For a discrete {-1, +1} simulated spin series, build a 0/1 flip
    indicator: 1 where the spin changed between consecutive timesteps.
    Output has one fewer timepoint than the input.
    """
    spin_ts = np.asarray(spin_ts)
    return (spin_ts[:, 1:] != spin_ts[:, :-1]).astype(int)


def flip_series_thresholded(ts_arr, threshold=FLIP_THRESHOLD):
    """
    For a continuous empirical series, build a 0/1 change indicator: 1 where
    the magnitude of change between consecutive timesteps exceeds
    `threshold`. Output has one fewer timepoint than the input.
    """
    ts_arr = np.asarray(ts_arr)
    diffs = np.abs(ts_arr[:, 1:] - ts_arr[:, :-1])
    return (diffs > threshold).astype(int)


# ----------------------------------------------------------------------------
# KS computation
# ----------------------------------------------------------------------------
def ks_per_node(sim_ts, emp_ts):
    """
    KS statistic between simulated and empirical time series, computed
    independently per node/region.

    sim_ts, emp_ts: (regions x timepoints) arrays, same number of regions.
    Timepoint counts don't need to match between sim and empirical.

    Returns: 1D array of length `regions`, one KS statistic per node.
    """
    sim_ts = np.asarray(sim_ts)
    emp_ts = np.asarray(emp_ts)
    n_regions = sim_ts.shape[0]

    ks_vals = np.empty(n_regions)
    for node in range(n_regions):
        stat, _ = ks_2samp(sim_ts[node, :], emp_ts[node, :])
        ks_vals[node] = stat
    return ks_vals


def ks_folder_average(sim_ts, subject_matrices, transform=None):
    """
    Average node-averaged KS statistic across all subjects in one folder.

    transform: optional function applied to each empirical subject matrix
    before the KS test (e.g. sign_binarize, flip_series_thresholded). For
    the FLIP variant, pass in `sim_ts` already converted via
    flip_series_discrete.

    Returns (folder_mean, list_of_per_subject_means).
    """
    subject_means = []
    for subj in subject_matrices:
        emp = transform(subj) if transform is not None else subj
        node_ks = ks_per_node(sim_ts, emp)
        subject_means.append(np.mean(node_ks))
    return np.mean(subject_means), subject_means


def run_ks_suite(sim_ts, empirical_data):
    """
    Run all three KS test variants (raw, sign, flip) against all three
    empirical subfolders plus their combined average.

    sim_ts: simulated (regions x timepoints) spin series, values in {-1, +1}.
    empirical_data: dict from load_all_empirical().

    Returns nested dict: results[variant][subfolder_or_'TS_avg'] = mean KS value.
    """
    results = {"raw": {}, "sign": {}, "flip": {}}
    sim_flip = flip_series_discrete(sim_ts)

    all_raw, all_sign, all_flip = [], [], []

    for sub in TS_SUBFOLDERS:
        subj_mats = empirical_data[sub]

        mean_raw, subj_raw = ks_folder_average(sim_ts, subj_mats, transform=None)
        mean_sign, subj_sign = ks_folder_average(sim_ts, subj_mats, transform=sign_binarize)
        mean_flip, subj_flip = ks_folder_average(
            sim_flip, subj_mats, transform=flip_series_thresholded
        )

        results["raw"][sub] = mean_raw
        results["sign"][sub] = mean_sign
        results["flip"][sub] = mean_flip

        all_raw.extend(subj_raw)
        all_sign.extend(subj_sign)
        all_flip.extend(subj_flip)

    results["raw"]["TS_avg"] = np.mean(all_raw)
    results["sign"]["TS_avg"] = np.mean(all_sign)
    results["flip"]["TS_avg"] = np.mean(all_flip)

    return results


# ----------------------------------------------------------------------------
# Temperature sweep - sibling to simulated_FC_vs_T_global
# ----------------------------------------------------------------------------
class simulated_KS_vs_T_global:
    """
    Runs Ising simulations across a range of global temperatures and, at
    each one, computes the full KS test suite (raw / sign / flip, each
    broken out as TS_1 / TS_2 / TS_3 / TS_avg) against empirical data,
    instead of generate_FC()/correlation(). Constructor and simulate()
    signatures mirror simulated_FC_vs_T_global so it slots in next to it.

    :param min_temp, max_temp, temp_step: define self.T_global via
           np.linspace, same as simulated_FC_vs_T_global.
    :param ising: an Ising subclass (e.g. Jij_sorted_ising), passed in the
           same way simulated_FC_vs_T_global takes its `ising` argument.
    :param alpha: temperature fitting exponent.
    :param Jij: structural connectivity matrix.
    :param empirical_data: dict from load_all_empirical().
    :param multiplier: per-neuron scaling array (T_i = T_global *
           multiplier ** alpha) - pass utils.normalize_array(np.mean(Jij, 0))
           to match what you confirmed, same role as simulated_FC_vs_T_global's
           `multiplier` argument.
    """

    def __init__(self, min_temp, max_temp, ising, temp_step, alpha, Jij, empirical_data, multiplier):
        self.T_global = np.linspace(min_temp, max_temp, temp_step)
        self.alpha = alpha
        self.multiplier = multiplier
        self.Jij = Jij
        self.ising = ising
        self.empirical_data = empirical_data

        self.ising_ar = []
        self.avg_temp_ar = []
        self.ks_results = {
            variant: {sub: [] for sub in TS_SUBFOLDERS + ["TS_avg"]}
            for variant in ("raw", "sign", "flip")
        }

        self.save = False

    def simulate(self, steps, thermalization, spin_array, text=True):
        for temp in self.T_global:
            temp_ar = temp * (self.multiplier ** self.alpha)
            avg_temp = np.mean(temp_ar)

            ising_obj = self.ising(temp_ar, Jij=self.Jij, spin_ar=spin_array)
            ising_obj.simulate(steps, thermalization)
            sim_ts = ising_obj.spin_series[:, :-1]  # drop the never-written trailing column

            results = run_ks_suite(sim_ts, self.empirical_data)
            for variant in ("raw", "sign", "flip"):
                for sub in TS_SUBFOLDERS + ["TS_avg"]:
                    self.ks_results[variant][sub].append(results[variant][sub])

            self.avg_temp_ar.append(avg_temp)
            self.ising_ar.append(ising_obj)

            if text:
                print(
                    f"T_global={temp:.3f}  avg_T={avg_temp:.3f}  "
                    f"raw TS_avg={results['raw']['TS_avg']:.4f}  "
                    f"sign TS_avg={results['sign']['TS_avg']:.4f}  "
                    f"flip TS_avg={results['flip']['TS_avg']:.4f}"
                )

    def graph_ks(self, variant="raw", show=True, save_path=None, coupling_label="Ising"):
        """
        Plot KS values for TS_1, TS_2, TS_3, TS_avg vs. global temperature
        for one test variant, in the same style as
        simulated_FC_vs_T_global.graph_data().
        """
        figure, axis = plt.subplots(1)
        axis.set_box_aspect(1)
        for sub in TS_SUBFOLDERS + ["TS_avg"]:
            style = "--" if sub == "TS_avg" else "-"
            axis.plot(self.T_global, self.ks_results[variant][sub], style, label=sub)
        axis.set_xlabel("global temperature")
        axis.set_ylabel("KS statistic")
        axis.set_title(
            f"KS similarity vs. global temperature — {variant} variant\n"
            f"{coupling_label}"
        )
        axis.legend()

        if save_path:
            figure.savefig(save_path)

        if show:
            plt.show()
        else:
            return figure, axis


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    empirical_data = load_all_empirical()

    jij = load_jij()
    ising_class = (
        I.random_ising
        if KS_USE_RANDOM_SPIN_FLIP
        else I.Jij_sorted_ising
    )
    alpha = KS_ALPHA

    coupling_models = [
        ("jij", jij, "structural Jij-driven Ising"),
    ]
    if RUN_FUNCTIONAL_COMPARISON:
        coupling_models.insert(
            0, ("functional", cf.avg_FC, "functional FC-driven Ising")
        )

    print("KS is using FINAL_GIM/CONFIG.py settings")
    print(f"Jij: {JIJ_PATH}")
    print(
        "spin flips: "
        f"{'random' if KS_USE_RANDOM_SPIN_FLIP else 'Jij_sorted'}"
    )
    print(f"alpha: {alpha:g}")

    for filename_label, coupling_matrix, plot_label in coupling_models:
        # Positive regional scaling is required for fractional alpha values.
        multiplier = utils.normalize_array(
            np.mean(np.abs(coupling_matrix), axis=0)
        )

        print(f"\nRunning KS sweep with {plot_label}")
        sweep = simulated_KS_vs_T_global(
            min_temp=KS_T_MIN,
            max_temp=KS_T_MAX,
            ising=ising_class,
            temp_step=KS_T_STEPS,
            alpha=alpha,
            Jij=coupling_matrix,
            empirical_data=empirical_data,
            multiplier=multiplier,
        )
        sweep.simulate(
            steps=KS_SWEEP_STEPS,
            thermalization=KS_SWEEP_THERMALIZATION,
            spin_array=np.ones(coupling_matrix.shape[0]),
        )

        for variant in ("raw", "sign", "flip"):
            output_path = RESULTS_DIR / f"ks_sweep_{filename_label}_{variant}.png"
            figure, _ = sweep.graph_ks(
                variant=variant,
                show=False,
                save_path=output_path,
                coupling_label=plot_label,
            )
            plt.close(figure)
            print(f"Saved: {output_path}")
