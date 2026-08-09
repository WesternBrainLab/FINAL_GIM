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

Confirmed against your code:
  - TS_Data/TS_1, TS_Data/TS_2, TS_Data/TS_3 each contain files named
    time_series_<n>.csv (TS_1 has 23 files on GitHub, not 24 - the loader
    doesn't hardcode a count, so this doesn't need to match exactly)
  - each file is (timepoints x regions) - 232 rows x 84 columns in the
    files inspected - so each subject matrix gets transposed to
    (regions x timepoints) before use
  - utils.matrix_from_dir(directory) stacks every file in a folder into
    one array - this is the "function in utils" you meant
  - per-neuron temperature: T_i = T_global * (mu ** alpha), where mu is
    the "multiplier" your simulated_FC_vs_T_global class already takes as
    a constructor arg - confirmed with you directly, matches how that
    class builds temp_ar
  - the temperature-sweep architecture mirrors simulated_FC_vs_T_global
    exactly: same constructor shape, same per-temperature loop calling
    ising(temp_ar, Jij=..., spin_matrix=...).simulate(steps,
    thermalization), except each run computes the KS suite against
    empirical data instead of generate_FC()/correlation()

One thing to flag: the ising.py I pulled from your GitHub repo defines the
Ising constructor's third argument as `spin_ar`, not `spin_matrix`, which
is what simulated_FC_vs_T_global.simulate() actually passes as a keyword.
I mirrored your working call as-is (Jij=..., spin_matrix=...) below on the
assumption your local ising.py has since diverged from what's pushed to
GitHub - if that call errors for you, the repo copy is just stale and this
comment is your reminder of where the mismatch is.

Also: in ising.py's Ising.simulate(), spin_series is allocated with shape
(size, steps + 1), but the recording loop only ever writes indices
0..steps-1 - the last column (index `steps`) is never assigned and stays
at its zero-initialized value. simulated_KS_vs_T_global drops that
trailing column before running KS tests on it.

Still open (I don't have simulate_fc.py/analysis.py to check against):
  - steps / thermalization values, and whether the 5-restart /
    ANNEAL_STEPS=10000 convention from your other scripts should apply
    here too
  - FLIP_THRESHOLD - empirical values run roughly -1.5 to 1.5 in the
    sample file, but you know what counts as a meaningful signal change
"""

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ks_2samp

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_DIR.parent

# ============================================================================
# RESULTS
# ============================================================================
# All GET_RESULTS figures, CSV summaries, and other final outputs are saved
# here.  Change RUN_FOLDER_NAME before a new analysis run to keep results
# separate. Use a Month-Day-Year_Time name, for example: "08-09-2026_1330".
#
# Input matrices and intermediate simulation/optimization files belong in
# FINAL_GIM/DATA; final reportable outputs belong in FINAL_GIM/RESULTS.
DATA_DIR = PROJECT_DIR / "DATA"
RESULTS_ROOT = PROJECT_DIR / "RESULTS"
RUN_FOLDER_NAME = "08-09-2026_1300"  # Change this for each new run.
RESULTS_DIR = RESULTS_ROOT / RUN_FOLDER_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEVEN_DATA_ROOT = DATA_DIR
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import steven.Scripts.utils as utils
import steven.Scripts.config as cf
import steven.Scripts.ising3 as I

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TS_DATA_DIR = STEVEN_DATA_ROOT / "TS_Data"
TS_SUBFOLDERS = ["TS_1", "TS_2", "TS_3"]

FLIP_THRESHOLD = 0.5  # >>> tune this to your empirical signal's scale

PEARSON_JIJ_PATH = DATA_DIR / "Jij_new_pearson" / "avg_Jij_new_pearson.csv"
PEARSON_SUMMARY_PATH = RESULTS_DIR / "pearson_2_summary.csv"

# Run both coupling models for the KS analysis.  This generates six plots:
# raw, sign, and flip KS sweeps for Pearson-Jij coupling and for
# functional-FC coupling.
RUN_FUNCTIONAL_COMPARISON = True


def load_pearson_run_config():
    """Load settings saved by the most recent pearson_2.py analysis run."""
    if not PEARSON_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Run pearson_2.py before this KS analysis so its configuration "
            f"is saved at {PEARSON_SUMMARY_PATH}."
        )

    with PEARSON_SUMMARY_PATH.open(newline="") as summary_file:
        summary = next(csv.DictReader(summary_file), None)

    required_keys = {
        "use_random_spin_flip",
        "used_alpha",
        "final_sweep_T_min",
        "final_sweep_T_max",
        "final_sweep_T_steps",
        "sweep_steps",
        "sweep_thermalization",
        "jij_threshold",
    }
    if summary is None or not required_keys.issubset(summary):
        raise ValueError(
            f"{PEARSON_SUMMARY_PATH} is missing KS settings. Run pearson_2.py again."
        )

    return {
        "use_random_spin_flip": summary["use_random_spin_flip"].strip().lower() == "true",
        "alpha": float(summary["used_alpha"]),
        "t_min": float(summary["final_sweep_T_min"]),
        "t_max": float(summary["final_sweep_T_max"]),
        "t_steps": int(float(summary["final_sweep_T_steps"])),
        "sweep_steps": int(float(summary["sweep_steps"])),
        "sweep_thermalization": int(float(summary["sweep_thermalization"])),
        "threshold": float(summary["jij_threshold"]),
    }


def load_pearson_jij(threshold):
    """Load and preprocess the same structural Jij matrix as pearson_2.py."""
    jij = np.genfromtxt(PEARSON_JIJ_PATH, delimiter=",").astype(float)
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
    folder_path = Path(TS_DATA_DIR) / subfolder
    subject_matrices = []
    for file_path in sorted(folder_path.glob("time_series_*.csv")):
        subject_matrices.append(np.genfromtxt(file_path, delimiter=",").T)
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

    pearson_config = load_pearson_run_config()
    pearson_jij = load_pearson_jij(pearson_config["threshold"])
    ising_class = (
        I.random_ising
        if pearson_config["use_random_spin_flip"]
        else I.Jij_sorted_ising
    )
    alpha = pearson_config["alpha"]

    coupling_models = [
        ("jij", pearson_jij, "Pearson-2 structural Jij-driven Ising"),
    ]
    if RUN_FUNCTIONAL_COMPARISON:
        coupling_models.insert(
            0, ("functional", cf.avg_FC, "functional FC-driven Ising")
        )

    print("KS is using the pearson_2 simulation configuration")
    print(f"Jij: {PEARSON_JIJ_PATH}")
    print(
        "spin flips: "
        f"{'random' if pearson_config['use_random_spin_flip'] else 'Jij_sorted'}"
    )
    print(f"alpha: {alpha:g}")

    for filename_label, coupling_matrix, plot_label in coupling_models:
        # Positive regional scaling is required for fractional alpha values.
        multiplier = utils.normalize_array(
            np.mean(np.abs(coupling_matrix), axis=0)
        )

        print(f"\nRunning KS sweep with {plot_label}")
        sweep = simulated_KS_vs_T_global(
            min_temp=pearson_config["t_min"],
            max_temp=pearson_config["t_max"],
            ising=ising_class,
            temp_step=pearson_config["t_steps"],
            alpha=alpha,
            Jij=coupling_matrix,
            empirical_data=empirical_data,
            multiplier=multiplier,
        )
        sweep.simulate(
            steps=pearson_config["sweep_steps"],
            thermalization=pearson_config["sweep_thermalization"],
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

# DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# Cleaned version
# ═══════════════════════════════════════════════════════════════════════════

#Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr

# Make the workspace package importable when this file is run by absolute path.
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
   sys.path.insert(0, str(WORKSPACE_ROOT))

import steven.Scripts.ising3 as I
import steven.Scripts.utils as utils
import steven.Scripts.param_anneal as pa
import steven.Scripts.temp_sweep as ts

#Directory setup

DATA_ROOT = DATA_DIR
PROJECT_DATA_DIR = DATA_DIR


# ── config ────────────────────────────────────────────────────────────────
# RESULTS_DIR is defined once at the top of this file from RUN_FOLDER_NAME.
SIMULATION_DIR = PROJECT_DATA_DIR / "simulation data"
OPTIMIZATION_DIR = PROJECT_DATA_DIR / "optimization data"
TEMP_SWEEP_DIR = PROJECT_DATA_DIR / "temp sweep data"

JIJ_NEW_DIR = PROJECT_DATA_DIR / "Jij_new_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_Jij_new_pearson.csv"

FC1_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_3"

THRESHOLD = 0.0 # 0, 0,02,0.03
ZERO_FC_DIAGONAL = True


# True  = set empirical/simulated FC diagonals to 0 and compare off-diagonal FC only.
# False = set empirical/simulated FC diagonals to 1 and include diagonals in FC correlations.

SEED          = 1
avg_Jij       = np.genfromtxt(DATA_ROOT / "Jij data_processed" / "avg_Jij_no_outliers_norm", delimiter=",").astype(float)
N             = avg_Jij.shape[0]


ANNEAL_STEPS  = 1000 #
ANNEAL_MAXFUN = 500 #was 500
ANNEAL_THERM  = 3000 #
SWEEP_STEPS   = 1000#was 1000
SWEEP_THERM   = 3000 # was 3000


N_RESTARTS    = 1 #
# False: visit every spin in fixed mean-Jij order.
# True:  randomly sample spin-flip proposals each simulation step.
USE_RANDOM_SPIN_FLIP = False 
ANNEAL_BOUNDS = ((0.1, 10), (-3, 3))
REFINE_T_WINDOW = 1.0
REFINE_ALPHA_WINDOW = 0.5
REFINE_MAXFUN = 100
REFINE_MAX_ROUNDS = 2# was 2
REFINE_SHRINK = 0.5
REFINE_MIN_T_WINDOW = 0.001
REFINE_MIN_ALPHA_WINDOW = 0.0005


T_MIN         = 0.5 #was 0.5
T_MAX         = 10# WAS 50,8
T_STEPS       =150# WAS100,300
TEMP_REPEATS  = 2# increase for a stable susceptibility peak
PEAK_IGNORE_EDGE_POINTS = 3
PEAK_PROMINENCE_FRACTION = 0.15
SMOOTH_TEMPERATURE_PLOTS = True
SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 3


N_NULL        = 100
NULL_RUNS     = 2
NULL_STEPS    = 1000
NULL_THERM    = 1000
CONSTANT_NULL_VALUE = 1.0


BINS          = 50
N_POST_CRIT_MATRICES = 5


# Optional:
# If you want to just use Global Temperature,set this to True.
# For your own optimized result, keep it False.
USE_FIXED_ALPHA = False
FIXED_ALPHA     = 0


BLUE   = "#2E86AB"
SD_BAND = "#2CA25F"
RED    = "#E84855"
AMBER  = "#F4A261"


np.random.seed(SEED)
ISING_CLASS = I.random_ising if USE_RANDOM_SPIN_FLIP else I.Jij_sorted_ising
SPIN_FLIP_METHOD = "random" if USE_RANDOM_SPIN_FLIP else "Jij_sorted"
print(f"Spin-flip proposal method: {SPIN_FLIP_METHOD}")




# ── helper functions ──────────────────────────────────────────────────────
def upper_tri_vec(mat):
   idx = np.triu_indices(mat.shape[0], k=1)
   return mat[idx]




def set_fc_diagonal(mat):
   np.fill_diagonal(mat, 0 if ZERO_FC_DIAGONAL else 1)
   return mat




def fc_compare_vec(mat):
   if ZERO_FC_DIAGONAL:
       return upper_tri_vec(mat)
   return mat.ravel()




def clean_vec(vec):
   """Return a finite vector so null distributions/plots cannot become all-NaN."""
   return np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)




def safe_pearson(x, y):
   """Pearson r that returns 0 instead of NaN for constant or non-finite vectors."""
   x = clean_vec(x)
   y = clean_vec(y)


   mask = np.isfinite(x) & np.isfinite(y)
   x = x[mask]
   y = y[mask]


   if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
       return 0.0


   r = pearsonr(x, y)[0]
   return 0.0 if not np.isfinite(r) else float(r)




def finite_vals(vals, name):
   """Drop non-finite null values and stop with a clear error if none are usable."""
   vals = np.asarray(vals, dtype=float)
   good = vals[np.isfinite(vals)]


   dropped = len(vals) - len(good)
   if dropped > 0:
       print(f"WARNING: dropped {dropped}/{len(vals)} non-finite values from {name}")


   if len(good) == 0:
       raise ValueError(
           f"All values in {name} are NaN/inf. Check Ising simulation output and temperature array."
       )


   return good




def evenly_spaced_indices(indices, n_select):
   if len(indices) <= n_select:
       return indices


   positions = np.linspace(0, len(indices) - 1, n_select, dtype=int)
   return indices[positions]




def refine_bounds(center, T_window, alpha_window, base_bounds=ANNEAL_BOUNDS):
   T_center, alpha_center = center
   (T_low, T_high), (alpha_low, alpha_high) = base_bounds


   refined_T = (
       max(T_low, T_center - T_window),
       min(T_high, T_center + T_window),
   )
   refined_alpha = (
       max(alpha_low, alpha_center - alpha_window),
       min(alpha_high, alpha_center + alpha_window),
   )


   return refined_T, refined_alpha




def smooth_temperature_curve(values, clip_min=None):
   """Smooth plotted temperature-sweep curves without changing the statistics."""
   values = np.asarray(values, dtype=float)
   if not SMOOTH_TEMPERATURE_PLOTS or values.size < 5:
       return values


   x = np.arange(values.size)
   finite = np.isfinite(values)
   if np.sum(finite) < 5:
       return values


   filled = values.copy()
   filled[~finite] = np.interp(x[~finite], x[finite], values[finite])


   window = min(SMOOTH_WINDOW, values.size if values.size % 2 == 1 else values.size - 1)
   if window <= SMOOTH_POLYORDER:
       window = SMOOTH_POLYORDER + 2
       if window % 2 == 0:
           window += 1


   if window > values.size:
       return values


   smoothed = savgol_filter(filled, window_length=window, polyorder=SMOOTH_POLYORDER)
   if clip_min is not None:
       smoothed = np.maximum(smoothed, clip_min)


   return smoothed




def temperature_mean_and_sd_band(values, sd):
   """
   Plot the raw temperature-sweep mean with a repeat-to-repeat SD band.
   """
   values = np.asarray(values, dtype=float)
   sd = np.asarray(sd, dtype=float)


   mean_plot = values
   sd_plot = smooth_temperature_curve(sd, clip_min=0.0)


   return mean_plot, sd_plot




def run_temperature_sweep(label, t_min, t_max, t_steps, alpha, save_dir):
   print("\n" + "=" * 65)
   print(f"STEP 2 : TEMPERATURE SWEEP - {label}  (alpha = {alpha:.4f})")
   print(f"         T range = {t_min:.3f} to {t_max:.3f}  |  steps = {t_steps}")
   print("=" * 65)


   sweep_obj = ts.simulated_FC_vs_T_global(
      min_temp   = t_min,
      max_temp   = t_max,
      temp_step  = t_steps,
      alpha      = alpha,
      Jij        = J_real,
      ising      = ISING_CLASS,
      multiplier = multiplier,
      save       = True
   )


   sweep_obj.simulate(
      steps          = SWEEP_STEPS,
      thermalization = SWEEP_THERM,
      partial        = False,
      diag           = not ZERO_FC_DIAGONAL,
      text           = True,
      n_repeats      = TEMP_REPEATS,
      emp_FC1        = emp_FC1,
      emp_FC2        = emp_FC2,
      emp_FC3        = emp_FC3,
      avg_FC         = rho_emp,
      path           = save_dir
   )


   return sweep_obj



def peak_candidate_indices(values):
   values = np.asarray(values, dtype=float)


   work = values.copy()
   finite = np.isfinite(work)
   if np.sum(finite) < 3:
      return np.array([], dtype=int), work


   edge = max(0, int(PEAK_IGNORE_EDGE_POINTS))
   if work.size > 2 * edge:
      work[:edge] = np.nan
      work[-edge:] = np.nan


   finite = np.isfinite(work)
   finite_vals_local = work[finite]
   spread = np.nanmax(finite_vals_local) - np.nanmin(finite_vals_local)
   prominence = PEAK_PROMINENCE_FRACTION * spread if spread > 0 else 0.0


   filled = work.copy()
   filled[~np.isfinite(filled)] = -np.inf
   peaks, _ = find_peaks(filled, prominence=prominence)
   return peaks.astype(int), work



def stable_peak_index(values):
   values = np.asarray(values, dtype=float)
   peaks, work = peak_candidate_indices(values)
   if peaks.size:
      return int(peaks[np.nanargmax(work[peaks])])
   return int(np.nanargmax(work))



# ── data ──────────────────────────────────────────────────────────────────
J_real  = np.genfromtxt(AVG_JIJ_NEW_PATH, delimiter=",").astype(float)
np.fill_diagonal(J_real, 0)
emp_FC1 = np.genfromtxt(FC1_PATH, delimiter=",").astype(float)
emp_FC2 = np.genfromtxt(FC2_PATH, delimiter=",").astype(float)
emp_FC3 = np.genfromtxt(FC3_PATH, delimiter=",").astype(float)
rho_emp = (emp_FC1 + emp_FC2 + emp_FC3) / 3.0          # Pearson empirical FC throughout
# Use non-negative coupling strength for temperature scaling.
# Signed Pearson Jij can otherwise produce negative multipliers, and
# negative ** fractional alpha becomes NaN.
multiplier = utils.normalize_array(np.mean(np.abs(J_real), axis=0))


# Set FC diagonal for plotting/saving and choose whether comparisons include it.
set_fc_diagonal(rho_emp)
set_fc_diagonal(emp_FC1)
set_fc_diagonal(emp_FC2)
set_fc_diagonal(emp_FC3)


rho_emp_vec = clean_vec(fc_compare_vec(rho_emp))


print("J_real min:          ", J_real.min())
print("J_real max:          ", J_real.max())
print("J_real has negatives:", np.any(J_real < 0))
print("emp FC neg fraction: ", np.mean(rho_emp_vec < 0))




# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 : PARAMETER ANNEALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 : PARAMETER ANNEALING — broad search (T*, alpha*)")
print("=" * 65)


best_result = None
best_optim = None
best_fun = np.inf


for restart_idx in range(N_RESTARTS):
   np.random.seed(SEED + restart_idx)


   print(f"\nRestart {restart_idx + 1}/{N_RESTARTS}")


   optim = pa.optimize(
       ising      = ISING_CLASS,
       Jij        = J_real,
       partial    = False,
       multiplier = multiplier,
       save       = (restart_idx == 0),
       save_dir   = OPTIMIZATION_DIR
   )
  
   result = optim.anneal(
       steps           = ANNEAL_STEPS,
       maxfun          = ANNEAL_MAXFUN,
       emp_FC          = rho_emp,
       therm           = ANNEAL_THERM,
       no_local_search = False,
       show            = False,
       bounds          = ANNEAL_BOUNDS
   )


   print(f"broad restart best r = {max(optim.correlate):.4f}")
   print(f"restart fun    = {result.fun:.6f}")


   if result.fun < best_fun:
       best_fun = result.fun
       best_result = result
       best_optim = optim


result = best_result
optim = best_optim
T_window = REFINE_T_WINDOW
alpha_window = REFINE_ALPHA_WINDOW


for refine_round in range(REFINE_MAX_ROUNDS):
   refined_bounds = refine_bounds(result.x, T_window, alpha_window)
   print("\n" + "=" * 65)
   print(
       f"STEP 1B.{refine_round + 1} : PARAMETER ANNEALING — refined search "
       f"T={refined_bounds[0]}, alpha={refined_bounds[1]}"
   )
   print("=" * 65)


   np.random.seed(SEED + N_RESTARTS + refine_round)
   refine_optim = pa.optimize(
       ising      = ISING_CLASS,
       Jij        = J_real,
       partial    = False,
       multiplier = multiplier,
       save       = False,
       save_dir   = OPTIMIZATION_DIR
   )


   refine_result = refine_optim.anneal(
       steps           = ANNEAL_STEPS,
       maxfun          = REFINE_MAXFUN,
       emp_FC          = rho_emp,
       therm           = ANNEAL_THERM,
       no_local_search = False,
       show            = False,
       bounds          = refined_bounds
   )


   print(f"refined round best r = {max(refine_optim.correlate):.4f}")
   print(f"refined round fun    = {refine_result.fun:.6f}")
   print(f"candidate T, alpha   = ({refine_result.x[0]:.6f}, {refine_result.x[1]:.6f})")


   if refine_result.fun < best_fun:
       best_fun = refine_result.fun
       result = refine_result
       optim = refine_optim
       print("accepted refined pair")
   else:
       print("kept previous best pair")


   T_window *= REFINE_SHRINK
   alpha_window *= REFINE_SHRINK
   if T_window <= REFINE_MIN_T_WINDOW and alpha_window <= REFINE_MIN_ALPHA_WINDOW:
       print(
           "Stopping refinement: "
           f"T window={T_window:.6f}, alpha window={alpha_window:.6f}"
       )
       break


T_star_annealed     = result.x[0]
alpha_star_annealed = result.x[1]


print(f"\nAnnealed T*     = {T_star_annealed:.4f}")
print(f"Annealed alpha* = {alpha_star_annealed:.4f}")
print(f"Annealing best r = {max(optim.correlate):.4f}")


optim.plot_error(show=False)
plt.savefig(RESULTS_DIR / "param_anneal_error_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error_3.png")




# ── choose alpha ──────────────────────────────────────────────────────────
if USE_FIXED_ALPHA:
   alpha_star = FIXED_ALPHA
   print(f"\nUsing fixed alpha = {alpha_star:.4f}")
else:
   alpha_star = alpha_star_annealed
   print(f"\nUsing annealed alpha* = {alpha_star:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 : TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
sweep = run_temperature_sweep(
   label    = "single sweep",
   t_min    = T_MIN,
   t_max    = T_MAX,
   t_steps  = T_STEPS,
   alpha    = alpha_star,
   save_dir = TEMP_SWEEP_DIR
)


# ── NaN guard ─────────────────────────────────────────────────────────────
corr_arr      = np.array(sweep.corr_ar_total)
spec_heat_arr = np.array(sweep.spec_heat_ar)
suscept_arr   = np.array(sweep.suscept_ar)
T_global = sweep.T_global


n_nan = np.sum(np.isnan(corr_arr))
print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")


# override T_crit and T_best using cleaned arrays
T_suscept_peak = sweep.T_global[stable_peak_index(suscept_arr)]
T_spec_heat_peak = sweep.T_global[stable_peak_index(spec_heat_arr)]
crit_idx = stable_peak_index(spec_heat_arr)
best_idx = np.nanargmax(corr_arr)
T_crit    = sweep.T_global[crit_idx]
T_best    = sweep.T_global[best_idx]
best_corr = np.nanmax(corr_arr)


# also patch the sweep object so downstream code is consistent
sweep.crit_temp  = T_crit
sweep.best_temp  = T_best
sweep.best_corr  = best_corr
sweep.best_ising = sweep.ising_ar[best_idx]
sweep.crit_ising = sweep.ising_ar[crit_idx]


print(f"\nSusceptibility peak temperature        : {T_suscept_peak:.4f}")
print(f"Specific heat peak temperature         : {T_spec_heat_peak:.4f}")
print(f"Critical temperature (specific heat)   : {T_crit:.4f}")
print(
   "Peak detection                         : "
   f"prominence>={PEAK_PROMINENCE_FRACTION:.2f} of curve range, "
   f"edge points ignored={PEAK_IGNORE_EDGE_POINTS}"
)
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")




# ── observables ───────────────────────────────────────────────────────────
avg_energy = np.array(sweep.avg_energy_ar)
avg_energy_sd = np.array(sweep.avg_energy_sd_ar)
avg_mag = np.array(sweep.avg_mag_ar)
avg_mag_sd = np.array(sweep.avg_mag_sd_ar)
suscept = np.array(sweep.suscept_ar)
suscept_sd = np.array(sweep.suscept_sd_ar)
spec_heat = np.array(sweep.spec_heat_ar)
spec_heat_sd = np.array(sweep.spec_heat_sd_ar)




# ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
   f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
   fontsize=14,
   fontweight="bold"
)

# reference lines shown on every panel
ref_lines = [
   (T_suscept_peak,   RED,    "--", rf"$T_{{\chi\,peak}}$ = {T_suscept_peak:.2f}"),
   (T_spec_heat_peak, SD_BAND, "--", rf"$T_{{C\,peak}}$ = {T_spec_heat_peak:.2f}"),
   (T_best,           AMBER,   ":",  rf"$T_{{best}}$ = {T_best:.2f}"),
]

panels = [
   (axes1[0, 0], avg_energy, avg_energy_sd, r"average energy $\langle E \rangle$", "Energy vs T"),
   (axes1[0, 1], avg_mag, avg_mag_sd, r"average $|M|$", "|Magnetization| vs T"),
   (axes1[1, 1], suscept, suscept_sd, r"susceptibility $\chi$", "Susceptibility vs T"),
   (axes1[1, 0], spec_heat, spec_heat_sd, r"specific heat $C$", "Specific Heat vs T"),
]

for ax, data, sd, ylabel, title in panels:
   data_plot, sd_plot = temperature_mean_and_sd_band(data, sd)

   ax.plot(T_global, data_plot, color=BLUE, lw=2.0)
   ax.fill_between(T_global, data_plot - sd_plot, data_plot + sd_plot, color=SD_BAND, alpha=0.28, linewidth=0)

   for temp, color, ls, label in ref_lines:
      ax.axvline(temp, color=color, linestyle=ls, lw=1.6, label=label)

   ax.set_xlabel("global temperature  T", fontsize=11)
   ax.set_ylabel(ylabel, fontsize=11)
   ax.set_title(title, fontsize=12)
   ax.legend(fontsize=8, framealpha=0.3)
   ax.spines[["top", "right"]].set_visible(False)

plt.savefig(RESULTS_DIR / "temperature_sweep_3.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: temperature_sweep_3.png")


# ── Figure 2: correlation vs T ────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

corr_total = np.array(sweep.corr_ar_total)
corr_total_sd = np.array(sweep.corr_sd_ar_total)
corr_total_plot, corr_total_sd_plot = temperature_mean_and_sd_band(corr_total, corr_total_sd)

ax_corr.plot(
   T_global,
   corr_total_plot,
   color=BLUE,
   lw=2.0,
   label="avg FC"
)
ax_corr.fill_between(
   T_global,
   corr_total_plot - corr_total_sd_plot,
   corr_total_plot + corr_total_sd_plot,
   color=SD_BAND,
   alpha=0.28,
   linewidth=0,
   label="standard deviation"
)


ax_corr.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")


ax_corr.set_xlabel("Global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim Pearson FC vs emp Pearson FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "correlation_vs_T_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_corr)
print("Saved: correlation_vs_T_3.png")




# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : MATRIX COMPARISON  (T_best) — Pearson FC only
# ════
print("STEP 3 : MATRIX COMPARISON  (T_best, Pearson FC)")
print("=" * 65)


best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()


set_fc_diagonal(sim_FC)


sim_FC_vec = clean_vec(fc_compare_vec(sim_FC))


r_best    = safe_pearson(sim_FC_vec, rho_emp_vec)
dist_best = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best = 1.0 - r_best


print(f"sim FC neg fraction : {np.mean(sim_FC_vec  < 0):.4f}")
print(f"emp FC neg fraction : {np.mean(rho_emp_vec < 0):.4f}")
print(f"sim FC range        : {sim_FC_vec.min():.4f} → {sim_FC_vec.max():.4f}")
print(f"emp FC range        : {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")




# ── color normalization ──────────────────────────────────────────────────
# Use one fixed shared norm for simulated and empirical FC.
fc_lim = 0.5
fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)


# Use separate norm for Jij because it may have a different scale.
j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
j_lim = np.percentile(np.abs(j_offdiag), 99)


if not np.isfinite(j_lim) or j_lim < 0.05:
   j_lim = 0.2


j_norm = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)


print(f"FC color limit  : ±{fc_lim:.4f}")
print(f"Jij color limit : ±{j_lim:.4f}")




# ── matrix figure ────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)


fig3.suptitle(
   f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}  |  r={r_best:.4f}  |  threshold={THRESHOLD:g}",
   fontsize=13,
   fontweight="bold"
)


matrix_panels = [
   (sim_FC,  f"Simulated Pearson FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
   (rho_emp, "Empirical Pearson FC", fc_norm),
   (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
]


for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
   im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
   ax.set_title(title, fontsize=11, pad=12)
   ax.set_xlabel("region", fontsize=9)
   ax.set_ylabel("region", fontsize=9)
   plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.savefig(RESULTS_DIR / "matrix_comparison_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3)




# ── scatter: sim vs emp ──────────────────────────────────────────────────
fig3s, ax3s = plt.subplots(figsize=(6, 5), constrained_layout=True)


ax3s.scatter(
   rho_emp_vec,
   sim_FC_vec,
   s=2,
   alpha=0.3,
   color=BLUE,
   rasterized=True
)


m, b = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)


ax3s.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")


ax3s.set_xlabel("empirical Pearson FC", fontsize=11)
ax3s.set_ylabel("simulated Pearson FC", fontsize=11)
ax3s.set_title(f"Sim vs Emp Pearson FC  (r = {r_best:.4f})", fontsize=12)
ax3s.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "scatter_sim_vs_emp_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3s)


print("Saved: matrix_comparison_3.png, scatter_sim_vs_emp_3.png")




# ── additional matrix comparisons after Tcrit ────────────────────────────
post_crit_indices = np.where(T_global > T_crit)[0]
best_idx = int(np.nanargmax(corr_arr))
post_crit_indices = post_crit_indices[post_crit_indices != best_idx]
post_crit_indices = evenly_spaced_indices(post_crit_indices, N_POST_CRIT_MATRICES)


if len(post_crit_indices) > 0:
   fig3_post, axes3_post = plt.subplots(
       len(post_crit_indices),
       3,
       figsize=(15, 3.8 * len(post_crit_indices)),
       constrained_layout=True,
       squeeze=False,
   )


   fig3_post.suptitle(
       f"Post-critical matrix comparisons  |  Tcrit={T_crit:.2f}  |  alpha={alpha_star:.2f}",
       fontsize=13,
       fontweight="bold"
   )


   print("\nPost-critical matrix comparisons:")


   for row, idx in enumerate(post_crit_indices):
       T_here = T_global[idx]
       gd_here = sweep.ising_ar[idx]
       sim_here = gd_here.FC.copy()
       set_fc_diagonal(sim_here)


       sim_here_vec = clean_vec(fc_compare_vec(sim_here))
       r_here = safe_pearson(sim_here_vec, rho_emp_vec)
       dist_here = np.linalg.norm(sim_here_vec - rho_emp_vec)


       print(f"  T={T_here:.4f}  r={r_here:.4f}  dist={dist_here:.4f}")


       row_panels = [
           (sim_here, f"Simulated Pearson FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
           (rho_emp, "Empirical Pearson FC", fc_norm),
           (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
       ]


       for ax, (mat, title, norm_to_use) in zip(axes3_post[row], row_panels):
           im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
           ax.set_title(title, fontsize=10, pad=10)
           ax.set_xlabel("region", fontsize=8)
           ax.set_ylabel("region", fontsize=8)
           plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


   plt.savefig(RESULTS_DIR / "matrix_comparisons_post_Tcrit_3.png", dpi=150, bbox_inches="tight")
   plt.close(fig3_post)
   print("Saved: matrix_comparisons_post_Tcrit_3.png")
else:
   print("No post-critical temperatures available for extra matrix comparisons.")




# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 : NULL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL}, partial=False)")
print(f"         T_best = {T_best:.3f}  |  alpha = {alpha_star:.3f}")
print("=" * 65)




def shuffle_jij(J):
   J_null = J.copy()


   idx  = np.triu_indices(J.shape[0], k=1)
   vals = J_null[idx].copy()


   np.random.shuffle(vals)


   J_null[idx]            = vals
   J_null[idx[1], idx[0]] = vals


   return J_null




def pearson_threshold_jij(J, Rho, threshold):
   Rho_thresh = Rho.copy()
   Rho_thresh[np.abs(Rho_thresh) < threshold] = 0.0


   J_thresh = J.copy()
   offdiag = ~np.eye(J_thresh.shape[0], dtype=bool)
   keep = offdiag & (Rho_thresh != 0.0)
   J_thresh[keep] = np.sign(Rho_thresh[keep]) * np.abs(J_thresh[keep])
   np.fill_diagonal(J_thresh, 0)


   return (J_thresh + J_thresh.T) / 2.0




def constant_jij_like(J, value=CONSTANT_NULL_VALUE):
   J_constant = np.full_like(J, value, dtype=float)
   np.fill_diagonal(J_constant, 0)
   return J_constant




def run_ising_avg(J, T_global_value, alpha, n_runs=NULL_RUNS):
   """
   Run null Ising model and return a finite Pearson FC matrix.


   Important fix: the null Jij matrices can have negative row/column means.
   With non-integer alpha, negative_mu ** alpha creates NaNs. Temperatures
   must be non-negative, so we build mu from absolute mean coupling strength.
   """
   J = np.asarray(J, dtype=float)


   # Signed/shuffled Jij can have negative means. Temperature multipliers must be >= 0.
   mu_loc = np.abs(np.mean(J, axis=0))
   mu_loc = np.nan_to_num(mu_loc, nan=0.0, posinf=0.0, neginf=0.0)


   max_mu = np.max(mu_loc)
   if not np.isfinite(max_mu) or max_mu <= 0:
       print("WARNING: null Jij has zero/non-finite mean coupling; using uniform temperature multipliers.")
       mu_loc = np.ones(J.shape[0], dtype=float)
   else:
       mu_loc = mu_loc / max_mu


   mu_loc_sorted = mu_loc[utils.cross_sort(mu_loc)]


   temp_arr = T_global_value * (mu_loc_sorted ** alpha)
   temp_arr = np.nan_to_num(temp_arr, nan=T_global_value, posinf=T_global_value, neginf=T_global_value)
   temp_arr[temp_arr <= 0] = 1e-12


   fc_sum = np.zeros((N, N), dtype=float)


   for _ in range(n_runs):
       sim = ISING_CLASS(temp_arr, Jij=J)
       sim.simulate(NULL_STEPS, NULL_THERM)
       sim.generate_FC(partial=False)


       fc = np.nan_to_num(sim.functional_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
       fc_sum += fc


   rho = fc_sum / n_runs
   rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
   set_fc_diagonal(rho)


   return rho


null_dist = []
null_diss = []
J_null_plot = pearson_threshold_jij(
   shuffle_jij(avg_Jij),
   rho_emp,
   THRESHOLD,
)
J_ones = pearson_threshold_jij(
   constant_jij_like(J_real),
   rho_emp,
   THRESHOLD,
)


null_matrix_vals = np.concatenate([
   J_real[~np.eye(J_real.shape[0], dtype=bool)],
   J_null_plot[~np.eye(J_null_plot.shape[0], dtype=bool)],
   J_ones[~np.eye(J_ones.shape[0], dtype=bool)],
])
null_matrix_lim = np.percentile(np.abs(null_matrix_vals), 99)
if not np.isfinite(null_matrix_lim) or null_matrix_lim < 0.05:
   null_matrix_lim = 0.2
null_matrix_norm = TwoSlopeNorm(vmin=-null_matrix_lim, vcenter=0, vmax=null_matrix_lim)


fig_null_jij, axes_null_jij = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig_null_jij.suptitle(
   f"Pearson null Jij matrices  |  threshold={THRESHOLD:g}",
   fontsize=13,
   fontweight="bold"
)
null_jij_panels = [
   (J_real, "Real thresholded Jij"),
   (J_null_plot, "Example random shuffled Jij\nthen thresholded"),
   (J_ones, f"84x84 constant Jij = {CONSTANT_NULL_VALUE:g}\nthen thresholded"),
]
for ax, (mat, title) in zip(axes_null_jij, null_jij_panels):
   im = ax.matshow(mat, cmap="RdBu_r", norm=null_matrix_norm)
   ax.set_title(title, fontsize=11, pad=12)
   ax.set_xlabel("region", fontsize=9)
   ax.set_ylabel("region", fontsize=9)
   plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.savefig(RESULTS_DIR / "null_jij_matrices_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_null_jij)
print("Saved: null_jij_matrices_3.png")


for i in range(N_NULL):
   J_null = pearson_threshold_jij(
       shuffle_jij(avg_Jij),
       rho_emp,
       THRESHOLD,
   )


   # IMPORTANT:
   # Use T_best here because the real model was evaluated at T_best.
   rho_null = run_ising_avg(J_null, T_best, alpha_star)


   vec_null = clean_vec(fc_compare_vec(rho_null))


   r_null = safe_pearson(vec_null, rho_emp_vec)


   null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
   null_diss.append(1.0 - r_null)


   if (i + 1) % 10 == 0:
       print(
           f"  {i+1}/{N_NULL}  "
           f"dist={null_dist[-1]:.4f}  "
           f"diss={null_diss[-1]:.4f}  "
           f"r={r_null:.4f}"
       )


null_dist = finite_vals(null_dist, "null_dist")
null_diss = finite_vals(null_diss, "null_diss")


p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)


print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")




ones_dist = []
ones_diss = []


print(f"\nRunning thresholded constant Jij null distribution (value={CONSTANT_NULL_VALUE:g})")


for i in range(N_NULL):
   rho_ones = run_ising_avg(J_ones, T_best, alpha_star)


   vec_ones = clean_vec(fc_compare_vec(rho_ones))


   r_ones = safe_pearson(vec_ones, rho_emp_vec)


   ones_dist.append(np.linalg.norm(vec_ones - rho_emp_vec))
   ones_diss.append(1.0 - r_ones)


   if (i + 1) % 10 == 0:
       print(
           f"  ones {i+1}/{N_NULL}  "
           f"dist={ones_dist[-1]:.4f}  "
           f"diss={ones_diss[-1]:.4f}  "
           f"r={r_ones:.4f}"
       )


ones_dist = finite_vals(ones_dist, "ones_dist")
ones_diss = finite_vals(ones_diss, "ones_diss")


p_ones_dist = np.mean(ones_dist <= dist_best)
p_ones_diss = np.mean(ones_diss <= diss_best)


print(f"\nreal dist  = {dist_best:.4f} | ones null mean = {ones_dist.mean():.4f} | p = {p_ones_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | ones null mean = {ones_diss.mean():.4f} | p = {p_ones_diss:.4f}")




# ── effect sizes ──────────────────────────────────────────────────────────
def cohens_d(null_vals, real_val):
   null_vals = finite_vals(null_vals, "cohens_d input")
   sd = null_vals.std(ddof=1)
   if not np.isfinite(sd) or sd == 0:
       return 0.0
   return (real_val - null_vals.mean()) / sd




def cliffs_delta(null_vals, real_val):
   null_vals = finite_vals(null_vals, "cliffs_delta input")
   greater = np.sum(null_vals > real_val)
   less    = np.sum(null_vals < real_val)


   return (greater - less) / len(null_vals)




def cliffs_magnitude(delta):
   a = abs(delta)


   if a < 0.147:
       return "negligible"
   if a < 0.330:
       return "small"
   if a < 0.474:
       return "medium"


   return "large"




def cohens_magnitude(d):
   a = abs(d)


   if a < 0.2:
       return "negligible"
   if a < 0.5:
       return "small"
   if a < 0.8:
       return "medium"


   return "large"




cd_dist  = cohens_d(null_dist, dist_best)
cd_diss  = cohens_d(null_diss, diss_best)
cd_ones_dist = cohens_d(ones_dist, dist_best)
cd_ones_diss = cohens_d(ones_diss, diss_best)


cld_dist = cliffs_delta(null_dist, dist_best)
cld_diss = cliffs_delta(null_diss, diss_best)
cld_ones_dist = cliffs_delta(ones_dist, dist_best)
cld_ones_diss = cliffs_delta(ones_diss, diss_best)


print(f"\nCohen's d  (dist) = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d  (diss) = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ  (dist) = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ  (diss) = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d  (ones dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d  (ones diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ  (ones dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ  (ones diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")




# ── Figure 4 ──────────────────────────────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"




def plot_null(ax, null_vals, real_val, p_val, cd, cld, xlabel, title, xlim=None):
   null_vals = finite_vals(null_vals, title)
   real_val = float(np.nan_to_num(real_val, nan=0.0, posinf=0.0, neginf=0.0))


   counts, edges = np.histogram(null_vals, bins=BINS)
   widths = np.diff(edges)


   for c, left, w in zip(counts, edges[:-1], widths):
       ax.bar(
           left,
           c,
           width=w,
           align="edge",
           color=REAL_COLOR if (left + w) <= real_val else NULL_COLOR,
           alpha=0.40 if (left + w) <= real_val else 0.80,
           edgecolor="white",
           linewidth=0.5
       )


   ax.axvline(
       real_val,
       color=REAL_COLOR,
       linestyle="--",
       lw=2.2,
       label=f"real $J_{{ij}}$  ({real_val:.4f})"
   )


   ax.text(
       0.97,
       0.95,
       f"p = {p_val:.4f}\n"
       f"Cohen's d = {cd:.3f}  [{cohens_magnitude(cd)}]\n"
       f"Cliff's δ = {cld:.3f}  [{cliffs_magnitude(cld)}]",
       transform=ax.transAxes,
       ha="right",
       va="top",
       fontsize=10,
       color=REAL_COLOR,
       fontweight="medium",
       linespacing=1.6,
       bbox=dict(
           boxstyle="round,pad=0.3",
           fc="white",
           ec=REAL_COLOR,
           alpha=0.6
       )
   )


   ax.set_xlabel(xlabel, fontsize=11)
   ax.set_ylabel("count", fontsize=11)
   ax.set_title(title, fontsize=12)
   if xlim is not None:
       ax.set_xlim(xlim)
   ax.legend(fontsize=9, framealpha=0.3)
   ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
   ax.spines[["top", "right"]].set_visible(False)


def overlay_reference_null(ax, ref_vals, label="random Jij null"):
   ref_vals = finite_vals(ref_vals, label)
   ax.hist(
       ref_vals,
       bins=BINS,
       histtype="step",
       color="black",
       linewidth=1.8,
       label=label
   )
   ax.legend(fontsize=9, framealpha=0.3)


def combined_xlim(*arrays):
   vals = np.concatenate([
       np.ravel(np.asarray(array, dtype=float))
       for array in arrays
   ])
   vals = vals[np.isfinite(vals)]
   if vals.size == 0:
       return None
   pad = 0.03 * max(np.ptp(vals), 1e-12)
   return float(vals.min() - pad), float(vals.max() + pad)




fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)


fig4.suptitle(
   f"Ising null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
   fontsize=13,
   fontweight="bold"
)


plot_null(
   axes4[0],
   null_dist,
   dist_best,
   p_dist,
   cd_dist,
   cld_dist,
   xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
   title="null distribution — euclidean distance"
)


plot_null(
   axes4[1],
   null_diss,
   diss_best,
   p_diss,
   cd_diss,
   cld_diss,
   xlabel="dissimilarity  (1 − r)",
   title="null distribution — dissimilarity"
)


plt.savefig(RESULTS_DIR / "ising_null_distributions_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4)


print("Saved: ising_null_distributions_3.png")


dist_xlim = combined_xlim(null_dist, ones_dist, dist_best)
diss_xlim = combined_xlim(null_diss, ones_diss, diss_best)



fig4_ones, axes4_ones = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)


fig4_ones.suptitle(
   f"Constant-ones Jij null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
   fontsize=13,
   fontweight="bold"
)


plot_null(
   axes4_ones[0],
   ones_dist,
   dist_best,
   p_ones_dist,
   cd_ones_dist,
   cld_ones_dist,
   xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
   title="ones Jij null — euclidean distance",
   xlim=dist_xlim
)
overlay_reference_null(axes4_ones[0], null_dist)


plot_null(
   axes4_ones[1],
   ones_diss,
   diss_best,
   p_ones_diss,
   cd_ones_diss,
   cld_ones_diss,
   xlabel="dissimilarity  (1 − r)",
   title="ones Jij null — dissimilarity",
   xlim=diss_xlim
)
overlay_reference_null(axes4_ones[1], null_diss)


plt.savefig(RESULTS_DIR / "ising_null_distributions_ones_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4_ones)


print("Saved: ising_null_distributions_ones_3.png")




# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)


print(f"Annealed T*       = {T_star_annealed:.4f}")
print(f"Annealed alpha*   = {alpha_star_annealed:.4f}")


if USE_FIXED_ALPHA:
   print(f"Used alpha         = {alpha_star:.4f}  [fixed previous value]")
else:
   print(f"Used alpha         = {alpha_star:.4f}  [annealed value]")


print(f"T_crit            = {T_crit:.4f}  (specific heat peak)")
print(f"T_best            = {T_best:.4f}  (peak Pearson r, Pearson FC)")
print(f"best r            = {r_best:.4f}  (Pearson FC)")
print(f"eucl. distance    = {dist_best:.4f}")
print(f"dissimilarity     = {diss_best:.4f}")
print(f"p (dist)          = {p_dist:.4f}")
print(f"p (diss)          = {p_diss:.4f}")
print(f"p ones (dist)     = {p_ones_dist:.4f}")
print(f"p ones (diss)     = {p_ones_diss:.4f}")
print(f"Cohen's d (dist)  = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d (diss)  = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ (dist)  = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ (diss)  = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d ones (dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d ones (diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ ones (dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ ones (diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")


summary_path = RESULTS_DIR / "pearson_2_summary.csv"
summary = {
   "seed": SEED,
   "spin_flip_method": SPIN_FLIP_METHOD,
   "use_random_spin_flip": USE_RANDOM_SPIN_FLIP,
   "n_regions": N,
   "jij_threshold": THRESHOLD,
   "zero_fc_diagonal": ZERO_FC_DIAGONAL,
   "used_fixed_alpha": USE_FIXED_ALPHA,
   "anneal_steps": ANNEAL_STEPS,
   "anneal_thermalization": ANNEAL_THERM,
   "anneal_maxfun": ANNEAL_MAXFUN,
   "anneal_restarts": N_RESTARTS,
   "anneal_T_min": ANNEAL_BOUNDS[0][0],
   "anneal_T_max": ANNEAL_BOUNDS[0][1],
   "anneal_alpha_min": ANNEAL_BOUNDS[1][0],
   "anneal_alpha_max": ANNEAL_BOUNDS[1][1],
   "refine_max_rounds": REFINE_MAX_ROUNDS,
   "refine_shrink": REFINE_SHRINK,
   "annealed_T": T_star_annealed,
   "annealed_alpha": alpha_star_annealed,
   "used_alpha": alpha_star,
   "manual_T_min": T_MIN,
   "manual_T_max": T_MAX,
   "manual_T_steps": T_STEPS,
   "final_sweep_T_min": float(np.nanmin(T_global)),
   "final_sweep_T_max": float(np.nanmax(T_global)),
   "final_sweep_T_steps": len(T_global),
   "sweep_steps": SWEEP_STEPS,
   "sweep_thermalization": SWEEP_THERM,
   "temperature_repeats": TEMP_REPEATS,
   "peak_prominence_fraction": PEAK_PROMINENCE_FRACTION,
   "peak_ignore_edge_points": PEAK_IGNORE_EDGE_POINTS,
   "smooth_temperature_plots": SMOOTH_TEMPERATURE_PLOTS,
   "smooth_window": SMOOTH_WINDOW,
   "smooth_polyorder": SMOOTH_POLYORDER,
   "T_susceptibility_peak": T_suscept_peak,
   "T_specific_heat_peak": T_spec_heat_peak,
   "T_crit": T_crit,
   "T_best": T_best,
   "best_r": r_best,
   "best_distance": dist_best,
   "best_dissimilarity": diss_best,
   "sim_fc_negative_fraction": float(np.mean(sim_FC_vec < 0)),
   "emp_fc_negative_fraction": float(np.mean(rho_emp_vec < 0)),
   "sim_fc_min": float(np.min(sim_FC_vec)),
   "sim_fc_max": float(np.max(sim_FC_vec)),
   "emp_fc_min": float(np.min(rho_emp_vec)),
   "emp_fc_max": float(np.max(rho_emp_vec)),
   "n_null": N_NULL,
   "null_runs_per_matrix": NULL_RUNS,
   "null_steps": NULL_STEPS,
   "null_thermalization": NULL_THERM,
   "random_null_distance_mean": float(np.mean(null_dist)),
   "random_null_distance_sd": float(np.std(null_dist, ddof=1)) if len(null_dist) > 1 else 0.0,
   "random_null_dissimilarity_mean": float(np.mean(null_diss)),
   "random_null_dissimilarity_sd": float(np.std(null_diss, ddof=1)) if len(null_diss) > 1 else 0.0,
   "p_distance": p_dist,
   "p_dissimilarity": p_diss,
   "cohens_d_distance": cd_dist,
   "cohens_d_dissimilarity": cd_diss,
   "cliffs_delta_distance": cld_dist,
   "cliffs_delta_dissimilarity": cld_diss,
   "ones_null_distance_mean": float(np.mean(ones_dist)),
   "ones_null_distance_sd": float(np.std(ones_dist, ddof=1)) if len(ones_dist) > 1 else 0.0,
   "ones_null_dissimilarity_mean": float(np.mean(ones_diss)),
   "ones_null_dissimilarity_sd": float(np.std(ones_diss, ddof=1)) if len(ones_diss) > 1 else 0.0,
   "p_ones_distance": p_ones_dist,
   "p_ones_dissimilarity": p_ones_diss,
   "cohens_d_ones_distance": cd_ones_dist,
   "cohens_d_ones_dissimilarity": cd_ones_diss,
   "cliffs_delta_ones_distance": cld_ones_dist,
   "cliffs_delta_ones_dissimilarity": cld_ones_diss,
}


summary_header = ",".join(summary.keys()) + "\n"
summary_values = ",".join(str(value) for value in summary.values()) + "\n"
summary_path.write_text(summary_header + summary_values)
print(f"Saved: {summary_path}")


print("\nOutput files:")
for f in [
   "param_anneal_error_3.png",
   "temperature_sweep_3.png",
   "correlation_vs_T_3.png",
   "matrix_comparison_3.png",
   "scatter_sim_vs_emp_3.png",
   "matrix_comparisons_post_Tcrit_3.png",
   "null_jij_matrices_3.png",
   "ising_null_distributions_3.png",
   "ising_null_distributions_ones_3.png"
]:
   print(f"  {f}")

# DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# Cleaned version
# ═══════════════════════════════════════════════════════════════════════════

#Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import sys
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr

# Make the workspace package importable when this file is run by absolute path.
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
   sys.path.insert(0, str(WORKSPACE_ROOT))

import steven.Scripts.ising3 as I
import steven.Scripts.param_anneal as pa
import steven.Scripts.temp_sweep as ts

#Directory setup

DATA_ROOT = DATA_DIR
PROJECT_DATA_DIR = DATA_DIR


# ── config ────────────────────────────────────────────────────────────────
# RESULTS_DIR is defined once at the top of this file from RUN_FOLDER_NAME.
SIMULATION_DIR = PROJECT_DATA_DIR / "simulation data"
OPTIMIZATION_DIR = PROJECT_DATA_DIR / "optimization data"
TEMP_SWEEP_DIR = PROJECT_DATA_DIR / "temp sweep data"

JIJ_NEW_DIR = PROJECT_DATA_DIR / "Jij_new_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_Jij_new_pearson.csv"
PET_NO_OUTLIERS_PATH = PROJECT_DATA_DIR / "PET_data" / "PET_temp_no_outliers"

FC1_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_3"

THRESHOLD = 0.0 # 0, 0,02,0.03
ZERO_FC_DIAGONAL = True


# True  = set empirical/simulated FC diagonals to 0 and compare off-diagonal FC only.
# False = set empirical/simulated FC diagonals to 1 and include diagonals in FC correlations.

SEED          = 1
avg_Jij       = np.genfromtxt(DATA_ROOT / "Jij data_processed" / "avg_Jij_no_outliers_norm", delimiter=",").astype(float)
N             = avg_Jij.shape[0]


ANNEAL_STEPS  = 1000 #
ANNEAL_MAXFUN = 500 #was 500
ANNEAL_THERM  = 2000 #
SWEEP_STEPS   = 100#was 1000
SWEEP_THERM   = 2000 # was 3000


N_RESTARTS    = 1 #
# False: visit every spin in fixed mean-Jij order.
# True:  randomly sample spin-flip proposals each simulation step.
USE_RANDOM_SPIN_FLIP = False
ANNEAL_BOUNDS = ((0.1, 10), (-6, 3))

# False: use PET_i.
# True:  replace PET_i with 1 / PET_i as an inverse-PET consistency check.
USE_INVERSE_PET = True

REFINE_T_WINDOW = 1.0
REFINE_ALPHA_WINDOW = 0.5
REFINE_MAXFUN = 100
REFINE_MAX_ROUNDS = 4# was 2
REFINE_SHRINK = 0.5
REFINE_MIN_T_WINDOW = 0.001
REFINE_MIN_ALPHA_WINDOW = 0.0005


T_MIN         = 0.5 #was 0.5
T_MAX         = 12# WAS 50,8
T_STEPS       =100# WAS100,300
TEMP_REPEATS  = 100# increase for a stable susceptibility peak
PEAK_IGNORE_EDGE_POINTS = 3
PEAK_PROMINENCE_FRACTION = 0.15
SMOOTH_TEMPERATURE_PLOTS = True
SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 3


N_NULL        = 100
NULL_RUNS     = 2
NULL_STEPS    = 1000
NULL_THERM    = 1000
CONSTANT_NULL_VALUE = 1.0


BINS          = 50
N_POST_CRIT_MATRICES = 5


# Optional:
# If you want to just use Global Temperature,set this to True.
# For your own optimized result, keep it False.
USE_FIXED_ALPHA = False
FIXED_ALPHA     = 0


BLUE   = "#2E86AB"
SD_BAND = "#2CA25F"
RED    = "#E84855"
AMBER  = "#F4A261"


np.random.seed(SEED)
ISING_CLASS = I.random_ising if USE_RANDOM_SPIN_FLIP else I.Jij_sorted_ising
SPIN_FLIP_METHOD = "random" if USE_RANDOM_SPIN_FLIP else "Jij_sorted"
print(f"Spin-flip proposal method: {SPIN_FLIP_METHOD}")




# ── helper functions ──────────────────────────────────────────────────────
def upper_tri_vec(mat):
   idx = np.triu_indices(mat.shape[0], k=1)
   return mat[idx]




def set_fc_diagonal(mat):
   np.fill_diagonal(mat, 0 if ZERO_FC_DIAGONAL else 1)
   return mat




def fc_compare_vec(mat):
   if ZERO_FC_DIAGONAL:
       return upper_tri_vec(mat)
   return mat.ravel()




def clean_vec(vec):
   """Return a finite vector so null distributions/plots cannot become all-NaN."""
   return np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)




def safe_pearson(x, y):
   """Pearson r that returns 0 instead of NaN for constant or non-finite vectors."""
   x = clean_vec(x)
   y = clean_vec(y)


   mask = np.isfinite(x) & np.isfinite(y)
   x = x[mask]
   y = y[mask]


   if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
       return 0.0


   r = pearsonr(x, y)[0]
   return 0.0 if not np.isfinite(r) else float(r)




def finite_vals(vals, name):
   """Drop non-finite null values and stop with a clear error if none are usable."""
   vals = np.asarray(vals, dtype=float)
   good = vals[np.isfinite(vals)]


   dropped = len(vals) - len(good)
   if dropped > 0:
       print(f"WARNING: dropped {dropped}/{len(vals)} non-finite values from {name}")


   if len(good) == 0:
       raise ValueError(
           f"All values in {name} are NaN/inf. Check Ising simulation output and temperature array."
       )


   return good




def evenly_spaced_indices(indices, n_select):
   if len(indices) <= n_select:
       return indices


   positions = np.linspace(0, len(indices) - 1, n_select, dtype=int)
   return indices[positions]




def refine_bounds(center, T_window, alpha_window, base_bounds=ANNEAL_BOUNDS):
   T_center, alpha_center = center
   (T_low, T_high), (alpha_low, alpha_high) = base_bounds


   refined_T = (
       max(T_low, T_center - T_window),
       min(T_high, T_center + T_window),
   )
   refined_alpha = (
       max(alpha_low, alpha_center - alpha_window),
       min(alpha_high, alpha_center + alpha_window),
   )


   return refined_T, refined_alpha




def smooth_temperature_curve(values, clip_min=None):
   """Smooth plotted temperature-sweep curves without changing the statistics."""
   values = np.asarray(values, dtype=float)
   if not SMOOTH_TEMPERATURE_PLOTS or values.size < 5:
       return values


   x = np.arange(values.size)
   finite = np.isfinite(values)
   if np.sum(finite) < 5:
       return values


   filled = values.copy()
   filled[~finite] = np.interp(x[~finite], x[finite], values[finite])


   window = min(SMOOTH_WINDOW, values.size if values.size % 2 == 1 else values.size - 1)
   if window <= SMOOTH_POLYORDER:
       window = SMOOTH_POLYORDER + 2
       if window % 2 == 0:
           window += 1


   if window > values.size:
       return values


   smoothed = savgol_filter(filled, window_length=window, polyorder=SMOOTH_POLYORDER)
   if clip_min is not None:
       smoothed = np.maximum(smoothed, clip_min)


   return smoothed




def temperature_mean_and_sd_band(values, sd):
   """
   Plot the raw temperature-sweep mean with a repeat-to-repeat SD band.
   """
   values = np.asarray(values, dtype=float)
   sd = np.asarray(sd, dtype=float)


   mean_plot = values
   sd_plot = smooth_temperature_curve(sd, clip_min=0.0)


   return mean_plot, sd_plot




def run_temperature_sweep(label, t_min, t_max, t_steps, alpha, save_dir):
   print("\n" + "=" * 65)
   print(f"STEP 2 : TEMPERATURE SWEEP - {label}  (alpha = {alpha:.4f})")
   print(f"         T range = {t_min:.3f} to {t_max:.3f}  |  steps = {t_steps}")
   print("=" * 65)


   sweep_obj = ts.simulated_FC_vs_T_global(
      min_temp   = t_min,
      max_temp   = t_max,
      temp_step  = t_steps,
      alpha      = alpha,
      Jij        = J_real,
      ising      = ISING_CLASS,
      multiplier = multiplier,
      save       = True
   )


   sweep_obj.simulate(
      steps          = SWEEP_STEPS,
      thermalization = SWEEP_THERM,
      partial        = False,
      diag           = not ZERO_FC_DIAGONAL,
      text           = True,
      n_repeats      = TEMP_REPEATS,
      emp_FC1        = emp_FC1,
      emp_FC2        = emp_FC2,
      emp_FC3        = emp_FC3,
      avg_FC         = rho_emp,
      path           = save_dir
   )


   return sweep_obj



def peak_candidate_indices(values):
   values = np.asarray(values, dtype=float)


   work = values.copy()
   finite = np.isfinite(work)
   if np.sum(finite) < 3:
      return np.array([], dtype=int), work


   edge = max(0, int(PEAK_IGNORE_EDGE_POINTS))
   if work.size > 2 * edge:
      work[:edge] = np.nan
      work[-edge:] = np.nan


   finite = np.isfinite(work)
   finite_vals_local = work[finite]
   spread = np.nanmax(finite_vals_local) - np.nanmin(finite_vals_local)
   prominence = PEAK_PROMINENCE_FRACTION * spread if spread > 0 else 0.0


   filled = work.copy()
   filled[~np.isfinite(filled)] = -np.inf
   peaks, _ = find_peaks(filled, prominence=prominence)
   return peaks.astype(int), work



def stable_peak_index(values):
   values = np.asarray(values, dtype=float)
   peaks, work = peak_candidate_indices(values)
   if peaks.size:
      return int(peaks[np.nanargmax(work[peaks])])
   return int(np.nanargmax(work))



# ── data ──────────────────────────────────────────────────────────────────
J_real  = np.genfromtxt(AVG_JIJ_NEW_PATH, delimiter=",").astype(float)
np.fill_diagonal(J_real, 0)
emp_FC1 = np.genfromtxt(FC1_PATH, delimiter=",").astype(float)
emp_FC2 = np.genfromtxt(FC2_PATH, delimiter=",").astype(float)
emp_FC3 = np.genfromtxt(FC3_PATH, delimiter=",").astype(float)
rho_emp = (emp_FC1 + emp_FC2 + emp_FC3) / 3.0          # Pearson empirical FC throughout

# Load PET values for the PET or inverse-PET local-temperature model.
pet_values = np.genfromtxt(PET_NO_OUTLIERS_PATH, delimiter=",").astype(float).reshape(-1)
if pet_values.size != N:
   raise ValueError(
      f"Expected {N} PET values in {PET_NO_OUTLIERS_PATH}, found {pet_values.size}."
   )
if not np.all(np.isfinite(pet_values)):
   raise ValueError(f"PET values contain NaN/inf: {PET_NO_OUTLIERS_PATH}")
if np.any(pet_values <= 0):
   raise ValueError(
      "PET values must all be greater than zero because negative alpha values "
      "are included in the optimization."
   )

mu = np.mean(np.abs(J_real), axis=0)
mu_max = np.max(mu)
if not np.isfinite(mu_max) or mu_max <= 0:
   raise ValueError("Cannot normalize mu: mean absolute coupling has no positive finite maximum.")
mu = mu / mu_max

# Always save the original mu-versus-PET diagnostic, independent of which
# local-temperature model is selected below.
mu_pet_r, mu_pet_p = pearsonr(mu, pet_values)
fit_slope, fit_intercept = np.polyfit(mu, pet_values, 1)
fit_order = np.argsort(mu)

fig_mu_pet, ax_mu_pet = plt.subplots(figsize=(7, 6), constrained_layout=True)
ax_mu_pet.scatter(mu, pet_values, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)
ax_mu_pet.plot(
   mu[fit_order],
   fit_slope * mu[fit_order] + fit_intercept,
   color=RED,
   linewidth=2,
   label=f"Pearson r = {mu_pet_r:.3f}, p = {mu_pet_p:.3g}",
)
ax_mu_pet.set_xlabel(r"normalized structural hub strength $\mu_i$")
ax_mu_pet.set_ylabel(r"PET value $PET_i$")
ax_mu_pet.set_title(r"Structural hub strength $\mu$ vs PET")
ax_mu_pet.legend(framealpha=0.3)
ax_mu_pet.spines[["top", "right"]].set_visible(False)
mu_pet_plot_path = RESULTS_DIR / "mu_vs_pet.png"
fig_mu_pet.savefig(mu_pet_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig_mu_pet)
print(f"Saved: {mu_pet_plot_path}")
print(f"mu vs PET: Pearson r={mu_pet_r:.6f}, p={mu_pet_p:.6g}")

selected_pet_values = 1.0 / pet_values if USE_INVERSE_PET else pet_values
multiplier = selected_pet_values
if USE_INVERSE_PET:
   local_temperature_model = "inverse_pet"
   local_temperature_formula = "T_i = T_global * (1 / PET_i) ** alpha"
else:
   local_temperature_model = "pet"
   local_temperature_formula = "T_i = T_global * PET_i ** alpha"

if not np.all(np.isfinite(multiplier)) or np.any(multiplier <= 0):
   raise ValueError(
      "Every local-temperature multiplier must be positive and finite because negative alpha "
      "values are included in the optimization."
   )

print(f"Normalized mu values: min={mu.min():.6f}, max={mu.max():.6f}")
print(f"PET values: min={pet_values.min():.6f}, max={pet_values.max():.6f}")
print(f"Use inverse PET: {USE_INVERSE_PET}")
print(f"Selected multiplier values: min={multiplier.min():.6f}, max={multiplier.max():.6f}")
print(f"Local temperature model: {local_temperature_model}")
print(f"Local temperature formula: {local_temperature_formula}")
print(f"Alpha search bounds: {ANNEAL_BOUNDS[1]}")


# Set FC diagonal for plotting/saving and choose whether comparisons include it.
set_fc_diagonal(rho_emp)
set_fc_diagonal(emp_FC1)
set_fc_diagonal(emp_FC2)
set_fc_diagonal(emp_FC3)


rho_emp_vec = clean_vec(fc_compare_vec(rho_emp))


print("J_real min:          ", J_real.min())
print("J_real max:          ", J_real.max())
print("J_real has negatives:", np.any(J_real < 0))
print("emp FC neg fraction: ", np.mean(rho_emp_vec < 0))




# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 : PARAMETER ANNEALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 : PARAMETER ANNEALING — broad search (T*, alpha*)")
print("=" * 65)


best_result = None
best_optim = None
best_fun = np.inf


for restart_idx in range(N_RESTARTS):
   np.random.seed(SEED + restart_idx)


   print(f"\nRestart {restart_idx + 1}/{N_RESTARTS}")


   optim = pa.optimize(
       ising      = ISING_CLASS,
       Jij        = J_real,
       partial    = False,
       multiplier = multiplier,
       save       = (restart_idx == 0),
       save_dir   = OPTIMIZATION_DIR
   )
  
   result = optim.anneal(
       steps           = ANNEAL_STEPS,
       maxfun          = ANNEAL_MAXFUN,
       emp_FC          = rho_emp,
       therm           = ANNEAL_THERM,
       no_local_search = False,
       show            = False,
       bounds          = ANNEAL_BOUNDS
   )


   print(f"broad restart best r = {max(optim.correlate):.4f}")
   print(f"restart fun    = {result.fun:.6f}")


   if result.fun < best_fun:
       best_fun = result.fun
       best_result = result
       best_optim = optim


result = best_result
optim = best_optim
T_window = REFINE_T_WINDOW
alpha_window = REFINE_ALPHA_WINDOW


for refine_round in range(REFINE_MAX_ROUNDS):
   refined_bounds = refine_bounds(result.x, T_window, alpha_window)
   print("\n" + "=" * 65)
   print(
       f"STEP 1B.{refine_round + 1} : PARAMETER ANNEALING — refined search "
       f"T={refined_bounds[0]}, alpha={refined_bounds[1]}"
   )
   print("=" * 65)


   np.random.seed(SEED + N_RESTARTS + refine_round)
   refine_optim = pa.optimize(
       ising      = ISING_CLASS,
       Jij        = J_real,
       partial    = False,
       multiplier = multiplier,
       save       = False,
       save_dir   = OPTIMIZATION_DIR
   )


   refine_result = refine_optim.anneal(
       steps           = ANNEAL_STEPS,
       maxfun          = REFINE_MAXFUN,
       emp_FC          = rho_emp,
       therm           = ANNEAL_THERM,
       no_local_search = False,
       show            = False,
       bounds          = refined_bounds
   )


   print(f"refined round best r = {max(refine_optim.correlate):.4f}")
   print(f"refined round fun    = {refine_result.fun:.6f}")
   print(f"candidate T, alpha   = ({refine_result.x[0]:.6f}, {refine_result.x[1]:.6f})")


   if refine_result.fun < best_fun:
       best_fun = refine_result.fun
       result = refine_result
       optim = refine_optim
       print("accepted refined pair")
   else:
       print("kept previous best pair")


   T_window *= REFINE_SHRINK
   alpha_window *= REFINE_SHRINK
   if T_window <= REFINE_MIN_T_WINDOW and alpha_window <= REFINE_MIN_ALPHA_WINDOW:
       print(
           "Stopping refinement: "
           f"T window={T_window:.6f}, alpha window={alpha_window:.6f}"
       )
       break


T_star_annealed     = result.x[0]
alpha_star_annealed = result.x[1]


print(f"\nAnnealed T*     = {T_star_annealed:.4f}")
print(f"Annealed alpha* = {alpha_star_annealed:.4f}")
print(f"Annealing best r = {max(optim.correlate):.4f}")


optim.plot_error(show=False)
plt.savefig(RESULTS_DIR / "param_anneal_error_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error_3.png")




# ── choose alpha ──────────────────────────────────────────────────────────
if USE_FIXED_ALPHA:
   alpha_star = FIXED_ALPHA
   print(f"\nUsing fixed alpha = {alpha_star:.4f}")
else:
   alpha_star = alpha_star_annealed
   print(f"\nUsing annealed alpha* = {alpha_star:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 : TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
sweep = run_temperature_sweep(
   label    = "single sweep",
   t_min    = T_MIN,
   t_max    = T_MAX,
   t_steps  = T_STEPS,
   alpha    = alpha_star,
   save_dir = TEMP_SWEEP_DIR
)


# ── NaN guard ─────────────────────────────────────────────────────────────
corr_arr      = np.array(sweep.corr_ar_total)
spec_heat_arr = np.array(sweep.spec_heat_ar)
suscept_arr   = np.array(sweep.suscept_ar)
T_global = sweep.T_global


n_nan = np.sum(np.isnan(corr_arr))
print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")


# override T_crit and T_best using cleaned arrays
T_suscept_peak = sweep.T_global[stable_peak_index(suscept_arr)]
T_spec_heat_peak = sweep.T_global[stable_peak_index(spec_heat_arr)]
crit_idx = stable_peak_index(spec_heat_arr)
best_idx = np.nanargmax(corr_arr)
T_crit    = sweep.T_global[crit_idx]
T_best    = sweep.T_global[best_idx]
best_corr = np.nanmax(corr_arr)


# also patch the sweep object so downstream code is consistent
sweep.crit_temp  = T_crit
sweep.best_temp  = T_best
sweep.best_corr  = best_corr
sweep.best_ising = sweep.ising_ar[best_idx]
sweep.crit_ising = sweep.ising_ar[crit_idx]


print(f"\nSusceptibility peak temperature        : {T_suscept_peak:.4f}")
print(f"Specific heat peak temperature         : {T_spec_heat_peak:.4f}")
print(f"Critical temperature (specific heat)   : {T_crit:.4f}")
print(
   "Peak detection                         : "
   f"prominence>={PEAK_PROMINENCE_FRACTION:.2f} of curve range, "
   f"edge points ignored={PEAK_IGNORE_EDGE_POINTS}"
)
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")




# ── observables ───────────────────────────────────────────────────────────
avg_energy = np.array(sweep.avg_energy_ar)
avg_energy_sd = np.array(sweep.avg_energy_sd_ar)
avg_mag = np.array(sweep.avg_mag_ar)
avg_mag_sd = np.array(sweep.avg_mag_sd_ar)
suscept = np.array(sweep.suscept_ar)
suscept_sd = np.array(sweep.suscept_sd_ar)
spec_heat = np.array(sweep.spec_heat_ar)
spec_heat_sd = np.array(sweep.spec_heat_sd_ar)




# ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
   f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
   fontsize=14,
   fontweight="bold"
)

# reference lines shown on every panel
ref_lines = [
   (T_suscept_peak,   RED,    "--", rf"$T_{{\chi\,peak}}$ = {T_suscept_peak:.2f}"),
   (T_spec_heat_peak, SD_BAND, "--", rf"$T_{{C\,peak}}$ = {T_spec_heat_peak:.2f}"),
   (T_best,           AMBER,   ":",  rf"$T_{{best}}$ = {T_best:.2f}"),
]

panels = [
   (axes1[0, 0], avg_energy, avg_energy_sd, r"average energy $\langle E \rangle$", "Energy vs T"),
   (axes1[0, 1], avg_mag, avg_mag_sd, r"average $|M|$", "|Magnetization| vs T"),
   (axes1[1, 1], suscept, suscept_sd, r"susceptibility $\chi$", "Susceptibility vs T"),
   (axes1[1, 0], spec_heat, spec_heat_sd, r"specific heat $C$", "Specific Heat vs T"),
]

for ax, data, sd, ylabel, title in panels:
   data_plot, sd_plot = temperature_mean_and_sd_band(data, sd)

   ax.plot(T_global, data_plot, color=BLUE, lw=2.0)
   ax.fill_between(T_global, data_plot - sd_plot, data_plot + sd_plot, color=SD_BAND, alpha=0.28, linewidth=0)

   for temp, color, ls, label in ref_lines:
      ax.axvline(temp, color=color, linestyle=ls, lw=1.6, label=label)

   ax.set_xlabel("global temperature  T", fontsize=11)
   ax.set_ylabel(ylabel, fontsize=11)
   ax.set_title(title, fontsize=12)
   ax.legend(fontsize=8, framealpha=0.3)
   ax.spines[["top", "right"]].set_visible(False)

plt.savefig(RESULTS_DIR / "temperature_sweep_3.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: temperature_sweep_3.png")


# ── Figure 2: correlation vs T ────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

corr_total = np.array(sweep.corr_ar_total)
corr_total_sd = np.array(sweep.corr_sd_ar_total)
corr_total_plot, corr_total_sd_plot = temperature_mean_and_sd_band(corr_total, corr_total_sd)

ax_corr.plot(
   T_global,
   corr_total_plot,
   color=BLUE,
   lw=2.0,
   label="avg FC"
)
ax_corr.fill_between(
   T_global,
   corr_total_plot - corr_total_sd_plot,
   corr_total_plot + corr_total_sd_plot,
   color=SD_BAND,
   alpha=0.28,
   linewidth=0,
   label="standard deviation"
)


ax_corr.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")


ax_corr.set_xlabel("Global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim Pearson FC vs emp Pearson FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "correlation_vs_T_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_corr)
print("Saved: correlation_vs_T_3.png")




# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : MATRIX COMPARISON  (T_best) — Pearson FC only
# ════
print("STEP 3 : MATRIX COMPARISON  (T_best, Pearson FC)")
print("=" * 65)


best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()


set_fc_diagonal(sim_FC)


sim_FC_vec = clean_vec(fc_compare_vec(sim_FC))


r_best    = safe_pearson(sim_FC_vec, rho_emp_vec)
dist_best = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best = 1.0 - r_best


print(f"sim FC neg fraction : {np.mean(sim_FC_vec  < 0):.4f}")
print(f"emp FC neg fraction : {np.mean(rho_emp_vec < 0):.4f}")
print(f"sim FC range        : {sim_FC_vec.min():.4f} → {sim_FC_vec.max():.4f}")
print(f"emp FC range        : {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")




# ── color normalization ──────────────────────────────────────────────────
# Use one fixed shared norm for simulated and empirical FC.
fc_lim = 0.5
fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)


# Use separate norm for Jij because it may have a different scale.
j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
j_lim = np.percentile(np.abs(j_offdiag), 99)


if not np.isfinite(j_lim) or j_lim < 0.05:
   j_lim = 0.2


j_norm = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)


print(f"FC color limit  : ±{fc_lim:.4f}")
print(f"Jij color limit : ±{j_lim:.4f}")




# ── matrix figure ────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)


fig3.suptitle(
   f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}  |  r={r_best:.4f}  |  threshold={THRESHOLD:g}",
   fontsize=13,
   fontweight="bold"
)


matrix_panels = [
   (sim_FC,  f"Simulated Pearson FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
   (rho_emp, "Empirical Pearson FC", fc_norm),
   (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
]


for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
   im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
   ax.set_title(title, fontsize=11, pad=12)
   ax.set_xlabel("region", fontsize=9)
   ax.set_ylabel("region", fontsize=9)
   plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.savefig(RESULTS_DIR / "matrix_comparison_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3)




# ── scatter: sim vs emp ──────────────────────────────────────────────────
fig3s, ax3s = plt.subplots(figsize=(6, 5), constrained_layout=True)


ax3s.scatter(
   rho_emp_vec,
   sim_FC_vec,
   s=2,
   alpha=0.3,
   color=BLUE,
   rasterized=True
)


m, b = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)


ax3s.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")


ax3s.set_xlabel("empirical Pearson FC", fontsize=11)
ax3s.set_ylabel("simulated Pearson FC", fontsize=11)
ax3s.set_title(f"Sim vs Emp Pearson FC  (r = {r_best:.4f})", fontsize=12)
ax3s.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "scatter_sim_vs_emp_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3s)


print("Saved: matrix_comparison_3.png, scatter_sim_vs_emp_3.png")




# ── additional matrix comparisons after Tcrit ────────────────────────────
post_crit_indices = np.where(T_global > T_crit)[0]
best_idx = int(np.nanargmax(corr_arr))
post_crit_indices = post_crit_indices[post_crit_indices != best_idx]
post_crit_indices = evenly_spaced_indices(post_crit_indices, N_POST_CRIT_MATRICES)


if len(post_crit_indices) > 0:
   fig3_post, axes3_post = plt.subplots(
       len(post_crit_indices),
       3,
       figsize=(15, 3.8 * len(post_crit_indices)),
       constrained_layout=True,
       squeeze=False,
   )


   fig3_post.suptitle(
       f"Post-critical matrix comparisons  |  Tcrit={T_crit:.2f}  |  alpha={alpha_star:.2f}",
       fontsize=13,
       fontweight="bold"
   )


   print("\nPost-critical matrix comparisons:")


   for row, idx in enumerate(post_crit_indices):
       T_here = T_global[idx]
       gd_here = sweep.ising_ar[idx]
       sim_here = gd_here.FC.copy()
       set_fc_diagonal(sim_here)


       sim_here_vec = clean_vec(fc_compare_vec(sim_here))
       r_here = safe_pearson(sim_here_vec, rho_emp_vec)
       dist_here = np.linalg.norm(sim_here_vec - rho_emp_vec)


       print(f"  T={T_here:.4f}  r={r_here:.4f}  dist={dist_here:.4f}")


       row_panels = [
           (sim_here, f"Simulated Pearson FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
           (rho_emp, "Empirical Pearson FC", fc_norm),
           (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
       ]


       for ax, (mat, title, norm_to_use) in zip(axes3_post[row], row_panels):
           im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
           ax.set_title(title, fontsize=10, pad=10)
           ax.set_xlabel("region", fontsize=8)
           ax.set_ylabel("region", fontsize=8)
           plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


   plt.savefig(RESULTS_DIR / "matrix_comparisons_post_Tcrit_3.png", dpi=150, bbox_inches="tight")
   plt.close(fig3_post)
   print("Saved: matrix_comparisons_post_Tcrit_3.png")
else:
   print("No post-critical temperatures available for extra matrix comparisons.")




# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 : NULL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL}, partial=False)")
print(f"         T_best = {T_best:.3f}  |  alpha = {alpha_star:.3f}")
print("=" * 65)




def shuffle_jij(J):
   J_null = J.copy()


   idx  = np.triu_indices(J.shape[0], k=1)
   vals = J_null[idx].copy()


   np.random.shuffle(vals)


   J_null[idx]            = vals
   J_null[idx[1], idx[0]] = vals


   return J_null




def pearson_threshold_jij(J, Rho, threshold):
   Rho_thresh = Rho.copy()
   Rho_thresh[np.abs(Rho_thresh) < threshold] = 0.0


   J_thresh = J.copy()
   offdiag = ~np.eye(J_thresh.shape[0], dtype=bool)
   keep = offdiag & (Rho_thresh != 0.0)
   J_thresh[keep] = np.sign(Rho_thresh[keep]) * np.abs(J_thresh[keep])
   np.fill_diagonal(J_thresh, 0)


   return (J_thresh + J_thresh.T) / 2.0




def constant_jij_like(J, value=CONSTANT_NULL_VALUE):
   J_constant = np.full_like(J, value, dtype=float)
   np.fill_diagonal(J_constant, 0)
   return J_constant




def run_ising_avg(J, T_global_value, alpha, n_runs=NULL_RUNS):
   """
   Run a null Ising model using the same PET-derived local temperatures.
   """
   J = np.asarray(J, dtype=float)
   if J.shape != (N, N):
       raise ValueError(f"Expected a {(N, N)} Jij matrix, found {J.shape}.")

   temp_arr = T_global_value * (multiplier ** alpha)
   temp_arr = np.nan_to_num(temp_arr, nan=T_global_value, posinf=T_global_value, neginf=T_global_value)
   temp_arr[temp_arr <= 0] = 1e-12


   fc_sum = np.zeros((N, N), dtype=float)


   for _ in range(n_runs):
       sim = ISING_CLASS(temp_arr, Jij=J)
       sim.simulate(NULL_STEPS, NULL_THERM)
       sim.generate_FC(partial=False)


       fc = np.nan_to_num(sim.functional_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
       fc_sum += fc


   rho = fc_sum / n_runs
   rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
   set_fc_diagonal(rho)


   return rho


null_dist = []
null_diss = []
J_null_plot = pearson_threshold_jij(
   shuffle_jij(avg_Jij),
   rho_emp,
   THRESHOLD,
)
J_ones = pearson_threshold_jij(
   constant_jij_like(J_real),
   rho_emp,
   THRESHOLD,
)


null_matrix_vals = np.concatenate([
   J_real[~np.eye(J_real.shape[0], dtype=bool)],
   J_null_plot[~np.eye(J_null_plot.shape[0], dtype=bool)],
   J_ones[~np.eye(J_ones.shape[0], dtype=bool)],
])
null_matrix_lim = np.percentile(np.abs(null_matrix_vals), 99)
if not np.isfinite(null_matrix_lim) or null_matrix_lim < 0.05:
   null_matrix_lim = 0.2
null_matrix_norm = TwoSlopeNorm(vmin=-null_matrix_lim, vcenter=0, vmax=null_matrix_lim)


fig_null_jij, axes_null_jij = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig_null_jij.suptitle(
   f"Pearson null Jij matrices  |  threshold={THRESHOLD:g}",
   fontsize=13,
   fontweight="bold"
)
null_jij_panels = [
   (J_real, "Real thresholded Jij"),
   (J_null_plot, "Example random shuffled Jij\nthen thresholded"),
   (J_ones, f"84x84 constant Jij = {CONSTANT_NULL_VALUE:g}\nthen thresholded"),
]
for ax, (mat, title) in zip(axes_null_jij, null_jij_panels):
   im = ax.matshow(mat, cmap="RdBu_r", norm=null_matrix_norm)
   ax.set_title(title, fontsize=11, pad=12)
   ax.set_xlabel("region", fontsize=9)
   ax.set_ylabel("region", fontsize=9)
   plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.savefig(RESULTS_DIR / "null_jij_matrices_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_null_jij)
print("Saved: null_jij_matrices_3.png")


for i in range(N_NULL):
   J_null = pearson_threshold_jij(
       shuffle_jij(avg_Jij),
       rho_emp,
       THRESHOLD,
   )


   # IMPORTANT:
   # Use T_best here because the real model was evaluated at T_best.
   rho_null = run_ising_avg(J_null, T_best, alpha_star)


   vec_null = clean_vec(fc_compare_vec(rho_null))


   r_null = safe_pearson(vec_null, rho_emp_vec)


   null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
   null_diss.append(1.0 - r_null)


   if (i + 1) % 10 == 0:
       print(
           f"  {i+1}/{N_NULL}  "
           f"dist={null_dist[-1]:.4f}  "
           f"diss={null_diss[-1]:.4f}  "
           f"r={r_null:.4f}"
       )


null_dist = finite_vals(null_dist, "null_dist")
null_diss = finite_vals(null_diss, "null_diss")


p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)


print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")




ones_dist = []
ones_diss = []


print(f"\nRunning thresholded constant Jij null distribution (value={CONSTANT_NULL_VALUE:g})")


for i in range(N_NULL):
   rho_ones = run_ising_avg(J_ones, T_best, alpha_star)


   vec_ones = clean_vec(fc_compare_vec(rho_ones))


   r_ones = safe_pearson(vec_ones, rho_emp_vec)


   ones_dist.append(np.linalg.norm(vec_ones - rho_emp_vec))
   ones_diss.append(1.0 - r_ones)


   if (i + 1) % 10 == 0:
       print(
           f"  ones {i+1}/{N_NULL}  "
           f"dist={ones_dist[-1]:.4f}  "
           f"diss={ones_diss[-1]:.4f}  "
           f"r={r_ones:.4f}"
       )


ones_dist = finite_vals(ones_dist, "ones_dist")
ones_diss = finite_vals(ones_diss, "ones_diss")


p_ones_dist = np.mean(ones_dist <= dist_best)
p_ones_diss = np.mean(ones_diss <= diss_best)


print(f"\nreal dist  = {dist_best:.4f} | ones null mean = {ones_dist.mean():.4f} | p = {p_ones_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | ones null mean = {ones_diss.mean():.4f} | p = {p_ones_diss:.4f}")




# ── effect sizes ──────────────────────────────────────────────────────────
def cohens_d(null_vals, real_val):
   null_vals = finite_vals(null_vals, "cohens_d input")
   sd = null_vals.std(ddof=1)
   if not np.isfinite(sd) or sd == 0:
       return 0.0
   return (real_val - null_vals.mean()) / sd




def cliffs_delta(null_vals, real_val):
   null_vals = finite_vals(null_vals, "cliffs_delta input")
   greater = np.sum(null_vals > real_val)
   less    = np.sum(null_vals < real_val)


   return (greater - less) / len(null_vals)




def cliffs_magnitude(delta):
   a = abs(delta)


   if a < 0.147:
       return "negligible"
   if a < 0.330:
       return "small"
   if a < 0.474:
       return "medium"


   return "large"




def cohens_magnitude(d):
   a = abs(d)


   if a < 0.2:
       return "negligible"
   if a < 0.5:
       return "small"
   if a < 0.8:
       return "medium"


   return "large"




cd_dist  = cohens_d(null_dist, dist_best)
cd_diss  = cohens_d(null_diss, diss_best)
cd_ones_dist = cohens_d(ones_dist, dist_best)
cd_ones_diss = cohens_d(ones_diss, diss_best)


cld_dist = cliffs_delta(null_dist, dist_best)
cld_diss = cliffs_delta(null_diss, diss_best)
cld_ones_dist = cliffs_delta(ones_dist, dist_best)
cld_ones_diss = cliffs_delta(ones_diss, diss_best)


print(f"\nCohen's d  (dist) = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d  (diss) = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ  (dist) = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ  (diss) = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d  (ones dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d  (ones diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ  (ones dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ  (ones diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")




# ── Figure 4 ──────────────────────────────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"




def plot_null(ax, null_vals, real_val, p_val, cd, cld, xlabel, title, xlim=None):
   null_vals = finite_vals(null_vals, title)
   real_val = float(np.nan_to_num(real_val, nan=0.0, posinf=0.0, neginf=0.0))


   counts, edges = np.histogram(null_vals, bins=BINS)
   widths = np.diff(edges)


   for c, left, w in zip(counts, edges[:-1], widths):
       ax.bar(
           left,
           c,
           width=w,
           align="edge",
           color=REAL_COLOR if (left + w) <= real_val else NULL_COLOR,
           alpha=0.40 if (left + w) <= real_val else 0.80,
           edgecolor="white",
           linewidth=0.5
       )


   ax.axvline(
       real_val,
       color=REAL_COLOR,
       linestyle="--",
       lw=2.2,
       label=f"real $J_{{ij}}$  ({real_val:.4f})"
   )


   ax.text(
       0.97,
       0.95,
       f"p = {p_val:.4f}\n"
       f"Cohen's d = {cd:.3f}  [{cohens_magnitude(cd)}]\n"
       f"Cliff's δ = {cld:.3f}  [{cliffs_magnitude(cld)}]",
       transform=ax.transAxes,
       ha="right",
       va="top",
       fontsize=10,
       color=REAL_COLOR,
       fontweight="medium",
       linespacing=1.6,
       bbox=dict(
           boxstyle="round,pad=0.3",
           fc="white",
           ec=REAL_COLOR,
           alpha=0.6
       )
   )


   ax.set_xlabel(xlabel, fontsize=11)
   ax.set_ylabel("count", fontsize=11)
   ax.set_title(title, fontsize=12)
   if xlim is not None:
       ax.set_xlim(xlim)
   ax.legend(fontsize=9, framealpha=0.3)
   ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
   ax.spines[["top", "right"]].set_visible(False)


def overlay_reference_null(ax, ref_vals, label="random Jij null"):
   ref_vals = finite_vals(ref_vals, label)
   ax.hist(
       ref_vals,
       bins=BINS,
       histtype="step",
       color="black",
       linewidth=1.8,
       label=label
   )
   ax.legend(fontsize=9, framealpha=0.3)


def combined_xlim(*arrays):
   vals = np.concatenate([
       np.ravel(np.asarray(array, dtype=float))
       for array in arrays
   ])
   vals = vals[np.isfinite(vals)]
   if vals.size == 0:
       return None
   pad = 0.03 * max(np.ptp(vals), 1e-12)
   return float(vals.min() - pad), float(vals.max() + pad)




fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)


fig4.suptitle(
   f"Ising null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
   fontsize=13,
   fontweight="bold"
)


plot_null(
   axes4[0],
   null_dist,
   dist_best,
   p_dist,
   cd_dist,
   cld_dist,
   xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
   title="null distribution — euclidean distance"
)


plot_null(
   axes4[1],
   null_diss,
   diss_best,
   p_diss,
   cd_diss,
   cld_diss,
   xlabel="dissimilarity  (1 − r)",
   title="null distribution — dissimilarity"
)


plt.savefig(RESULTS_DIR / "ising_null_distributions_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4)


print("Saved: ising_null_distributions_3.png")


dist_xlim = combined_xlim(null_dist, ones_dist, dist_best)
diss_xlim = combined_xlim(null_diss, ones_diss, diss_best)



fig4_ones, axes4_ones = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)


fig4_ones.suptitle(
   f"Constant-ones Jij null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
   fontsize=13,
   fontweight="bold"
)


plot_null(
   axes4_ones[0],
   ones_dist,
   dist_best,
   p_ones_dist,
   cd_ones_dist,
   cld_ones_dist,
   xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
   title="ones Jij null — euclidean distance",
   xlim=dist_xlim
)
overlay_reference_null(axes4_ones[0], null_dist)


plot_null(
   axes4_ones[1],
   ones_diss,
   diss_best,
   p_ones_diss,
   cd_ones_diss,
   cld_ones_diss,
   xlabel="dissimilarity  (1 − r)",
   title="ones Jij null — dissimilarity",
   xlim=diss_xlim
)
overlay_reference_null(axes4_ones[1], null_diss)


plt.savefig(RESULTS_DIR / "ising_null_distributions_ones_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4_ones)


print("Saved: ising_null_distributions_ones_3.png")




# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)


print(f"Annealed T*       = {T_star_annealed:.4f}")
print(f"Annealed alpha*   = {alpha_star_annealed:.4f}")


if USE_FIXED_ALPHA:
   print(f"Used alpha         = {alpha_star:.4f}  [fixed previous value]")
else:
   print(f"Used alpha         = {alpha_star:.4f}  [annealed value]")


print(f"T_crit            = {T_crit:.4f}  (specific heat peak)")
print(f"T_best            = {T_best:.4f}  (peak Pearson r, Pearson FC)")
print(f"best r            = {r_best:.4f}  (Pearson FC)")
print(f"eucl. distance    = {dist_best:.4f}")
print(f"dissimilarity     = {diss_best:.4f}")
print(f"p (dist)          = {p_dist:.4f}")
print(f"p (diss)          = {p_diss:.4f}")
print(f"p ones (dist)     = {p_ones_dist:.4f}")
print(f"p ones (diss)     = {p_ones_diss:.4f}")
print(f"Cohen's d (dist)  = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d (diss)  = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ (dist)  = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ (diss)  = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d ones (dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d ones (diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ ones (dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ ones (diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")


summary_path = RESULTS_DIR / "PET_Hypothesis_summary.csv"
summary = {
   "seed": SEED,
   "spin_flip_method": SPIN_FLIP_METHOD,
   "use_random_spin_flip": USE_RANDOM_SPIN_FLIP,
   "n_regions": N,
   "local_temperature_model": local_temperature_model,
   "use_inverse_pet": USE_INVERSE_PET,
   "mu_pet_pearson_r": float(mu_pet_r),
   "mu_pet_pearson_p": float(mu_pet_p),
   "jij_threshold": THRESHOLD,
   "zero_fc_diagonal": ZERO_FC_DIAGONAL,
   "used_fixed_alpha": USE_FIXED_ALPHA,
   "anneal_steps": ANNEAL_STEPS,
   "anneal_thermalization": ANNEAL_THERM,
   "anneal_maxfun": ANNEAL_MAXFUN,
   "anneal_restarts": N_RESTARTS,
   "anneal_T_min": ANNEAL_BOUNDS[0][0],
   "anneal_T_max": ANNEAL_BOUNDS[0][1],
   "anneal_alpha_min": ANNEAL_BOUNDS[1][0],
   "anneal_alpha_max": ANNEAL_BOUNDS[1][1],
   "refine_max_rounds": REFINE_MAX_ROUNDS,
   "refine_shrink": REFINE_SHRINK,
   "annealed_T": T_star_annealed,
   "annealed_alpha": alpha_star_annealed,
   "used_alpha": alpha_star,
   "manual_T_min": T_MIN,
   "manual_T_max": T_MAX,
   "manual_T_steps": T_STEPS,
   "final_sweep_T_min": float(np.nanmin(T_global)),
   "final_sweep_T_max": float(np.nanmax(T_global)),
   "final_sweep_T_steps": len(T_global),
   "sweep_steps": SWEEP_STEPS,
   "sweep_thermalization": SWEEP_THERM,
   "temperature_repeats": TEMP_REPEATS,
   "peak_prominence_fraction": PEAK_PROMINENCE_FRACTION,
   "peak_ignore_edge_points": PEAK_IGNORE_EDGE_POINTS,
   "smooth_temperature_plots": SMOOTH_TEMPERATURE_PLOTS,
   "smooth_window": SMOOTH_WINDOW,
   "smooth_polyorder": SMOOTH_POLYORDER,
   "T_susceptibility_peak": T_suscept_peak,
   "T_specific_heat_peak": T_spec_heat_peak,
   "T_crit": T_crit,
   "T_best": T_best,
   "best_r": r_best,
   "best_distance": dist_best,
   "best_dissimilarity": diss_best,
   "sim_fc_negative_fraction": float(np.mean(sim_FC_vec < 0)),
   "emp_fc_negative_fraction": float(np.mean(rho_emp_vec < 0)),
   "sim_fc_min": float(np.min(sim_FC_vec)),
   "sim_fc_max": float(np.max(sim_FC_vec)),
   "emp_fc_min": float(np.min(rho_emp_vec)),
   "emp_fc_max": float(np.max(rho_emp_vec)),
   "n_null": N_NULL,
   "null_runs_per_matrix": NULL_RUNS,
   "null_steps": NULL_STEPS,
   "null_thermalization": NULL_THERM,
   "random_null_distance_mean": float(np.mean(null_dist)),
   "random_null_distance_sd": float(np.std(null_dist, ddof=1)) if len(null_dist) > 1 else 0.0,
   "random_null_dissimilarity_mean": float(np.mean(null_diss)),
   "random_null_dissimilarity_sd": float(np.std(null_diss, ddof=1)) if len(null_diss) > 1 else 0.0,
   "p_distance": p_dist,
   "p_dissimilarity": p_diss,
   "cohens_d_distance": cd_dist,
   "cohens_d_dissimilarity": cd_diss,
   "cliffs_delta_distance": cld_dist,
   "cliffs_delta_dissimilarity": cld_diss,
   "ones_null_distance_mean": float(np.mean(ones_dist)),
   "ones_null_distance_sd": float(np.std(ones_dist, ddof=1)) if len(ones_dist) > 1 else 0.0,
   "ones_null_dissimilarity_mean": float(np.mean(ones_diss)),
   "ones_null_dissimilarity_sd": float(np.std(ones_diss, ddof=1)) if len(ones_diss) > 1 else 0.0,
   "p_ones_distance": p_ones_dist,
   "p_ones_dissimilarity": p_ones_diss,
   "cohens_d_ones_distance": cd_ones_dist,
   "cohens_d_ones_dissimilarity": cd_ones_diss,
   "cliffs_delta_ones_distance": cld_ones_dist,
   "cliffs_delta_ones_dissimilarity": cld_ones_diss,
}


summary_header = ",".join(summary.keys()) + "\n"
summary_values = ",".join(str(value) for value in summary.values()) + "\n"
summary_path.write_text(summary_header + summary_values)
print(f"Saved: {summary_path}")


print("\nOutput files:")
for f in [
   "param_anneal_error_3.png",
   "temperature_sweep_3.png",
   "correlation_vs_T_3.png",
   "matrix_comparison_3.png",
   "scatter_sim_vs_emp_3.png",
   "matrix_comparisons_post_Tcrit_3.png",
   "null_jij_matrices_3.png",
   "ising_null_distributions_3.png",
   "ising_null_distributions_ones_3.png"
]:
   print(f"  {f}")

# DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

"""
threshold_correlation_sweep.py

Reproduces the "Mean correlation coefficients between the simulated data
(at Tc) and the empirical FC as a function of threshold" analysis from your
slides, for a single Jij / empirical FC setup (no count/MS/FA variants -
per your note, just the regular simulation).

At a fixed temperature (intended to be T_c, your critical temperature),
runs several independent Ising simulations. For each restart and each
threshold value, both the simulated and empirical FC matrices are
independently thresholded (|value| < threshold -> 0, otherwise kept as-is),
then the Pearson correlation between the two thresholded matrices is
computed. The mean and std of that correlation across restarts is plotted
against threshold, matching the shaded-std-band style in your slides - one
version excluding the diagonal (image 1), one including it (image 2).

Assumptions, since these weren't nailed down:
  - The shaded std band comes from repeating the simulation N times at the
    same T_c (matching the 5-restart convention elsewhere in your
    pipeline), not from averaging multiple subjects. Swap `n_restarts` or
    the run() loop if you actually meant subject-level variability.
  - T_c itself isn't computed here - it's the output of your existing
    simulated_FC_vs_T_global sweep (its .crit_temp attribute). Plug that
    value in below rather than re-deriving it.
  - Threshold range is 0.1 to 0.9 in steps of 0.1, matching the x-axis in
    your plots (the text mentions 0-1, so widen `thresholds` if you want
    the endpoints included too).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_DIR.parent
STEVEN_DATA_ROOT = DATA_DIR
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import steven.Scripts.ising3 as I
import steven.Scripts.utils as utils


PEARSON_SUMMARY_PATH = RESULTS_DIR / "pearson_2_summary.csv"
JIJ_NEW_DIR = DATA_DIR / "Jij_new_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_Jij_new_pearson.csv"
FC1_PATH = STEVEN_DATA_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = STEVEN_DATA_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = STEVEN_DATA_ROOT / "FC data_processed" / "avg_TS_3"


def load_pearson_alpha_tcrit(summary_path=PEARSON_SUMMARY_PATH):
    """
    Read the alpha and Tcrit saved by pearson_2.py.
    Run pearson_2.py first so this file exists and matches your latest run.
    """
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run pearson_2.py first so it saves "
            "used_alpha and T_crit."
        )

    values = np.genfromtxt(summary_path, delimiter=",", names=True)
    return float(values["used_alpha"]), float(values["T_crit"])


def threshold_matrix(matrix, threshold):
    """Zero out entries with |value| < threshold; keep entries >= threshold as-is."""
    out = matrix.copy()
    out[np.abs(out) < threshold] = 0
    return out


def thresholded_correlation(sim_FC, emp_FC, threshold, diag=False):
    """
    Pearson correlation between sim_FC and emp_FC after independently
    thresholding each. diag=False excludes the diagonal (utils.flat_remove_diag),
    diag=True flattens the full matrix including the diagonal.
    """
    sim_t = threshold_matrix(sim_FC, threshold)
    emp_t = threshold_matrix(emp_FC, threshold)
    if diag:
        sim_t = sim_t.copy()
        emp_t = emp_t.copy()
        np.fill_diagonal(sim_t, 1.0)
        np.fill_diagonal(emp_t, 1.0)
        x = sim_t.flatten()
        y = emp_t.flatten()
    else:
        x = utils.flat_remove_diag(sim_t)
        y = utils.flat_remove_diag(emp_t)

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    r = pearsonr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


class threshold_correlation_sweep:
    """
    :param ising_class: an Ising subclass (e.g. Jij_sorted_ising), same role
           as the `ising` argument in simulated_FC_vs_T_global.
    :param Jij: structural connectivity matrix (e.g. cf.avg_Jij).
    :param emp_FC: empirical FC matrix to compare against (e.g. cf.avg_FCp
           for partial correlation, cf.avg_FC for Pearson).
    :param T_c: the temperature to run at - your critical temperature.
    :param alpha: temperature fitting exponent.
    :param multiplier: per-neuron temp scaling array, e.g.
           utils.normalize_array(np.mean(Jij, 0)) - same role as
           simulated_FC_vs_T_global's `multiplier`.
    :param partial: whether generate_FC() uses partial correlation.
    """

    def __init__(self, ising_class, Jij, emp_FC, T_c, alpha, multiplier, partial=True):
        self.ising_class = ising_class
        self.Jij = Jij
        self.emp_FC = emp_FC
        self.T_c = T_c
        self.alpha = alpha
        self.multiplier = multiplier
        self.partial = partial
        self.sim_FCs = []

    def run(self, n_restarts, steps, thermalization, spin_array):
        """Run n_restarts independent simulations at T_c and store each sim FC."""
        temp_ar = self.T_c * (self.multiplier ** self.alpha)
        self.sim_FCs = []
        for _ in range(n_restarts):
            ising_obj = self.ising_class(temp_ar, Jij=self.Jij, spin_ar=spin_array)
            ising_obj.simulate(steps, thermalization)
            sim_FC = ising_obj.generate_FC(self.partial)
            sim_FC = np.nan_to_num(sim_FC, nan=0.0, posinf=0.0, neginf=0.0)
            self.sim_FCs.append(sim_FC)

    def sweep_thresholds(self, thresholds, diag=False):
        """
        Returns (mean_corr, std_corr) arrays aligned with `thresholds`,
        computed across the simulated FC matrices from run().
        """
        mean_corr, std_corr = [], []
        for t in thresholds:
            corrs = [
                thresholded_correlation(sim_FC, self.emp_FC, t, diag=diag)
                for sim_FC in self.sim_FCs
            ]
            mean_corr.append(np.mean(corrs))
            std_corr.append(np.std(corrs))
        return np.array(mean_corr), np.array(std_corr)

    def graph_threshold_sweep(self, thresholds, diag=False, show=True, save_path=None, label=None):
        mean_corr, std_corr = self.sweep_thresholds(thresholds, diag=diag)

        figure, axis = plt.subplots(1)
        axis.plot(thresholds, mean_corr, label=label or ("include diagonal" if diag else "exclude diagonal"))
        axis.fill_between(thresholds, mean_corr - std_corr, mean_corr + std_corr, alpha=0.3)
        axis.set_xlabel("Threshold")
        axis.set_ylabel("Correlation coefficient")
        axis.set_title(f"Mean correlation vs threshold (T_c) - {'incl.' if diag else 'excl.'} diagonal")
        axis.legend()

        if save_path:
            figure.savefig(save_path)
            csv_path = Path(save_path).with_suffix(".csv")
            np.savetxt(
                csv_path,
                np.column_stack([thresholds, mean_corr, std_corr]),
                delimiter=",",
                header="threshold,mean_pearson_r,std_pearson_r",
                comments="",
            )

        if show:
            plt.show()
        else:
            return figure, axis, mean_corr, std_corr

    def graph_threshold_sweep_combined(self, thresholds, show=True, save_path=None):
        excl_mean, excl_std = self.sweep_thresholds(thresholds, diag=False)
        incl_mean, incl_std = self.sweep_thresholds(thresholds, diag=True)

        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)

        axis.plot(thresholds, excl_mean, color="#2E86AB", lw=2.0, label="exclude diagonal")
        axis.fill_between(
            thresholds,
            excl_mean - excl_std,
            excl_mean + excl_std,
            color="#2E86AB",
            alpha=0.22,
            linewidth=0,
        )

        axis.plot(thresholds, incl_mean, color="#E84855", lw=2.0, label="include diagonal")
        axis.fill_between(
            thresholds,
            incl_mean - incl_std,
            incl_mean + incl_std,
            color="#E84855",
            alpha=0.18,
            linewidth=0,
        )

        axis.set_xlabel("Threshold")
        axis.set_ylabel("Pearson correlation coefficient r")
        axis.set_title("Correlation vs threshold at T_c")
        axis.legend()
        axis.spines[["top", "right"]].set_visible(False)

        if save_path:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
            csv_path = Path(save_path).with_suffix(".csv")
            np.savetxt(
                csv_path,
                np.column_stack([thresholds, excl_mean, excl_std, incl_mean, incl_std]),
                delimiter=",",
                header=(
                    "threshold,"
                    "exclude_diag_mean_pearson_r,exclude_diag_std_pearson_r,"
                    "include_diag_mean_pearson_r,include_diag_std_pearson_r"
                ),
                comments="",
            )

        if show:
            plt.show()
        else:
            return figure, axis


if __name__ == "__main__":
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    Jij = np.genfromtxt(AVG_JIJ_NEW_PATH, delimiter=",").astype(float)
    np.fill_diagonal(Jij, 0)

    emp_FC1 = np.genfromtxt(FC1_PATH, delimiter=",").astype(float)
    emp_FC2 = np.genfromtxt(FC2_PATH, delimiter=",").astype(float)
    emp_FC3 = np.genfromtxt(FC3_PATH, delimiter=",").astype(float)
    emp_FC = (emp_FC1 + emp_FC2 + emp_FC3) / 3.0
    np.fill_diagonal(emp_FC, 0)

    multiplier = utils.normalize_array(np.mean(np.abs(Jij), axis=0))

    alpha, T_c = load_pearson_alpha_tcrit()
    print(f"Using pearson_2.py values: alpha = {alpha:.4f}, T_c = {T_c:.4f}")

    thresholds = np.arange(0.1, 1.0, 0.1)  # matches the x-axis range in your slides

    sweep = threshold_correlation_sweep(
        ising_class=I.Jij_sorted_ising,  # swap for whichever Ising subclass your FC sweeps actually use
        Jij=Jij,
        emp_FC=emp_FC,
        T_c=T_c,
        alpha=alpha,
        multiplier=multiplier,
        partial=False,
    )
    sweep.run(n_restarts=5, steps=3000, thermalization=1000, spin_array=np.ones(Jij.shape[0]))

    sweep.graph_threshold_sweep_combined(
        thresholds,
        show=False,
        save_path=output_dir / "threshold_corr_combined_diag_vs_no_diag.png",
    )


# ── Optional single-simulation diagnostic plots ──────────────────────────
# This small container is the plotting interface formerly defined in GIM.py.
# Create it with a completed Ising object, then call any graph_* method.  The
# object must have ``functional_connectivity``, ``energy_series``,
# ``mag_series``, ``steps``, ``partial``, ``timer``, ``spin.Jij``,
# ``correlation()``, ``susceptibility()``, and ``specific_heat()``—the same
# completed-simulation interface used throughout this project.
class SimulationGraphData:
    """Diagnostic graph helpers for one completed Ising simulation."""

    def __init__(self, ising, beta, T_global, alpha, emp_FC, diag=False,
                 save=False, save_dir=None):
        self.ising = ising
        self.FC = ising.functional_connectivity
        self.emp_FC = emp_FC
        self.Jij = ising.spin.Jij
        self.beta = beta
        self.T_global = T_global
        self.alpha = alpha
        self.save = save
        self.partial = ising.partial
        self.time = ising.timer
        self.correlation = ising.correlation(emp_FC, diag)
        self.suscept = ising.susceptibility(beta)
        self.spec_heat = ising.specific_heat(beta)

        if save:
            self.path = Path(save_dir) if save_dir is not None else RESULTS_DIR
            self.path.mkdir(parents=True, exist_ok=True)

    def graph_mag_energy(self, show=True):
        """Plot magnetization and energy over time; save energy_mag_graph.png."""
        energy_series = self.ising.energy_series
        mag_series = self.ising.mag_series
        iterations = np.arange(self.ising.steps + 1)

        mpl.rcParams["lines.markersize"] = 3
        figure, axis = plt.subplots(1, 2)
        axis[0].scatter(iterations, mag_series)
        axis[0].plot(iterations, utils.average_series(mag_series), "r", label="average mag")
        axis[0].set(xlabel="steps", ylabel="magnetization", ylim=[0, 1])
        axis[0].legend()
        axis[1].set_ylim([np.min(energy_series), np.max(energy_series)])
        axis[1].scatter(iterations, energy_series)
        axis[1].plot(iterations, utils.average_series(energy_series), "r", label="average energy")
        axis[1].set(xlabel="steps", ylabel="energy")
        axis[1].legend()

        if self.save:
            figure.savefig(self.path / "energy_mag_graph.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            return figure, axis

    def graph_ROC(self, show=True):
        """Plot simulated-FC ROC curves; save ROC_graphs.png when enabled."""
        FC_tpr, FC_fpr, FC_auc = utils.receiver_operating_characteristic(self.FC, self.emp_FC)
        Jij_tpr, Jij_fpr, Jij_auc = utils.receiver_operating_characteristic(self.FC, self.Jij)

        figure, axis = plt.subplots(1, 2)
        axis[0].plot(FC_fpr, FC_tpr, label=f"AUC={FC_auc:.4f}")
        axis[0].set(title="ROC sim FC vs emp FC", xlabel="false positive ratio", ylabel="true positive ratio")
        axis[0].legend()
        axis[1].plot(Jij_fpr, Jij_tpr, label=f"AUC={Jij_auc:.4f}")
        axis[1].set(title="ROC sim FC vs Jij", xlabel="false positive ratio", ylabel="true positive ratio")
        axis[1].legend()

        if self.save:
            figure.savefig(self.path / "ROC_graphs.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            return figure, axis

    def graph_FC(self, show=True, title="simulated FC"):
        """Plot simulated FC, empirical FC, and Jij; save matrix_graphs.png."""
        figure, axis = plt.subplots(1, 3, figsize=(10, 4))
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        axis[0].matshow(self.FC, cmap="coolwarm", norm=norm)
        axis[0].set_title(title)
        empirical_image = axis[1].matshow(self.emp_FC, cmap="coolwarm", norm=norm)
        axis[1].set_title("empirical FC (partial)" if self.partial else "empirical FC (Pearson)")
        axis[2].matshow(self.Jij, cmap="coolwarm", norm=norm)
        axis[2].set_title("Jij")
        figure.colorbar(empirical_image, fraction=0.046, pad=0.04)

        if self.save:
            figure.savefig(self.path / "matrix_graphs.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            return figure, axis

    def graph_everything(self, show=True):
        """Create the six-panel diagnostic figure; save everything_graphs.png."""
        FC_tpr, FC_fpr, _ = utils.receiver_operating_characteristic(self.FC, self.emp_FC)
        Jij_tpr, Jij_fpr, _ = utils.receiver_operating_characteristic(self.FC, self.Jij)
        energy_series = self.ising.energy_series
        mag_series = self.ising.mag_series
        iterations = np.arange(self.ising.steps + 1)

        mpl.rcParams["lines.markersize"] = 3
        figure, axis = plt.subplots(2, 3, figsize=(12, 9), constrained_layout=True)
        axis[0, 0].scatter(iterations, mag_series)
        axis[0, 0].plot(iterations, utils.average_series(mag_series), "r")
        axis[0, 0].set(xlabel="steps", ylabel="magnetization", ylim=[0, 1])
        axis[0, 1].set_ylim([np.min(energy_series), np.max(energy_series)])
        axis[0, 1].scatter(iterations, energy_series)
        axis[0, 1].plot(iterations, utils.average_series(energy_series), "r")
        axis[0, 1].set(xlabel="steps", ylabel="energy")
        axis[0, 2].matshow(self.FC)
        axis[0, 2].set_title("simulated FC")
        axis[1, 2].matshow(self.emp_FC)
        axis[1, 2].set_title("empirical FC")
        axis[1, 0].plot(FC_fpr, FC_tpr)
        axis[1, 0].set(title="ROC sim FC vs emp FC", xlabel="false positive ratio", ylabel="true positive ratio")
        axis[1, 1].plot(Jij_fpr, Jij_tpr)
        axis[1, 1].set(title="ROC sim FC vs Jij", xlabel="false positive ratio", ylabel="true positive ratio")

        if self.save:
            figure.savefig(self.path / "everything_graphs.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            return figure, axis
