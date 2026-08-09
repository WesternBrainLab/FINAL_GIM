import os
import numpy as np
import pandas as pd

import scipy as sp
from concurrent.futures import ThreadPoolExecutor
import pickle
import time

"""Shared data configuration for the FINAL_GIM analysis pipeline.

This module performs no simulations.  Its job is to load the matrices used by
the other analysis scripts once, calculate a few useful network-level
summaries, and expose them under descriptive names.  Keeping these values in
one place avoids each script loading a different version of the data.

The arrays loaded here are all 84 × 84 connectivity matrices.  Rows and
columns refer to the same brain-region ordering, so matrices must remain
aligned before they are compared with simulated functional connectivity.

Planned consolidation: these definitions can later be moved into UTILS.py.
If that happens, update dependent imports at the same time, rather than
copying the code into both files; otherwise two configurations could drift.
"""

from pathlib import Path

# ``config`` is imported by ``steven.Scripts.ising3`` as part of the package,
# but this fallback preserves direct execution from this directory.
try:
    from . import utils
except ImportError:  # pragma: no cover - retained for direct script use
    import utils
import numpy as np
import os

# FINAL_GIM/CONFIG.py -> FINAL_GIM/DATA.  Using ``resolve`` makes this work
# regardless of the terminal directory from which a script is launched.
DATA_DIR = Path(__file__).resolve().parent / "DATA"


def get_data_matrix(relative_path):
    """Load one comma-delimited matrix from FINAL_GIM/DATA.

    ``relative_path`` is deliberately relative to ``DATA_DIR`` (for example,
    ``'FC data_processed/avg_TS_1'``), so callers do not need to hard-code
    machine-specific absolute paths.
    """
    return utils.get_matrix(relative_path, directory=DATA_DIR)

# ── Structural connectivity used by the Ising model ──────────────────────
# ``avg_Jij`` is the group-average structural coupling matrix after removing
# outliers and normalizing its scale.  Ising simulations use it as Jij: a
# positive/negative value determines how strongly two regions influence each
# other's spin states.  ``regions`` is the number of model nodes and should
# match every FC matrix loaded below.
avg_Jij = get_data_matrix('Jij data_processed/avg_Jij_no_outliers_norm')
regions = np.shape(avg_Jij)[0]

# ── Empirical functional-connectivity (FC) reference matrices ────────────
# Each ``FC_n`` is a Pearson-correlation FC matrix from one recording/session
# group. ``avg_FC`` is their elementwise mean and is the usual empirical
# reference when judging a simulated FC matrix.
#
# Filename suffixes:
#   p  = partial-correlation FC (relationship after controlling other nodes)
#   b  = binarized FC
#   pb = binarized partial-correlation FC
# The unsuffixed matrices below are standard Pearson FC values.
FC_1p, FC_2p, FC_3p = get_data_matrix('FC data_processed/avg_TS_1p'), \
                      get_data_matrix('FC data_processed/avg_TS_2p'), \
                      get_data_matrix('FC data_processed/avg_TS_3p')
# Group-average partial-correlation FC.
avg_FCp = utils.average_matrices(FC_1p, FC_2p, FC_3p)

FC_1, FC_2, FC_3 = get_data_matrix('FC data_processed/avg_TS_1'), \
                   get_data_matrix('FC data_processed/avg_TS_2'), \
                   get_data_matrix('FC data_processed/avg_TS_3')
avg_FC = utils.average_matrices(FC_1, FC_2, FC_3)

FC_1pb, FC_2pb, FC_3pb = get_data_matrix('FC data_processed/avg_TS_1pb'), \
                         get_data_matrix('FC data_processed/avg_TS_2pb'), \
                         get_data_matrix('FC data_processed/avg_TS_3pb')
avg_FCpb = utils.average_matrices(FC_1pb, FC_2pb, FC_3pb)

# ── Per-region summaries used for heterogeneous temperatures ─────────────
# The mean of each *column* measures a region's average connectivity to all
# other regions.  ``ind_avg_Jij`` is derived from the structural matrix and
# is commonly normalized before using it as a per-region temperature
# multiplier.  ``ind_avg_FC`` is the corresponding empirical FC summary.
ind_avg_Jij = np.mean(avg_Jij, 0)
norm_ind_avg_Jij = utils.normalize_array(ind_avg_Jij)
ind_avg_FC = np.mean(avg_FC, 0)

# Sorted copies are intended for distribution/rank plots.  ``ndarray.sort``
# sorts ascending in place; these are therefore least-connected to
# most-connected, despite legacy names that do not state the direction.
sort_ind_avg_FC = ind_avg_FC.copy()
sort_ind_avg_FC.sort()
sort_ind_avg_Jij = ind_avg_Jij.copy()
sort_ind_avg_Jij.sort()



 # This module contains helper functions 
 # When running UTILS.py, thresholded jij is saved to DATA folder.


def get_folder(folder_name, directory = os.path.dirname(__file__)):
    current_directory = directory
    folder_from_directory = os.path.join(current_directory, folder_name)
    return folder_from_directory


def get_matrix(file_name, directory = os.path.dirname(__file__)):
    file_path = os.path.join(directory, file_name)
    with open(file_path, newline='') as csvfile:
        matrix_from_file = np.genfromtxt(csvfile, delimiter = ',')
        return matrix_from_file


def save_matrix(matrix, name):
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(name, index = False, header = False)


def matrix_from_dir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + '/' + f)]
    matrix_ar = []

    for file_name in files:
        #file_path = get_matrix(os.path.join(directory, file_name))
        file_path = get_matrix(file_name, directory)
        matrix_ar.append(file_path)
    return np.array(matrix_ar)


def minmax_norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def df_to_text(data, directory, file_name):
    path = directory + '/' + file_name
    with open(path, 'w') as file:
        data_string = data.to_string()
        file.write(data_string)


def part_corr(time_series, ridge=1e-8):
    """
    Pairwise partial-correlation matrix conditioned on all other variables.

    Parameters
    ----------
    time_series : ndarray, shape (num_neurons, num_timepoints)
    ridge : float
        Small diagonal regularization for numerical stability.

    Returns
    -------
    partial_corr : ndarray, shape (num_neurons, num_neurons)
    """

    X = np.asarray(time_series, dtype=float)

    if X.ndim != 2:
        raise ValueError("time_series must have shape (num_neurons, num_timepoints).")

    num_neurons, num_steps = X.shape

    if num_steps < 2:
        raise ValueError("At least two time points are required.")

    # Remove the mean from each neuron time series
    X = X - X.mean(axis=1, keepdims=True)

    # Sample covariance matrix
    covariance = (X @ X.T) / (num_steps - 1)

    # Regularization helps when covariance is singular or nearly singular
    scale = np.trace(covariance) / num_neurons
    covariance += ridge * scale * np.eye(num_neurons)

    # Precision matrix
    precision = np.linalg.pinv(covariance)

    diagonal = np.diag(precision)

    # Guard against invalid / near-zero diagonal entries
    if np.any(diagonal <= 0):
        raise ValueError(
            "Precision matrix has non-positive diagonal entries. "
            "Increase ridge or remove constant time series."
        )

    partial_corr = -precision / np.sqrt(np.outer(diagonal, diagonal))

    # The diagonal is defined as 1 for a correlation matrix
    np.fill_diagonal(partial_corr, 1.0)

    # Remove tiny numerical asymmetries
    partial_corr = 0.5 * (partial_corr + partial_corr.T)

    return partial_corr


def normalize_array(array):
    return array / np.max(np.abs(array))


def cross_sort(sort_array, *args, hi_lo = True):
    if hi_lo:
        copy_array = np.sort(np.unique(sort_array))[::-1]
    else:
        copy_array = np.sort(np.unique(sort_array))

    index = []
    for i in copy_array:
        current_index = np.where(sort_array == i)[0]
        if np.size(current_index) > 1:
            for j in current_index:
                index.append(j)
        else:
            index.append(current_index[0])
    if args:
        return args[index]
    return index


def percent_error(actual, expected):
    return np.abs((actual - expected) / expected)


def average_matrices(*arrays):
    avg_ar = np.zeros(np.shape(arrays[0]))

    for ar in arrays:
        avg_ar += ar

    avg_ar /= len(arrays)
    return avg_ar


def flat_remove_diag(array):
    new_ar = []
    length = range(np.shape(array)[0])
    for y in length:
        for x in length:
            if x != y:
                new_ar.append(array[y, x])
    return np.array(new_ar)


def average_series(series):
    return np.cumsum(series) / (np.arange(np.size(series)) + 1)


def receiver_operating_characteristic(input_matrix, check_matrix):
    fpr_ar, tpr_ar = [0], [0]

    for thresh in np.linspace(1, 0.01, 100):
        check_matrix_copy = check_matrix.copy()
        input_matrix_copy = input_matrix.copy()

        input_matrix_copy[input_matrix < thresh] = 0
        input_matrix_copy[input_matrix >= thresh] = 1

        check_matrix_copy[check_matrix_copy < thresh] = 0
        check_matrix_copy[check_matrix_copy >= thresh] = 1
        _, counts = np.unique(check_matrix_copy, return_counts = True)

        compare_matrix = input_matrix_copy == check_matrix_copy

        true_positive = compare_matrix * input_matrix_copy
        false_positive = np.abs(compare_matrix - 1) * input_matrix_copy

        fpr_ar.append(np.size(false_positive[false_positive == 1]) / counts[0])
        tpr_ar.append(np.size(true_positive[true_positive == 1]) / counts[1])

    fpr_ar.append(1)
    tpr_ar.append(1)
    return tpr_ar, fpr_ar, sp.integrate.trapezoid(tpr_ar, x = fpr_ar)


def get_pickle_file(directory, file_name):
    directory = directory + '/' + file_name
    with open(directory, 'rb') as picklefile:
        return pickle.load(picklefile)


def threshold_matrix(matrix, threshold):
    matrix_copy = matrix.copy() / np.max(np.abs(matrix))
    thresh_mat = np.ones(np.shape(matrix_copy))
    thresh_mat[abs(matrix_copy) < threshold] = 0
    thresh_mat[abs(matrix_copy) >= threshold] = 1
    matrix_copy = matrix_copy * thresh_mat
    matrix_copy[matrix_copy >= 0] = 1
    matrix_copy[matrix_copy < 0] = -1
    return matrix_copy

# This is a directory path to the project folder. It is used to define paths to data and output directories.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "DATA"
PROJECT_DATA_DIR = DATA_ROOT

JIJ_DIR = DATA_ROOT / "Jij data_raw"
JIJ_PATTERN = "Jij_{}.csv"
JIJ_NEW_DIR = PROJECT_DATA_DIR / "thresholded_Jij_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_thresholded_Jij_pearson.csv"

FC1_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_3"

SUBJECT_IDS = list(range(2, 26))
THRESHOLD = 0.0 # was 0.03
SEED = 1


# For each of the 25 subjects:
#   1. Load Jij_initial_i
#   2. Load the matching session Pearson FC (avg_TS_1 / avg_TS_2 / avg_TS_3)
#   3. Apply threshold  →  Jij_new_i
#   4. Save as csv →  thresholded_Jij_pearson_i.csv
#
# Then average all 25  →  avg_thresholded_Jij_pearson.csv
#
# Threshold rule (off-diagonal entries only):
#   |Rho_ij| <  THRESHOLD  →  keep Jij_initial[i,j]               (structural sign)
#   |Rho_ij| >= THRESHOLD  →  abs(Jij_initial[i,j]) * sign(Rho_ij) (empirical sign)
#   diagonal               →  0

# ── subject → session mapping ─────────────────────────────────────────────
# Maps subject index (1-based) to session key.
# Adjust if your subject-to-session assignment differs.
def subject_session(s: int) -> str:
    if s <= 9:   return "FC1"
    if s <= 17:  return "FC2"
    return "FC3"


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


def save_csv(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)


def build_Jij_new(
    Jij_initial: np.ndarray,
    Rho_empirical: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sign-corrected Jij update using Pearson FC.
    Returns (Jij_new, Rho_thresh).
    """
    Jij       = Jij_initial.copy().astype(float)
    Rho       = Rho_empirical.copy().astype(float)
    N         = Jij.shape[0]

    Rho_thresh = Rho.copy()
    Rho_thresh[np.abs(Rho) < threshold] = 0.0

    Jij_new = Jij.copy()
    for i in range(N):
        for j in range(N):
            if i == j:
                Jij_new[i, j] = 0.0
            elif Rho_thresh[i, j] != 0.0:
                Jij_new[i, j] = np.sign(Rho_thresh[i, j]) * abs(Jij[i, j])
            # else: structural sign kept unchanged

    return Jij_new, Rho_thresh


def enforce_symmetry(mat: np.ndarray, label: str = "") -> np.ndarray:
    err = np.max(np.abs(mat - mat.T))
    if err > 1e-10:
        print(f"    [symmetry] {label}: max asymmetry={err:.2e} — enforcing (A+Aᵀ)/2")
        mat = (mat + mat.T) / 2.0
    return mat


def print_diagnostics(Jij_init, Jij_new, Rho_thresh, s):
    N         = Jij_init.shape[0]
    off       = ~np.eye(N, dtype=bool)
    n_above   = np.sum(Rho_thresh[off] != 0)
    n_total   = np.sum(off)
    n_flipped = np.sum(np.sign(Jij_new[off]) != np.sign(Jij_init[off]))
    print(
        f"  Subject {s:02d} [{subject_session(s)}]: "
        f"above-thresh={n_above}/{n_total} ({100*n_above/n_total:.1f}%)  "
        f"sign-flips={n_flipped} ({100*n_flipped/n_total:.1f}%)"
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)
    subject_ids = SUBJECT_IDS

    print("=" * 65)
    print(f"BUILD JIJ_NEW (Pearson)  —  {len(subject_ids)} subjects  "
          f"|  threshold={THRESHOLD}")
    print("=" * 65)

    # pre-load the 3 session Pearson FC matrices
    session_fc = {
        "FC1": load_csv(FC1_PATH).astype(float),
        "FC2": load_csv(FC2_PATH).astype(float),
        "FC3": load_csv(FC3_PATH).astype(float),
    }
    for mat in session_fc.values():
        np.fill_diagonal(mat, 0)

    print("\nLoaded session FC matrices: avg_TS_1, avg_TS_2, avg_TS_3")

    os.makedirs(JIJ_NEW_DIR, exist_ok=True)
    all_Jij_new = []

    for s in subject_ids:

        jij_path = JIJ_DIR / JIJ_PATTERN.format(s)
        Jij_init  = load_csv(jij_path).astype(float)
        Rho_emp_s = session_fc[subject_session(s)]

        assert Jij_init.shape == Rho_emp_s.shape, (
            f"Subject {s:02d}: shape mismatch "
            f"Jij {Jij_init.shape} vs Rho_emp {Rho_emp_s.shape}"
        )

        Jij_new, Rho_thresh = build_Jij_new(Jij_init, Rho_emp_s, THRESHOLD)
        Jij_new = enforce_symmetry(Jij_new, f"Subject {s:02d}")
        print_diagnostics(Jij_init, Jij_new, Rho_thresh, s)

        out_path = JIJ_NEW_DIR / f"Jij_new_pearson_{s:02d}.csv"
        save_csv(Jij_new, out_path)
        all_Jij_new.append(Jij_new)

    # average across subjects
    print(f"\n[averaging]  avg_Jij_new across {len(subject_ids)} subjects …")
    avg_Jij_new = np.mean(all_Jij_new, axis=0)
    np.fill_diagonal(avg_Jij_new, 0)

    save_csv(avg_Jij_new, AVG_JIJ_NEW_PATH)

    off = avg_Jij_new[~np.eye(avg_Jij_new.shape[0], dtype=bool)]
    print(
        f"  avg_Jij_new_pearson : shape={avg_Jij_new.shape}  "
        f"min={off.min():.4f}  max={off.max():.4f}  "
        f"neg_fraction={np.mean(off < 0):.3f}"
    )
    print(f"  Saved → {AVG_JIJ_NEW_PATH}")
    print(f"  Per-subject files → {JIJ_NEW_DIR}/")



if __name__ == "__main__":
    main()
