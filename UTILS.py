"""
UTILS.py — Shared helper functions and Jij sign-correction pipeline.

When imported: provides pure utility functions with no side effects.
When run directly (__main__): builds sign-corrected per-subject Jij matrices
and saves them under DATA/thresholded_Jij_pearson/.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp

import CONFIG as C

# ── Path constants (Jij build pipeline) ───────────────────────────────────
JIJ_RAW_DIR     = C.DATA_DIR / "Jij data_raw"
JIJ_RAW_PATTERN = "Jij_{}.csv"
JIJ_NEW_DIR     = C.DATA_DIR / "thresholded_Jij_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_thresholded_Jij_pearson.csv"

SUBJECT_IDS = list(range(2, 26))   # subjects 2 – 25 (24 total)
SEED        = C.SEED


# ═══════════════════════════════════════════════════════════════════════════
# I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════

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


def load_csv(path: str | Path) -> np.ndarray:
    """Thin wrapper: load a CSV as a float64 array."""
    return np.genfromtxt(path, delimiter=",", dtype=float)


def save_csv(matrix: np.ndarray, path: str | Path) -> None:
    """Save *matrix* as CSV; creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)


def get_pickle_file(directory, file_name):
    directory = directory + '/' + file_name
    with open(directory, 'rb') as picklefile:
        return pickle.load(picklefile)




def df_to_text(data, directory, file_name):
    path = directory + '/' + file_name
    with open(path, 'w') as file:
        data_string = data.to_string()
        file.write(data_string)

# ═══════════════════════════════════════════════════════════════════════════
# ARRAY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def normalize_array(array):
    return array / np.max(np.abs(array))


def minmax_norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

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

def percent_error(actual, expected):
    return np.abs((actual - expected) / expected)


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


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════


def threshold_matrix(matrix, threshold):
    matrix_copy = matrix.copy() / np.max(np.abs(matrix))
    thresh_mat = np.ones(np.shape(matrix_copy))
    thresh_mat[abs(matrix_copy) < threshold] = 0
    thresh_mat[abs(matrix_copy) >= threshold] = 1
    matrix_copy = matrix_copy * thresh_mat
    matrix_copy[matrix_copy >= 0] = 1
    matrix_copy[matrix_copy < 0] = -1
    return matrix_copy


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


# ═══════════════════════════════════════════════════════════════════════════
# JIJ BUILD PIPELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def subject_session(s: int) -> str:
    """Map subject index (1-based) to session key "FC1" | "FC2" | "FC3"."""
    if s <= 9:  return "FC1"
    if s <= 17: return "FC2"
    return "FC3"


def build_jij_new(
    jij_initial:   np.ndarray,
    rho_empirical: np.ndarray,
    threshold:     float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Pearson-FC sign correction to *jij_initial*.

    Off-diagonal rule
    -----------------
    |Rho_ij| <  threshold → keep structural sign (Jij unchanged)
    |Rho_ij| >= threshold → abs(Jij_ij) * sign(Rho_ij)
    Diagonal              → 0

    Returns (jij_new, rho_thresh).
    """
    jij = jij_initial.copy().astype(float)
    rho = rho_empirical.copy().astype(float)
    n   = jij.shape[0]

    rho_thresh = rho.copy()
    rho_thresh[np.abs(rho) < threshold] = 0.0

    jij_new = jij.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                jij_new[i, j] = 0.0
            elif rho_thresh[i, j] != 0.0:
                jij_new[i, j] = np.sign(rho_thresh[i, j]) * abs(jij[i, j])
            # else: keep structural sign unchanged

    return jij_new, rho_thresh


def enforce_symmetry(mat: np.ndarray, label: str = "") -> np.ndarray:
    """Force symmetry via (A + Aᵀ)/2 when max asymmetry exceeds 1e-10."""
    err = np.max(np.abs(mat - mat.T))
    if err > 1e-10:
        print(f"  [symmetry] {label}: max asymmetry = {err:.2e} — enforcing (A+Aᵀ)/2")
        mat = (mat + mat.T) / 2.0
    return mat


def _print_jij_diagnostics(
    jij_init:   np.ndarray,
    jij_new:    np.ndarray,
    rho_thresh: np.ndarray,
    s:          int,
) -> None:
    n       = jij_init.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    n_above   = np.sum(rho_thresh[offdiag] != 0)
    n_total   = np.sum(offdiag)
    n_flipped = np.sum(np.sign(jij_new[offdiag]) != np.sign(jij_init[offdiag]))
    print(
        f"  Subject {s:02d} [{subject_session(s)}]: "
        f"above-thresh = {n_above}/{n_total} ({100 * n_above / n_total:.1f}%)  "
        f"sign-flips = {n_flipped} ({100 * n_flipped / n_total:.1f}%)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — build and save sign-corrected Jij matrices
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    np.random.seed(SEED)
    threshold = C.THRESHOLD

    print("=" * 65)
    print(
        f"BUILD JIJ_NEW (Pearson)  —  {len(SUBJECT_IDS)} subjects  "
        f"|  threshold = {threshold}"
    )
    print("=" * 65)

    # Pre-load the three session Pearson FC matrices
    session_fc = {
        "FC1": load_csv(C.FC1_PATH).astype(float),
        "FC2": load_csv(C.FC2_PATH).astype(float),
        "FC3": load_csv(C.FC3_PATH).astype(float),
    }
    for mat in session_fc.values():
        np.fill_diagonal(mat, 0.0)
    print("Loaded session FC matrices: avg_TS_1, avg_TS_2, avg_TS_3\n")

    JIJ_NEW_DIR.mkdir(parents=True, exist_ok=True)
    all_jij_new = []

    for s in SUBJECT_IDS:
        jij_path  = JIJ_RAW_DIR / JIJ_RAW_PATTERN.format(s)
        jij_init  = load_csv(jij_path).astype(float)
        rho_emp_s = session_fc[subject_session(s)]

        if jij_init.shape != rho_emp_s.shape:
            raise ValueError(
                f"Subject {s:02d}: shape mismatch — "
                f"Jij {jij_init.shape} vs Rho_emp {rho_emp_s.shape}"
            )

        jij_new, rho_thresh = build_jij_new(jij_init, rho_emp_s, threshold)
        jij_new = enforce_symmetry(jij_new, f"Subject {s:02d}")
        _print_jij_diagnostics(jij_init, jij_new, rho_thresh, s)

        out_path = JIJ_NEW_DIR / f"Jij_new_pearson_{s:02d}.csv"
        save_csv(jij_new, out_path)
        all_jij_new.append(jij_new)

    # Average across all subjects
    print(f"\n[averaging]  avg_Jij_new across {len(SUBJECT_IDS)} subjects …")
    avg_jij_new = np.mean(all_jij_new, axis=0)
    np.fill_diagonal(avg_jij_new, 0.0)
    save_csv(avg_jij_new, AVG_JIJ_NEW_PATH)

    offdiag = avg_jij_new[~np.eye(avg_jij_new.shape[0], dtype=bool)]
    print(
        f"  avg_Jij_new: shape = {avg_jij_new.shape}  "
        f"min = {offdiag.min():.4f}  max = {offdiag.max():.4f}  "
        f"neg_fraction = {np.mean(offdiag < 0):.3f}"
    )
    print(f"  Saved → {AVG_JIJ_NEW_PATH}")
    print(f"  Per-subject files → {JIJ_NEW_DIR}/")


if __name__ == "__main__":
    main()
