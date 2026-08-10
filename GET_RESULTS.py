"""
GET_RESULTS.py — Main analysis pipeline for the Generalized Ising Model.

Runs the following steps and saves all outputs to RESULTS_DIR:

  Step 0  — Mu-vs-PET diagnostic: compare structural hub strength to PET
            (independent of which LOCAL_TEMPERATURE_MODEL is active).
  Step 1  — Parameter annealing: find (T*, alpha*) via dual_annealing,
            with iterative bounds refinement.
  Step 1B — Optuna cross-validated search: refines (T*, alpha*) further
            around the annealed optimum using repeated-simulation folds,
            with the *same* iterative window-shrinking as Step 1's refine
            loop (REFINE_MAX_ROUNDS / REFINE_SHRINK / REFINE_MIN_*_WINDOW).
  Step 2  — Temperature sweep: measure FC-correlation across a range of T.
  Step 3  — Matrix comparison: compare simulated and empirical FC at T_best.
  Step 4  — Null distribution: test significance against shuffled-Jij and
            constant-Jij null models.

Two further optional analyses (KS test suite, threshold-correlation sweep)
run after Step 4 if enabled.

LOCAL_TEMPERATURE_MODEL ("pearson" | "pet" | "inverse_pet") and FC_MODE
("pearson" | "partial") in CONFIG.py select, respectively, how the local
temperature multiplier is built and which empirical FC (full vs partial
correlation) is fit against.

Every step and every figure is individually toggleable from CONFIG.py —
flip a RUN_* or MAKE_*/SAVE_* boolean to False to skip it without
touching this file. See CONFIG.py for the full list and defaults.

Change C.RUN_FOLDER_NAME in CONFIG.py before each new run to keep outputs
separate (recommended format: "MM-DD-YYYY_HHMM").

Additional classes / functions:
  SimulationGraphData         — diagnostic plots for one completed simulation.
  simulated_KS_vs_T_global    — KS-test sibling to simulated_FC_vs_T_global.
  threshold_correlation_sweep — correlation vs Jij threshold at T_crit.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import ks_2samp, pearsonr

import CONFIG     as C
import UTILS      as utils
import GIM        as I
import OPTIMIZE   as pa_module
import TEMP_SWEEP as ts

# Ensure RESULTS_DIR exists before any output is written
C.ensure_results_dir()

# ── FC mode: full Pearson FC vs partial-correlation FC ─────────────────────
PARTIAL   = (C.FC_MODE == "partial")
_FC1_PATH = C.PARTIAL_FC1_PATH if PARTIAL else C.FC1_PATH
_FC2_PATH = C.PARTIAL_FC2_PATH if PARTIAL else C.FC2_PATH
_FC3_PATH = C.PARTIAL_FC3_PATH if PARTIAL else C.FC3_PATH


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL-TEMPERATURE MODEL
# ═══════════════════════════════════════════════════════════════════════════

def load_pet_values() -> np.ndarray:
    """Load the regional PET regressor vector (no size/sign validation)."""
    pet_values = utils.load_csv(C.PET_NO_OUTLIERS_PATH).astype(float).reshape(-1)
    if not np.all(np.isfinite(pet_values)):
        raise ValueError(f"PET values contain NaN/inf: {C.PET_NO_OUTLIERS_PATH}")
    return pet_values


def build_multiplier(J_real: np.ndarray) -> np.ndarray:
    """
    Build the per-node local-temperature multiplier per
    C.LOCAL_TEMPERATURE_MODEL:

      "pearson"     -> normalize(mean(|Jij|, axis=0))   — structural hub strength
      "pet"         -> PET_i
      "inverse_pet" -> 1 / PET_i

    T_i = T_global * multiplier_i ** alpha, so pet/inverse_pet multipliers
    must be strictly positive (alpha search ranges include negative values).
    """
    if C.LOCAL_TEMPERATURE_MODEL == "pearson":
        return utils.normalize_array(np.mean(np.abs(J_real), axis=0))

    pet_values = load_pet_values()
    if pet_values.size != J_real.shape[0]:
        raise ValueError(
            f"Expected {J_real.shape[0]} PET values in {C.PET_NO_OUTLIERS_PATH}, "
            f"found {pet_values.size}."
        )
    if np.any(pet_values <= 0):
        raise ValueError(
            "PET values must all be > 0 because negative alpha values are searched."
        )

    if C.LOCAL_TEMPERATURE_MODEL == "inverse_pet":
        return 1.0 / pet_values
    if C.LOCAL_TEMPERATURE_MODEL == "pet":
        return pet_values

    raise ValueError(f"Unknown CONFIG.LOCAL_TEMPERATURE_MODEL: {C.LOCAL_TEMPERATURE_MODEL!r}")


# ── Shared data ───────────────────────────────────────────────────────────
J_real  = utils.load_csv(C.AVG_JIJ_NEW_PATH).astype(float)
np.fill_diagonal(J_real, 0.0)

avg_Jij = utils.load_csv(C.JIJ_PROCESSED_PATH).astype(float)

emp_FC1 = utils.load_csv(_FC1_PATH).astype(float)
emp_FC2 = utils.load_csv(_FC2_PATH).astype(float)
emp_FC3 = utils.load_csv(_FC3_PATH).astype(float)
rho_emp = (emp_FC1 + emp_FC2 + emp_FC3) / 3.0

N          = J_real.shape[0]
multiplier = build_multiplier(J_real)

ISING_CLASS      = I.random_ising if C.USE_RANDOM_SPIN_FLIP else I.Jij_sorted_ising
SPIN_FLIP_METHOD = "random" if C.USE_RANDOM_SPIN_FLIP else "Jij_sorted"

print(f"Spin-flip method  : {SPIN_FLIP_METHOD}")
print(f"FC mode           : {'partial' if PARTIAL else 'pearson'}")
print(f"Local temp model  : {C.LOCAL_TEMPERATURE_MODEL}")
print(f"Results folder    : {C.RESULTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# VECTOR / MATRIX HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def upper_tri_vec(mat: np.ndarray) -> np.ndarray:
    """Return the upper-triangle (k=1) of *mat* as a 1-D vector."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def set_fc_diagonal(mat: np.ndarray) -> np.ndarray:
    """Set diagonal to 0 (ZERO_FC_DIAGONAL=True) or 1 (False) in-place."""
    np.fill_diagonal(mat, 0 if C.ZERO_FC_DIAGONAL else 1)
    return mat


def fc_compare_vec(mat: np.ndarray) -> np.ndarray:
    """Return the comparison vector: upper-triangle or full ravel."""
    return upper_tri_vec(mat) if C.ZERO_FC_DIAGONAL else mat.ravel()


def clean_vec(vec: np.ndarray) -> np.ndarray:
    """Replace NaN / ±inf with 0 so downstream stats don't silently break."""
    return np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r that returns 0.0 instead of NaN for degenerate vectors."""
    x = clean_vec(x)
    y = clean_vec(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = pearsonr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


def finite_vals(vals: list | np.ndarray, name: str) -> np.ndarray:
    """
    Drop non-finite values and raise if none remain.

    Prints a warning when values are dropped so silent failures are visible.
    """
    arr  = np.asarray(vals, dtype=float)
    good = arr[np.isfinite(arr)]
    dropped = len(arr) - len(good)
    if dropped:
        print(f"WARNING: dropped {dropped}/{len(arr)} non-finite values from '{name}'")
    if len(good) == 0:
        raise ValueError(
            f"All values in '{name}' are NaN/inf. "
            "Check Ising simulation output and temperature array."
        )
    return good


def evenly_spaced_indices(indices: np.ndarray, n_select: int) -> np.ndarray:
    """Return *n_select* evenly-spaced entries from *indices*."""
    if len(indices) <= n_select:
        return indices
    positions = np.linspace(0, len(indices) - 1, n_select, dtype=int)
    return indices[positions]


def refine_bounds(
    center:       tuple[float, float],
    T_window:     float,
    alpha_window: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute narrowed annealing bounds centred on *center*."""
    T_c, a_c = center
    (T_lo, T_hi), (a_lo, a_hi) = C.ANNEAL_BOUNDS
    return (
        (max(T_lo, T_c - T_window),     min(T_hi, T_c + T_window)),
        (max(a_lo, a_c - alpha_window), min(a_hi, a_c + alpha_window)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SMOOTHING / PEAK DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def smooth_curve(values: np.ndarray, clip_min: float | None = None) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to a temperature-sweep curve for display.

    The underlying statistics are computed on the raw values; this function
    is called only immediately before plotting.
    """
    values = np.asarray(values, dtype=float)
    if not C.SMOOTH_TEMPERATURE_PLOTS or values.size < 5:
        return values

    x      = np.arange(values.size)
    finite = np.isfinite(values)
    if finite.sum() < 5:
        return values

    filled          = values.copy()
    filled[~finite] = np.interp(x[~finite], x[finite], values[finite])

    # Ensure window is odd and larger than polyorder
    window = min(C.SMOOTH_WINDOW, values.size if values.size % 2 == 1 else values.size - 1)
    if window <= C.SMOOTH_POLYORDER:
        window = C.SMOOTH_POLYORDER + 2
    if window % 2 == 0:
        window += 1
    if window > values.size:
        return values

    smoothed = savgol_filter(filled, window_length=window, polyorder=C.SMOOTH_POLYORDER)
    if clip_min is not None:
        smoothed = np.maximum(smoothed, clip_min)
    return smoothed


def mean_and_sd_band(
    values: np.ndarray,
    sd:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, smoothed_sd) for a fill_between SD band."""
    return np.asarray(values, dtype=float), smooth_curve(np.asarray(sd, dtype=float), clip_min=0.0)


def peak_candidate_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peak candidates in *values* using prominence filtering and
    edge-point masking (C.PEAK_IGNORE_EDGE_POINTS on each side).

    Returns (peak_indices, work_array) where *work_array* has edges set
    to NaN so callers can use it directly for argmax.
    """
    values = np.asarray(values, dtype=float)
    work   = values.copy()
    finite = np.isfinite(work)
    if finite.sum() < 3:
        return np.array([], dtype=int), work

    edge = max(0, C.PEAK_IGNORE_EDGE_POINTS)
    if work.size > 2 * edge:
        work[:edge]  = np.nan
        work[-edge:] = np.nan

    finite_work = work[np.isfinite(work)]
    spread      = np.nanmax(finite_work) - np.nanmin(finite_work)
    prominence  = C.PEAK_PROMINENCE_FRACTION * spread if spread > 0 else 0.0

    filled = work.copy()
    filled[~np.isfinite(filled)] = -np.inf
    peaks, _ = find_peaks(filled, prominence=prominence)
    return peaks.astype(int), work


def stable_peak_index(values: np.ndarray) -> int:
    """
    Return the index of the highest-prominence peak in *values*.
    Falls back to nanargmax when no peaks are detected.
    """
    values = np.asarray(values, dtype=float)
    peaks, work = peak_candidate_indices(values)
    if peaks.size:
        return int(peaks[np.nanargmax(work[peaks])])
    return int(np.nanargmax(work))


# ═══════════════════════════════════════════════════════════════════════════
# MU-VS-PET DIAGNOSTIC (independent of LOCAL_TEMPERATURE_MODEL)
# ═══════════════════════════════════════════════════════════════════════════

def run_mu_vs_pet_diagnostic(J_real: np.ndarray) -> None:
    """
    Compare normalized structural hub strength (mu) against regional PET
    values, regardless of which LOCAL_TEMPERATURE_MODEL is active. Saves a
    scatter + linear fit and prints the Pearson correlation.
    """
    mu = utils.normalize_array(np.mean(np.abs(J_real), axis=0))
    try:
        pet_values = load_pet_values()
    except FileNotFoundError:
        print(f"WARNING: skipping mu-vs-PET diagnostic — {C.PET_NO_OUTLIERS_PATH} not found.")
        return

    if pet_values.size != mu.size:
        print(
            f"WARNING: skipping mu-vs-PET diagnostic — "
            f"{pet_values.size} PET values vs {mu.size} regions."
        )
        return

    r, p = pearsonr(mu, pet_values)
    print(f"mu vs PET: Pearson r={r:.6f}, p={p:.6g}")

    if not C.MAKE_MU_VS_PET_PLOT:
        return

    slope, intercept = np.polyfit(mu, pet_values, 1)
    order = np.argsort(mu)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(mu, pet_values, color=C.BLUE, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.plot(
        mu[order], slope * mu[order] + intercept, color=C.RED, linewidth=2,
        label=f"Pearson r = {r:.3f}, p = {p:.3g}",
    )
    ax.set_xlabel(r"normalized structural hub strength $\mu_i$")
    ax.set_ylabel(r"PET value $PET_i$")
    ax.set_title(r"Structural hub strength $\mu$ vs PET")
    ax.legend(framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _maybe_show_and_save(
        fig, C.RESULTS_DIR / "mu_vs_pet.png",
        C.MAKE_MU_VS_PET_PLOT, C.SAVE_MU_VS_PET_PLOT,
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPTUNA CROSS-VALIDATED SEARCH (Step 1B)
# ═══════════════════════════════════════════════════════════════════════════

def run_optuna_cv(
    Jij:          np.ndarray,
    multiplier:   np.ndarray,
    emp_FC_vec:   np.ndarray,
    center:       tuple[float, float],
    T_window:     float,
    alpha_window: float,
):
    """
    Iteratively-refined, cross-validated search for (T, alpha) using Optuna,
    centred on *center* (typically the dual-annealing result from Step 1).

    Each trial's score is the mean Pearson r across C.OPTUNA_TEST_FOLDS
    independent held-out simulation reruns at that (T, alpha) — a "fold"
    here is one full independent Ising simulation rather than a data split
    (there's a single empirical FC target to fit), so this estimates how
    much of a candidate's fit is reproducible signal vs a lucky single run.

    The search window shrinks each round using the *same* schedule as
    Step 1's refine loop: C.REFINE_MAX_ROUNDS rounds, each window scaled by
    C.REFINE_SHRINK and re-centred on that round's best point, stopping
    early once both windows fall below C.REFINE_MIN_T_WINDOW /
    C.REFINE_MIN_ALPHA_WINDOW.

    Returns (T_cv, alpha_cv, best_cv_score, all_trial_values).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "RUN_STEP1B_OPTUNA_CV requires the 'optuna' package. "
            "Install it with: pip install optuna"
        ) from exc

    def objective(trial, T_bounds, a_bounds):
        T     = trial.suggest_float("T", *T_bounds)
        alpha = trial.suggest_float("alpha", *a_bounds)

        temp_ar = T * (multiplier ** alpha)
        temp_ar = np.nan_to_num(temp_ar, nan=T, posinf=T, neginf=T)
        temp_ar[temp_ar <= 0] = 1e-12

        fold_scores = []
        for _ in range(C.OPTUNA_TEST_FOLDS):
            sim = ISING_CLASS(temp_ar, Jij=Jij)
            sim.simulate(C.ANNEAL_STEPS, C.ANNEAL_THERM)
            fc = np.nan_to_num(sim.generate_FC(PARTIAL), nan=0.0, posinf=0.0, neginf=0.0)
            fold_scores.append(safe_pearson(clean_vec(fc_compare_vec(fc)), emp_FC_vec))

        trial.set_user_attr("T", T)
        trial.set_user_attr("alpha", alpha)
        return float(np.mean(fold_scores))

    best_center = center
    best_score  = -np.inf
    all_values:  list[float] = []
    round_boundaries: list[int] = []

    for rnd in range(C.REFINE_MAX_ROUNDS):
        T_bounds, a_bounds = refine_bounds(best_center, T_window, alpha_window)
        print(
            f"Optuna CV round {rnd + 1}/{C.REFINE_MAX_ROUNDS}  "
            f"T={T_bounds}  alpha={a_bounds}  "
            f"(T_window={T_window:.6f}, alpha_window={alpha_window:.6f})"
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=C.SEED + rnd),
        )
        study.optimize(
            lambda trial: objective(trial, T_bounds, a_bounds),
            n_trials=C.OPTUNA_TRAIN_TRIALS,
            show_progress_bar=False,
        )

        round_values = [t.value for t in study.trials if t.value is not None]
        all_values.extend(round_values)
        round_boundaries.append(len(all_values))

        round_T     = study.best_params["T"]
        round_alpha = study.best_params["alpha"]
        round_score = study.best_value
        print(
            f"  round best: T={round_T:.4f}  alpha={round_alpha:.4f}  "
            f"mean fold r={round_score:.4f}"
        )

        if round_score > best_score:
            best_score  = round_score
            best_center = (round_T, round_alpha)
            print("  accepted round result")
        else:
            print("  kept previous best (round did not improve)")

        T_window     *= C.REFINE_SHRINK
        alpha_window *= C.REFINE_SHRINK
        if T_window <= C.REFINE_MIN_T_WINDOW and alpha_window <= C.REFINE_MIN_ALPHA_WINDOW:
            print(
                "  stopping Optuna CV refinement: "
                f"T window={T_window:.6f}, alpha window={alpha_window:.6f}"
            )
            break

    T_cv, alpha_cv = best_center
    print(f"\nOptuna CV best  : T={T_cv:.4f}  alpha={alpha_cv:.4f}  mean fold r={best_score:.4f}")
    print(f"Dual-anneal ref : T={center[0]:.4f}  alpha={center[1]:.4f}")

    if C.MAKE_OPTUNA_CV_PLOT and all_values:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        trial_idx = range(1, len(all_values) + 1)
        ax.plot(trial_idx, all_values, "o", color=C.BLUE, ms=3, alpha=0.5, label="trial")
        running_best = np.maximum.accumulate(all_values)
        ax.plot(trial_idx, running_best, color=C.RED, lw=2, label="running best")
        for b in round_boundaries[:-1]:
            ax.axvline(b + 0.5, color="gray", linestyle=":", lw=1)
        ax.set_xlabel("trial (across refinement rounds)")
        ax.set_ylabel("mean fold Pearson r")
        ax.set_title(
            f"Optuna CV search — {len(round_boundaries)} rounds, "
            f"{C.OPTUNA_TEST_FOLDS} folds/trial"
        )
        ax.legend(framealpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        _maybe_show_and_save(
            fig, C.RESULTS_DIR / "optuna_cv_convergence.png",
            C.MAKE_OPTUNA_CV_PLOT, C.SAVE_OPTUNA_CV_PLOT,
        )

    return T_cv, alpha_cv, float(best_score), all_values


# ═══════════════════════════════════════════════════════════════════════════
# NULL DISTRIBUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def shuffle_jij(J: np.ndarray) -> np.ndarray:
    """Symmetrically shuffle the off-diagonal values of *J*."""
    J_null = J.copy()
    idx    = np.triu_indices(J.shape[0], k=1)
    vals   = J_null[idx].copy()
    np.random.shuffle(vals)
    J_null[idx]            = vals
    J_null[idx[1], idx[0]] = vals
    return J_null


def pearson_threshold_jij(
    J:         np.ndarray,
    Rho:       np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply Pearson-sign correction then threshold *J* using *Rho*.

    Entries where |Rho_ij| >= threshold are sign-corrected; others keep
    their structural sign. Result is symmetrised.
    """
    Rho_t = Rho.copy()
    Rho_t[np.abs(Rho_t) < threshold] = 0.0

    J_t     = J.copy()
    offdiag = ~np.eye(J_t.shape[0], dtype=bool)
    keep    = offdiag & (Rho_t != 0.0)
    J_t[keep] = np.sign(Rho_t[keep]) * np.abs(J_t[keep])
    np.fill_diagonal(J_t, 0.0)
    return (J_t + J_t.T) / 2.0


def constant_jij(J: np.ndarray, value: float = C.CONSTANT_NULL_VALUE) -> np.ndarray:
    """Return a matrix filled with *value* (diagonal = 0)."""
    J_c = np.full_like(J, value, dtype=float)
    np.fill_diagonal(J_c, 0.0)
    return J_c


def run_ising_avg(
    J:            np.ndarray,
    T_global_val: float,
    alpha:        float,
    n_runs:       int = C.NULL_RUNS,
) -> np.ndarray:
    """
    Run *n_runs* Ising simulations with coupling matrix *J* and return the
    mean FC (Pearson or partial, per C.FC_MODE).

    Uses the absolute mean coupling for the temperature multiplier so that
    negative-sign or shuffled Jij matrices don't produce NaN temperatures
    with fractional alpha values. Nulls always use the Pearson mu-based
    multiplier (not PET), since shuffled/constant Jij have no meaningful
    PET pairing.
    """
    J  = np.asarray(J, dtype=float)
    mu = np.abs(np.mean(J, axis=0))
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    mu_max = np.max(mu)
    mu     = (mu / mu_max) if (np.isfinite(mu_max) and mu_max > 0) else np.ones(N)

    temp = T_global_val * (mu ** alpha)
    temp = np.nan_to_num(temp, nan=T_global_val, posinf=T_global_val, neginf=T_global_val)
    temp[temp <= 0] = 1e-12

    fc_sum = np.zeros((N, N), dtype=float)
    for _ in range(n_runs):
        sim = ISING_CLASS(temp, Jij=J)
        sim.simulate(C.NULL_STEPS, C.NULL_THERM)
        sim.generate_FC(partial=PARTIAL)
        fc_sum += np.nan_to_num(
            sim.functional_connectivity, nan=0.0, posinf=0.0, neginf=0.0
        )

    rho = fc_sum / n_runs
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    set_fc_diagonal(rho)
    return rho


# ═══════════════════════════════════════════════════════════════════════════
# EFFECT-SIZE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def cohens_d(null_vals: np.ndarray, real_val: float) -> float:
    """Cohen's d: (real − null_mean) / null_std."""
    null_vals = finite_vals(null_vals, "cohens_d")
    sd = null_vals.std(ddof=1)
    return 0.0 if (not np.isfinite(sd) or sd == 0) else float((real_val - null_vals.mean()) / sd)


def cliffs_delta(null_vals: np.ndarray, real_val: float) -> float:
    """Cliff's delta: (n_greater − n_less) / n_total."""
    null_vals = finite_vals(null_vals, "cliffs_delta")
    greater   = np.sum(null_vals > real_val)
    less      = np.sum(null_vals < real_val)
    return float((greater - less) / len(null_vals))


def cohens_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def cliffs_magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"


# ═══════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _add_ref_vlines(ax, ref_lines: list) -> None:
    """Add vertical reference lines from [(T, color, linestyle, label), ...]."""
    for T, color, ls, label in ref_lines:
        ax.axvline(T, color=color, linestyle=ls, lw=1.6, label=label)


def plot_null_histogram(
    ax:       plt.Axes,
    null_vals: np.ndarray,
    real_val:  float,
    p_val:     float,
    cd:        float,
    cld:       float,
    xlabel:    str,
    title:     str,
    xlim:      tuple | None = None,
) -> None:
    """
    Draw a null-distribution histogram with the real-model value marked
    and effect-size statistics annotated in the upper-right corner.
    """
    null_vals = finite_vals(null_vals, title)
    real_val  = float(np.nan_to_num(real_val, nan=0.0, posinf=0.0, neginf=0.0))

    counts, edges = np.histogram(null_vals, bins=C.BINS)
    widths        = np.diff(edges)

    for count, left, width in zip(counts, edges[:-1], widths):
        is_left = (left + width) <= real_val
        ax.bar(
            left, count, width=width, align="edge",
            color=C.REAL_COLOR if is_left else C.NULL_COLOR,
            alpha=0.40 if is_left else 0.80,
            edgecolor="white", linewidth=0.5,
        )

    ax.axvline(real_val, color=C.REAL_COLOR, linestyle="--", lw=2.2,
               label=f"real Jij  ({real_val:.4f})")

    ax.text(
        0.97, 0.95,
        (f"p = {p_val:.4f}\n"
         f"Cohen's d = {cd:.3f}  [{cohens_magnitude(cd)}]\n"
         f"Cliff's δ = {cld:.3f}  [{cliffs_magnitude(cld)}]"),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=C.REAL_COLOR, fontweight="medium", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C.REAL_COLOR, alpha=0.6),
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(title, fontsize=12)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


def overlay_null_step(
    ax:       plt.Axes,
    ref_vals: np.ndarray,
    label:    str = "random Jij null",
) -> None:
    """Overlay a reference null as a step histogram outline."""
    ax.hist(
        finite_vals(ref_vals, label),
        bins=C.BINS, histtype="step",
        color="black", linewidth=1.8, label=label,
    )
    ax.legend(fontsize=9, framealpha=0.3)


def combined_xlim(*arrays) -> tuple[float, float] | None:
    """Compute a shared x-axis limit with 3 % padding across all arrays."""
    vals = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    pad = 0.03 * max(float(np.ptp(vals)), 1e-12)
    return float(vals.min() - pad), float(vals.max() + pad)


def _maybe_show_and_save(fig, save_path: Path, make: bool, save: bool) -> None:
    """Save / show / close a figure according to CONFIG toggles."""
    if not make:
        return
    if save:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path.name}")
    if C.SHOW_PLOTS_INTERACTIVELY:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPERATURE SWEEP RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_temperature_sweep(
    label:    str,
    t_min:    float,
    t_max:    float,
    t_steps:  int,
    alpha:    float,
    save_dir: Path,
) -> ts.simulated_FC_vs_T_global:
    """
    Construct and run a simulated_FC_vs_T_global sweep, then return it.

    All simulation parameters are taken from CONFIG; FC mode (Pearson vs
    partial) follows the module-level PARTIAL flag.
    """
    print("\n" + "=" * 65)
    print(f"STEP 2 : TEMPERATURE SWEEP — {label}  (alpha = {alpha:.4f})")
    print(f"         T = {t_min:.3f} → {t_max:.3f}  |  steps = {t_steps}")
    print("=" * 65)

    sweep_obj = ts.simulated_FC_vs_T_global(
        min_temp  = t_min,
        max_temp  = t_max,
        temp_step = t_steps,
        alpha     = alpha,
        Jij       = J_real,
        ising     = ISING_CLASS,
        multiplier = multiplier,
        save       = True,
    )
    sweep_obj.simulate(
        steps          = C.SWEEP_STEPS,
        thermalization = C.SWEEP_THERM,
        partial        = PARTIAL,
        diag           = not C.ZERO_FC_DIAGONAL,
        text           = C.PRINT_PROGRESS_TEXT,
        n_repeats      = C.TEMP_REPEATS,
        emp_FC1        = emp_FC1,
        emp_FC2        = emp_FC2,
        emp_FC3        = emp_FC3,
        avg_FC         = rho_emp,
        path           = save_dir,
    )
    return sweep_obj


# ═══════════════════════════════════════════════════════════════════════════
# SimulationGraphData — diagnostic plots for a single completed simulation
# ═══════════════════════════════════════════════════════════════════════════

class SimulationGraphData:
    """
    Diagnostic plotting helper wrapping one completed Ising object.

    Parameters
    ----------
    ising : Ising
        Completed simulation (simulate() already called).
    beta, T_global, alpha : float
        Thermodynamic parameters for this run.
    emp_FC : ndarray
        Empirical FC used for correlation and ROC metrics.
    diag : bool
        Whether to include the diagonal in correlation calculations.
    save : bool
        Save figures to *save_dir* when True.
    save_dir : Path-like, optional
        Output directory. Defaults to C.RESULTS_DIR.
    """

    def __init__(
        self,
        ising:    I.Ising,
        beta:     float,
        T_global: float,
        alpha:    float,
        emp_FC:   np.ndarray,
        diag:     bool = False,
        save:     bool = False,
        save_dir: Path | None = None,
    ) -> None:
        self.ising       = ising
        self.FC          = ising.functional_connectivity
        self.emp_FC      = emp_FC
        self.Jij         = ising.spin.Jij
        self.beta        = beta
        self.T_global    = T_global
        self.alpha       = alpha
        self.partial     = ising.partial
        self.time        = ising.timer
        self.correlation = ising.correlation(emp_FC, diag)
        self.suscept     = ising.susceptibility(beta)
        self.spec_heat   = ising.specific_heat(beta)
        self.save        = save
        self.path        = Path(save_dir) if save_dir is not None else C.RESULTS_DIR

    def graph_mag_energy(self, show: bool = True):
        if not C.MAKE_MAG_ENERGY_PLOT:
            return
        steps = np.arange(self.ising.steps)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, data, label in [
            (axes[0], self.ising.mag_series,    "magnetisation"),
            (axes[1], self.ising.energy_series, "energy"),
        ]:
            ax.scatter(steps, data, s=2, alpha=0.4)
            ax.plot(steps, utils.average_series(data), "r", label="running mean")
            ax.set(xlabel="steps", ylabel=label)
            ax.legend()
        if self.save:
            fig.savefig(self.path / "energy_mag_graph.png", dpi=150, bbox_inches="tight")
        if show: plt.show()
        else:    return fig, axes

    def graph_ROC(self, show: bool = True):
        if not C.MAKE_ROC_PLOT:
            return
        FC_tpr, FC_fpr, FC_auc   = utils.receiver_operating_characteristic(self.FC, self.emp_FC)
        J_tpr,  J_fpr,  J_auc   = utils.receiver_operating_characteristic(self.FC, self.Jij)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(FC_fpr, FC_tpr, label=f"AUC = {FC_auc:.4f}")
        axes[0].set(title="ROC: sim FC vs emp FC", xlabel="FPR", ylabel="TPR")
        axes[0].legend()
        axes[1].plot(J_fpr,  J_tpr,  label=f"AUC = {J_auc:.4f}")
        axes[1].set(title="ROC: sim FC vs Jij",    xlabel="FPR", ylabel="TPR")
        axes[1].legend()
        if self.save:
            fig.savefig(self.path / "ROC_graphs.png", dpi=150, bbox_inches="tight")
        if show: plt.show()
        else:    return fig, axes

    def graph_FC(self, show: bool = True, title: str = "Simulated FC"):
        if not C.MAKE_FC_MATRIX_PLOT:
            return
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        axes[0].matshow(self.FC,     cmap="RdBu_r", norm=norm); axes[0].set_title(title)
        im = axes[1].matshow(self.emp_FC, cmap="RdBu_r", norm=norm)
        axes[1].set_title("Empirical FC")
        axes[2].matshow(self.Jij,    cmap="RdBu_r", norm=norm); axes[2].set_title("Jij")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        if self.save:
            fig.savefig(self.path / "matrix_graphs.png", dpi=150, bbox_inches="tight")
        if show: plt.show()
        else:    return fig, axes

    def graph_everything(self, show: bool = True):
        if not C.MAKE_EVERYTHING_PLOT:
            return
        FC_tpr,  FC_fpr,  _ = utils.receiver_operating_characteristic(self.FC, self.emp_FC)
        Jij_tpr, Jij_fpr, _ = utils.receiver_operating_characteristic(self.FC, self.Jij)
        steps = np.arange(self.ising.steps)
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        axes[0, 0].scatter(steps, self.ising.mag_series,    s=2, alpha=0.4)
        axes[0, 0].plot(steps, utils.average_series(self.ising.mag_series),    "r")
        axes[0, 0].set(xlabel="steps", ylabel="magnetisation", ylim=[0, 1])
        axes[0, 1].scatter(steps, self.ising.energy_series, s=2, alpha=0.4)
        axes[0, 1].plot(steps, utils.average_series(self.ising.energy_series), "r")
        axes[0, 1].set(xlabel="steps", ylabel="energy")
        axes[0, 2].matshow(self.FC);      axes[0, 2].set_title("Simulated FC")
        axes[1, 2].matshow(self.emp_FC);  axes[1, 2].set_title("Empirical FC")
        axes[1, 0].plot(FC_fpr,  FC_tpr);  axes[1, 0].set(title="ROC: sim FC vs emp FC", xlabel="FPR", ylabel="TPR")
        axes[1, 1].plot(Jij_fpr, Jij_tpr); axes[1, 1].set(title="ROC: sim FC vs Jij",    xlabel="FPR", ylabel="TPR")
        if self.save:
            fig.savefig(self.path / "everything_graphs.png", dpi=150, bbox_inches="tight")
        if show: plt.show()
        else:    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# simulated_KS_vs_T_global
# ═══════════════════════════════════════════════════════════════════════════

def _load_empirical_folder(subfolder: str) -> np.ndarray:
    """
    Load all subject time-series CSVs from TS_Data/<subfolder>/.

    Each file is (timepoints × regions); transposed to (regions × timepoints).
    Returned as an object array of arrays so different file lengths are supported.
    """
    folder    = C.TS_DATA_DIR / subfolder
    matrices  = [
        np.genfromtxt(f, delimiter=",").T
        for f in sorted(folder.glob("time_series_*.csv"))
    ]
    arr       = np.empty(len(matrices), dtype=object)
    arr[:]    = matrices
    return arr


def load_all_empirical() -> dict[str, np.ndarray]:
    """Return {subfolder: subject_array} for all three TS subfolders."""
    return {sub: _load_empirical_folder(sub) for sub in C.TS_SUBFOLDERS}


def _sign_binarize(ts: np.ndarray) -> np.ndarray:
    """Binarise a (regions × T) array to ±1 by sign."""
    out = np.ones_like(ts)
    out[ts < 0] = -1
    return out


def _flip_discrete(spin_ts: np.ndarray) -> np.ndarray:
    """0/1 flip indicator for a discrete {-1,+1} spin series (T-1 steps)."""
    return (spin_ts[:, 1:] != spin_ts[:, :-1]).astype(int)


def _flip_thresholded(ts: np.ndarray, threshold: float = C.FLIP_THRESHOLD) -> np.ndarray:
    """0/1 change indicator for a continuous series: 1 where |diff| > threshold."""
    return (np.abs(np.diff(ts, axis=1)) > threshold).astype(int)


def _ks_per_node(sim_ts: np.ndarray, emp_ts: np.ndarray) -> np.ndarray:
    """Per-node KS statistic between sim and empirical distributions."""
    n    = sim_ts.shape[0]
    vals = np.empty(n)
    for node in range(n):
        vals[node], _ = ks_2samp(sim_ts[node], emp_ts[node])
    return vals


def _ks_folder_average(
    sim_ts:          np.ndarray,
    subject_matrices: np.ndarray,
    transform=None,
) -> tuple[float, list[float]]:
    """Mean per-subject, per-node KS statistic across all subjects in a folder."""
    means = []
    for subj in subject_matrices:
        emp = transform(subj) if transform is not None else subj
        means.append(float(np.mean(_ks_per_node(sim_ts, emp))))
    return float(np.mean(means)), means


def _run_ks_suite(
    sim_ts:         np.ndarray,
    empirical_data: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """
    Run all three KS variants (raw / sign / flip) against every subfolder
    and their combined average.

    Returns nested dict: results[variant][subfolder_or_'TS_avg'].
    """
    results = {"raw": {}, "sign": {}, "flip": {}}
    sim_flip = _flip_discrete(sim_ts)

    all_raw, all_sign, all_flip = [], [], []
    for sub in C.TS_SUBFOLDERS:
        subj_mats = empirical_data[sub]

        m_raw,  s_raw  = _ks_folder_average(sim_ts,  subj_mats)
        m_sign, s_sign = _ks_folder_average(sim_ts,  subj_mats, transform=_sign_binarize)
        m_flip, s_flip = _ks_folder_average(sim_flip, subj_mats, transform=_flip_thresholded)

        results["raw"][sub]  = m_raw
        results["sign"][sub] = m_sign
        results["flip"][sub] = m_flip
        all_raw.extend(s_raw); all_sign.extend(s_sign); all_flip.extend(s_flip)

    results["raw"]["TS_avg"]  = float(np.mean(all_raw))
    results["sign"]["TS_avg"] = float(np.mean(all_sign))
    results["flip"]["TS_avg"] = float(np.mean(all_flip))
    return results


class simulated_KS_vs_T_global:
    """
    Sibling to simulated_FC_vs_T_global that replaces FC correlation with
    three KS-test metrics (raw / sign / flip) against empirical time-series.

    Parameters mirror simulated_FC_vs_T_global for consistency.
    """

    def __init__(
        self,
        min_temp:       float,
        max_temp:       float,
        ising:          type,
        temp_step:      int,
        alpha:          float,
        Jij:            np.ndarray,
        empirical_data: dict[str, np.ndarray],
        multiplier:     np.ndarray,
    ) -> None:
        self.T_global       = np.linspace(min_temp, max_temp, temp_step)
        self.alpha          = alpha
        self.multiplier     = multiplier
        self.Jij            = Jij
        self.ising          = ising
        self.empirical_data = empirical_data
        self.ising_ar:      list = []
        self.avg_temp_ar:   list[float] = []
        self.ks_results     = {
            v: {s: [] for s in C.TS_SUBFOLDERS + ["TS_avg"]}
            for v in ("raw", "sign", "flip")
        }

    def simulate(
        self,
        steps:          int,
        thermalization: int,
        spin_array:     np.ndarray,
        text:           bool = True,
    ) -> None:
        """Run the KS sweep across all temperatures."""
        for T in self.T_global:
            temp_ar = T * (self.multiplier ** self.alpha)
            temp_ar = np.nan_to_num(temp_ar, nan=T, posinf=T, neginf=T)
            temp_ar[temp_ar <= 0] = 1e-12

            sim = self.ising(temp_ar, Jij=self.Jij, spin_ar=spin_array)
            sim.simulate(steps, thermalization)
            sim_ts = sim.spin_series  # (N, steps)

            results = _run_ks_suite(sim_ts, self.empirical_data)
            for v in ("raw", "sign", "flip"):
                for sub in C.TS_SUBFOLDERS + ["TS_avg"]:
                    self.ks_results[v][sub].append(results[v][sub])

            self.avg_temp_ar.append(float(np.mean(temp_ar)))
            self.ising_ar.append(sim)

            if text:
                print(
                    f"T = {T:.3f}  avg_T = {np.mean(temp_ar):.3f}  "
                    f"raw={results['raw']['TS_avg']:.4f}  "
                    f"sign={results['sign']['TS_avg']:.4f}  "
                    f"flip={results['flip']['TS_avg']:.4f}"
                )

    def graph_ks(
        self,
        variant:        str = "raw",
        show:           bool = True,
        save_path:      Path | None = None,
        coupling_label: str = "Ising",
    ):
        """Plot KS statistic vs global temperature for one variant."""
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.set_box_aspect(1)
        for sub in C.TS_SUBFOLDERS + ["TS_avg"]:
            style = "--" if sub == "TS_avg" else "-"
            ax.plot(self.T_global, self.ks_results[variant][sub], style, label=sub)
        ax.set_xlabel("global temperature")
        ax.set_ylabel("KS statistic")
        ax.set_title(f"KS vs temperature — {variant} variant\n{coupling_label}")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# threshold_correlation_sweep
# ═══════════════════════════════════════════════════════════════════════════

class threshold_correlation_sweep:
    """
    Run N independent Ising simulations at T_crit and measure how Pearson r
    (sim FC vs emp FC) varies as both matrices are independently thresholded.

    Reproduces the "mean correlation vs threshold" figure from the slides.

    Parameters
    ----------
    ising_class : Ising subclass
    Jij, emp_FC : ndarray
        Structural coupling and empirical FC matrices.
    T_c : float
        Temperature to simulate at (typically T_crit).
    alpha, multiplier : float, ndarray
        Temperature fitting parameters.
    partial : bool
        Use partial correlation for the simulated FC.
    """

    def __init__(
        self,
        ising_class: type,
        Jij:         np.ndarray,
        emp_FC:      np.ndarray,
        T_c:         float,
        alpha:       float,
        multiplier:  np.ndarray,
        partial:     bool = False,
    ) -> None:
        self.ising_class = ising_class
        self.Jij         = Jij
        self.emp_FC      = emp_FC
        self.T_c         = T_c
        self.alpha       = alpha
        self.multiplier  = multiplier
        self.partial     = partial
        self.sim_FCs:    list[np.ndarray] = []

    def run(
        self,
        n_restarts:     int,
        steps:          int,
        thermalization: int,
        spin_array:     np.ndarray,
    ) -> None:
        """Run *n_restarts* simulations at T_c and cache the FC matrices."""
        temp_ar = self.T_c * (self.multiplier ** self.alpha)
        temp_ar = np.nan_to_num(temp_ar, nan=self.T_c, posinf=self.T_c, neginf=self.T_c)
        temp_ar[temp_ar <= 0] = 1e-12
        self.sim_FCs = []

        for _ in range(n_restarts):
            sim = self.ising_class(temp_ar, Jij=self.Jij, spin_ar=spin_array)
            sim.simulate(steps, thermalization)
            fc = np.nan_to_num(sim.generate_FC(self.partial), nan=0.0, posinf=0.0, neginf=0.0)
            self.sim_FCs.append(fc)

    def _thresholded_corr(
        self,
        sim_FC: np.ndarray,
        thresh: float,
        diag:   bool,
    ) -> float:
        """Pearson r after independently thresholding sim and emp FC."""
        def apply_thresh(m: np.ndarray) -> np.ndarray:
            out = m.copy()
            out[np.abs(out) < thresh] = 0.0
            return out

        s = apply_thresh(sim_FC)
        e = apply_thresh(self.emp_FC)
        if diag:
            np.fill_diagonal(s, 1.0)
            np.fill_diagonal(e, 1.0)
            x, y = s.flatten(), e.flatten()
        else:
            x = utils.flat_remove_diag(s)
            y = utils.flat_remove_diag(e)

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        r = pearsonr(x, y)[0]
        return 0.0 if not np.isfinite(r) else float(r)

    def sweep_thresholds(
        self,
        thresholds: np.ndarray,
        diag:       bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_r, std_r) arrays aligned with *thresholds*."""
        mean_r, std_r = [], []
        for t in thresholds:
            rs = [self._thresholded_corr(fc, t, diag) for fc in self.sim_FCs]
            mean_r.append(float(np.mean(rs)))
            std_r.append(float(np.std(rs)))
        return np.array(mean_r), np.array(std_r)

    def graph_combined(
        self,
        thresholds: np.ndarray,
        show:       bool = True,
        save_path:  Path | None = None,
    ):
        """Plot include- and exclude-diagonal curves on one axis."""
        excl_mean, excl_std = self.sweep_thresholds(thresholds, diag=False)
        incl_mean, incl_std = self.sweep_thresholds(thresholds, diag=True)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for mean, std, color, label in [
            (excl_mean, excl_std, C.BLUE, "exclude diagonal"),
            (incl_mean, incl_std, C.RED,  "include diagonal"),
        ]:
            ax.plot(thresholds, mean, color=color, lw=2.0, label=label)
            ax.fill_between(
                thresholds, mean - std, mean + std,
                color=color, alpha=0.22, linewidth=0,
            )
        ax.set_xlabel("threshold", fontsize=11)
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title("Correlation vs threshold at T_crit", fontsize=12)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            csv_path = Path(save_path).with_suffix(".csv")
            np.savetxt(
                csv_path,
                np.column_stack([thresholds, excl_mean, excl_std, incl_mean, incl_std]),
                delimiter=",",
                header="threshold,excl_mean_r,excl_std_r,incl_mean_r,incl_std_r",
                comments="",
            )
        if show:
            plt.show()
        else:
            return fig, ax


def _load_pearson_alpha_tcrit() -> tuple[float, float]:
    """Read alpha and T_crit from the summary CSV written by this script."""
    summary_path = C.PEARSON_SUMMARY_PATH if C.LOCAL_TEMPERATURE_MODEL == "pearson" else C.PET_SUMMARY_PATH
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run the pipeline first so the summary is available."
        )
    values = np.genfromtxt(summary_path, delimiter=",", names=True)
    return float(values["used_alpha"]), float(values["T_crit"])


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — full analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(C.SEED)

    # ── Prepare shared FC matrices ────────────────────────────────────────
    for mat in (rho_emp, emp_FC1, emp_FC2, emp_FC3):
        set_fc_diagonal(mat)

    rho_emp_vec = clean_vec(fc_compare_vec(rho_emp))

    print(f"J_real min           : {J_real.min():.4f}")
    print(f"J_real max           : {J_real.max():.4f}")
    print(f"J_real has negatives : {np.any(J_real < 0)}")
    print(f"emp FC neg fraction  : {np.mean(rho_emp_vec < 0):.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 0 — MU vs PET DIAGNOSTIC (independent of LOCAL_TEMPERATURE_MODEL)
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_MU_VS_PET_DIAGNOSTIC:
        print("\n" + "=" * 65)
        print("STEP 0 : MU vs PET DIAGNOSTIC")
        print("=" * 65)
        run_mu_vs_pet_diagnostic(J_real)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1 — PARAMETER ANNEALING
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_STEP1_ANNEALING:
        print("\n" + "=" * 65)
        print("STEP 1 : PARAMETER ANNEALING — broad search (T*, alpha*)")
        print("=" * 65)

        best_result, best_optim, best_fun = None, None, np.inf

        for restart in range(C.N_RESTARTS):
            np.random.seed(C.SEED + restart)
            print(f"\nRestart {restart + 1}/{C.N_RESTARTS}")

            optim = pa_module.optimize(
                ising       = ISING_CLASS,
                therm       = C.ANNEAL_THERM,
                steps       = C.ANNEAL_STEPS,
                train_trials= 1,   # not used by anneal(); placeholder
                temp_range  = list(C.ANNEAL_BOUNDS[0]),
                alpha_range = list(C.ANNEAL_BOUNDS[1]),
                partial     = PARTIAL,
                multiplier  = multiplier,
                save        = (restart == 0),
            )

            result = optim.anneal(
                steps           = C.ANNEAL_STEPS,
                maxfun          = C.ANNEAL_MAXFUN,
                emp_FC          = rho_emp,
                therm           = C.ANNEAL_THERM,
                bounds          = C.ANNEAL_BOUNDS,
                no_local_search = False,
                show            = False,
            )

            print(f"Restart best r = {max(optim.correlate):.4f}")
            if result.fun < best_fun:
                best_fun, best_result, best_optim = result.fun, result, optim

        result, optim = best_result, best_optim
        T_window, alpha_window = C.REFINE_T_WINDOW, C.REFINE_ALPHA_WINDOW

        # Iterative refinement
        for rnd in range(C.REFINE_MAX_ROUNDS):
            refined = refine_bounds(result.x, T_window, alpha_window)
            print("\n" + "=" * 65)
            print(
                f"STEP 1A.{rnd + 1} : REFINE  "
                f"T={refined[0]}  alpha={refined[1]}"
            )
            print("=" * 65)

            np.random.seed(C.SEED + C.N_RESTARTS + rnd)
            ref_optim = pa_module.optimize(
                ising       = ISING_CLASS,
                therm       = C.ANNEAL_THERM,
                steps       = C.ANNEAL_STEPS,
                train_trials= 1,
                temp_range  = list(C.ANNEAL_BOUNDS[0]),
                alpha_range = list(C.ANNEAL_BOUNDS[1]),
                partial     = PARTIAL,
                multiplier  = multiplier,
                save        = False,
            )
            ref_result = ref_optim.anneal(
                steps           = C.ANNEAL_STEPS,
                maxfun          = C.REFINE_MAXFUN,
                emp_FC          = rho_emp,
                therm           = C.ANNEAL_THERM,
                bounds          = refined,
                no_local_search = False,
                show            = False,
            )

            print(f"Refined best r = {max(ref_optim.correlate):.4f}")
            if ref_result.fun < best_fun:
                best_fun, result, optim = ref_result.fun, ref_result, ref_optim
                print("Accepted refined pair")
            else:
                print("Kept previous best pair")

            T_window     *= C.REFINE_SHRINK
            alpha_window *= C.REFINE_SHRINK
            if T_window <= C.REFINE_MIN_T_WINDOW and alpha_window <= C.REFINE_MIN_ALPHA_WINDOW:
                print("Stopping refinement: window too small")
                break

        T_star_annealed     = float(result.x[0])
        alpha_star_annealed = float(result.x[1])
        print(f"\nAnnealed T*     = {T_star_annealed:.4f}")
        print(f"Annealed alpha* = {alpha_star_annealed:.4f}")

        if C.MAKE_PARAM_ANNEAL_ERROR_PLOT:
            fig_ann, _ = optim.plot_error(show=False)
            _maybe_show_and_save(
                fig_ann,
                C.RESULTS_DIR / "param_anneal_error_3.png",
                C.MAKE_PARAM_ANNEAL_ERROR_PLOT,
                C.SAVE_PARAM_ANNEAL_ERROR_PLOT,
            )
    else:
        T_star_annealed     = C.FIXED_ALPHA
        alpha_star_annealed = C.FIXED_ALPHA

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1B — OPTUNA CROSS-VALIDATED SEARCH (refines Step 1's result)
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_STEP1B_OPTUNA_CV:
        print("\n" + "=" * 65)
        print("STEP 1B : OPTUNA CROSS-VALIDATED SEARCH (around annealed optimum)")
        print("=" * 65)
        T_cv, alpha_cv, cv_score, _cv_trials = run_optuna_cv(
            Jij          = J_real,
            multiplier   = multiplier,
            emp_FC_vec   = rho_emp_vec,
            center       = (T_star_annealed, alpha_star_annealed),
            T_window     = C.REFINE_T_WINDOW,
            alpha_window = C.REFINE_ALPHA_WINDOW,
        )
        if C.PREFER_OPTUNA_CV_RESULT:
            print("Using Optuna CV result in place of the dual-annealing result.")
            T_star_annealed, alpha_star_annealed = T_cv, alpha_cv
    else:
        T_cv = alpha_cv = cv_score = None

    alpha_star = C.FIXED_ALPHA if C.USE_FIXED_ALPHA else alpha_star_annealed
    print(f"\nUsing alpha* = {alpha_star:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2 — TEMPERATURE SWEEP
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_STEP2_TEMP_SWEEP:
        sweep = run_temperature_sweep(
            label    = "single sweep",
            t_min    = C.T_MIN,
            t_max    = C.T_MAX,
            t_steps  = C.T_STEPS,
            alpha    = alpha_star,
            save_dir = C.TEMP_SWEEP_DIR,
        )

        T_global      = sweep.T_global
        corr_arr      = np.array(sweep.corr_ar_total)
        spec_heat_arr = np.array(sweep.spec_heat_ar)
        suscept_arr   = np.array(sweep.suscept_ar)

        n_nan = int(np.sum(np.isnan(corr_arr)))
        print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")

        T_suscept_peak   = T_global[stable_peak_index(suscept_arr)]
        T_spec_heat_peak = T_global[stable_peak_index(spec_heat_arr)]
        crit_idx         = stable_peak_index(spec_heat_arr)
        best_idx         = int(np.nanargmax(corr_arr))
        T_crit           = T_global[crit_idx]
        T_best           = T_global[best_idx]
        best_corr        = float(np.nanmax(corr_arr))

        # Patch sweep object so downstream steps stay consistent
        sweep.crit_temp  = T_crit
        sweep.best_temp  = T_best
        sweep.best_corr  = best_corr
        sweep.best_ising = sweep.ising_ar[best_idx]
        sweep.crit_ising = sweep.ising_ar[crit_idx]

        print(f"Susceptibility peak  : {T_suscept_peak:.4f}")
        print(f"Specific heat peak   : {T_spec_heat_peak:.4f}")
        print(f"T_crit               : {T_crit:.4f}")
        print(f"T_best               : {T_best:.4f}")
        print(f"Best Pearson r       : {best_corr:.4f}")

        # Optional: persist the raw best/crit Ising objects for later reuse
        if C.SAVE_ISING_OBJECTS:
            import pickle
            C.ISING_DATA_DIR.mkdir(parents=True, exist_ok=True)
            for tag, obj in (("best", sweep.best_ising), ("crit", sweep.crit_ising)):
                out_path = C.ISING_DATA_DIR / f"ising_{tag}_{C.RUN_FOLDER_NAME or 'run'}.pkl"
                with open(out_path, "wb") as fh:
                    pickle.dump(obj, fh)
                print(f"Saved: {out_path}")

        avg_energy    = np.array(sweep.avg_energy_ar)
        avg_energy_sd = np.array(sweep.avg_energy_sd_ar)
        avg_mag       = np.array(sweep.avg_mag_ar)
        avg_mag_sd    = np.array(sweep.avg_mag_sd_ar)
        suscept       = np.array(sweep.suscept_ar)
        suscept_sd    = np.array(sweep.suscept_sd_ar)
        spec_heat     = np.array(sweep.spec_heat_ar)
        spec_heat_sd  = np.array(sweep.spec_heat_sd_ar)

        # Reference vertical lines used in multiple figures
        ref_lines = [
            (T_suscept_peak,   C.RED,    "--", rf"$T_{{\chi\,peak}}$ = {T_suscept_peak:.2f}"),
            (T_spec_heat_peak, C.SD_BAND,"--", rf"$T_{{C\,peak}}$ = {T_spec_heat_peak:.2f}"),
            (T_best,           C.AMBER,  ":",  rf"$T_{{best}}$ = {T_best:.2f}"),
        ]

        # ── Figure 1: thermodynamic observables ───────────────────────
        if C.MAKE_TEMPERATURE_SWEEP_PLOT:
            fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
            fig1.suptitle(
                f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
                fontsize=14, fontweight="bold",
            )
            panels = [
                (axes1[0, 0], avg_energy,  avg_energy_sd,  r"$\langle E \rangle$",        "Energy vs T"),
                (axes1[0, 1], avg_mag,     avg_mag_sd,     r"$\langle |M| \rangle$",       "|Magnetisation| vs T"),
                (axes1[1, 1], suscept,     suscept_sd,     r"susceptibility $\chi$",        "Susceptibility vs T"),
                (axes1[1, 0], spec_heat,   spec_heat_sd,   r"specific heat $C$",            "Specific Heat vs T"),
            ]
            for ax, data, sd, ylabel, title in panels:
                d_plot, sd_plot = mean_and_sd_band(data, sd)
                ax.plot(T_global, d_plot, color=C.BLUE, lw=2.0)
                ax.fill_between(T_global, d_plot - sd_plot, d_plot + sd_plot,
                                color=C.SD_BAND, alpha=0.28, linewidth=0)
                _add_ref_vlines(ax, ref_lines)
                ax.set_xlabel("global temperature T", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.legend(fontsize=8, framealpha=0.3)
                ax.spines[["top", "right"]].set_visible(False)
            _maybe_show_and_save(
                fig1, C.RESULTS_DIR / "temperature_sweep_3.png",
                C.MAKE_TEMPERATURE_SWEEP_PLOT, C.SAVE_TEMPERATURE_SWEEP_PLOT,
            )

        # ── Figure 2: correlation vs T ────────────────────────────────
        if C.MAKE_CORRELATION_VS_T_PLOT:
            corr_total    = np.array(sweep.corr_ar_total)
            corr_total_sd = np.array(sweep.corr_sd_ar_total)
            c_plot, csd_plot = mean_and_sd_band(corr_total, corr_total_sd)

            fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)
            ax_corr.plot(T_global, c_plot, color=C.BLUE, lw=2.0, label="avg FC")
            ax_corr.fill_between(T_global, c_plot - csd_plot, c_plot + csd_plot,
                                 color=C.SD_BAND, alpha=0.28, linewidth=0, label="SD")
            ax_corr.axvline(T_crit, color=C.RED,   linestyle="--", lw=1.5,
                            label=f"T_crit = {T_crit:.2f}")
            ax_corr.axvline(T_best, color=C.AMBER, linestyle=":",  lw=1.5,
                            label=f"T_best = {T_best:.2f}")
            ax_corr.set_xlabel("global temperature T", fontsize=11)
            ax_corr.set_ylabel("Pearson r  (sim vs emp FC)", fontsize=11)
            ax_corr.set_title("Correlation vs Temperature", fontsize=12)
            ax_corr.legend(fontsize=9, framealpha=0.3)
            ax_corr.spines[["top", "right"]].set_visible(False)
            _maybe_show_and_save(
                fig_corr, C.RESULTS_DIR / "correlation_vs_T_3.png",
                C.MAKE_CORRELATION_VS_T_PLOT, C.SAVE_CORRELATION_VS_T_PLOT,
            )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3 — MATRIX COMPARISON at T_best
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_STEP3_MATRIX_COMPARISON:
        print("\n" + "=" * 65)
        print("STEP 3 : MATRIX COMPARISON  (T_best)")
        print("=" * 65)

        best_gd = sweep.best_ising
        sim_FC  = best_gd.FC.copy()
        Jij_mat = best_gd.Jij.copy()
        set_fc_diagonal(sim_FC)

        sim_FC_vec = clean_vec(fc_compare_vec(sim_FC))
        r_best     = safe_pearson(sim_FC_vec, rho_emp_vec)
        dist_best  = float(np.linalg.norm(sim_FC_vec - rho_emp_vec))
        diss_best  = 1.0 - r_best

        print(f"sim FC neg fraction : {np.mean(sim_FC_vec  < 0):.4f}")
        print(f"emp FC neg fraction : {np.mean(rho_emp_vec < 0):.4f}")
        print(f"r                   : {r_best:.4f}")
        print(f"Eucl. distance      : {dist_best:.4f}")
        print(f"Dissimilarity       : {diss_best:.4f}")

        # Colour limits
        fc_lim  = 0.5
        fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)
        j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
        j_lim   = max(float(np.percentile(np.abs(j_offdiag), 99)), 0.05)
        j_norm  = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)

        fc_label = "partial" if PARTIAL else "Pearson"

        # Matrix heatmaps
        if C.MAKE_MATRIX_COMPARISON_PLOT:
            fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            fig3.suptitle(
                f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}"
                f"  |  r={r_best:.4f}  |  threshold={C.THRESHOLD:g}",
                fontsize=13, fontweight="bold",
            )
            for ax, (mat, title, norm_used) in zip(axes3, [
                (sim_FC,  f"Simulated {fc_label} FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
                (rho_emp, f"Empirical {fc_label} FC",                                            fc_norm),
                (Jij_mat, r"Structural connectivity $J_{ij}$",                                  j_norm),
            ]):
                im = ax.matshow(mat, cmap="RdBu_r", norm=norm_used)
                ax.set_title(title, fontsize=11, pad=12)
                ax.set_xlabel("region", fontsize=9)
                ax.set_ylabel("region", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _maybe_show_and_save(
                fig3, C.RESULTS_DIR / "matrix_comparison_3.png",
                C.MAKE_MATRIX_COMPARISON_PLOT, C.SAVE_MATRIX_COMPARISON_PLOT,
            )

        # Scatter plot
        if C.MAKE_SCATTER_SIM_VS_EMP_PLOT:
            fig3s, ax3s = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax3s.scatter(rho_emp_vec, sim_FC_vec, s=2, alpha=0.3, color=C.BLUE, rasterized=True)
            m_fit, b_fit = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
            x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)
            ax3s.plot(x_line, m_fit * x_line + b_fit, color="black", lw=1.5, linestyle="--")
            ax3s.set_xlabel(f"empirical {fc_label} FC", fontsize=11)
            ax3s.set_ylabel(f"simulated {fc_label} FC", fontsize=11)
            ax3s.set_title(f"Sim vs Emp FC  (r = {r_best:.4f})", fontsize=12)
            ax3s.spines[["top", "right"]].set_visible(False)
            _maybe_show_and_save(
                fig3s, C.RESULTS_DIR / "scatter_sim_vs_emp_3.png",
                C.MAKE_SCATTER_SIM_VS_EMP_PLOT, C.SAVE_SCATTER_SIM_VS_EMP_PLOT,
            )

        # Post-critical matrix comparisons
        if C.MAKE_POST_CRIT_MATRIX_PLOT:
            post_idx = np.where(T_global > T_crit)[0]
            post_idx = post_idx[post_idx != best_idx]
            post_idx = evenly_spaced_indices(post_idx, C.N_POST_CRIT_MATRICES)

            if len(post_idx) > 0:
                fig3p, axes3p = plt.subplots(
                    len(post_idx), 3,
                    figsize=(15, 3.8 * len(post_idx)),
                    constrained_layout=True, squeeze=False,
                )
                fig3p.suptitle(
                    f"Post-critical matrices  |  T_crit={T_crit:.2f}  |  alpha={alpha_star:.2f}",
                    fontsize=13, fontweight="bold",
                )
                for row, idx in enumerate(post_idx):
                    T_here   = T_global[idx]
                    gd_here  = sweep.ising_ar[idx]
                    sim_here = gd_here.FC.copy()
                    set_fc_diagonal(sim_here)
                    r_here   = safe_pearson(clean_vec(fc_compare_vec(sim_here)), rho_emp_vec)
                    for ax, (mat, title, norm_used) in zip(axes3p[row], [
                        (sim_here, f"Sim FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
                        (rho_emp,  f"Empirical {fc_label} FC",                fc_norm),
                        (Jij_mat,  r"Structural $J_{ij}$",                   j_norm),
                    ]):
                        im = ax.matshow(mat, cmap="RdBu_r", norm=norm_used)
                        ax.set_title(title, fontsize=10, pad=10)
                        ax.set_xlabel("region", fontsize=8)
                        ax.set_ylabel("region", fontsize=8)
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _maybe_show_and_save(
                    fig3p, C.RESULTS_DIR / "matrix_comparisons_post_Tcrit_3.png",
                    C.MAKE_POST_CRIT_MATRIX_PLOT, C.SAVE_POST_CRIT_MATRIX_PLOT,
                )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4 — NULL DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_STEP4_NULL_DISTRIBUTION:
        print("\n" + "=" * 65)
        print(f"STEP 4 : NULL DISTRIBUTION  (N={C.N_NULL}, T_best={T_best:.3f})")
        print("=" * 65)

        J_null_example = pearson_threshold_jij(shuffle_jij(avg_Jij), rho_emp, C.THRESHOLD)
        J_ones         = pearson_threshold_jij(constant_jij(J_real),   rho_emp, C.THRESHOLD)

        # Null Jij matrix figure
        if C.MAKE_NULL_JIJ_MATRICES_PLOT:
            null_all = np.concatenate([
                J_real[~np.eye(N, dtype=bool)],
                J_null_example[~np.eye(N, dtype=bool)],
                J_ones[~np.eye(N, dtype=bool)],
            ])
            null_lim  = max(float(np.percentile(np.abs(null_all), 99)), 0.05)
            null_norm = TwoSlopeNorm(vmin=-null_lim, vcenter=0, vmax=null_lim)

            fig_nj, axes_nj = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            fig_nj.suptitle(
                f"Null Jij matrices  |  threshold = {C.THRESHOLD:g}",
                fontsize=13, fontweight="bold",
            )
            for ax, (mat, title) in zip(axes_nj, [
                (J_real,         "Real thresholded Jij"),
                (J_null_example, "Example shuffled Jij\n(thresholded)"),
                (J_ones,         f"Constant Jij = {C.CONSTANT_NULL_VALUE:g}\n(thresholded)"),
            ]):
                im = ax.matshow(mat, cmap="RdBu_r", norm=null_norm)
                ax.set_title(title, fontsize=11, pad=12)
                ax.set_xlabel("region", fontsize=9)
                ax.set_ylabel("region", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _maybe_show_and_save(
                fig_nj, C.RESULTS_DIR / "null_jij_matrices_3.png",
                C.MAKE_NULL_JIJ_MATRICES_PLOT, C.SAVE_NULL_JIJ_MATRICES_PLOT,
            )

        # Shuffled-Jij null loop
        null_dist, null_diss = [], []
        for i in range(C.N_NULL):
            J_null   = pearson_threshold_jij(shuffle_jij(avg_Jij), rho_emp, C.THRESHOLD)
            rho_null = run_ising_avg(J_null, T_best, alpha_star)
            vec_null = clean_vec(fc_compare_vec(rho_null))
            r_null   = safe_pearson(vec_null, rho_emp_vec)
            null_dist.append(float(np.linalg.norm(vec_null - rho_emp_vec)))
            null_diss.append(1.0 - r_null)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{C.N_NULL}  r={r_null:.4f}  dist={null_dist[-1]:.4f}")

        null_dist = finite_vals(null_dist, "null_dist")
        null_diss = finite_vals(null_diss, "null_diss")
        p_dist    = float(np.mean(null_dist <= dist_best))
        p_diss    = float(np.mean(null_diss <= diss_best))
        cd_dist   = cohens_d(null_dist,  dist_best)
        cd_diss   = cohens_d(null_diss,  diss_best)
        cld_dist  = cliffs_delta(null_dist, dist_best)
        cld_diss  = cliffs_delta(null_diss, diss_best)

        print(f"\nShuffled null — dist: real={dist_best:.4f}  null mean={null_dist.mean():.4f}  p={p_dist:.4f}")
        print(f"Shuffled null — diss: real={diss_best:.4f}  null mean={null_diss.mean():.4f}  p={p_diss:.4f}")

        if C.MAKE_NULL_DISTRIBUTION_PLOT:
            fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
            fig4.suptitle(
                f"Null distribution  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}  |  r={r_best:.4f}",
                fontsize=13, fontweight="bold",
            )
            plot_null_histogram(
                axes4[0], null_dist, dist_best, p_dist, cd_dist, cld_dist,
                xlabel=r"Euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
                title="null distribution — Euclidean distance",
            )
            plot_null_histogram(
                axes4[1], null_diss, diss_best, p_diss, cd_diss, cld_diss,
                xlabel="dissimilarity  (1 − r)",
                title="null distribution — dissimilarity",
            )
            _maybe_show_and_save(
                fig4, C.RESULTS_DIR / "ising_null_distributions_3.png",
                C.MAKE_NULL_DISTRIBUTION_PLOT, C.SAVE_NULL_DISTRIBUTION_PLOT,
            )

        # Constant-Jij null loop
        ones_dist, ones_diss = [], []
        if C.RUN_STEP4_ONES_NULL:
            print(f"\nConstant-Jij null  (value={C.CONSTANT_NULL_VALUE:g})")
            for i in range(C.N_NULL):
                rho_ones = run_ising_avg(J_ones, T_best, alpha_star)
                vec_ones = clean_vec(fc_compare_vec(rho_ones))
                r_ones   = safe_pearson(vec_ones, rho_emp_vec)
                ones_dist.append(float(np.linalg.norm(vec_ones - rho_emp_vec)))
                ones_diss.append(1.0 - r_ones)
                if (i + 1) % 10 == 0:
                    print(f"  ones {i+1}/{C.N_NULL}  r={r_ones:.4f}  dist={ones_dist[-1]:.4f}")

            ones_dist = finite_vals(ones_dist, "ones_dist")
            ones_diss = finite_vals(ones_diss, "ones_diss")
            p_ones_dist  = float(np.mean(ones_dist <= dist_best))
            p_ones_diss  = float(np.mean(ones_diss <= diss_best))
            cd_ones_dist = cohens_d(ones_dist,  dist_best)
            cd_ones_diss = cohens_d(ones_diss,  diss_best)
            cld_ones_dist = cliffs_delta(ones_dist, dist_best)
            cld_ones_diss = cliffs_delta(ones_diss, diss_best)

            print(f"\nOnes null — dist: p={p_ones_dist:.4f}")
            print(f"Ones null — diss: p={p_ones_diss:.4f}")

            if C.MAKE_ONES_NULL_DISTRIBUTION_PLOT:
                dist_xlim = combined_xlim(null_dist, ones_dist, dist_best)
                diss_xlim = combined_xlim(null_diss, ones_diss, diss_best)
                fig4o, axes4o = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
                fig4o.suptitle(
                    f"Constant Jij null  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}",
                    fontsize=13, fontweight="bold",
                )
                plot_null_histogram(
                    axes4o[0], ones_dist, dist_best, p_ones_dist,
                    cd_ones_dist, cld_ones_dist,
                    xlabel=r"Euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
                    title="ones Jij null — Euclidean distance", xlim=dist_xlim,
                )
                overlay_null_step(axes4o[0], null_dist)
                plot_null_histogram(
                    axes4o[1], ones_diss, diss_best, p_ones_diss,
                    cd_ones_diss, cld_ones_diss,
                    xlabel="dissimilarity  (1 − r)",
                    title="ones Jij null — dissimilarity", xlim=diss_xlim,
                )
                overlay_null_step(axes4o[1], null_diss)
                _maybe_show_and_save(
                    fig4o, C.RESULTS_DIR / "ising_null_distributions_ones_3.png",
                    C.MAKE_ONES_NULL_DISTRIBUTION_PLOT, C.SAVE_ONES_NULL_DISTRIBUTION_PLOT,
                )
        else:
            p_ones_dist = p_ones_diss = np.nan
            cd_ones_dist = cd_ones_diss = cld_ones_dist = cld_ones_diss = np.nan
            ones_dist = ones_diss = np.array([np.nan])

    # ═══════════════════════════════════════════════════════════════════
    # OPTIONAL — KS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_KS_ANALYSIS:
        empirical_data = load_all_empirical()
        for filename_label, coupling_matrix, plot_label in [
            ("jij", J_real, "Pearson structural Jij-driven Ising"),
        ] + ([("functional", I.avg_FC, "functional FC-driven Ising")] if C.RUN_FUNCTIONAL_COMPARISON else []):
            mu_loc = utils.normalize_array(np.mean(np.abs(coupling_matrix), axis=0))
            ks_sweep = simulated_KS_vs_T_global(
                min_temp       = C.T_MIN,
                max_temp       = C.T_MAX,
                ising          = ISING_CLASS,
                temp_step      = C.T_STEPS,
                alpha          = alpha_star,
                Jij            = coupling_matrix,
                empirical_data = empirical_data,
                multiplier     = mu_loc,
            )
            ks_sweep.simulate(
                steps          = C.SWEEP_STEPS,
                thermalization = C.SWEEP_THERM,
                spin_array     = np.ones(coupling_matrix.shape[0]),
                text           = C.PRINT_PROGRESS_TEXT,
            )
            if C.MAKE_KS_SWEEP_PLOTS:
                for variant in ("raw", "sign", "flip"):
                    out = C.RESULTS_DIR / f"ks_sweep_{filename_label}_{variant}.png"
                    fig_ks, _ = ks_sweep.graph_ks(
                        variant=variant, show=False,
                        save_path=out if C.SAVE_KS_SWEEP_PLOTS else None,
                        coupling_label=plot_label,
                    )
                    plt.close(fig_ks)

    # ═══════════════════════════════════════════════════════════════════
    # OPTIONAL — THRESHOLD CORRELATION SWEEP
    # ═══════════════════════════════════════════════════════════════════
    if C.RUN_THRESHOLD_CORRELATION_SWEEP:
        thresh_sweep = threshold_correlation_sweep(
            ising_class = ISING_CLASS,
            Jij         = J_real,
            emp_FC      = rho_emp,
            T_c         = T_crit,
            alpha       = alpha_star,
            multiplier  = multiplier,
            partial     = PARTIAL,
        )
        thresh_sweep.run(
            n_restarts     = C.THRESH_SWEEP_N_RESTARTS,
            steps          = C.THRESH_SWEEP_STEPS,
            thermalization = C.THRESH_SWEEP_THERM,
            spin_array     = np.ones(J_real.shape[0]),
        )
        if C.MAKE_THRESHOLD_CORRELATION_PLOT:
            thresholds = np.arange(
                C.THRESH_SWEEP_VALUES_START,
                C.THRESH_SWEEP_VALUES_STOP,
                C.THRESH_SWEEP_VALUES_STEP,
            )
            save_path = (
                C.RESULTS_DIR / "threshold_corr_combined_diag_vs_no_diag.png"
                if C.SAVE_THRESHOLD_CORRELATION_PLOT else None
            )
            fig_tc, _ = thresh_sweep.graph_combined(thresholds, show=False, save_path=save_path)
            if C.SAVE_THRESHOLD_CORRELATION_PLOT:
                print("Saved: threshold_corr_combined_diag_vs_no_diag.png")
            plt.close(fig_tc)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Local temp model    = {C.LOCAL_TEMPERATURE_MODEL}")
    print(f"FC mode             = {'partial' if PARTIAL else 'pearson'}")
    print(f"Annealed T*         = {T_star_annealed:.4f}")
    print(f"Annealed alpha*     = {alpha_star_annealed:.4f}")
    if C.RUN_STEP1B_OPTUNA_CV:
        print(f"Optuna CV T*        = {T_cv:.4f}")
        print(f"Optuna CV alpha*    = {alpha_cv:.4f}")
        print(f"Optuna CV score     = {cv_score:.4f}")
    print(f"Used alpha          = {alpha_star:.4f}")
    print(f"T_crit              = {T_crit:.4f}")
    print(f"T_best              = {T_best:.4f}")
    print(f"Best r              = {r_best:.4f}")
    print(f"Eucl. distance      = {dist_best:.4f}")
    print(f"Dissimilarity       = {diss_best:.4f}")
    print(f"p (dist)            = {p_dist:.4f}")
    print(f"p (diss)            = {p_diss:.4f}")
    print(f"Cohen's d  (dist)   = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
    print(f"Cohen's d  (diss)   = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
    print(f"Cliff's δ  (dist)   = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
    print(f"Cliff's δ  (diss)   = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")

    if C.SAVE_RUN_SUMMARY:
        summary_path = (
            C.PEARSON_SUMMARY_PATH if C.LOCAL_TEMPERATURE_MODEL == "pearson"
            else C.PET_SUMMARY_PATH
        )

        summary = {
            "seed":                      C.SEED,
            "spin_flip_method":          SPIN_FLIP_METHOD,
            "use_random_spin_flip":      C.USE_RANDOM_SPIN_FLIP,
            "n_regions":                 N,
            "local_temperature_model":   C.LOCAL_TEMPERATURE_MODEL,
            "fc_mode":                   C.FC_MODE,
            "partial":                   PARTIAL,
            "jij_threshold":             C.THRESHOLD,
            "zero_fc_diagonal":          C.ZERO_FC_DIAGONAL,
            "used_fixed_alpha":          C.USE_FIXED_ALPHA,
            "anneal_steps":              C.ANNEAL_STEPS,
            "anneal_thermalization":     C.ANNEAL_THERM,
            "anneal_maxfun":             C.ANNEAL_MAXFUN,
            "anneal_restarts":           C.N_RESTARTS,
            "anneal_T_min":              C.ANNEAL_BOUNDS[0][0],
            "anneal_T_max":              C.ANNEAL_BOUNDS[0][1],
            "anneal_alpha_min":          C.ANNEAL_BOUNDS[1][0],
            "anneal_alpha_max":          C.ANNEAL_BOUNDS[1][1],
            "refine_max_rounds":         C.REFINE_MAX_ROUNDS,
            "refine_shrink":             C.REFINE_SHRINK,
            "annealed_T":                T_star_annealed,
            "annealed_alpha":            alpha_star_annealed,
            "run_optuna_cv":             C.RUN_STEP1B_OPTUNA_CV,
            "optuna_train_trials":       C.OPTUNA_TRAIN_TRIALS,
            "optuna_test_folds":         C.OPTUNA_TEST_FOLDS,
            "optuna_T_cv":               T_cv,
            "optuna_alpha_cv":           alpha_cv,
            "optuna_cv_score":           cv_score,
            "optuna_result_preferred":   C.PREFER_OPTUNA_CV_RESULT,
            "used_alpha":                alpha_star,
            "final_sweep_T_min":         float(np.nanmin(T_global)),
            "final_sweep_T_max":         float(np.nanmax(T_global)),
            "final_sweep_T_steps":       len(T_global),
            "sweep_steps":               C.SWEEP_STEPS,
            "sweep_thermalization":      C.SWEEP_THERM,
            "temperature_repeats":       C.TEMP_REPEATS,
            "T_susceptibility_peak":     T_suscept_peak,
            "T_specific_heat_peak":      T_spec_heat_peak,
            "T_crit":                    T_crit,
            "T_best":                    T_best,
            "best_r":                    r_best,
            "best_distance":             dist_best,
            "best_dissimilarity":        diss_best,
            "n_null":                    C.N_NULL,
            "null_runs_per_matrix":      C.NULL_RUNS,
            "null_steps":                C.NULL_STEPS,
            "null_thermalization":       C.NULL_THERM,
            "null_dist_mean":            float(null_dist.mean()),
            "null_diss_mean":            float(null_diss.mean()),
            "p_distance":                p_dist,
            "p_dissimilarity":           p_diss,
            "cohens_d_distance":         cd_dist,
            "cohens_d_dissimilarity":    cd_diss,
            "cliffs_delta_distance":     cld_dist,
            "cliffs_delta_dissimilarity":cld_diss,
            "ones_dist_mean":            float(np.nanmean(ones_dist)),
            "ones_diss_mean":            float(np.nanmean(ones_diss)),
            "p_ones_distance":           p_ones_dist,
            "p_ones_dissimilarity":      p_ones_diss,
            "cohens_d_ones_distance":    cd_ones_dist,
            "cohens_d_ones_dissimilarity":cd_ones_diss,
            "cliffs_delta_ones_distance": cld_ones_dist,
            "cliffs_delta_ones_dissimilarity":cld_ones_diss,
        }

        header = ",".join(summary.keys()) + "\n"
        values = ",".join(str(v) for v in summary.values()) + "\n"
        summary_path.write_text(header + values)
        print(f"\nSaved summary → {summary_path}")

