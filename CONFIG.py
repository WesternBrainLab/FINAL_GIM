"""
CONFIG.py — universal, shared configuration for the FINAL_GIM analysis scripts
(pearson_2 / PET hypothesis / threshold_correlation_sweep / ks_analysis, etc.)

HOW TO USE
----------
Put this file in FINAL_GIM/ (next to your analysis scripts) and add, near the
top of each script, right after the sys.path setup:

    import CONFIG as C

Then replace the script's local constants with `C.SOMETHING`, and wrap each
figure/save block in `if C.MAKE_<X>:` and `if C.SAVE_<X>:` (see the
"HOW TO WRAP EXISTING CODE" section at the bottom of this file for the exact
pattern used in your scripts).

Nothing in this file runs simulations itself — it is pure configuration, so
it's safe to import from every script without side effects.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 0. WHICH LOCAL-TEMPERATURE MODEL / PIPELINE VARIANT TO RUN
# ═══════════════════════════════════════════════════════════════════════════
# "pearson"     -> mu-based multiplier, sign-corrected Pearson Jij (pearson_2.py)
# "pet"         -> multiplier = PET_i ** alpha
# "inverse_pet" -> multiplier = (1 / PET_i) ** alpha
LOCAL_TEMPERATURE_MODEL = "pearson"   # "pearson" | "pet" | "inverse_pet"

# Used only when LOCAL_TEMPERATURE_MODEL in {"pet", "inverse_pet"}; kept here
# so scripts don't need their own separate USE_INVERSE_PET flag.
USE_INVERSE_PET = (LOCAL_TEMPERATURE_MODEL == "inverse_pet")

# ═══════════════════════════════════════════════════════════════════════════
# 1. PATHS
# ═══════════════════════════════════════════════════════════════════════════
# Set this to the FINAL_GIM directory itself. Every script does:
#     PROJECT_DIR = Path(__file__).resolve().parent
# so this normally does not need to change if CONFIG.py lives there too.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"

# Change this before each new analysis run so results don't overwrite each
# other. Use a Month-Day-Year_Time style name, e.g. "08-09-2026_1300".
RUN_FOLDER_NAME = "08-09-2026_1300"
RESULTS_ROOT = PROJECT_DIR / "RESULTS"
RESULTS_DIR = RESULTS_ROOT / RUN_FOLDER_NAME

JIJ_NEW_DIR = DATA_DIR / "Jij_new_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_Jij_new_pearson.csv"
PET_NO_OUTLIERS_PATH = DATA_DIR / "PET_data" / "PET_temp_no_outliers"

FC1_PATH = DATA_DIR / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_DIR / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_DIR / "FC data_processed" / "avg_TS_3"

SIMULATION_DIR = DATA_DIR / "simulation data"
OPTIMIZATION_DIR = SIMULATION_DIR / "optimization data"
TEMP_SWEEP_DIR = SIMULATION_DIR / "temp sweep data"

PEARSON_SUMMARY_PATH = RESULTS_DIR / "pearson_2_summary.csv"
PET_SUMMARY_PATH = RESULTS_DIR / "PET_Hypothesis_summary.csv"

TS_DATA_DIR = DATA_DIR / "TS_Data"
TS_SUBFOLDERS = ["TS_1", "TS_2", "TS_3"]


def ensure_results_dir():
    """Call this once near the top of a script's __main__ block."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


# ═══════════════════════════════════════════════════════════════════════════
# 2. CORE SIMULATION / DATA-HANDLING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
SEED = 1
THRESHOLD = 0.0          # Jij build/threshold value (was 0, 0.02, 0.03)
ZERO_FC_DIAGONAL = True  # True: compare off-diagonal FC only. False: diag=1, include it.

# False: visit every spin in fixed mean-Jij order (Jij_sorted_ising)
# True:  randomly sample spin-flip proposals each simulation step (random_ising)
USE_RANDOM_SPIN_FLIP = False

# ── Annealing (parameter search for T*, alpha*) ──────────────────────────
ANNEAL_STEPS = 1000
ANNEAL_MAXFUN = 500
ANNEAL_THERM = 2000
ANNEAL_BOUNDS = ((0.1, 10), (-3, 3))   # (T range, alpha range) — widen alpha for PET runs, e.g. (-6, 3)
N_RESTARTS = 1

REFINE_T_WINDOW = 1.0
REFINE_ALPHA_WINDOW = 0.5
REFINE_MAXFUN = 100
REFINE_MAX_ROUNDS = 2
REFINE_SHRINK = 0.5
REFINE_MIN_T_WINDOW = 0.001
REFINE_MIN_ALPHA_WINDOW = 0.0005

# ── Temperature sweep ─────────────────────────────────────────────────────
T_MIN = 0.5
T_MAX = 10
T_STEPS = 150
SWEEP_STEPS = 1000
SWEEP_THERM = 3000
TEMP_REPEATS = 2   # increase for a more stable susceptibility peak

PEAK_IGNORE_EDGE_POINTS = 3
PEAK_PROMINENCE_FRACTION = 0.15
SMOOTH_TEMPERATURE_PLOTS = True
SMOOTH_WINDOW = 31
SMOOTH_POLYORDER = 3

# Optional: skip annealing entirely and force a fixed alpha.
USE_FIXED_ALPHA = False
FIXED_ALPHA = 0

# ── Null distributions ────────────────────────────────────────────────────
N_NULL = 100
NULL_RUNS = 2
NULL_STEPS = 1000
NULL_THERM = 1000
CONSTANT_NULL_VALUE = 1.0
BINS = 50
N_POST_CRIT_MATRICES = 5

# ── Threshold-correlation sweep (threshold_correlation_sweep.py) ─────────
THRESH_SWEEP_N_RESTARTS = 5
THRESH_SWEEP_STEPS = 3000
THRESH_SWEEP_THERM = 1000
THRESH_SWEEP_VALUES_START = 0.1
THRESH_SWEEP_VALUES_STOP = 1.0     # exclusive, per np.arange
THRESH_SWEEP_VALUES_STEP = 0.1

# ── KS analysis (ks_analysis.py) ──────────────────────────────────────────
FLIP_THRESHOLD = 0.5
RUN_FUNCTIONAL_COMPARISON = True   # also run the functional-FC-coupled Ising alongside Jij

# ── Plot colors, shared across all scripts ────────────────────────────────
BLUE = "#2E86AB"
SD_BAND = "#2CA25F"
RED = "#E84855"
AMBER = "#F4A261"
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"


# ═══════════════════════════════════════════════════════════════════════════
# 3. RUN TOGGLES — turn pipeline stages on/off
# ═══════════════════════════════════════════════════════════════════════════
RUN_STEP1_ANNEALING = True
RUN_STEP2_TEMP_SWEEP = True
RUN_STEP3_MATRIX_COMPARISON = True
RUN_STEP4_NULL_DISTRIBUTION = True
RUN_STEP4_ONES_NULL = True         # the constant/"ones" Jij null, separate from the shuffled-Jij null
RUN_MU_VS_PET_DIAGNOSTIC = True    # only meaningful for the PET/inverse_pet pipeline
RUN_KS_ANALYSIS = False
RUN_THRESHOLD_CORRELATION_SWEEP = False


# ═══════════════════════════════════════════════════════════════════════════
# 4. GRAPH TOGGLES — turn individual figures on/off (MAKE_) and whether each
#    one gets written to disk (SAVE_). MAKE_=False skips the figure entirely
#    (fastest). MAKE_=True, SAVE_=False builds it but doesn't write a file
#    (e.g. for interactive/show=True use).
# ═══════════════════════════════════════════════════════════════════════════
MAKE_PARAM_ANNEAL_ERROR_PLOT = True
SAVE_PARAM_ANNEAL_ERROR_PLOT = True

MAKE_MU_VS_PET_PLOT = True
SAVE_MU_VS_PET_PLOT = True

MAKE_TEMPERATURE_SWEEP_PLOT = True     # 2x2: energy, |M|, susceptibility, specific heat
SAVE_TEMPERATURE_SWEEP_PLOT = True

MAKE_CORRELATION_VS_T_PLOT = True
SAVE_CORRELATION_VS_T_PLOT = True

MAKE_MATRIX_COMPARISON_PLOT = True     # sim FC / emp FC / Jij at T_best
SAVE_MATRIX_COMPARISON_PLOT = True

MAKE_SCATTER_SIM_VS_EMP_PLOT = True
SAVE_SCATTER_SIM_VS_EMP_PLOT = True

MAKE_POST_CRIT_MATRIX_PLOT = True      # matrix comparisons at several T > T_crit
SAVE_POST_CRIT_MATRIX_PLOT = True

MAKE_NULL_JIJ_MATRICES_PLOT = True     # real / shuffled / constant Jij side by side
SAVE_NULL_JIJ_MATRICES_PLOT = True

MAKE_NULL_DISTRIBUTION_PLOT = True     # shuffled-Jij null: distance + dissimilarity histograms
SAVE_NULL_DISTRIBUTION_PLOT = True

MAKE_ONES_NULL_DISTRIBUTION_PLOT = True
SAVE_ONES_NULL_DISTRIBUTION_PLOT = True

MAKE_KS_SWEEP_PLOTS = True             # raw / sign / flip KS-vs-temperature plots
SAVE_KS_SWEEP_PLOTS = True

MAKE_THRESHOLD_CORRELATION_PLOT = True
SAVE_THRESHOLD_CORRELATION_PLOT = True

# Diagnostic single-simulation plots (SimulationGraphData helper class)
MAKE_MAG_ENERGY_PLOT = False
MAKE_ROC_PLOT = False
MAKE_FC_MATRIX_PLOT = False
MAKE_EVERYTHING_PLOT = False           # combined 6-panel diagnostic figure

# Whether to also pop up interactive windows (plt.show()) in addition to
# saving. Leave False for unattended/batch runs.
SHOW_PLOTS_INTERACTIVELY = False

# Whether to write the run's CSV summary (pearson_2_summary.csv /
# PET_Hypothesis_summary.csv) at the end.
SAVE_RUN_SUMMARY = True

# Whether to print progress/diagnostic text to the console during sweeps.
PRINT_PROGRESS_TEXT = True


# ═══════════════════════════════════════════════════════════════════════════
# HOW TO WRAP EXISTING CODE
# ═══════════════════════════════════════════════════════════════════════════
# In each script, replace hardcoded constants with C.NAME, e.g.:
#     THRESHOLD = 0.0                  ->  (delete; use C.THRESHOLD directly)
#     T_MIN, T_MAX, T_STEPS = 0.5, 10, 150   ->  C.T_MIN, C.T_MAX, C.T_STEPS
#
# Wrap each figure block like this:
#
#     if C.MAKE_TEMPERATURE_SWEEP_PLOT:
#         fig1, axes1 = plt.subplots(2, 2, ...)
#         ...
#         if C.SAVE_TEMPERATURE_SWEEP_PLOT:
#             plt.savefig(C.RESULTS_DIR / "temperature_sweep_3.png", dpi=150, bbox_inches="tight")
#         if C.SHOW_PLOTS_INTERACTIVELY:
#             plt.show()
#         plt.close(fig1)
#
# Wrap each pipeline stage like this:
#
#     if C.RUN_STEP4_NULL_DISTRIBUTION:
#         ... null distribution code ...
#         if C.RUN_STEP4_ONES_NULL:
#             ... ones-null code ...
#
# For LOCAL_TEMPERATURE_MODEL, replace each script's own USE_INVERSE_PET /
# multiplier-selection logic with a single check against C.LOCAL_TEMPERATURE_MODEL
# so pearson_2.py, the PET-hypothesis script, and any future variant all read
# the same switch instead of three separate copies of that logic.