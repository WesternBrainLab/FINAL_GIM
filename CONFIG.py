"""
CONFIG.py — Universal configuration for the FINAL_GIM


Import in every other script by using:
   import CONFIG as C


Then reference constants as C.SEED, C.T_MIN, etc.
Wrap figure blocks with C.MAKE_<X> / C.SAVE_<X> and pipeline stages
with C.RUN_<STEP>.


Nothing in this file runs simulations or loads data — it is pure
configuration and is safe to import without side effects.
"""


from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# 0. LOCAL-TEMPERATURE MODEL
# ═══════════════════════════════════════════════════════════════════════════
# "pearson"     → multiplier = normalize(|mean(Jij)|)  — mu-based scaling
# "pet"         → multiplier = PET_i ** alpha
# "inverse_pet" → multiplier = (1 / PET_i) ** alpha
LOCAL_TEMPERATURE_MODEL: str = "pearson"   # "pearson" | "pet" | "inverse_pet"


# Derived automatically — do not set manually.
USE_INVERSE_PET: bool = (LOCAL_TEMPERATURE_MODEL == "inverse_pet")


# ═══════════════════════════════════════════════════════════════════════════
# 1. PATHS
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_DIR  = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_DIR / "DATA"


# ── Results ───────────────────────────────────────────────────────────────
# All final artifacts are written directly into RESULTS and overwrite the
# prior run. Run metadata and the optimization split are saved as JSON.
RUN_FOLDER_NAME: str = ""
RESULTS_ROOT = PROJECT_DIR / "RESULTS"
RESULTS_DIR  = RESULTS_ROOT


# ── Data sub-paths ────────────────────────────────────────────────────────
JIJ_NEW_DIR       = DATA_DIR / "thresholded_Jij_pearson"
AVG_JIJ_NEW_PATH  = JIJ_NEW_DIR / "avg_thresholded_Jij_pearson.csv"
PET_NO_OUTLIERS_PATH = DATA_DIR / "PET_data" / "PET_temp_no_outliers"


FC1_PATH = DATA_DIR / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_DIR / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_DIR / "FC data_processed" / "avg_TS_3"
PARTIAL_FC1_PATH = DATA_DIR / "FC data_processed" / "avg_TS_1p"
PARTIAL_FC2_PATH = DATA_DIR / "FC data_processed" / "avg_TS_2p"
PARTIAL_FC3_PATH = DATA_DIR / "FC data_processed" / "avg_TS_3p"


JIJ_PROCESSED_PATH = DATA_DIR / "Jij data_processed" / "avg_Jij_no_outliers_norm"


# ── Simulation intermediate data ──────────────────────────────────────────
SIMULATION_DIR    = DATA_DIR / "simulation data"
OPTIMIZATION_DIR  = SIMULATION_DIR / "optimization data"
TEMP_SWEEP_DIR    = SIMULATION_DIR / "temp sweep data"
ISING_DATA_DIR    = SIMULATION_DIR / "ising data"


# ── Summary CSVs (written by GET_RESULTS.py) ─────────────────────────────
PEARSON_SUMMARY_PATH = RESULTS_DIR / "pearson_2_summary.csv"
PET_SUMMARY_PATH     = RESULTS_DIR / "PET_Hypothesis_summary.csv"


# ── Time-series data (for KS analysis) ───────────────────────────────────
TS_DATA_DIR   = DATA_DIR / "TS_Data"
TS_SUBFOLDERS = ["TS_1", "TS_2", "TS_3"]




def ensure_results_dir() -> Path:
   """Create RESULTS_DIR if it does not exist. Returns the path."""
   RESULTS_DIR.mkdir(parents=True, exist_ok=True)
   return RESULTS_DIR




# ═══════════════════════════════════════════════════════════════════════════
# 2. CORE SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════


# ── General ───────────────────────────────────────────────────────────────
SEED: int              = 1
THRESHOLD: float       = 0.0    # Jij sign-correction threshold (|Rho| cutoff)
ZERO_FC_DIAGONAL: bool = True   # True → compare off-diagonal FC only (diag set to 0)
                                # False → include diagonal (diag set to 1)


# False → visit every spin in fixed mean-Jij order  (Jij_sorted_ising)
# True  → randomly sample spin-flip proposals        (random_ising)
USE_RANDOM_SPIN_FLIP: bool = False


# "pearson" → fit full Pearson FC (FC1/2/3_PATH).
# "partial" → fit partial-correlation FC (PARTIAL_FC1/2/3_PATH); every
#             sim.generate_FC(...) / Ising partial= argument follows this.
FC_MODE: str = "pearson"   # "pearson" | "partial"


# ── Parameter annealing (broad search for T*, alpha*) ────────────────────
ANNEAL_STEPS:   int   = 1000
ANNEAL_MAXFUN:  int   = 500
ANNEAL_THERM:   int   = 2000
ANNEAL_BOUNDS         = ((0.1, 10), (-3, 3))   # ((T_low, T_high), (alpha_low, alpha_high))
                                                # Widen alpha to (-6, 3) for PET runs.
N_RESTARTS: int        = 1


# ── Cross-validated Optuna optimization ─────────────────────────────────
# Step 1B in GET_RESULTS.py: refines the dual-annealing result from Step 1
# using Optuna, scoring each candidate (T, alpha) as the mean Pearson r
# across OPTUNA_TEST_FOLDS independent simulation reruns (a "fold" here is
# a fresh simulation, not a data split — this estimates how reproducible a
# candidate's fit is rather than rewarding a lucky single run). The search
# window shrinks each round using the same REFINE_* schedule as Step 1.
OPTUNA_TRAIN_TRIALS: int = 30
OPTUNA_TEST_FOLDS: int   = 1

RUN_STEP1B_OPTUNA_CV: bool      = True    # run the Optuna CV refinement
PREFER_OPTUNA_CV_RESULT: bool   = False   # True → use the CV (T, alpha) downstream
                                           # instead of the dual-annealing result
MAKE_OPTUNA_CV_PLOT: bool       = True    # convergence plot across all rounds/trials
SAVE_OPTUNA_CV_PLOT: bool       = True


# Iterative refinement of the annealing result (also drives the Optuna CV
# window-shrink schedule in Step 1B)
REFINE_T_WINDOW:         float = 1.0
REFINE_ALPHA_WINDOW:     float = 0.5
REFINE_MAXFUN:           int   = 100
REFINE_MAX_ROUNDS:       int   = 2
REFINE_SHRINK:           float = 0.5
REFINE_MIN_T_WINDOW:     float = 0.001
REFINE_MIN_ALPHA_WINDOW: float = 0.0005


# ── Temperature sweep ─────────────────────────────────────────────────────
T_MIN:         float = 0.5
T_MAX:         float = 10.0
T_STEPS:       int   = 150
SWEEP_STEPS:   int   = 1000
SWEEP_THERM:   int   = 3000
TEMP_REPEATS:  int   = 2    # Repeats per temperature step (increase for stable peaks)


PEAK_IGNORE_EDGE_POINTS:    int   = 3
PEAK_PROMINENCE_FRACTION:   float = 0.15
SMOOTH_TEMPERATURE_PLOTS:   bool  = True
SMOOTH_WINDOW:              int   = 31
SMOOTH_POLYORDER:           int   = 3


# Skip annealing and force a fixed alpha value instead
USE_FIXED_ALPHA: bool  = False
FIXED_ALPHA:     float = 0.0


# ── Null distributions ────────────────────────────────────────────────────
N_NULL:               int   = 100
NULL_RUNS:            int   = 2
NULL_STEPS:           int   = 1000
NULL_THERM:           int   = 1000
CONSTANT_NULL_VALUE:  float = 1.0
BINS:                 int   = 50
N_POST_CRIT_MATRICES: int   = 5


# ── Threshold-correlation sweep ───────────────────────────────────────────
THRESH_SWEEP_N_RESTARTS:    int   = 5
THRESH_SWEEP_STEPS:         int   = 3000
THRESH_SWEEP_THERM:         int   = 1000
THRESH_SWEEP_VALUES_START:  float = 0.1
THRESH_SWEEP_VALUES_STOP:   float = 1.0    # exclusive (np.arange semantics)
THRESH_SWEEP_VALUES_STEP:   float = 0.1


# ── KS analysis ───────────────────────────────────────────────────────────
FLIP_THRESHOLD:             float = 0.5
RUN_FUNCTIONAL_COMPARISON:  bool  = True   # also run FC-coupled Ising alongside Jij


# ── Plot colours (shared across all scripts) ──────────────────────────────
BLUE      = "#2E86AB"
SD_BAND   = "#2CA25F"
RED       = "#E84855"
AMBER     = "#F4A261"
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"


# ═══════════════════════════════════════════════════════════════════════════
# 3. RUN TOGGLES — enable / disable entire pipeline stages
# ═══════════════════════════════════════════════════════════════════════════
RUN_STEP1_ANNEALING:              bool = True
RUN_STEP2_TEMP_SWEEP:             bool = True
RUN_STEP3_MATRIX_COMPARISON:      bool = True
RUN_STEP4_NULL_DISTRIBUTION:      bool = True
RUN_STEP4_ONES_NULL:              bool = True   # constant-Jij null (separate from shuffled)
RUN_MU_VS_PET_DIAGNOSTIC:        bool = True   # only meaningful for pet / inverse_pet
RUN_KS_ANALYSIS:                  bool = False
RUN_THRESHOLD_CORRELATION_SWEEP:  bool = False


# ═══════════════════════════════════════════════════════════════════════════
# 4. GRAPH TOGGLES
#    MAKE_X = False → skip the figure entirely (fastest for batch runs)
#    MAKE_X = True, SAVE_X = False → build but don't write to disk
#    MAKE_X = True, SAVE_X = True  → build and save (normal usage)
# ═══════════════════════════════════════════════════════════════════════════
MAKE_PARAM_ANNEAL_ERROR_PLOT:    bool = True
SAVE_PARAM_ANNEAL_ERROR_PLOT:    bool = True


MAKE_MU_VS_PET_PLOT:             bool = True
SAVE_MU_VS_PET_PLOT:             bool = True


MAKE_TEMPERATURE_SWEEP_PLOT:     bool = True   # 2×2: energy, |M|, susceptibility, Cv
SAVE_TEMPERATURE_SWEEP_PLOT:     bool = True


MAKE_CORRELATION_VS_T_PLOT:      bool = True
SAVE_CORRELATION_VS_T_PLOT:      bool = True


MAKE_MATRIX_COMPARISON_PLOT:     bool = True   # sim FC / emp FC / Jij at T_best
SAVE_MATRIX_COMPARISON_PLOT:     bool = True


MAKE_SCATTER_SIM_VS_EMP_PLOT:    bool = True
SAVE_SCATTER_SIM_VS_EMP_PLOT:    bool = True


MAKE_POST_CRIT_MATRIX_PLOT:      bool = True   # matrices at T > T_crit
SAVE_POST_CRIT_MATRIX_PLOT:      bool = True


MAKE_NULL_JIJ_MATRICES_PLOT:     bool = True   # real / shuffled / constant Jij
SAVE_NULL_JIJ_MATRICES_PLOT:     bool = True


MAKE_NULL_DISTRIBUTION_PLOT:     bool = True   # shuffled-Jij null histograms
SAVE_NULL_DISTRIBUTION_PLOT:     bool = True


MAKE_ONES_NULL_DISTRIBUTION_PLOT: bool = True
SAVE_ONES_NULL_DISTRIBUTION_PLOT: bool = True


MAKE_KS_SWEEP_PLOTS:             bool = True   # raw / sign / flip KS-vs-T plots
SAVE_KS_SWEEP_PLOTS:             bool = True


MAKE_THRESHOLD_CORRELATION_PLOT: bool = True
SAVE_THRESHOLD_CORRELATION_PLOT: bool = True


# Single-simulation diagnostic plots (SimulationGraphData in GET_RESULTS.py)
MAKE_MAG_ENERGY_PLOT:  bool = False
MAKE_ROC_PLOT:         bool = False
MAKE_FC_MATRIX_PLOT:   bool = False
MAKE_EVERYTHING_PLOT:  bool = False   # combined 6-panel diagnostic figure


# Pop up interactive windows in addition to saving (set False for batch runs)
SHOW_PLOTS_INTERACTIVELY: bool = False


# Write the run-summary CSV at the end of GET_RESULTS.py
SAVE_RUN_SUMMARY: bool = True


# Print progress/diagnostic text to the console during sweeps
PRINT_PROGRESS_TEXT: bool = True


# Persist the raw best/critical Ising objects from Step 2 (pickled to
# ISING_DATA_DIR) for later reanalysis without re-simulating. Off by
# default — these can be large, especially with long SWEEP_STEPS.
SAVE_ISING_OBJECTS: bool = False

