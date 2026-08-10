"""Cross-validated and dual-annealing optimization for the GIM model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.optimize import dual_annealing

import CONFIG as C
import GIM as I
import UTILS as utils


class optimize:
    """Optimize global temperature, alpha, and optional Jij thresholding."""

    def __init__(
        self,
        ising,
        therm: int,
        steps: int,
        train_trials: int,
        temp_range: tuple[float, float],
        alpha_range: tuple[float, float],
        spins: np.ndarray | None = None,
        partial: bool = True,
        multiplier: np.ndarray | None = None,
        FC_dir: str | Path | None = C.TS_DATA_DIR / "TS_1",
        save: bool = False,
    ):
        self.ising = ising
        self.therm = therm
        self.steps = steps
        self.train_trials = train_trials
        self.t_lower, self.t_upper = temp_range
        self.alpha_lower, self.alpha_upper = alpha_range
        self.partial = partial
        # Retained only for compatibility; GET_RESULTS.py owns all output.
        self.save = save

        # NEW: Set the spin-array length from the Jij matrix instead of
        # assuming every analysis has exactly 84 regions.
        n_regions = I.avg_Jij.shape[0]
        self.spins = (
            np.asarray(spins).copy()
            if spins is not None
            else np.random.choice([-1, 1], n_regions)
        )

        # NEW: Use mean absolute coupling as the default local-temperature
        # multiplier; callers may still supply a custom multiplier.
        default_multiplier = utils.normalize_array(np.mean(np.abs(I.avg_Jij), axis=0))
        self.multiplier = (
            np.asarray(multiplier, dtype=float).copy()
            if multiplier is not None
            else default_multiplier
        )
        self.Jij = I.avg_Jij.copy()

        # NEW: Build one empirical FC matrix from each time-series CSV.
        self.FC_arr = self._load_fc_matrices(FC_dir)

        # NEW: Store dual-annealing evaluations for diagnostics and saving.
        self.T_global: list[float] = []
        self.alpha_vals: list[float] = []
        self.error: list[float] = []
        self.correlate: list[float] = []
        self.FC: list[np.ndarray] = []

    # NEW: Small helpers isolate preparation, simulation, and scoring logic.
    @staticmethod
    def _load_fc_matrices(FC_dir: str | Path | None) -> list[np.ndarray]:
        if FC_dir is None:
            raise ValueError("FC_dir is required to construct empirical FC matrices.")
        directory = Path(FC_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"FC directory does not exist: {directory}")

        fc_matrices = []
        for file_path in sorted(path for path in directory.iterdir() if path.is_file()):
            time_series = utils.get_matrix(file_path.name, directory)
            fc_matrices.append(np.corrcoef(time_series.T))
        if not fc_matrices:
            raise ValueError(f"No time-series files were found in: {directory}")
        return fc_matrices

    def _build_temp_arr(self, T_global: float, alpha: float) -> np.ndarray:
        """Return a finite, strictly positive per-neuron temperature array."""
        temp = T_global * (self.multiplier ** alpha)
        temp = np.nan_to_num(temp, nan=T_global, posinf=T_global, neginf=T_global)
        temp[temp <= 0] = 1e-12
        return temp

    def _run_ising(self, temp: np.ndarray, Jij: np.ndarray) -> I.Ising:
        """Run a fresh Ising simulation and calculate its FC matrix."""
        simulation = self.ising(temp, Jij=Jij, spin_ar=self.spins.copy())
        simulation.simulate(self.steps, thermalization=self.therm)
        simulation.generate_FC(partial=self.partial)
        return simulation

    @staticmethod
    def _error_func(simulation: I.Ising, empirical_fc: np.ndarray) -> float:
        """Score a simulation using off-diagonal correlation plus FC RMSE."""
        correlation = simulation.correlation(empirical_fc, diag=False)
        if not np.isfinite(correlation):
            return float("inf")
        rmse = float(np.sqrt(np.mean((simulation.functional_connectivity - empirical_fc) ** 2)))
        return ((1.0 - correlation) + rmse) ** 2

    def _get_training_fc(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Create a random 70/30 train/test split of empirical FC matrices."""
        n_matrices = len(self.FC_arr)
        train_n = max(1, round(n_matrices * 0.7))
        train_indices = np.random.choice(n_matrices, size=train_n, replace=False)
        train_set = set(train_indices.tolist())
        self.train_indices = sorted(train_set)
        self.test_indices = [index for index in range(n_matrices) if index not in train_set]
        train_matrices = [self.FC_arr[index] for index in train_indices]
        test_matrices = [matrix for index, matrix in enumerate(self.FC_arr) if index not in train_set]
        return utils.average_matrices(*train_matrices), test_matrices

    # NEW: Optuna searches T_global, alpha, and an FC-derived Jij threshold.
    def _train_objective(self, trial: optuna.Trial) -> float:
        T_global = trial.suggest_float("t_glob", self.t_lower, self.t_upper)
        alpha = trial.suggest_float("alpha", self.alpha_lower, self.alpha_upper)
        threshold = trial.suggest_float("thresh", 0.0, 1.0)
        temp = self._build_temp_arr(T_global, alpha)
        Jij = self.Jij * utils.threshold_matrix(self.train_FC, threshold)
        return self._error_func(self._run_ising(temp, Jij), self.train_FC)

    # NEW: Cross-validation selects the parameter set with the lowest held-out error.
    def optimize_params(self, test_trials: int) -> np.ndarray:
        """Return the best [T_global, alpha, threshold] across validation folds."""
        if test_trials < 1:
            raise ValueError("test_trials must be at least 1.")

        fold_errors: list[float] = []
        fold_params = np.zeros((test_trials, 3))
        fold_splits: list[tuple[list[int], list[int]]] = []
        for fold in range(test_trials):
            self.train_FC, test_matrices = self._get_training_fc()
            fold_splits.append((self.train_indices.copy(), self.test_indices.copy()))
            study = optuna.create_study(direction="minimize")
            study.optimize(self._train_objective, n_trials=self.train_trials)

            best = study.best_params
            T_star, alpha_star, threshold_star = best["t_glob"], best["alpha"], best["thresh"]
            fold_params[fold] = [T_star, alpha_star, threshold_star]

            best_temp = self._build_temp_arr(T_star, alpha_star)
            best_Jij = self.Jij * utils.threshold_matrix(self.train_FC, threshold_star)
            simulation = self._run_ising(best_temp, best_Jij)
            evaluation_set = test_matrices or self.FC_arr
            fold_errors.append(float(np.mean([
                self._error_func(simulation, empirical_fc) for empirical_fc in evaluation_set
            ])))

        best_fold = int(np.argmin(fold_errors))
        self.best_params = fold_params[best_fold].copy()
        self.best_validation_error = fold_errors[best_fold]
        self.train_indices, self.test_indices = fold_splits[best_fold]
        return self.best_params

    # Compatibility with legacy calls that used optim.optimize(...).
    def optimize(self, test_trials: int) -> np.ndarray:
        return self.optimize_params(test_trials)

    # NEW: Dual annealing remains available for the two-parameter search used
    # by GET_RESULTS.py; it does not alter the Jij threshold.
    def anneal(
        self,
        steps: int,
        maxfun: int,
        emp_FC: np.ndarray,
        therm: int,
        bounds: tuple = C.ANNEAL_BOUNDS,
        no_local_search: bool = False,
        show: bool = False,
    ):
        self.steps, self.therm = steps, therm
        self.T_global, self.alpha_vals = [], []
        self.error, self.correlate, self.FC = [], [], []

        def objective(parameters: np.ndarray) -> float:
            T_global, alpha = map(float, parameters)
            simulation = self._run_ising(self._build_temp_arr(T_global, alpha), self.Jij)
            correlation = simulation.correlation(emp_FC, diag=False)
            error = 1.0 - correlation if np.isfinite(correlation) else float("inf")
            self.T_global.append(T_global)
            self.alpha_vals.append(alpha)
            self.error.append(error)
            self.correlate.append(correlation)
            self.FC.append(simulation.functional_connectivity.copy())
            return error

        self.optim_param = dual_annealing(
            objective,
            bounds=bounds,
            maxfun=maxfun,
            minimizer_kwargs={"method": None} if no_local_search else {},
        )
        if show:
            self.plot_error(show=True)
        return self.optim_param

    def plot_error(self, show: bool = True):
        """Build a 3-D temperature × alpha × error plot and return it."""
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(self.T_global, self.alpha_vals, self.error)
        axis.set(xlabel="T_global", ylabel="alpha", zlabel="error (1 − r)")
        if show:
            plt.show()
        return figure


def load_3d_plots(folder_name: str, file_name: str) -> None:
    """Load a saved diagnostic plot from the current RESULTS run folder."""
    utils.get_pickle_file(str(C.RESULTS_DIR / "optimization" / folder_name), file_name)
    plt.show()
