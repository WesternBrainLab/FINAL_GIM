import numpy as np
import scipy.stats as sp
import scipy.integrate as int
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import FUNC_CON as fc
import GIM as I
import UTILS as utils
import CONFIG as cf
import os
import pickle
import json
from copy import copy


class temp_sweep:

    def __init__(self,
                 min_temp,
                 max_temp,
                 temp_step,
                 alpha,
                 multiplier,
                 Jij,
                 ising = I.Jij_sorted_ising,
                 save = False,
                 name = 'temp_sweep',
                 path = cf.TEMP_SWEEP_DATA):

        '''
        Runs multiple Ising model simulations at increasing temperatures, then graphs temperature vs average energy,
        average magnetization, specific heat, and magnetic susceptibility at the end.

        :param min_temp: starting temperature value
        :param max_temp: ending temperature value
        :param temp_step: number of temperature steps to get from start to end
        :param alpha: alpha value
        :param Jij: Jij matrix used for all simulations
        :param ising: Ising timescale used for all simulations
        :param multiplier: temperature multiplier values used per neuron
        :param save: set to True if you want to save data under simulation data/temp sweep data
        '''

        self.T_global = np.linspace(min_temp, max_temp, temp_step)
        self.alpha = alpha
        self.multiplier = multiplier
        self.Jij = Jij
        self.ising = ising

        self.sim_dataset = {}
        self.ising_ar = []
        self.suscept_ar = []
        self.spec_heat_ar = []
        self.avg_temp_ar = []

        self.save = save
        if self.save:
            num_folders = len(next(os.walk(path))[1])
            self.folder_name = name + '_run_' + str(num_folders)
            self.path = path + '/' + self.folder_name
            os.mkdir(self.path)

    def simulate(self,
                 steps,
                 thermalization,
                 spin_array = None,
                 partial = False,
                 show = False,
                 diag = False):

        '''
        Main class for preforming the temperature sweep simulations

        :param steps: number of timesteps per simulation
        :param thermalization: number of thermalization steps per simulation
        :param spin_array: set initial spin configuration for all simulations. Default is randomized
        :param partial: if True, uses partial correlation to calculate FC matrix
        :param show: if True, displays a live plot of all parameters as simulation runs
        :param diag: if True, includes diagonal values in correlation calculation between emp. and sim. FC
        :param text: if True, prints out text for each simulation that displays the final parameter values
        :param name: set custom file name
        :param path: set custom save path
        :return:
        '''

        def save():
            '''
            If self.save == True, this function will generate a log containing information about the simulations
            '''
            fc.save_to_json(self.sim_dataset, self.path, 'sim_dataset.json')

            with open(self.path + '/log.json', 'w') as file:
                json.dump(self.log, file, indent=4)

            pickle.dump(self.crit_ising, open(self.path + '/crit_ising.pickle', 'wb'))

        if spin_array is None:
            spin_array = np.random.choice([-1, 1], 84)
        else:
            spin_array = np.asarray(spin_array).copy()

        if show:
            plt.ion()

        for id, temp in enumerate(self.T_global):
            temp_ar = temp * (self.multiplier ** self.alpha)
            avg_temp = np.mean(temp_ar)
            beta = 1 / temp

            ising = self.ising(temp_ar, Jij = self.Jij, beta = beta, spin_ar = spin_array.copy())
            log = ising.simulate(steps, thermalization, log = True)
            print(str(id) + ': ' + str(temp))
            self.sim_dataset[id] = ising.generate_FC(partial)

            self.avg_temp_ar.append(avg_temp)
            self.ising_ar.append(log)
            self.suscept_ar.append(ising.susceptibility(beta))
            self.spec_heat_ar.append(ising.specific_heat(beta))

            if show:
                if temp != self.T_global[0]:
                    plt.close()
                figure, axis = ising.graph_everything(show=False)
                figure.canvas.draw()
                figure.canvas.flush_events()
        if show:
            plt.close()
            plt.ioff()

        crit_index = np.nanargmax(self.spec_heat_ar)
        self.crit_temp = self.T_global[crit_index]
        self.crit_ising = self.ising_ar[crit_index]
        self.log = {'alpha': self.alpha,
                    'multiplier': self.multiplier.tolist(),
                    'min temp': self.T_global[0],
                    'max temp': self.T_global[-1],
                    'temp steps': np.size(self.T_global),
                    'critical temperature': self.crit_temp,
                    'mean critical temperature': np.mean((self.multiplier ** self.alpha) * self.crit_temp),
                    'partial correlation': partial,
                    'include diagonals': diag,
                    'time scale': ising.__str__()}

        print(self.log)

        if self.save:
            save()

    def correlate(self, emp_dataset):
        self.temp_corr = np.zeros([np.size(self.T_global), np.uint8(len(emp_dataset))])
        for sim_id in self.sim_dataset:
            for emp_id in emp_dataset:
                sim_FC = np.array(self.sim_dataset[sim_id]['matrix'])
                emp_FC = np.array(emp_dataset[emp_id]['matrix'])
                self.temp_corr[np.uint8(sim_id), np.uint8(emp_id)] = utils.mat_corr(sim_FC, emp_FC)

        return self.temp_corr

    def graph_corr(self):
        [temp, emp_FC] = np.shape(self.temp_corr)
        for emp_id in range(emp_FC):
            plt.plot(self.T_global, self.temp_corr[:, emp_id])
        plt.xlabel('Global Temperature')
        plt.ylabel('Correlation')

        plt.show()

    # ------------------------------------------------------------------ #
    # Everything below is plotting/graphing results                    #
    # ------------------------------------------------------------------ #

    def graph_data(self,
                    rho_emp,
                    rho_emp_vec,
                    results_dir,
                    alpha_star,
                    stable_peak_index,
                    temperature_mean_and_sd_band,
                    safe_pearson,
                    clean_vec,
                    fc_compare_vec,
                    set_fc_diagonal,
                    evenly_spaced_indices,
                    PEAK_PROMINENCE_FRACTION=0.1,
                    PEAK_IGNORE_EDGE_POINTS=True,
                    THRESHOLD=0,
                    N_POST_CRIT_MATRICES=3,
                    BLUE='steelblue',
                    RED='crimson',
                    AMBER='darkorange',
                    SD_BAND='lightsteelblue'):
        '''
        Generates and saves all temperature sweep plots:
          - Figure 1: E, |M|, susceptibility, specific heat vs T
          - Figure 2: correlation vs T
          - Figure 3: matrix comparison at T_best
          - Figure 3_post: post-critical matrix comparisons

        :param rho_emp: 2-D empirical Pearson FC matrix
        :param rho_emp_vec: flattened empirical FC vector (upper triangle)
        :param results_dir: pathlib.Path or str — directory to save figures
        :param alpha_star: best-fit alpha value (used in titles)
        :param stable_peak_index: callable(arr) -> int, returns robust peak index
        :param temperature_mean_and_sd_band: callable(data, sd) -> (data_plot, sd_plot)
        :param safe_pearson: callable(a, b) -> float
        :param clean_vec: callable(vec) -> vec with NaNs / Infs removed
        :param fc_compare_vec: callable(mat) -> upper-triangle vector
        :param set_fc_diagonal: callable(mat) -> None, zeros / sets FC diagonal in-place
        :param evenly_spaced_indices: callable(indices, n) -> n evenly-spaced subset
        :param PEAK_PROMINENCE_FRACTION: prominence threshold for peak detection
        :param PEAK_IGNORE_EDGE_POINTS: whether to ignore edge points in peak detection
        :param THRESHOLD: threshold used during simulation (for title labels)
        :param N_POST_CRIT_MATRICES: number of post-critical matrices to plot
        :param BLUE: main line colour
        :param RED: susceptibility peak marker colour
        :param AMBER: best-match temperature marker colour
        :param SD_BAND: SD shading colour
        '''

        os.makedirs(str(results_dir), exist_ok=True)

        # ── build the arrays graph_data needs, from what simulate() stored ──
        # `simulate()` runs ONE simulation per temperature (no repeats), so
        # there is no real spread to plot -> sd arrays are zero-filled.
        # `avg_energy` / `avg_mag` are pulled out of the per-temperature log
        # returned by ising.simulate(..., log=True). Different GIM versions
        # name these keys differently, so _get_log_mean() tries a few common
        # spellings before giving up.
        def _get_log_mean(log, candidates):
            if isinstance(log, dict):
                for key in candidates:
                    if key in log:
                        series = np.asarray(log[key])
                        return float(np.nanmean(series))
            else:
                for key in candidates:
                    if hasattr(log, key):
                        series = np.asarray(getattr(log, key))
                        return float(np.nanmean(series))
            return np.nan

        avg_energy_ar = np.array([
            _get_log_mean(log, ['energy', 'energy_series', 'E', 'avg_energy'])
            for log in self.ising_ar
        ])
        avg_mag_ar = np.array([
            _get_log_mean(log, ['magnetization', 'mag_series', 'M', 'avg_mag'])
            for log in self.ising_ar
        ])

        avg_energy = avg_energy_ar
        avg_energy_sd = np.zeros_like(avg_energy_ar)
        avg_mag = avg_mag_ar
        avg_mag_sd = np.zeros_like(avg_mag_ar)
        suscept = np.array(self.suscept_ar)
        suscept_sd = np.zeros_like(suscept)
        spec_heat = np.array(self.spec_heat_ar)
        spec_heat_sd = np.zeros_like(spec_heat)

        # ── correlation of each simulated FC against the empirical FC ──────
        corr_ar_total = np.full(len(self.T_global), np.nan)
        for sim_id, sim_entry in self.sim_dataset.items():
            sim_FC = np.array(sim_entry['matrix']).copy()
            set_fc_diagonal(sim_FC)
            sim_vec = clean_vec(fc_compare_vec(sim_FC))
            corr_ar_total[int(sim_id)] = safe_pearson(sim_vec, rho_emp_vec)
        corr_sd_ar_total = np.zeros_like(corr_ar_total)

        self.avg_energy_ar = avg_energy_ar.tolist()
        self.avg_energy_sd_ar = avg_energy_sd.tolist()
        self.avg_mag_ar = avg_mag_ar.tolist()
        self.avg_mag_sd_ar = avg_mag_sd.tolist()
        self.suscept_sd_ar = suscept_sd.tolist()
        self.spec_heat_sd_ar = spec_heat_sd.tolist()
        self.corr_ar_total = corr_ar_total.tolist()
        self.corr_sd_ar_total = corr_sd_ar_total.tolist()

        results_dir = results_dir  # keep as-is (Path or str both work with savefig)

        # ── derive peak temperatures from stored arrays ────────────────────
        corr_arr      = np.array(self.corr_ar_total)
        spec_heat_arr = np.array(self.spec_heat_ar)
        suscept_arr   = np.array(self.suscept_ar)

        n_nan = np.sum(np.isnan(corr_arr))
        print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")

        T_suscept_peak   = self.T_global[stable_peak_index(suscept_arr)]
        T_spec_heat_peak = self.T_global[stable_peak_index(spec_heat_arr)]
        crit_idx  = stable_peak_index(spec_heat_arr)
        best_idx  = int(np.nanargmax(corr_arr))
        T_crit    = self.T_global[crit_idx]
        T_best    = self.T_global[best_idx]
        best_corr = np.nanmax(corr_arr)

        # patch the object so downstream code stays consistent
        self.crit_temp  = T_crit
        self.best_temp  = T_best
        self.best_corr  = best_corr
        self.best_ising = self.ising_ar[best_idx]
        self.crit_ising = self.ising_ar[crit_idx]

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

        # ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────
        fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        fig1.suptitle(
            f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
            fontsize=14,
            fontweight="bold"
        )

        ref_lines = [
            (T_suscept_peak,   RED,     "--", rf"$T_{{\chi\,peak}}$ = {T_suscept_peak:.2f}"),
            (T_spec_heat_peak, SD_BAND, "--", rf"$T_{{C\,peak}}$ = {T_spec_heat_peak:.2f}"),
            (T_best,           AMBER,   ":",  rf"$T_{{best}}$ = {T_best:.2f}"),
        ]

        panels = [
            (axes1[0, 0], avg_energy, avg_energy_sd, r"average energy $\langle E \rangle$", "Energy vs T"),
            (axes1[0, 1], avg_mag,    avg_mag_sd,    r"average $|M|$",                      "|Magnetization| vs T"),
            (axes1[1, 1], suscept,    suscept_sd,    r"susceptibility $\chi$",               "Susceptibility vs T"),
            (axes1[1, 0], spec_heat,  spec_heat_sd,  r"specific heat $C$",                  "Specific Heat vs T"),
        ]

        for ax, data, sd, ylabel, title in panels:
            data_plot, sd_plot = temperature_mean_and_sd_band(data, sd)

            ax.plot(self.T_global, data_plot, color=BLUE, lw=2.0)
            ax.fill_between(
                self.T_global,
                data_plot - sd_plot,
                data_plot + sd_plot,
                color=SD_BAND, alpha=0.28, linewidth=0
            )

            for temp, color, ls, label in ref_lines:
                ax.axvline(temp, color=color, linestyle=ls, lw=1.6, label=label)

            ax.set_xlabel("global temperature  T", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=8, framealpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        plt.savefig(str(results_dir) + "/temperature_sweep_3.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print("Saved: temperature_sweep_3.png")

        # ── Figure 2: correlation vs T ────────────────────────────────────
        fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

        corr_total    = np.array(self.corr_ar_total)
        corr_total_sd = np.array(self.corr_sd_ar_total)
        corr_total_plot, corr_total_sd_plot = temperature_mean_and_sd_band(corr_total, corr_total_sd)

        ax_corr.plot(self.T_global, corr_total_plot, color=BLUE, lw=2.0, label="avg FC")
        ax_corr.fill_between(
            self.T_global,
            corr_total_plot - corr_total_sd_plot,
            corr_total_plot + corr_total_sd_plot,
            color=SD_BAND, alpha=0.28, linewidth=0,
            label="standard deviation"
        )

        ax_corr.axvline(T_crit, color=RED,   linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
        ax_corr.axvline(T_best, color=AMBER, linestyle=":",  lw=1.5, label=f"T_best = {T_best:.2f}")

        ax_corr.set_xlabel("Global temperature  T", fontsize=11)
        ax_corr.set_ylabel("Pearson r  (sim Pearson FC vs emp Pearson FC)", fontsize=11)
        ax_corr.set_title("Correlation vs Temperature", fontsize=12)
        ax_corr.legend(fontsize=9, framealpha=0.3)
        ax_corr.spines[["top", "right"]].set_visible(False)

        plt.savefig(str(results_dir) + "/correlation_vs_T_3.png", dpi=150, bbox_inches="tight")
        plt.close(fig_corr)
        print("Saved: correlation_vs_T_3.png")

        # ── Step 3: Matrix comparison at T_best ───────────────────────────
        print("STEP 3 : MATRIX COMPARISON  (T_best, Pearson FC)")
        print("=" * 65)

        best_gd = self.best_ising
        sim_FC  = np.array(self.sim_dataset[best_idx]['matrix']).copy()
        Jij_mat = np.array(self.Jij).copy()

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

        # ── colour normalisation ──────────────────────────────────────────
        fc_lim  = 0.5
        fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)

        j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
        j_lim     = np.percentile(np.abs(j_offdiag), 99)
        if not np.isfinite(j_lim) or j_lim < 0.05:
            j_lim = 0.2
        j_norm = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)

        print(f"FC color limit  : ±{fc_lim:.4f}")
        print(f"Jij color limit : ±{j_lim:.4f}")

        # ── matrix figure ─────────────────────────────────────────────────
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        fig3.suptitle(
            f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}"
            f"  |  r={r_best:.4f}  |  threshold={THRESHOLD:g}",
            fontsize=13,
            fontweight="bold"
        )

        matrix_panels = [
            (sim_FC,  f"Simulated Pearson FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
            (rho_emp, "Empirical Pearson FC",                                            fc_norm),
            (Jij_mat, "Structural connectivity  $J_{ij}$",                               j_norm),
        ]

        for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
            im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
            ax.set_title(title, fontsize=11, pad=12)
            ax.set_xlabel("region", fontsize=9)
            ax.set_ylabel("region", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.savefig(str(results_dir) + "/matrix_comparison_3.png", dpi=150, bbox_inches="tight")
        plt.close(fig3)

        # ── post-critical matrix comparisons ──────────────────────────────
        post_crit_indices = np.where(self.T_global > T_crit)[0]
        best_idx_local    = best_idx
        post_crit_indices = post_crit_indices[post_crit_indices != best_idx_local]
        post_crit_indices = evenly_spaced_indices(post_crit_indices, N_POST_CRIT_MATRICES)

        if len(post_crit_indices) > 0:
            fig3_post, axes3_post = plt.subplots(
                len(post_crit_indices), 3,
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
                T_here   = self.T_global[idx]
                sim_here = np.array(self.sim_dataset[idx]['matrix']).copy()
                set_fc_diagonal(sim_here)

                sim_here_vec = clean_vec(fc_compare_vec(sim_here))
                r_here    = safe_pearson(sim_here_vec, rho_emp_vec)
                dist_here = np.linalg.norm(sim_here_vec - rho_emp_vec)

                print(f"  T={T_here:.4f}  r={r_here:.4f}  dist={dist_here:.4f}")

                row_panels = [
                    (sim_here, f"Simulated Pearson FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
                    (rho_emp,  "Empirical Pearson FC",                                   fc_norm),
                    (Jij_mat,  "Structural connectivity  $J_{ij}$",                      j_norm),
                ]

                for ax, (mat, title, norm_to_use) in zip(axes3_post[row], row_panels):
                    im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
                    ax.set_title(title, fontsize=10, pad=10)
                    ax.set_xlabel("region", fontsize=8)
                    ax.set_ylabel("region", fontsize=8)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(
                str(results_dir) + "/matrix_comparisons_post_Tcrit_3.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig3_post)
            print("Saved: matrix_comparisons_post_Tcrit_3.png")
        else:
            print("No post-critical temperatures available for extra matrix comparisons.")


# ---------------------------------------------------------------------- #
# Helper callables that graph_data() takes as parameters.                #
# ---------------------------------------------------------------------- #

def stable_peak_index(arr, prominence_fraction=0.1, ignore_edges=True):
    '''Robust peak finder: picks the highest local max with enough
    prominence, ignoring NaNs and (optionally) the first/last points.
    Falls back to nanargmax if no such peak is found.'''
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return 0

    data_range = np.nanmax(arr) - np.nanmin(arr)
    min_prom = prominence_fraction * data_range if data_range > 0 else 0

    candidates = []
    start = 1 if ignore_edges else 0
    end = len(arr) - 1 if ignore_edges else len(arr)
    for i in range(start, end):
        if np.isnan(arr[i]):
            continue
        left = arr[max(0, i - 1)]
        right = arr[min(len(arr) - 1, i + 1)]
        if np.isnan(left) or np.isnan(right):
            continue
        if arr[i] >= left and arr[i] >= right:
            prominence = arr[i] - min(
                np.nanmin(arr[:i + 1]) if i > 0 else arr[i],
                np.nanmin(arr[i:]) if i < len(arr) - 1 else arr[i]
            )
            if prominence >= min_prom:
                candidates.append(i)

    if candidates:
        return max(candidates, key=lambda i: arr[i])
    return int(np.nanargmax(arr))


def temperature_mean_and_sd_band(data, sd):
    '''No repeated runs per temperature in this pipeline, so this is a
    pass-through; kept as a function so plotting code stays generic if
    multi-run averaging is added later.'''
    return np.asarray(data, dtype=float), np.asarray(sd, dtype=float)


def safe_pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
    if np.sum(mask) < 2:
        return np.nan
    r, _ = sp.pearsonr(a[mask], b[mask])
    return r


def clean_vec(vec):
    vec = np.asarray(vec, dtype=float)
    return vec[np.isfinite(vec)]


def fc_compare_vec(mat):
    mat = np.asarray(mat)
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def set_fc_diagonal(mat, value=0.0):
    np.fill_diagonal(mat, value)


def evenly_spaced_indices(indices, n):
    indices = np.asarray(indices)
    if len(indices) <= n:
        return indices
    pick = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return indices[pick]


if __name__ == '__main__':
    steps = 4000
    thermalization = 2000
    min_temp = 0.05
    max_temp = 10
    temp_step = 100
    alpha = 2
    Jij = cf.avg_Jij * utils.get_sign_matrix(cf.avg_FC)

    simulation = temp_sweep(
        min_temp,
        max_temp,
        temp_step,
        alpha,
        multiplier=cf.norm_ind_avg_Jij,
        Jij=Jij,
        ising=I.Jij_sorted_ising,
    )
    simulation.simulate(steps, thermalization, partial=False, show=False)

    results_dir = os.path.join(cf.PROJECT_ROOT, "RESULTS", "TEMPSWEEP_plotting")
    os.makedirs(results_dir, exist_ok=True)

    rho_emp = np.array(cf.avg_FC)
    rho_emp_diag_free = rho_emp.copy()
    set_fc_diagonal(rho_emp_diag_free)
    rho_emp_vec = clean_vec(fc_compare_vec(rho_emp_diag_free))

    simulation.graph_data(
        rho_emp=rho_emp,
        rho_emp_vec=rho_emp_vec,
        results_dir=results_dir,
        alpha_star=alpha,
        stable_peak_index=stable_peak_index,
        temperature_mean_and_sd_band=temperature_mean_and_sd_band,
        safe_pearson=safe_pearson,
        clean_vec=clean_vec,
        fc_compare_vec=fc_compare_vec,
        set_fc_diagonal=set_fc_diagonal,
        evenly_spaced_indices=evenly_spaced_indices,
    )
