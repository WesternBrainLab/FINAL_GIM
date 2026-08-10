"""
TEMP_SWEEP.py — Temperature sweep simulation for the Generalized Ising Model.

Sweeps a range of global temperatures, running one or more independent Ising
simulations at each step and recording energy, magnetisation, susceptibility,
specific heat, and FC-correlation observables.

This module is computation-only. GET_RESULTS.py is responsible for executing
sweeps and saving their results.
"""


from __future__ import annotations

import builtins
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import CONFIG as C
import UTILS  as utils
import GIM    as I

import os
import scipy.stats as sp
import sys



# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Adds the project root to sys.path so that modules in sibling folders
# (e.g., steven/, kayla_1/) can be imported regardless of where you run
# this script from.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))



# ─────────────────────────────────────────────────────────────────────────────
# DATA CONFIGURATION
#
# PROJECT1_JIJ_PATH is an optional alternate coupling matrix.  The normal
# fallback is GIM.avg_Jij.
# ─────────────────────────────────────────────────────────────────────────────
FINAL_GIM_DIR = Path(__file__).resolve().parent
PROJECT1_ROOT = PROJECT_ROOT / 'kayla_1' / 'Project_1'
PROJECT1_JIJ_PATH = C.AVG_JIJ_NEW_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Jij SOURCE TOGGLE
#
# Set USE_PROJECT1_JIJ = True  → loads Jij from Kayla's CSV file (recommended
#                                 for direct runs of this script)
# Set USE_PROJECT1_JIJ = False → falls back to the Jij matrix defined in
#                                 steven/Scripts/config.py
# ─────────────────────────────────────────────────────────────────────────────
USE_PROJECT1_JIJ = True




def default_jij():
   """
   Returns the default Jij coupling matrix.


   Tries to load avg_Jij_new_pearson.csv from Kayla's project folder when
   USE_PROJECT1_JIJ is True and the file exists; otherwise falls back to the
   matrix stored in config.py.


   The diagonal is zeroed out because self-coupling (a neuron coupled to
   itself) is not meaningful in the Ising model.
   """
   if USE_PROJECT1_JIJ and PROJECT1_JIJ_PATH.exists():
       Jij = np.genfromtxt(PROJECT1_JIJ_PATH, delimiter=',').astype(float)
       np.fill_diagonal(Jij, 0)
       return Jij
   return I.avg_Jij

# ── Shared empirical FC matrices (loaded once) ────────────────────────────
_FC1 = utils.load_csv(C.FC1_PATH).astype(float)
_FC2 = utils.load_csv(C.FC2_PATH).astype(float)
_FC3 = utils.load_csv(C.FC3_PATH).astype(float)
_avg_FC_pearson = utils.average_matrices(_FC1, _FC2, _FC3)
_partial_FC1 = utils.load_csv(C.PARTIAL_FC1_PATH).astype(float)
_partial_FC2 = utils.load_csv(C.PARTIAL_FC2_PATH).astype(float)
_partial_FC3 = utils.load_csv(C.PARTIAL_FC3_PATH).astype(float)
_avg_FC_partial = utils.average_matrices(_partial_FC1, _partial_FC2, _partial_FC3)


# Main class for running temperature sweep
#  simulations and analyzing results
class simulated_FC_vs_T_global:


   def __init__(self, min_temp, max_temp, temp_step, alpha,
                Jij = None, ising = I.Jij_sorted_ising,
                multiplier = None,
                save = False):


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
       :param save: retained for backwards compatibility; this module does not write results.
       '''
   # Builds an array of global temperatures to
   # sweep through, and sets up the Ising model parameters and data storage arrays.
       self.T_global = np.linspace(min_temp, max_temp, temp_step)
       self.alpha = alpha


       # Load the Jij matrix and multiplier array, using defaults
       # if not provided.
       if Jij is None:
           Jij = default_jij()
       if multiplier is None:
           multiplier = utils.normalize_array(np.mean(np.abs(Jij), axis=0))


       self.multiplier = multiplier
       self.Jij = Jij
       self.ising = ising


       self.ising_ar = []
       self.dist_ar = []
       self.suscept_ar = []
       self.suscept_sd_ar = []
       self.suscept_se_ar = []
       self.spec_heat_ar = []
       self.spec_heat_sd_ar = []
       self.spec_heat_se_ar = []
       self.avg_energy_ar = []
       self.avg_energy_sd_ar = []
       self.avg_energy_se_ar = []
       self.avg_mag_ar = []
       self.avg_mag_sd_ar = []
       self.avg_mag_se_ar = []
       self.avg_temp_ar = []
       self.Jij_auc_ar, self.FC_auc_ar = [], []
       self.corr_ar_1, self.corr_ar_2, self.corr_ar_3, self.corr_ar_total = [], [], [], []
       self.corr_sd_ar_1, self.corr_sd_ar_2, self.corr_sd_ar_3, self.corr_sd_ar_total = [], [], [], []
       self.corr_se_ar_1, self.corr_se_ar_2, self.corr_se_ar_3, self.corr_se_ar_total = [], [], [], []
       self.suscept_peak_temp = None
       self.spec_heat_peak_temp = None


       self.save = save

# Main simulation function that runs the temperature sweep, executes the Ising simulations, and records the results.
   def simulate(self, steps, thermalization = None, spin_array = np.random.choice([-1, 1], 84),
                partial = True, show = False, diag = False, text = True,
                name = 'temp_sweep', path = None,
                n_repeats = 1, emp_FC1 = None, emp_FC2 = None, emp_FC3 = None,
                avg_FC = None, reject_degenerate_partial = True,
                partial_min_std = 1e-4, partial_min_nonzero_fraction = 0.01):


       '''
       Main class for preforming the temperature sweep simulations


       :param steps: number of timesteps per simulation
       :param thermalization: number of thermalization steps per simulation
       :param spin_array: set initial spin configuration for all simulations. Default is randomized
       :param partial: if True, uses partial correlation to calculate FC matrix
       :param show: if True, displays a live plot of all parameters as simulation runs
       :param diag: if True, includes diagonal values in correlation calculation between emp. and sim. FC
       :param text: if True, prints out text for each simulation that displays the final parameter values
       :param name: retained for backwards compatibility.
       :param path: retained for backwards compatibility.
       :return:
       '''


       # Keep all scoring off-diagonal only. Diagonal FC values are
       # self-correlations and can make collapsed matrices look artificially good (can preserve correlation between matrices
       # even if the off-diagonal values are very different).
       diag = False


       def offdiag_vec(matrix):
           return utils.flat_remove_diag(np.asarray(matrix, dtype=float))


       # Here this function checks if the simulated FC matrix is degenerate (i.e., has very low variance or very few non-zero values) when using partial correlation.
       # If it is degenerate, the simulation results for that temperature are rejected.
       # We found that using pearson correlation (partial=False) gives a greater correlation value so only use when partial = True.
       def is_degenerate_partial_fc(matrix):
           offdiag = offdiag_vec(matrix)
           offdiag = offdiag[np.isfinite(offdiag)]
           if offdiag.size == 0:
               return True
           return (
               np.nanstd(offdiag) < partial_min_std
               or np.mean(np.abs(offdiag) > 1e-12) < partial_min_nonzero_fraction
           )


       if emp_FC1 is None or emp_FC2 is None or emp_FC3 is None or avg_FC is None:
           if partial:
               emp_FC1 = _partial_FC1
               emp_FC2 = _partial_FC2
               emp_FC3 = _partial_FC3
               avg_FC = _avg_FC_partial
           else:
               emp_FC1 = _FC1
               emp_FC2 = _FC2
               emp_FC3 = _FC3
               avg_FC = _avg_FC_pearson


       # Zero diagonals and convert empirical FC matrices to independent floats.
       emp_mats = [emp_FC1, emp_FC2, emp_FC3, avg_FC]
       emp_FC1, emp_FC2, emp_FC3, avg_FC = [
           np.asarray(matrix, dtype=float).copy() for matrix in emp_mats
       ]
       for matrix in (emp_FC1, emp_FC2, emp_FC3, avg_FC):
           np.fill_diagonal(matrix, 0.0)

       n_repeats = builtins.max(1, builtins.int(n_repeats))
       therm = thermalization if thermalization is not None else steps // 2


       if show:
           plt.ion()
       def mean_sd_se(values):
           values = np.asarray(values, dtype=float)
           values = values[np.isfinite(values)]
           if values.size == 0:
               return np.nan, np.nan, np.nan
           mean = np.nanmean(values)
           if values.size <= 1:
               return mean, 0.0, 0.0
           sd = np.nanstd(values, ddof=1)
           return mean, sd, sd / np.sqrt(values.size)


       temp_buckets = []
       for temp in self.T_global:
           temp_ar = temp * (self.multiplier ** self.alpha)
           temp_ar = np.nan_to_num(temp_ar, nan=temp, posinf=temp, neginf=temp)
           temp_ar[temp_ar <= 0] = 1e-12
           temp_buckets.append({
               'temp': temp,
               'temp_ar': temp_ar,
               'avg_temp': np.mean(temp_ar),
               'beta': 1 / temp,
               'data': [],
               'corr_1': [],
               'corr_2': [],
               'corr_3': [],
               'corr_total': [],
               'suscept': [],
               'spec_heat': [],
               'avg_energy': [],
               'avg_mag': [],
               'jij_auc': [],
               'fc_auc': [],
           })

       # Simulation loop: for each repeat, run a simulation at each temperature and record the results.
       for repeat_idx in range(n_repeats):
           if text and n_repeats > 1:
               print(f'temperature sweep repeat {repeat_idx + 1}/{n_repeats}')


           for temp_index, bucket in enumerate(temp_buckets):
               temp = bucket['temp']
               temp_ar = bucket['temp_ar']
               beta = bucket['beta']


               if n_repeats == 1 and spin_array is not None:
                   init_spin = spin_array.copy()
               else:
                   init_spin = np.random.choice([-1, 1], np.shape(self.Jij)[0])


               ising = self.ising(temp_ar, Jij=self.Jij, spin_ar=init_spin)
               ising.simulate(steps, therm)
               sim_FC = ising.generate_FC(partial)
               ising.functional_connectivity = np.nan_to_num(sim_FC, nan=0.0, posinf=0.0, neginf=0.0)
               np.fill_diagonal(ising.functional_connectivity, 0.0)
               sim_FC = ising.functional_connectivity
               ising_data = I.get_data(ising, beta, temp, self.alpha, emp_FC=avg_FC, diag=diag)


               if partial and reject_degenerate_partial and is_degenerate_partial_fc(sim_FC):
                   corr_1 = corr_2 = corr_3 = corr_total = np.nan
               else:
                   corr_1 = ising.correlation(emp_FC1, diag=False)
                   corr_2 = ising.correlation(emp_FC2, diag=False)
                   corr_3 = ising.correlation(emp_FC3, diag=False)
                   corr_total = ising.correlation(avg_FC, diag=False)


               try:
                   Jij_tpr_ar, Jij_fpr_ar, Jij_auc = utils.receiver_operating_characteristic(sim_FC, self.Jij)
               except Exception:
                   Jij_auc = np.nan
               try:
                   FC_tpr_ar, FC_fpr_ar, FC_auc = utils.receiver_operating_characteristic(sim_FC, avg_FC)
               except Exception:
                   FC_auc = np.nan


               bucket['data'].append(ising_data)
               bucket['corr_1'].append(corr_1)
               bucket['corr_2'].append(corr_2)
               bucket['corr_3'].append(corr_3)
               bucket['corr_total'].append(corr_total)
               bucket['suscept'].append(ising.susceptibility(beta))
               bucket['spec_heat'].append(ising.specific_heat(beta))
               bucket['avg_energy'].append(np.mean(ising.energy_series))
               bucket['avg_mag'].append(np.mean(np.abs(ising.mag_series)))
               bucket['jij_auc'].append(Jij_auc)
               bucket['fc_auc'].append(FC_auc)


               if text:
                   print(ising_data)
                   if n_repeats > 1:
                       print(f'repeat {repeat_idx + 1}/{n_repeats}; temperature {temp_index + 1}/{len(self.T_global)}')
                   print('_____________________________')


               if show:
                   if temp_index != 0 or repeat_idx != 0:
                       plt.close()
                   figure, axis = ising_data.graph_everything(show=False)
                   figure.canvas.draw()
                   figure.canvas.flush_events()

        # Aggregate results across repeats for each temperature, selecting the best repeat based on correlation.
       for bucket in temp_buckets:
           repeat_corr_total_arr = np.asarray(bucket['corr_total'], dtype=float)
           if np.any(np.isfinite(repeat_corr_total_arr)):
               best_repeat_index = np.nanargmax(repeat_corr_total_arr)
           else:
               best_repeat_index = 0
           ising_data = bucket['data'][best_repeat_index]


           if text and n_repeats > 1:
               mean_corr, _, _ = mean_sd_se(bucket['corr_total'])
               print(
                   f"temperature {bucket['temp']:.4f}; "
                   f"repeats: {n_repeats}; mean correlation: {mean_corr:.4f}"
               )


           jij_auc, jij_auc_sd, jij_auc_se = mean_sd_se(bucket['jij_auc'])
           fc_auc, fc_auc_sd, fc_auc_se = mean_sd_se(bucket['fc_auc'])
           corr_1, corr_1_sd, corr_1_se = mean_sd_se(bucket['corr_1'])
           corr_2, corr_2_sd, corr_2_se = mean_sd_se(bucket['corr_2'])
           corr_3, corr_3_sd, corr_3_se = mean_sd_se(bucket['corr_3'])
           corr_total, corr_total_sd, corr_total_se = mean_sd_se(bucket['corr_total'])
           suscept, suscept_sd, suscept_se = mean_sd_se(bucket['suscept'])
           spec_heat, spec_heat_sd, spec_heat_se = mean_sd_se(bucket['spec_heat'])
           avg_energy, avg_energy_sd, avg_energy_se = mean_sd_se(bucket['avg_energy'])
           avg_mag, avg_mag_sd, avg_mag_se = mean_sd_se(bucket['avg_mag'])


           self.Jij_auc_ar.append(jij_auc)
           self.FC_auc_ar.append(fc_auc)


           self.avg_temp_ar.append(bucket['avg_temp'])
           self.corr_ar_1.append(corr_1)
           self.corr_ar_2.append(corr_2)
           self.corr_ar_3.append(corr_3)
           self.corr_ar_total.append(corr_total)
           self.corr_sd_ar_1.append(corr_1_sd)
           self.corr_sd_ar_2.append(corr_2_sd)
           self.corr_sd_ar_3.append(corr_3_sd)
           self.corr_sd_ar_total.append(corr_total_sd)
           self.corr_se_ar_1.append(corr_1_se)
           self.corr_se_ar_2.append(corr_2_se)
           self.corr_se_ar_3.append(corr_3_se)
           self.corr_se_ar_total.append(corr_total_se)
           self.ising_ar.append(ising_data)
           self.suscept_ar.append(suscept)
           self.suscept_sd_ar.append(suscept_sd)
           self.suscept_se_ar.append(suscept_se)
           self.spec_heat_ar.append(spec_heat)
           self.spec_heat_sd_ar.append(spec_heat_sd)
           self.spec_heat_se_ar.append(spec_heat_se)
           self.avg_energy_ar.append(avg_energy)
           self.avg_energy_sd_ar.append(avg_energy_sd)
           self.avg_energy_se_ar.append(avg_energy_se)
           self.avg_mag_ar.append(avg_mag)
           self.avg_mag_sd_ar.append(avg_mag_sd)
           self.avg_mag_se_ar.append(avg_mag_se)
       if show:
           plt.ioff()

      # Identify the temperature corresponding to the peak susceptibility and specific heat.
      # and record the critical temperature and correlation values.
       suscept_peak_index = np.nanargmax(self.suscept_ar)
       spec_heat_peak_index = np.nanargmax(self.spec_heat_ar)
       crit_index = spec_heat_peak_index
       self.suscept_peak_temp = self.T_global[suscept_peak_index]
       self.spec_heat_peak_temp = self.T_global[spec_heat_peak_index]
       self.crit_temp = self.spec_heat_peak_temp
       self.crit_corr = self.corr_ar_total[crit_index]
       self.crit_ising = self.ising_ar[crit_index]
       best_corr_index = np.nanargmax(self.corr_ar_total)
       self.best_temp = self.T_global[best_corr_index]
       self.best_corr = self.corr_ar_total[best_corr_index]
       self.best_ising = self.ising_ar[best_corr_index]
       if text:
           print('susceptibility peak temperature:', self.suscept_peak_temp)
           print('specific heat peak temperature:', self.spec_heat_peak_temp)
           print('critical temperature:', self.crit_temp)
           print('highest correlation run:')
           print(self.best_ising)
