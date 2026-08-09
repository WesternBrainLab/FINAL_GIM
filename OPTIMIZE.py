import numpy as np
import matplotlib.pyplot as plt
import ising3 as I
import utils as utils
import config as cf
import optuna
import pandas as pd
from datetime import date
from pathlib import Path
import os
import pickle


# Optimization logs and serialized diagnostic figures are intermediate data.
DATA_DIR = Path(__file__).resolve().parent / "DATA"
OPTIMIZATION_DATA_DIR = DATA_DIR / "simulation data" / "optimization data"


class optimize():

    def __init__(self, ising, therm, steps, train_trials, temp_range, alpha_range,
                 spins = np.random.choice([-1, 1], 84), 
                 partial = True, 
                 multiplier = utils.normalize_array(cf.ind_avg_Jij),
                 FC_dir = 'FC_data_processed',
                 save = False):
        self.ising = ising
        self.spins = spins
        self.therm = therm
        self.steps = steps
        self.train_trials = train_trials
        [self.t_lower, self.t_upper] = temp_range
        [self.alpha_lower, self.alpha_upper] = alpha_range
        self.partial = partial
        self.multiplier = multiplier
        self.FC_arr = []
        for file in os.listdir(FC_dir):
            ts = utils.get_matrix(file, FC_dir)
            self.FC_arr.append(np.corrcoef(ts.T))
        self.Jij = cf.avg_Jij.copy()
        self.save = save
        self.cur_date = date.today()

    def save_run(self):
        data = {
            'error': self.error,
            'correlation': self.correlate,
            'global temp': self.T_global,
            'alpha': self.alpha
        }

        self.dataframe = pd.DataFrame(data)

        save_folder_name = 'parameter optimization run ' + self.run_index + '_' + self.cur_date.strftime("%d_%m_%Y")
        self.directory = OPTIMIZATION_DATA_DIR / save_folder_name
        self.directory.mkdir(parents=True, exist_ok=False)

        log_path = self.directory / 'log'
        with open(log_path, 'w') as file:
            error = self.optim_param.fun
            T_global, alpha = self.optim_param.x
            file.write('error: {:.2f} | highest correlation: {:.2f}\n'
                       'best global temp: {} | best alpha: {} | best average temperature: {:.2f}\n'
                       'time scale: {} | partial correlation: {}\n\n'
                       .format(error, np.max(self.correlate),
                               T_global, alpha, np.mean(T_global * self.multiplier ** alpha),
                               self.ising, self.partial))
            file.write(self.dataframe.to_string())

    def error_func(self, ising, FC):
                FC = ising.generate_FC(partial=self.partial)
                correlate = ising.correlation(self.test_arr)
                return ((1 - correlate) + np.sqrt(np.mean((FC - cf.avg_FC) ** 2))) ** 2

    def get_training_FC(self):
        train_size = round(len(self.FC_arr) * 0.7)
        train_id = np.random.choice(np.arange(len(self.FC_arr)), size = train_size, replace = False).astype(np.uint8)
        train_arr = utils.average_matrices(np.array(self.FC_arr)[train_id, :, :])
        test_arr = [x for x in self.FC_arr if x not in train_arr]
        return train_arr, test_arr
    
    def train_objective(self, trial):
        t_glob = trial.suggest_float('t_glob', self.t_lower, self.t_upper)
        alpha = trial.suggest_float('alpha', self.alpha_lower, self.alpha_upper)
        temp = t_glob * (self.multiplier ** alpha)
        thresh = trial.suggest_float('thresh', 0, 1)

        Jij = self.Jij * utils.threshold_matrix(self.train_FC, thresh)
        curr_ising = self.ising(temp, spin_ar = self.spins.copy(), Jij = Jij)
        curr_ising.simulate(self.steps, thermalization = self.therm)
        return self.error_func(curr_ising, self.train_FC)

    def optimize(self, test_trials):
        avg_error = []
        params = np.zeros((test_trials, 3))
        for i in range(test_trials):
            self.train_FC, self.test_arr = self.get_training_FC()
            study = optuna.create_study()
            study.optimize(self.train_objective, n_trials = self.train_trials)
            best_glob, best_alpha, best_thresh = study.best_params().values()
            params[i, :] = study.best_params().values()
            best_temp = best_glob * (self.multiplier ** best_alpha)
            best_Jij = self.Jij * utils.threshold_matrix(self.train_FC, best_thresh)
            test_run = self.ising(best_temp, Jij = best_Jij)
            test_run.simulate(self.steps, thermalization = self.therm)
            test_error = []
            for FC in self.FC_arr:
                test_error.append(self.error_func(test_run, FC))
            avg_error.append(np.mean(test_error))

        best_params = params[np.argmin(avg_error), :]
        return best_params

    def plot_error(self, show = True):
        np_T_global = np.array(self.T_global)
        np_alpha = np.array(self.alpha)
        np_error = np.array((self.error))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('global temp')
        ax.set_ylabel('alpha')
        ax.set_zlabel('error')

        ax.scatter(np_T_global, np_alpha, np_error)
        if self.save:
            pickle.dump(fig, open(self.directory / 'error graph.fig.pickle', 'wb'))

        if show:
            plt.show()

    def plot_auc(self, show = True):
        FC_auc, Jij_auc = [], []

        for sim_FC in self.FC:
            _, _, auc = utils.receiver_operating_characteristic(sim_FC, cf.avg_FC)
            FC_auc.append(auc)

            _, _, auc = utils.receiver_operating_characteristic(sim_FC, cf.avg_Jij)
            Jij_auc.append(auc)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('global temp')
        ax.set_ylabel('alpha')
        ax.set_zlabel('auc')

        ax.scatter(self.T_global, self.alpha, FC_auc)
        ax.scatter(self.T_global, self.alpha, Jij_auc)

        if self.save:
            pickle.dump(fig, open(self.directory / 'auc graph.fig.pickle', 'wb'))

        if show:
            plt.show()


def load_3d_plots(folder_name, file_name):
    directory = OPTIMIZATION_DATA_DIR / folder_name
    plot = utils.get_pickle_file(str(directory), file_name)
    plt.show()


if __name__ == '__main__':
    #timescale = [I.random_ising, I.Jij_sorted_ising]
    #partial = [True, False]
    #for _ in range(3):
    #    spins = np.random.choice([-1, 1], 84)
    #    for ts in timescale:
    #        for p in partial:
    #            optim = optimize(ts, spins, partial = p, save = True)
    #            optim.anneal(2000, 1000, therm = 1000, show = False)
    #            optim.plot_error(False)
    #            optim.plot_auc(False)
    optim = optimize(I.Jij_sorted_ising, 
                     therm = 1000, 
                     steps = 2000, 
                     train_trials = 100, 
                     temp_range = [0.5, 12], 
                     alpha_range = [0, 2], 
                     partial = False, 
                     FC_dir = '/media/brainlab-uwo/Elements1/Kayla/steven/Data/TS_Data/TS_1',
                     save = False)
    optim.optimize(test_trials = 100)
    #folder_name = 'parameter optimization run 34_12_03_2025'
    #file_name = 'error graph.fig.pickle'
    #load_3d_plots(folder_name, file_name)
