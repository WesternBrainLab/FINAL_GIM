import numpy as np
import matplotlib.pyplot as plt
import ising as I
import utils
import config as cf
import optuna
from datetime import date
import os
import pickle
import json
import pandas as pd
import FuncCon as fc
import scipy as sp
import matplotlib.colors as colors


class optimize():

    def __init__(self,
                 steps,
                 therm,
                 Jij,
                 multiplier,
                 ising = I.Jij_sorted_ising,
                 spins = np.random.choice([-1, 1], 84),
                 partial = False,
                 directory = cf.OPTIM_DATA,
                 save = False):
        '''
        Preforms parameter annealing to find optimal global temperature and alpha values. This is done by defining an
        error function between the simulated and empirical FC matrices and sampling random alpha and temperature values.
        The values sampled are addaptively selected based on the previous error value, such that if a simulation
        produces low error, the next set of global temperature and alpha values will be close to the previous values.

        :param ising: type of timescale used
        :param spins: set initial spin values
        :param Jij: set Jij matrix
        :param partial: if True, use partial correlation to generate simulated FC matrix
        :param multiplier: temperature multiplier per neuron
        :param save: if True, saves results under simulation data/optimization data
        '''

        self.ising = ising
        self.spins = spins
        self.Jij = Jij
        self.partial = partial
        self.multiplier = multiplier
        self.steps = steps
        self.therm = therm
        self.directory = directory
        self.save = save
        if save:
            run_index = str(len(next(os.walk(cf.OPTIM_DATA))[1]))
            cur_date = date.today()
            save_folder_name = '/parameter optimization run ' + run_index + '_' + cur_date.strftime("%d_%m_%Y")
            self.directory = self.directory + save_folder_name

            os.mkdir(self.directory)


    def train(self, train_FC, trials = 100, temp_range = [0, 1], alpha_range = [0, 3]):
        '''
        Preforms the annealing operation

        :param steps: number of simulations ran
        :param maxfun: sets the "group size" for an annealing run. This sets a limit for how many times a series of
                       update algorithms can be run before resetting the annealing process and selecting new random
                       variables. This prevents the algorithm from being stuck at a local error minimum, which may not
                       be the lowest error the system may achieve
        :param emp_FC: set the emprical FC matrix to be compared to
        :param therm: set number of thermalization steps
        :param no_local_search: If true, the algorithm won't preform a "sub-annealing" operation, which basically does
                                a seperate annealing search that only restricts itself around 1 set of parameters. This
                                is usually done to get super accurate ideal parameters, but for our purposes it's not
                                necessary and is generally just a waste of time
        :param show: If True, shows a live plot of energy and spin data for each annealing run
        '''

        def objective(trial):
            beta = trial.suggest_float('t_glob', t_lower, t_upper)
            alpha = trial.suggest_float('alpha', alpha_lower, alpha_upper)
            t_glob = 1/beta
            temp = t_glob * self.multiplier ** alpha
            thresh = trial.suggest_float('thresh', 0, 1)
            error_arr = []
            train_id = np.arange(nFC)
            np.random.shuffle(train_id)

            for step, id in enumerate(train_id):
                emp_FC = np.array(train_FC[str(id)]['matrix'])
                Jij = self.Jij * utils.get_sign_matrix(emp_FC, thresh)
                time_series = self.ising(temp, Jij = Jij, spin_ar = self.spins.copy())
                time_series.simulate(self.steps, thermalization=self.therm)
                FC = np.array(time_series.generate_FC(partial = self.partial)['matrix'])
                correlate = utils.mat_corr(FC, emp_FC)
                error = ((1 - correlate) + np.sqrt(np.mean((FC - emp_FC) ** 2))) ** 2
                error_arr.append(error)
                avg_error = np.mean(error_arr)
                print('tests complete: ', step + 1, '/', nFC, ' | error: ', error, ' | average error: ', avg_error)
                trial.report(avg_error, step=step)

                if trial.should_prune() or np.isnan(error):
                    raise optuna.TrialPruned()

            return avg_error

        self.study = optuna.create_study(pruner =
                                         optuna.pruners.MedianPruner(n_startup_trials=3,
                                                                     n_warmup_steps=1,
                                                                     interval_steps=1))
        [t_lower, t_upper] = temp_range
        [alpha_lower, alpha_upper] = alpha_range
        nFC = len(train_FC)

        self.study.optimize(objective, n_trials = trials)
        self.dataframe = self.study.trials_dataframe()
        if self.save:
            self.dataframe.to_csv(self.directory + "/log.csv", index=False)
            best_trial = {'lowest error': self.study.best_value,
                          'best parameters': self.study.best_params,
                          'Jij': self.Jij.tolist(),
                          'multiplier': self.multiplier.tolist(),
                          'partial': self.partial}
            with open(self.directory + '/best_trial.json', 'w') as file:
                json.dump(best_trial, file, indent=4)
        return self.dataframe

    def test(self, test_FC):
        t_glob = 1 / self.study.best_params['t_glob']
        alpha = self.study.best_params['alpha']
        thresh = self.study.best_params['thresh']
        temp = t_glob * self.multiplier ** alpha
        nFC = len(test_FC)
        print(nFC)
        error_arr = []

        for id in test_FC:
            print(id)
            emp_FC = test_FC[id]['matrix']
            Jij = self.Jij * utils.get_sign_matrix(emp_FC, thresh)
            time_series = self.ising(temp, Jij=Jij, spin_ar=self.spins.copy())
            time_series.simulate(self.steps, thermalization=self.therm)
            FC = time_series.generate_FC(partial=self.partial)['matrix']

            correlate = utils.mat_corr(FC, emp_FC)
            error = ((1 - correlate) + np.sqrt(np.mean((FC - emp_FC) ** 2))) ** 2
            error_arr.append(error)
            print('tests complete: ', id, '/', nFC, ' | error: ', error)
        avg_error = np.mean(error_arr)
        print('average error: ', avg_error)

    def plot(self, x_axis, y_axis):
        fig = plot_error(x_axis, y_axis, self.dataframe)

        if self.save:
            pickle.dump(fig, open(f'{self.directory}/error_graph.fig.pickle', 'wb'))

        return fig


def plot_error(x_axis, y_axis, dataframe):
    x_axis = 'params_' + x_axis
    y_axis = 'params_' + y_axis
    z = dataframe['value']
    x = dataframe[x_axis]
    y = dataframe[y_axis]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel('value')
    ax.scatter(x, y, z)
    return fig


def error_heatmap(x_axis, y_axis, dataframe, cmap = 'magma', log_x = False, log_y = False, ax = None):
    # Create regular grid
    x_axis = 'params_' + x_axis
    y_axis = 'params_' + y_axis
    z = dataframe['value']
    x = dataframe[x_axis]
    y = dataframe[y_axis]
    nan_id = np.argwhere(np.isnan(z))
    z = np.delete(z, nan_id)
    x = np.delete(x, nan_id)
    y = np.delete(y, nan_id)
    # 2. Determine interpolation coordinates (always linear transformations)
    x_interp = np.log10(x) if log_x else x
    y_interp = np.log10(y) if log_y else y

    # 3. Create a uniform evaluation grid based on interpolation spaces
    xi_linear = np.linspace(x_interp.min(), x_interp.max(), 100)
    yi_linear = np.linspace(y_interp.min(), y_interp.max(), 100)
    Xi_linear, Yi_linear = np.meshgrid(xi_linear, yi_linear)

    # 4. Perform the interpolation in linear-log math space
    # (Using modern RBFInterpolator as sp.interpolate.Rbf is deprecated)
    points = np.vstack([x_interp, y_interp]).T
    rbf = sp.interpolate.RBFInterpolator(points, z, kernel='linear')

    grid_points = np.vstack([Xi_linear.ravel(), Yi_linear.ravel()]).T
    zi = rbf(grid_points).reshape(Xi_linear.shape)

    # 5. Convert grid metrics back to raw scales for pcolormesh mapping
    xi_raw = 10 ** Xi_linear if log_x else Xi_linear
    yi_raw = 10 ** Yi_linear if log_y else Yi_linear

    # 6. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.colorbar(ax)

    hm = ax.pcolormesh(xi_raw, yi_raw, zi, cmap=cmap, shading='auto', norm=colors.LogNorm(vmin=z.min(), vmax=z.max()))
    ax.scatter(x, y, color='white', edgecolor='black', s=20, label='Data Points')

    # Apply log scaling to the axes visuals
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Careful: ax.set_aspect('equal') behaves unpredictably on log-scale axes
    # It attempts to equalize raw data spans rather than visual screen inches.

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return hm


def error_multi_heatmap(dataframe, cmap = 'magma'):
    fig, ax = plt.subplots(1, 4, width_ratios=[.48, .48, .48, .04])
    ax0, ax1, ax2, ax_cb = ax
    plt0 = error_heatmap('t_glob', 'alpha', dataframe, cmap = cmap, log_x = True, ax = ax0)
    plt1 = error_heatmap('t_glob', 'thresh', dataframe, cmap = cmap, log_x = True, log_y = True, ax = ax1)
    plt2 = error_heatmap('thresh', 'alpha', dataframe, cmap = cmap, log_x = True, ax = ax2)
    fig.colorbar(plt0, cax = ax_cb)
    return fig


def load_3d_plots(directory, file_name):
    plot = utils.get_pickle_file(directory, file_name)
    plt.show()
    return plot


if __name__ == '__main__':
    dataframe = pd.read_csv('E:\Python\GIM_FINAL\simulation data\optimization data\parameter optimization run 1_14_08_2026\log.csv')
    heatmap = plot_error('alpha', 'thresh', dataframe)
    plt.show()