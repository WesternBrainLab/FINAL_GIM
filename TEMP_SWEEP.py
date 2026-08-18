import numpy as np
import scipy.stats as sp
import scipy.integrate as int
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    steps = 4000
    thermalization = 2000
    min_temp = 0.05
    max_temp = 10
    temp_step = 100
    alpha = 2
    Jij = cf.avg_Jij * utils.get_sign_matrix(cf.avg_FC)
    simulation = temp_sweep(min_temp, max_temp, temp_step, alpha, Jij = Jij, ising = I.Jij_sorted_ising)
    simulation.simulate(steps, thermalization, partial = False, show = True)
    simulation.graph_data(True)