import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import UTILS as utils
import time
import CONFIG as cf
import pickle
from abc import ABC, abstractmethod
from matplotlib.colors import TwoSlopeNorm
from copy import copy
import json
import FUNC_CON as fc

'''
This file contains the class objects which define the Ising model simulation. 3 (base) classes are defined; one for
storing system information (Spins), another to run the simulation (Ising), and one for storing and visualizing 
simulation data (get_data). 

In short, the simulation runs off of a Metropolis Hastings algorithm, which is a type of Monte Carlo algorithm. Don't 
worry! These terms aren't as daunting as they seem. Essentially, Monte Carlo algorithms encompass any type of 
simulation where a bunch of random samples are taken in order to get a probability distribution for the samples. It 
gets it's name from the Monte Carlo casino, where it was originally developed to test the win/loss rates of casino 
games by simulating a large number of rounds and tracking the results. A Metropolis-Hastings algorithm is a type of 
Monte Carlo algorithm which first draws a random sample from a system, then checks if the system wants to replace it's
current sample with the new one using a set of rules. A brief outline on how the simulation works is given below:

INITIALIZATION
    The simulation requires 3 initial parameters: 
              Spins - An array which stores the positions and initial spin orientation of each neuron in the system.
              
        Temperature - An array which defines how susceptible each neuron is to changing spin state (higher temperature
                      values means more likely to change states).
                      
                Jij - The structural connectivity matrix which stores the strength of synaptic connections between
                      neurons. This matrix should be square, with side lengths equal to the number of neurons in the 
                      system. Each row and column represents a specific neuron, such that the value of the matrix at 
                      row i and column j represents the strength of the connection going from neuron i to j. In general,
                      these connections should be undirected (meaning going from i to j is the same as j to i), however
                      the program also allows an optional parameter for directed matrices as well.

ISING MODEL
    To simulate the Ising model, we implement the Metropolis-Hastings algorithm by first selecting neurons which we 
    propose to have their spins flipped. The acceptance criteria is then set so that if the total energy of the system 
    is decreased, the proposal is immediately accepted, and if the total energy increases, the neuron still has a 
    chance of flipping, based on the magnitude of the change in energy and it's temperature (where larger increases to 
    energy, and lower temperature values corresponds to a lower acceptance chance). This is based on the physical 
    principle that any system prefers to stay in it's lowest energy state, however interactions with the environment 
    (in this case, represented by the temperature of the system) can provide excess energy. Essentially, we're
    simulating how neurons evolve in time, with each step of our simulation representing a "time step", or small jump
    in time.
    
    The simulation begins by preforming a "thermalization" process, where it's let to run for a given amount of time 
    steps without any data being collected, to allow the initial state of the system to "adapt" to the environment 
    defined by the simulation. Then, the energy level and other metrics are collected per time step.

'''


class Spins:
    ''''
    This class stores all nueron arrangements and spin information, and provides a function which allows spins of
    individual nuerons to be updated
    '''

    def __init__(self, Jij, spin_ar=None):
        '''
        Initialize structural connectivity (Jij) matrix and array containing spin configurations. Magnetization and
        energy are calculated and stored for the initial spin state.

        :param Jij: Structural connectivity matrix defining the strength of synaptic connections between neurons.

        :param spin_ar: Defines the spin orientation of all neurons. Up spin (excitory) neurons are represented as +1
               and down spin (inhibitory) neurons are represented as -1. If no array is provided, one is randomly
               generated based on the size of the Jij matrix.
        '''

        self.Jij = utils.normalize_array(Jij)
        self.size = np.shape(Jij)[0]
        if spin_ar is None:
            self.spins = np.random.choice([-1, 1], self.size)
        else:
            self.spins = spin_ar.copy()
        self.total_energy = self.hamiltonian()
        self.mag = self.magnetization()

    def energy(self, i, j):
        '''
        Calculate the energy between neurons i and j, defined as the negative Jij value corresponding to the two
        neurons, multiplied by the spin value of both neurons. Energy is positive (increased) if the neurons have
        opposite spin and is negative (decreased) if they have the same spin.

        :param i: First neuron.

        :param j: Second neuron.

        :return: Energy between neurons.
        '''

        return -np.sum(self.Jij[i, j] * self.spins[i] * self.spins[j])

    def magnetization(self):
        '''
        Calculate the magnetization value of the system. This essentially gives a percentage of how synchronized the
        spins of each neuron is, with a value of 1 indicating all neurons have the same spins (either all +1 or all -1)
        and a value of 0 indicating a mixed state of an equal amount of +1 and -1 spins.

        :return: System magnetization.
        '''

        self.mag = abs(np.sum(self.spins)) / self.size
        return self.mag

    def find_dE(self, index):
        '''
        Calculate the change in system energy if a specific set of neurons were to flip spins.

        :param index: Array containing the indices of proposed spin flip neurons.

        :return: Change in system energy.
        '''

        temp_Jij = self.Jij[index, :]
        if type(index) is list:
            temp_Jij[:, index] = 0
        return -2 * np.sum(temp_Jij * (self.spins * -self.spins[index].reshape((np.size(index), 1))))

    def hamiltonian(self):
        '''
        Calculate the total system energy.

        :return: Total system energy.
        '''

        return -np.sum(self.Jij * (self.spins * self.spins.reshape((np.size(self.spins), 1)))) / 2

    def update(self, index, energy=None):
        '''
        Update the system by changing the spins of a given set of neurons, and recalculating the system energy.

        :param index: Array containing the indices of spin flip neurons.

        :param energy: Optional parameter that let's the user input an energy change value. If left as None, energy
               change is found using the find_dE method.

        :return: The updated system.
        '''

        if energy is None:
            energy = self.find_dE(index)
        self.total_energy += energy
        self.spins[index] *= -1


class Ising(ABC):
    '''
    This class is used as the base structure for implementations of the Ising model.
    '''

    def __init__(self, temp, Jij, beta = 1, spin_ar=None, save=False, directory=cf.ISING_DATA):
        '''
        Initialize simulation parameters. System energy and magnetization is stored per timestep of the simulation in
        arrays. A timer variable is also defined to measure how long the simulation takes.

        :param temp: Array which stores the temperature of each neuron. Typically, temperature is defined as Tu^a
                     where T is the global temperature, alpha is fitting parameter, and u is an array which defines the
                     scaling for each neuron.

        :param Jij: Structural connectivity matrix which stores the strength of synaptic connections between neurons

        :param spin_ar: Array which stores the position and spins of all neurons in the system
        '''

        self.temp = temp
        self.Jij = Jij
        self.spin = Spins(Jij, spin_ar)
        self.timer = 0
        self.save = save
        self.beta = beta
        self.directory = directory

    def metropolis_step(self, index, dE=None):
        '''
        Sets the update conditions for the simulation per time step. A set of neurons are proposed to flip, and their
        potential change in energy (dE) is calculated. if dE is negative (system goes into a lower energy state), the
        flip proposal is accepted and the system updates. If it is positive, the chance of the proposal being accepted
        is then dictated by the Boltzmann distribution, with temperature set to the temperature of the neuron.

        :param index: Array of neuron flip proposal indices.

        :param dE: Optional parameter to set the change in energy value. If left as none, the find_dE method from the
                   Spins class is used to find the change in energy

        :return: None, or updated system
        '''

        if dE is None:
            dE = self.spin.find_dE(index)
        if dE < 0 or random.random() < np.exp(-dE / self.temp[index]):
            return dE
        else:
            return None

    @abstractmethod
    def time_scale(self):
        '''
        This method is defined in child classes of the Ising parent class and is used to define the rules for neuron
        flip proposal selection.

        :return: Array of neuron flip proposal indices.
        '''

        pass

    def simulate(self, steps, thermalization, log = False):
        '''
        Performs the Ising model simulation and records the system state, total energy and magnetization for each
        time step.

        :param steps: Define total amount of time steps to simulate after thermalization.

        :param thermalization: Define the amount of thermalization steps. If set to None, creates a temporary
                               magnetization array and calculates the variance of the array per time step.
                               Thermalization is complete when the variance is less than 5e-2. (Note: this doesn't work
                               very well, and you'll mostly be stuck in an infinite loop of thermalization.)

        :return: Arrays containing the system state, total energy and magnetization for each time step of the
                 simulation.
        '''

        self.steps = steps
        self.therm = thermalization
        self.spin_series = np.zeros((self.spin.size, steps))
        self.energy_series = np.zeros((steps, 1))
        self.mag_series = np.zeros((steps, 1))

        start = time.time()

        for _ in range(thermalization):
            self.time_scale()

        for i in range(steps):
            self.time_scale()
            self.spin_series[:, i] = self.spin.spins
            self.energy_series[i] = self.spin.total_energy
            self.mag_series[i] = self.spin.magnetization()

        self.timer = time.time() - start

        if log:
            self.log = {
                'temperature': self.temp,
                'Jij': self.Jij,
                'thermalization': thermalization,
                'steps': steps,
                'time scale': self.__str__(),
                'run time': self.timer,
                'susceptibility': self.susceptibility(self.beta),
                'specific heat': self.specific_heat(self.beta)}

        if self.save:
            num_folders = len(next(os.walk(self.directory))[1])
            self.folder_name = 'ising_simulation_run_' + str(num_folders)
            self.directory = cf.ISING_DATA + self.folder_name
            os.mkdir(self.directory)

            if log:
                with open(self.directory + '/log.json', 'w') as file:
                    json.dump(self.log, file, indent=4)

            pickle.dump(self, open(self.directory + '/ising.pickle', 'wb'))

        if log:
            return self.log

    def generate_FC(self, partial = False):
        '''
        Generates a functional connectivity matrix based on the results of the simulation. The matrix is calculated
        by applying a correlation function over the spin configuration array to find how correlated each neuron is
        with each other. Note that neuron pairs with high correlation tend to switch spins at the same frequency.

        :return: Functional connectivity matrix
        '''
        self.partial = partial
        self.functional_connectivity = fc.generate_FC(self.spin_series, sim = True, partial = partial)

        return self.functional_connectivity

    def susceptibility(self, beta):
        '''
        Finds the susceptibility of the system after simulation. This measures how volatile the system is, with a high
        susceptibility indicating large changes to global magnetization per time step.

        :return: The system susceptibility.
        '''
        return (np.var(np.abs(self.mag_series))) * beta

    def specific_heat(self, beta):
        '''
        Finds the specific heat capacity of the system after simulation. This measures the variability of energy during
        the simulation and can be used to find the critical temperature. At criticality, the system is inbetween states
        of order and disorder, and neurons begin clustering their spins in specific patterns, which resembles the
        dynamics of neurons in the brain. The critical temperature can be found by identifying the temperature value
        which maximizes specific heat.

        :return:
        '''
        return (np.var(self.energy_series)) * (beta ** 2)

    def graph_mag_energy(self, show=True):
        '''
        Produces 2 graphs, one showing mean magnetization vs time and another showing mean energy vs time. If self.save
        == True, running this method will also save the graphs as a png image.

        :param show: If True, the graph is displayed.

        :return: Saved/displayed graphs
        '''
        iterations = np.arange(self.steps)

        mpl.rcParams['lines.markersize'] = 3
        figure, axis = plt.subplots(1, 2)
        axis[0].scatter(iterations, self.mag_series)
        axis[0].plot(iterations, utils.average_series(self.mag_series), 'r', label='average mag')
        axis[0].set_xlabel('steps')
        axis[0].set_ylabel('magnetization')
        axis[0].set_ylim([-1, 1])
        axis[0].legend()
        axis[1].set_ylim([np.min(self.energy_series), np.max(self.energy_series)])
        axis[1].scatter(iterations, self.energy_series)
        axis[1].plot(iterations, utils.average_series(self.energy_series), 'r', label='average energy')
        axis[1].set_xlabel('steps')
        axis[1].set_ylabel('energy')
        axis[1].legend()

        if self.save:
            figure.savefig(self.directory + '/energy_mag_graph.png')

        if show:
            plt.show()
        else:
            return figure, axis

    def graph_FC(self, show=True, title='simulated FC'):
        '''
        Produces heat map plots showing the simulated and empirical FC matrices, as well as the Jij matrix.If
        self.save == True, running this method will also save the graphs as a png image.

        :param show: If True, the graph is displayed.

        :return: Saved/displayed heatmaps.
        '''

        figure, axis = plt.subplots(1, 2, figsize=(10, 4))
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        sim_FC = axis[0].matshow(self.functional_connectivity['matrix'], cmap='coolwarm', norm=norm)
        axis[0].set_title(title)
        axis[1].matshow(self.Jij, cmap='coolwarm', norm=norm)
        axis[1].set_title('Jij')
        figure.colorbar(sim_FC, fraction=0.046, pad=0.04)

        if self.save:
            figure.savefig(self.directory + '/matrix_graphs.png')

        if show:
            plt.show()
        else:
            return figure, axis

    def graph_everything(self, show = True):
        '''
        Produces all the previous plots meantioned above. If self.save == True, running this method will also save
        the graphs as a png image.

        :param show: If True, the graph is displayed.

        :return: Saved/displayed heatmaps.
        '''

        energy_series = self.energy_series
        mag_series = self.mag_series
        iterations = np.arange(self.steps)

        mpl.rcParams['lines.markersize'] = 3
        figure, axis = plt.subplots(2, 2, figsize = (12, 9), constrained_layout = True)
        axis[0, 0].scatter(iterations, mag_series)
        axis[0, 0].plot(iterations, utils.average_series(mag_series), 'r')
        axis[0, 0].set_ylim([0, 1])
        axis[0, 0].set_xlabel('steps')
        axis[0, 0].set_ylabel('magnetization')

        axis[0, 1].set_ylim([np.min(energy_series), np.max(energy_series)])
        axis[0, 1].scatter(iterations, energy_series)
        axis[0, 1].plot(iterations, utils.average_series(energy_series), 'r')
        axis[0, 1].set_xlabel('steps')
        axis[0, 1].set_ylabel('energy')

        axis[1, 0].matshow(self.functional_connectivity['matrix'])
        axis[1, 0].set_title('simulated FC')
        axis[1, 1].matshow(self.Jij)
        axis[1, 1].set_title('Jij')

        if self.save:
            figure.savefig(self.directory + '/everything_graphs.png')

        if show:
            plt.show()
        else:
            return figure, axis


class Jij_sorted_ising(Ising):

    def __init__(self, temp, Jij, beta = 1, spin_ar=None, save=False, directory=cf.ISING_DATA):
        super().__init__(temp, Jij, beta, spin_ar, save, directory)
        ind_avg_Jij = np.mean(Jij, 0)
        self.index = utils.cross_sort(ind_avg_Jij)

    def __str__(self):
        return 'Jij sorted Ising'

    def time_scale(self):
        for i in self.index:
            dE = self.metropolis_step(i)
            if dE is not None:
                self.spin.update(i, dE)


class random_ising(Ising):

    def __init__(self, temp, Jij, beta = 1, spin_ar=None, clusters=1, num_index = None, save=False, directory=cf.ISING_DATA):
        super().__init__(temp, Jij, beta, spin_ar, save, directory)
        self.regions = np.shape(Jij)[0]
        self.clusters = clusters
        if num_index is None:
            self.num_index = self.regions
        else:
            self.num_index = num_index

    def __str__(self):
        return f'random Ising with num_index = {self.num_index}'

    def time_scale(self):
        rng = np.random.default_rng()
        for _ in range(self.clusters):
            random_index = rng.choice(self.regions, size=self.num_index, replace=False)
            for i in random_index:
                dE = self.metropolis_step(i)
                if dE is not None:
                    self.spin.update(i, dE)


if __name__ == '__main__':
    t_global = 5.69
    alpha = 1.5
    temp = t_global * (cf.norm_ind_avg_Jij ** alpha)
    beta = 1 / np.mean(temp)
    steps = 2000
    Jij = cf.avg_Jij * utils.get_sign_matrix(cf.avg_FC)

    simulation = Jij_sorted_ising(temp, Jij=Jij)
    simulation.simulate(steps, 1000)
    simulation.generate_FC(False)
    plt.matshow(simulation.functional_connectivity)
    plt.show()

    data = get_data(simulation, beta, t_global, alpha, emp_FC=cf.avg_FC, save=False)
    data.graph_mag_energy()
    data.graph_FC()
