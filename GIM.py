import numpy as np
import os
from scipy.stats import pearsonr
from scipy.stats import kstest
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
# Support both package imports (``steven.Scripts.ising3``) and running this
# legacy module directly from the Scripts directory.
try:
    from . import utils
    from . import config as cf
except ImportError:  # pragma: no cover - retained for direct script use
    import utils
    import config as cf
import time
import pickle
from abc import ABC, abstractmethod
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
# Saved simulation logs and pickles are intermediate data, not final figures.
DEFAULT_ISING_DATA_DIR = PROJECT_DIR / "DATA" / "simulation data" / "ising data"


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
    
    def __init__(self, Jij, spin_ar = None, directed = False):
        '''
        Initialize structural connectivity (Jij) matrix and array containing spin configurations. Magnetization and
        energy are calculated and stored for the initial spin state.

        :param Jij: Structural connectivity matrix defining the strength of synaptic connections between neurons.

        :param spin_ar: Defines the spin orientation of all neurons. Up spin (excitory) neurons are represented as +1
               and down spin (inhibitory) neurons are represented as -1. If no array is provided, one is randomly
               generated based on the size of the Jij matrix.

        :param directed: Parameter which indicates if the Jij matrix represents a directed or undirected graph (if
               connections between neurons i and j are the same as connections between j to i.
        '''

        self.directed = directed
        self.Jij = utils.normalize_array(Jij)
        self.size = np.shape(Jij)[0]
        if spin_ar is None:
            self.spins = np.random.choice([-1, 1], self.size)
        else:
            self.spins = spin_ar
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

        self.mag = np.sum(self.spins) / self.size
        return self.mag

    def find_dE(self, index):
        '''
        Calculate the change in system energy if a specific set of neurons were to flip spins.

        :param index: Array containing the indices of proposed spin flip neurons.

        :return: Change in system energy.
        '''

        temp_Jij = self.Jij[index].copy()
        temp_Jij[index] = 0
        return -2 * np.sum(temp_Jij * (self.spins * -self.spins[index].reshape((np.size(index), 1))))

    def hamiltonian(self):
        '''
        Calculate the total system energy.

        :return: Total system energy.
        '''

        if not self.directed:
            return -np.sum(self.Jij * (self.spins * self.spins.reshape((np.size(self.spins), 1)))) / 2
        else:
            return -np.sum(self.Jij * (self.spins * self.spins.reshape((np.size(self.spins), 1))))

    def update(self, index, energy = None):
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

    def __init__(self, temp, Jij = cf.avg_Jij, spin_ar = None, directed = False):
        '''
        Initialize simulation parameters. System energy and magnetization is stored per timestep of the simulation in
        arrays. A timer variable is also defined to measure how long the simulation takes.

        :param temp: Array which stores the temperature of each neuron. Typically, temperature is defined as Tu^a
                     where T is the global temperature, alpha is fitting parameter, and u is an array which defines the
                     scaling for each neuron.

        :param Jij: Structural connectivity matrix which stores the strength of synaptic connections between neurons

        :param spin_ar: Array which stores the position and spins of all neurons in the system

        :param directed: Optional parameter which indicates if the Jij matrix is directed
        '''

        self.temp = temp
        self.spin = Spins(Jij, spin_ar, directed)
        self.timer = 0

    def metropolis_step(self, index, dE = None):
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

    def simulate(self, steps, thermalization = None):
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
        self.energy_series = np.zeros(self.steps)
        self.mag_series = np.zeros(self.steps)
        self.spin_series = np.zeros((self.spin.size, self.steps))

        self.therm = thermalization
        for _ in range(thermalization):
            start = time.time()
            self.time_scale()
            self.timer += time.time() - start

        for i in range(self.steps):
            start = time.time()
            self.time_scale()
            self.spin_series[:, i] = self.spin.spins
            self.energy_series[i] = self.spin.total_energy
            self.mag_series[i] = self.spin.magnetization()
            self.timer += time.time() - start

    def generate_FC(self, partial = False):
        '''
        Generates a functional connectivity matrix based on the results of the simulation. The matrix is calculated
        by applying a correlation function over the spin configuration array to find how correlated each neuron is
        with each other. Note that neuron pairs with high correlation tend to switch spins at the same frequency.

        :param partial: Toggles between partial correlation and Pearson correlation. Partial correlation tends to be
                        more accurate as it removes correlation effects from indirect neuron interactions, however
                        takes longer to process.

        :return: Functional connectivity matrix
        '''

        self.partial = partial

        if not self.partial:
            self.functional_connectivity = np.nan_to_num(np.corrcoef(self.spin_series))
        else:
            for ridge in (1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    self.functional_connectivity = np.nan_to_num(
                        utils.part_corr(self.spin_series, ridge=ridge)
                    )
                    break
                except ValueError:
                    continue
            else:
                self.functional_connectivity = np.zeros(
                    (self.spin_series.shape[0], self.spin_series.shape[0])
                )

        return self.functional_connectivity

    def susceptibility(self, beta):
        '''
        Finds the susceptibility of the system after simulation. This measures how volatile the system is, with a high
        susceptibility indicating large changes to global magnetization per time step.

        :param beta: Scaling parameter, often set to 1/the global temperature

        :return: The system susceptibility.
        '''

        m = np.asarray(self.mag_series, dtype=float)
        return self.spin.size * beta * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2)

    def specific_heat(self, beta):
        '''
        Finds the specific heat capacity of the system after simulation. This measures the variability of energy during
        the simulation and can be used to find the critical temperature. At criticality, the system is inbetween states
        of order and disorder, and neurons begin clustering their spins in specific patterns, which resembles the
        dynamics of neurons in the brain. The critical temperature can be found by identifying the temperature value
        which maximizes specific heat.

        :param beta: Scaling parameter, often set to 1/the global temperature
        :return:
        '''

        e = np.asarray(self.energy_series, dtype=float)
        return (beta ** 2) * np.var(e) / self.spin.size

    def correlation(self, emp_FC = cf.avg_FC, diag = False):
        '''
        Calculates the correlation between the simulated and empirical functional connectivity matrices.

        :param emp_FC: Empirical functional connectivity matrix.

        :param diag: If False, calculates the correlation without the diagonal values of each matrix. Diagonal valuess
                     for the FC matrix are always 1, so they will always be perfectly correlated between the simulated
                     and empirical matrices. This can skew results by making the correlation look higher than it
                     actually is.

        :return: Pearson correlation between the simulated and empirical FC.
        '''

        if diag:
            sim_vec = self.functional_connectivity.flatten()
            emp_vec = emp_FC.flatten()
        else:
            sim_vec = utils.flat_remove_diag(self.functional_connectivity)
            emp_vec = utils.flat_remove_diag(emp_FC)

        sim_vec = np.nan_to_num(np.asarray(sim_vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        emp_vec = np.nan_to_num(np.asarray(emp_vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.isfinite(sim_vec) & np.isfinite(emp_vec)
        sim_vec = sim_vec[mask]
        emp_vec = emp_vec[mask]

        if sim_vec.size < 2 or np.std(sim_vec) == 0 or np.std(emp_vec) == 0:
            return 0.0

        r = pearsonr(sim_vec, emp_vec)[0]
        return 0.0 if not np.isfinite(r) else float(r)

    def distance(self):
        avg_TS_1 = utils.get_matrix('avg_TS_1')
        avg_TS_2 = utils.get_matrix('avg_TS_2')
        avg_TS_3 = utils.get_matrix('avg_TS_3')

        dist_1 = kstest(self.spin_series, avg_TS_1).statistic
        dist_2 = kstest(self.spin_series, avg_TS_2).statistic
        dist_3 = kstest(self.spin_series, avg_TS_3).statistic
        avg_dist = np.mean([dist_1, dist_2, dist_3])
        return avg_dist
    


class get_data:
    '''
    This class acts as a wrapper for the Ising class, storing simulation data and providing methods that allows easy
    visualization of parameters.
    '''

    def __init__(self, ising, beta, T_global, alpha, emp_FC = cf.avg_FCp, diag = False, save = False, save_dir = None):
        '''
        Initializes parameters and stores a log of simulation data. Optionally, the class allows the user to save the
        simulation log, along with graphs of time series data acquired during the simulation.

        :param ising: Ising class object post simulation.

        :param beta: Scaling parameter, often set to 1/the global temperature.

        :param T_global: Global temperature.

        :param alpha: temperature fitting parameter.

        :param emp_FC: empirical functional connectivity matrix.

        :param diag: If False, calculates the correlation without the diagonal values of each matrix.

        :param save: If True, saves the simulation log and graphs of simulation data.
        '''

        self.ising = ising
        self.FC = ising.functional_connectivity
        self.emp_FC = emp_FC
        self.Jij = ising.spin.Jij
        self.beta = beta
        self.T_global = T_global
        self.alpha = alpha
        self.save = save

        self.partial = ising.partial
        self.time = ising.timer
        self.correlation = ising.correlation(self.emp_FC, diag)
        self.suscept = ising.susceptibility(beta)
        self.spec_heat = ising.specific_heat(beta)
        self.message = f'SIMULATION LOG\n' \
                  f'global temperature: {T_global}\n' \
                  f'alpha: {alpha}\n' \
                  f'thermalization: {self.ising.therm}\n' \
                  f'partial correlation: {self.partial}\n' \
                  f'time scale: {self.ising}\n' \
                  f'run time: {self.time}s\n' \
                  f'correlation: {self.correlation}\n' \
                  f'susceptibility: {self.suscept}\n' \
                  f'specific heat: {self.spec_heat}'

        if save:
            save_root = Path(save_dir) if save_dir is not None else DEFAULT_ISING_DATA_DIR
            save_root.mkdir(parents=True, exist_ok=True)
            num_folders = len(next(os.walk(save_root))[1])
            self.folder_name = 'ising_simulation_run_' + str(num_folders)
            self.path = save_root / self.folder_name
            os.mkdir(self.path)

            with open(self.path / 'log.txt', 'w') as dir:
                dir.write(self.message)

            pickle.dump(self.ising, open(self.path / 'ising.pickle', 'wb'))

    def __str__(self):
        return self.message

class default_ising(Ising):

    def __init__(self, temp, Jij = cf.avg_Jij, spin_ar = None):
        super().__init__(temp, Jij, spin_ar)
        self.index = np.arange(self.spin.size)

    def __str__(self):
        return 'default Ising'

    def time_scale(self):
        update_index = []
        for i in self.index:
            if self.metropolis_step(i) is not None:
                update_index.append(i)
        self.spin.update(update_index)


class Jij_sorted_ising(Ising):

    def __init__(self, temp, Jij = cf.avg_Jij, directed = False, spin_ar = None):
        super().__init__(temp, Jij, spin_ar, directed)
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

    def __init__(self, temp, Jij = cf.avg_Jij, spin_ar = None, directed = False, num_index = None):
        super().__init__(temp, Jij, spin_ar, directed)
        if num_index is None:
            self.num_index = np.shape(Jij)[0]
        else:
            self.num_index = num_index

    def __str__(self):
        return f'random Ising with num_index = {self.num_index}'

    def time_scale(self):
        random_index = np.random.choice(cf.regions, size = self.num_index)
        for i in random_index:
            dE = self.metropolis_step(i)
            if dE is not None:
                self.spin.update(i, dE)


if __name__ == '__main__':
    t_global = 8 #8.15
    alpha = 2.07 # 2.07
    temp = t_global * (cf.norm_ind_avg_Jij ** alpha)
    beta = 1 / np.mean(temp)
    steps = 3000
    Jij = cf.avg_Jij

    simulation = random_ising(temp, Jij = Jij)
    simulation.simulate(steps, 1000)
    simulation.generate_FC(True)
    plt.matshow(simulation.functional_connectivity)
    plt.show()

    data = get_data(simulation, beta, t_global, alpha, emp_FC = cf.avg_FCp, save = False)
