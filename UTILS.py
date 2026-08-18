import numpy as np
import pandas as pd
import scipy as sp
import os
import pickle
import time
from copy import copy
from scipy.stats import pearsonr


def get_folder(folder_name, directory = os.path.dirname(__file__)):
    current_directory = directory
    folder_from_directory = os.path.join(current_directory, folder_name)
    return folder_from_directory


def get_matrix(file_name, directory = os.path.dirname(__file__)):
    file_path = os.path.join(directory, file_name)
    with open(file_path, newline='') as csvfile:
        matrix_from_file = np.genfromtxt(csvfile, delimiter = ',')
        return matrix_from_file


def save_matrix(matrix, name):
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(name, index = False, header = False)


def matrix_from_dir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + '/' + f)]
    matrix_ar = []

    for file_name in files:
        #file_path = get_matrix(os.path.join(directory, file_name))
        file_path = get_matrix(file_name, directory)
        matrix_ar.append(file_path)
    return matrix_ar


def minmax_norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def df_to_text(data, directory, file_name):
    path = directory + '/' + file_name
    with open(path, 'w') as file:
        data_string = data.to_string()
        file.write(data_string)


def part_corr(time_series, ridge=1e-8):
    """
    Pairwise partial-correlation matrix conditioned on all other variables.

    Parameters
    ----------
    time_series : ndarray, shape (num_neurons, num_timepoints)
    ridge : float
        Small diagonal regularization for numerical stability.

    Returns
    -------
    partial_corr : ndarray, shape (num_neurons, num_neurons)
    """

    X = np.asarray(time_series, dtype=float)

    if X.ndim != 2:
        raise ValueError("time_series must have shape (num_neurons, num_timepoints).")

    num_neurons, num_steps = X.shape

    if num_steps < 2:
        raise ValueError("At least two time points are required.")

    # Remove the mean from each neuron time series
    X = X - X.mean(axis=1, keepdims=True)

    # Sample covariance matrix
    covariance = (X @ X.T) / (num_steps - 1)

    # Regularization helps when covariance is singular or nearly singular
    scale = np.trace(covariance) / num_neurons
    covariance += ridge * scale * np.eye(num_neurons)

    # Precision matrix
    precision = np.linalg.pinv(covariance)

    diagonal = np.diag(precision)

    # Guard against invalid / near-zero diagonal entries
    if np.any(diagonal <= 0):
        raise ValueError(
            "Precision matrix has non-positive diagonal entries. "
            "Increase ridge or remove constant time series."
        )

    partial_corr = -precision / np.sqrt(np.outer(diagonal, diagonal))

    # The diagonal is defined as 1 for a correlation matrix
    np.fill_diagonal(partial_corr, 1.0)

    # Remove tiny numerical asymmetries
    partial_corr = 0.5 * (partial_corr + partial_corr.T)

    return partial_corr


def normalize_array(array):
    return array / np.max(np.abs(array))


def cross_sort(sort_array, *args, hi_lo = True):
    if hi_lo:
        copy_array = np.sort(np.unique(sort_array))[::-1]
    else:
        copy_array = np.sort(np.unique(sort_array))

    index = []
    for i in copy_array:
        current_index = np.where(sort_array == i)[0]
        if np.size(current_index) > 1:
            for j in current_index:
                index.append(j)
        else:
            index.append(current_index[0])
    if args:
        return args[index]
    return index


def percent_error(actual, expected):
    return np.abs((actual - expected) / expected)


def average_matrices(*arrays):
    avg_ar = np.zeros(np.shape(arrays[0]))

    for ar in arrays:
        avg_ar += ar

    avg_ar /= len(arrays)
    return avg_ar


def flat_remove_diag(array):
    new_ar = []
    length = range(np.shape(array)[0])
    for y in length:
        for x in length:
            if x != y:
                new_ar.append(array[y, x])
    return np.array(new_ar)


def average_series(series):
    return np.cumsum(series) / (np.arange(np.size(series)) + 1)


def receiver_operating_characteristic(input_matrix, check_matrix):
    fpr_ar, tpr_ar = [0], [0]

    for thresh in np.linspace(1, 0.01, 100):
        check_matrix_copy = check_matrix.copy()
        input_matrix_copy = input_matrix.copy()

        input_matrix_copy[input_matrix < thresh] = 0
        input_matrix_copy[input_matrix >= thresh] = 1

        check_matrix_copy[check_matrix_copy < thresh] = 0
        check_matrix_copy[check_matrix_copy >= thresh] = 1
        _, counts = np.unique(check_matrix_copy, return_counts = True)

        compare_matrix = input_matrix_copy == check_matrix_copy

        true_positive = compare_matrix * input_matrix_copy
        false_positive = np.abs(compare_matrix - 1) * input_matrix_copy

        fpr_ar.append(np.size(false_positive[false_positive == 1]) / counts[0])
        tpr_ar.append(np.size(true_positive[true_positive == 1]) / counts[1])

    fpr_ar.append(1)
    tpr_ar.append(1)
    return tpr_ar, fpr_ar, sp.integrate.trapezoid(tpr_ar, x = fpr_ar)


def get_pickle_file(directory, file_name):
    directory = directory + '/' + file_name
    with open(directory, 'rb') as picklefile:
        return pickle.load(picklefile)


def get_sign_matrix(matrix, thresh = 0):
    sign_mat = matrix.copy() / np.max(np.abs(matrix))
    sign_mat[np.abs(sign_mat) < thresh] = 1
    sign_mat[sign_mat > 0] = 1
    sign_mat[sign_mat < 0] = -1
    return sign_mat


def mat_corr(M1, M2, diag=False):
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
        return pearsonr(M1.flatten(), M2.flatten())[0]
    else:
        return pearsonr(flat_remove_diag(M1), flat_remove_diag(M2))[0]