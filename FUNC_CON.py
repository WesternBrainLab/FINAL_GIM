import numpy as np
import UTILS as utils
from pathlib import Path
import json
import CONFIG as cf


def generate_FC(matrix, sim = False, partial = False):
    if matrix.shape[0] != matrix.shape[1] or matrix != matrix.T:
        if not partial:
            matrix = np.nan_to_num(np.corrcoef(matrix)).tolist()
        else:
            matrix = utils.part_corr(matrix).tolist()

    FC = {'sim': sim,
          'partial': partial,
          'matrix': matrix}
    return FC


def generate_dataset(FC_directory, partial = False, transpose = False):
    dir_path = Path(FC_directory)
    csv_files = [file.as_posix() for file in dir_path.glob("*.csv")]
    dataset = {}

    for i, file in enumerate(csv_files):
        FC_matrix = np.loadtxt(file, delimiter = ',')
        if transpose:
            FC_matrix = FC_matrix.T
        FC = generate_FC(FC_matrix, False, partial)
        dataset[i] = FC

    return dataset


def generate_avg_FC(FC_directory, partial = False, transpose = False):
    dir_path = Path(FC_directory)
    csv_files = [str(file) for file in dir_path.glob("*.csv")]
    avg_matrix = []

    for file in csv_files:
        matrix = np.loadtxt(file, delimiter = ',')
        if transpose:
            matrix = matrix.T
        if matrix.shape[0] != matrix.shape[1] or matrix != matrix.T:
            if not partial:
                matrix = np.nan_to_num(np.corrcoef(matrix))
            else:
                matrix = utils.part_corr(matrix)

        avg_matrix.append(matrix)

    avg_matrix = np.mean(np.array(avg_matrix), 0)
    FC = {'sim': False,
          'partial': partial,
          'matrix': avg_matrix.tolist()}
    return FC


def save_to_json(FC, save_path, name):
    save_path = save_path + '/' + name + '.json'
    save_json = json.dumps(FC, indent=4)

    with open(save_path, 'w') as file:
        file.write(save_json)

    with open(cf.PROJECT_ROOT / 'FC_DATASET_PATH.json', 'r') as file:
        dataset_path = json.load(file)
        dataset_path[name] = save_path
    with open(cf.PROJECT_ROOT / 'FC_DATASET_PATH.json', 'w') as file:
        json.dump(dataset_path, file, indent=4)


def load_FC(name):
    path = cf.FC_PATHS[name]
    with open(path, 'r') as file:
        FC = json.load(file)
        return FC
