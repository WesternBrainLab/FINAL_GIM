from pathlib import Path
import json
import os
import numpy as np
import UTILS as utils

PROJECT_ROOT = str(Path(__file__).parent.resolve().as_posix()) + '/'
DATA_DIR = PROJECT_ROOT + 'DATA/'
RESULTS_DIR = PROJECT_ROOT + 'RESULTS/'
ISING_DATA = RESULTS_DIR + 'GIM/'
OPTIM_DATA = RESULTS_DIR + 'OPTIMIZE/'
TEMP_SWEEP_DATA = RESULTS_DIR + 'TEMP_SWEEP/'
for _results_path in (ISING_DATA, OPTIM_DATA, TEMP_SWEEP_DATA):
    os.makedirs(_results_path, exist_ok=True)

with open(PROJECT_ROOT + 'FC_DATASET_PATH.json', 'r') as file:
    FC_PATHS = json.load(file)

avg_Jij = utils.get_matrix(
    'Jij data_processed/avg_Jij_no_outliers_norm', directory=DATA_DIR
)
avg_FC = utils.get_matrix(
    'FC data_processed/avg_TS_1', directory=DATA_DIR
)
norm_ind_avg_Jij = utils.normalize_array(np.mean(np.abs(avg_Jij), axis=0))
