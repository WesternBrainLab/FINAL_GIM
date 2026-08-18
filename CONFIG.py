from pathlib import Path
import json

PROJECT_ROOT = str(Path(__file__).parent.resolve().as_posix()) + '/'
DATA_DIR = PROJECT_ROOT + 'DATA'
ISING_DATA = DATA_DIR + 'simulation data/ising data'
OPTIM_DATA = DATA_DIR + 'simulation data/optimization data'
TEMP_SWEEP_DATA = DATA_DIR + 'simulation data/temp sweep data'
with open(PROJECT_ROOT + 'FC_DATASET_PATH.json', 'r') as file:
    FC_PATHS = json.load(file)