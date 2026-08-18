# FINAL_GIM

FINAL_GIM contains the Ising/GIM simulation pipeline, empirical functional-connectivity data, parameter optimization, temperature sweeps, and analysis scripts used by Western Brain Lab.

## Requirements

- Python 3.10 or newer
- A writable working copy of this repository
- The packages listed in `requirements.txt`

The repository uses case-sensitive module names (`CONFIG`, `GIM`, `UTILS`, and `FUNC_CON`). Run commands from the `FINAL_GIM` directory so these local imports resolve correctly.

## Installation

### Linux and macOS

```bash
cd FINAL_GIM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd FINAL_GIM
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

To leave the virtual environment, run `deactivate`.

## Repository layout

```text
FINAL_GIM/
├── CONFIG.py                 Shared paths and loaded empirical matrices
├── GIM.py                    Ising/GIM simulation classes
├── OPTIMIZE.py              Parameter optimization utilities
├── TEMP_SWEEP.py            Temperature-sweep utilities
├── UTILS.py                 Matrix, FC, and numerical helpers
├── FUNC_CON.py              Functional-connectivity dataset helpers
├── FC_DATASET_PATH.json     Dataset metadata
├── DATA/                    Local Jij, FC, PET, and time-series data
├── Tests/                   Analysis and validation scripts
└── RESULTS/                 Generated figures and summaries
```

## Configuration

`CONFIG.py` derives `PROJECT_ROOT` from the location of the file and uses the repository-local `DATA/` directory. It loads:

- `DATA/Jij data_processed/avg_Jij_no_outliers_norm` as `avg_Jij`
- `DATA/FC data_processed/avg_TS_1` as `avg_FC`
- `DATA/` as `DATA_DIR`

If you use a different dataset, update the corresponding paths in `CONFIG.py` and ensure the replacement matrices have compatible dimensions. `FC_DATASET_PATH.json` contains dataset metadata; its paths must be changed if datasets are moved outside this repository.

## Running the analyses -- IGNORE FOR NOW AS IT IS ADDITIONAL TESTS

From the repository root:

```bash
python Tests/Mu_vs_pet.py
python Tests/Threshold_Correlation_Sweep.py --help
python Tests/Null_distribution.py --help
python Tests/KS_Analysis.py
```

The simulation scripts can also be imported from Python:

```python
import CONFIG as cf
import GIM
import UTILS as utils
```

Most simulation settings, including temperature ranges, thermalization, number of steps, and output locations, are defined in the script where the simulation is specifically run. Start with small values when validating a new installation because full sweeps can be computationally expensive.

## Reproducibility

Set random seeds in analysis scripts when reproducible simulations are required. Record the configuration values, Python version, package versions, and dataset commit used for each analysis.

