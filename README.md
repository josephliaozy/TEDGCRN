# TE-DGCRN

Disclosure of the data and key code associated with the TE-DGCRN paper.

## About the Project

TE-DGCRN is a PyTorch model for multi-step traffic flow forecasting. The repository contains the key model implementation, experiment configurations, and PeMS benchmark data used in this study.

## Repository Contents

- `model/`: implementation of TE-DGCRN
- `lib/`: data loading, preprocessing, and evaluation utilities
- `config_file/`: dataset-specific configurations
- `data/`: PeMS03, PeMS04, PeMS07, and PeMS08 datasets
- `run.py`: training and testing entry point

## Requirements

- Python 3.10
- PyTorch
- NumPy
- pandas

## Usage

Train TE-DGCRN on a selected dataset:

```bash
python run.py --dataset PEMSD8 --mode train
```

Available dataset names are `PEMSD3`, `PEMSD4`, `PEMSD7`, and `PEMSD8`.

Test a trained model:

```bash
python run.py --dataset PEMSD8 --mode test
```

## Data

The traffic-flow data are stored as `.npz` files under `data/`. The road-connection CSV files are provided as supplementary data but are not used as model inputs because TE-DGCRN learns dynamic graph structures from traffic observations and temporal conditions.
