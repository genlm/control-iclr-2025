# Python for data science / DS1000

This directory contains the code for running inference on the DS1000 dataset.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for DS1000 are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git[ds1000]
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

### 2. Setup code Environment

We need to setup a separate sandbox environment, in which the code gets executed. In the root directory, run:

```bash
python -m venv .ds1000env
source .ds1000env/bin/activate
pip install -U pip
pip install -e .[ds1000_code_env]
python -c "import torch, pandas as pd; print(torch.__version__, pd.__version__)"
```

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.
