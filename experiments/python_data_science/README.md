# Python for Data Science / DS1000

This directory contains the code for running inference on the DS1000 dataset.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for DS1000 are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git[ds1000]
```

or, if that command fails, run:
```bash
pip install genlm-eval[ds1000] @ git+https://github.com/genlm/genlm-eval.git
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

### 2. Setup code Environment

We need to setup a separate sandbox environment, in which the code gets executed. In the root directory, run:

```bash
python -m venv .ds1000env
source .ds1000env/bin/activate
pip install -U pip
pip install git+https://github.com/genlm/genlm-eval.git[ds1000_code_env]
python -c "import torch, pandas as pd; print(torch.__version__, pd.__version__)"
```

### 3. Install line-based sampling branch

Unlike the other domains, DS1000 doesn't sample token-by-token, but instead samples entire Python lines before performing a resampling step. To do so, install the following branch:

```bash
pip install git+https://github.com/genlm/genlm-control.git@samuki/sample-until-python
```

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.

**Note:** This is a replication of the DS1000 benchmark, not the original implementation.
