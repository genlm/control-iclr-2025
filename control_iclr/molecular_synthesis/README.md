# Molecular Synthesis

This directory contains the code for running inference in the Molecular Synthesis domain.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for Molecular Synthesis are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git[molecules]
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

### 2. Download the GBD-17 dataset

Download the GBD-17 dataset (`GDB17.50000000.smi.gz`) from [here](https://gdb.unibe.ch/downloads/) and unzip it in this directory (via e.g., `gunzip GDB17.50000000.smi.gz`).

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.
