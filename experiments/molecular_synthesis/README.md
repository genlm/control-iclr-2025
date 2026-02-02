# Molecular Synthesis

This directory contains the code for running inference in the Molecular Synthesis domain.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for Molecular Synthesis are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git[molecules]
```

or, if that command fails, run:
```bash
pip install genlm-eval[molecules] @ git+https://github.com/genlm/genlm-eval.git
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

### 2. Download the GDB-17 dataset

Download the GDB-17 dataset (`GDB17.50000000.smi.gz`) from the [GDB downloads page](https://gdb.unibe.ch/downloads/) and unzip it in this directory:

```bash
# After downloading GDB17.50000000.smi.gz to this directory:
gunzip GDB17.50000000.smi.gz
```

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.

## Note: Reproducing Results

The original experiments reported in the paper were run with RoPE scaling enabled model config:

```json
{"engine_opts": {"rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}
```

The current repository defaults do not include RoPE scaling settings. To reproduce the original experiments, add the RoPE scaling configuration to the `--lm-args` parameter.
