# Goal Inference

This directory contains the code for running inference on the Planetarium dataset.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for Planetarium are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git[goal_inference]
```

or, if that command fails, run:
```bash
pip install genlm-eval[goal_inference] @ git+https://github.com/genlm/genlm-eval.git
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

### 2. Install Fast-Downward and VAL

The benchmarks require the Fast-Downward planner and the VAL plan validator.

To install them on Linux, follow the instructions from the [Planetarium repository](https://github.com/BatsResearch/planetarium):
```
apptainer pull fast-downward.sif docker://aibasel/downward:latest
mkdir tmp
curl -o tmp/VAL.zip https://dev.azure.com/schlumberger/4e6bcb11-cd68-40fe-98a2-e3777bfec0a6/_apis/build/builds/77/artifacts?artifactName=linux64\&api-version=7.1\&%24format=zip
unzip tmp/VAL.zip -d tmp/
tar -xzvf tmp/linux64/*.tar.gz -C tmp/ --strip-components=1
```

For other platforms, follow the instructions in the [Fast-Downward build documentation](https://github.com/aibasel/downward/blob/main/BUILD.md).

**Note:** Make sure to add `fast-downward.sif` and `VAL` to your PATH or create aliases for them.

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.

## Note: Reproducing Results

The original experiments reported in the paper were run with RoPE scaling enabled model config:

```json
{"engine_opts": {"rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}
```

The current repository defaults do not include RoPE scaling settings. To reproduce the original experiments, add the RoPE scaling configuration to the `--lm-args` parameter.