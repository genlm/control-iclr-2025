# Text to SQL

This directory contains the code for running inference on the Spider dataset.

## Setup

### 1. Install dependencies

Make sure that `genlm-eval` and its dependencies for Spider are installed:

```bash
pip install git+https://github.com/genlm/genlm-eval.git@v0.1.0[spider]
```

Note: It is recommended to use a virtual environment, e.g., `conda`, to install the dependencies.

You may also need to download the following `nltk` data:
```bash
python -m nltk.downloader punkt_tab
```

### 2. Download the Spider dataset

Download and unzip the Spider dataset in this directory with the following command:
```bash
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
```

## Models and baselines

All models can be run through the `cli.py` script. See `run_all.sh` for the argument configurations used to run the models and baselines in the paper.

