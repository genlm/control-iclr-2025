# Syntactic and Semantic Control for Large Language Models via Sequential Monte Carlo (ICLR 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2504.13139-b31b1b.svg)](https://arxiv.org/abs/2504.13139)

This repository contains the code for the experiments in the paper *Syntactic and Semantic Control for Large Language Models via sequential Monte Carlo* (ICLR 2025). 

For the controlled generation library underlying the models implemented here, see [genlm/genlm-control](https://github.com/genlm/genlm-control). Our the experiments are run using the [genlm/genlm-eval](https://github.com/genlm/genlm-eval) library.

## Setup

### Requirements

- Python >=3.11
- A GPU
- The dependencies in `pyproject.toml`

### Installation

Clone this repository:
```
git clone https://github.com/genlm/control-iclr-2025.git
cd control-iclr-2025
```

Install the dependencies:
```
pip install -e .
```

Note: It is recommended to use a virtual environment to manage the dependencies. For example, with conda:

```
conda create -n control_iclr python=3.11
conda activate control_iclr
pip install -e .
```

## Running the experiments

The experiments are split by domain. Each domain has its own directory with a README with instructions for installing domain-specific dependencies and running the models. All experiments are run through the `cli.py` scripts in each directory.

### Model naming convention

The `cli.py` scripts accept a `--model-type` argument that corresponds to the model or baseline used in the experiments. It can be one of the following:

- `base`: The base language model.
- `lcd`: The locally-constrained decoding model.
- `grammar-only-is`: The grammar-only model with importance sampling.
- `grammar-only-smc`: The grammar-only model with sequential Monte Carlo.
- `sample-rerank`: The sample rerank model.
- `full-is`: The full model with importance sampling.
- `full-smc`: The full model with sequential Monte Carlo.

### Output saving

The `cli.py` scripts save model and evaluation output in the directory provided through the `--output-dir` argument. If not provided, the default is to not save any output. The files are named as follows:

- `{instance_id}-{replacate_n}-output.json`: The output posterior distribution for the given instance and replicate.
- `{instance_id}-{replacate_n}-results.json`: The result of the evaluation for the given instance and replicate.
- `{instance_id}-{replacate_n}-record.json`: A record of the inference process, recording the particle beam at each step of inference, for the given instance and replicate.


## Common Issues

### GPU Compute Capability

When running on GPUs with compute capability less than 8.0 (e.g., RTX 2080, Quadro RTX 8000), bfloat16 is not supported and will cause errors. For example, you may see an error like:

```
ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your Quadro RTX 8000 GPU has compute capability 7.5. You can use float16 instead by explicitly setting the `dtype` flag in CLI, for example: --dtype=half.
```

To fix this, you can pass arguments that need to be passed to the vLLM engine initialization through the `--lm-args` parameter. To use float16 instead of bfloat16, you can pass: `--lm-args '{"engine_opts": {"dtype": "half"}}'` to `cli.py`. This same mechanism can be used to pass any other vLLM engine configuration options; all options can be passed in the value associated with the `engine_opts` key. Note that this must be a JSON string.

### Maximum Model Length

Sometimes the model's max sequence length is larger than the maximum number of tokens that can be stored in KV cache. For example, you may see an error like:

```
ValueError: The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (93936). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

We can fix this in the same way as above, by passing arguments that need to be passed to the vLLM engine initialization through the `--lm-args` parameter. For example, to decrease the maximum model length, you can pass: `--lm-args '{"engine_opts": {"max_model_len": 10000}}'` to `cli.py`.