#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results
GPU_LM_ARGS="{\"temperature\": 1.0,\"engine_opts\":{\"max_model_len\":8192}}"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
#MODEL_NAME="meta-llama/Meta-Llama-3-70B"


# Base language model
python cli.py --lm-name $MODEL_NAME --model-type base --lm-args $GPU_LM_ARGS --output-dir $RESULTS_DIR/base_lm --n-replicates 5


# Full Importance Sampling
python cli.py --lm-name $MODEL_NAME --model-type critic-is \
    --n-particles 10 \
    --lm-args $GPU_LM_ARGS \
    --output-dir $RESULTS_DIR/critic-is \
    --n-replicates 5

# Full SMC
python cli.py --lm-name $MODEL_NAME --model-type critic-smc \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method stratified \
    --lm-args $GPU_LM_ARGS \
    --output-dir $RESULTS_DIR/critic-smc \
    --n-replicates 5
