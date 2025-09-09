#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results
GPU_LM_ARGS="{\"engine_opts\":{\"max_model_len\":7760,\"rope_scaling\":{\"rope_type\":\"dynamic\",\"factor\":8.0}}}"
#MODEL_NAME="meta-llama/Meta-Llama-3.1-70B"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"


# Base language model
python cli.py --lm-name $MODEL_NAME --model-type base --smiles-file $SMILES_FILE --lm-args $GPU_LM_ARGS --output-dir $RESULTS_DIR/base_lm


# Full Importance Sampling
python cli.py --lm-name $MODEL_NAME --model-type critic-is \
    --n-particles 10 \
    --lm-args $GPU_LM_ARGS \
    --output-dir $RESULTS_DIR/critic-is

# Full SMC
python cli.py --lm-name $MODEL_NAME --model-type critic-smc \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method stratified \
    --lm-args $GPU_LM_ARGS \
    --output-dir $RESULTS_DIR/critic-smc
