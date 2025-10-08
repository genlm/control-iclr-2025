#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results
LM_NAME="meta-llama/Meta-Llama-3.1-8B"

GPU_LM_ARGS="{\"engine_opts\":{\"max_model_len\":7760,\"rope_scaling\":{\"rope_type\":\"dynamic\",\"factor\":8.0}}}"

# Base language model
python cli.py --model-type base --lm-name $LM_NAME --output-dir $RESULTS_DIR/base_lm --lm-args $GPU_LM_ARGS

# Locally-constrained decoding
python cli.py --model-type lcd --lm-name $LM_NAME --output-dir $RESULTS_DIR/lcd --lm-args $GPU_LM_ARGS

# Grammar-only Importance Sampling
python cli.py --model-type grammar-only-is \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/grammar_only_is \
    --lm-args $GPU_LM_ARGS

# Grammar-only SMC
python cli.py --model-type grammar-only-smc \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/grammar_only_smc \
    --lm-args $GPU_LM_ARGS

# Sample Rerank
python cli.py --model-type sample-rerank \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/sample_rerank \
    --lm-args $GPU_LM_ARGS

# Full Importance Sampling
python cli.py --model-type full-is \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/full_is \
    --lm-args $GPU_LM_ARGS

# Full SMC
python cli.py --model-type full-smc \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/full_smc \
    --lm-args $GPU_LM_ARGS
