#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results
SMILES_FILE="GDB17.50000000.smi"
GPU_LM_ARGS="{\"engine_opts\":{\"max_model_len\":7760,\"rope_scaling\":{\"rope_type\":\"dynamic\",\"factor\":8.0}}}"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

# Base language model
python cli.py --model-type base --smiles-file $SMILES_FILE --lm-args $GPU_LM_ARGS --output-dir $RESULTS_DIR/base_lm

# Locally-constrained decoding
python cli.py --model-type lcd --smiles-file $SMILES_FILE --lm-args $GPU_LM_ARGS --output-dir $RESULTS_DIR/lcd

# Grammar-only Importance Sampling
python cli.py --model-type grammar-only-is \
    --smiles-file $SMILES_FILE \
    --lm-args $GPU_LM_ARGS \
    --model-name $MODEL_NAME \
    --n-particles 10 \
    --output-dir $RESULTS_DIR/grammar_only_is

# Grammar-only SMC
python cli.py --model-type grammar-only-smc \
    --smiles-file $SMILES_FILE \
    --lm-args $GPU_LM_ARGS \
    --model-name $MODEL_NAME \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method stratified \
    --output-dir $RESULTS_DIR/grammar_only_smc

# Sample Rerank
python cli.py --model-type sample-rerank \
    --smiles-file $SMILES_FILE \
    --n-particles 10 \
    --lm-args $GPU_LM_ARGS \
    --model-name $MODEL_NAME \
    --output-dir $RESULTS_DIR/sample_rerank

# Full Importance Sampling
python cli.py --model-type full-is \
    --smiles-file $SMILES_FILE \
    --n-particles 10 \
    --lm-args $GPU_LM_ARGS \
    --model-name $MODEL_NAME \
    --output-dir $RESULTS_DIR/full_is

# Full SMC
python cli.py --model-type full-smc \
    --smiles-file $SMILES_FILE \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method stratified \
    --lm-args $GPU_LM_ARGS \
    --model-name $MODEL_NAME \
    --output-dir $RESULTS_DIR/full_smc
