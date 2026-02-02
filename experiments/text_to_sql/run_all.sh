#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results
MAX_INSTANCES=1034 # Can decrease this to run fewer instances
LM_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
GPU_LM_ARGS="{\"engine_opts\":{\"max_model_len\":10000}}" # Comment out if running on CPU.

# Base language model
python cli.py --model-type base --lm-name $LM_NAME --output-dir $RESULTS_DIR/base_lm --max-instances $MAX_INSTANCES --lm-args $GPU_LM_ARGS --n-replicates 5

# Locally-constrained decoding
python cli.py --model-type lcd --lm-name $LM_NAME --output-dir $RESULTS_DIR/lcd --max-instances $MAX_INSTANCES --lm-args $GPU_LM_ARGS --n-replicates 5

# Grammar-only Importance Sampling
python cli.py --model-type grammar-only-is \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/grammar_only_is \
    --max-instances $MAX_INSTANCES \
    --lm-args $GPU_LM_ARGS \
    --n-replicates 5

# Grammar-only SMC
python cli.py --model-type grammar-only-smc \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/grammar_only_smc \
    --max-instances $MAX_INSTANCES \
    --lm-args $GPU_LM_ARGS \
    --n-replicates 5

# Sample Rerank
python cli.py --model-type sample-rerank \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/sample_rerank \
    --max-instances $MAX_INSTANCES \
    --lm-args $GPU_LM_ARGS \
    --n-replicates 5

# Full Importance Sampling
python cli.py --model-type full-is \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/full_is \
    --max-instances $MAX_INSTANCES \
    --lm-args $GPU_LM_ARGS \
    --n-replicates 5

# Full SMC
python cli.py --model-type full-smc \
    --lm-name $LM_NAME \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/full_smc \
    --max-instances $MAX_INSTANCES \
    --lm-args $GPU_LM_ARGS \
    --n-replicates 5
