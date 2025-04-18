#!/bin/bash

# Note: use `python cli.py --help` to see the available options and defaults.

RESULTS_DIR=results

# Base language model
python cli.py --model-type base --output-dir $RESULTS_DIR/base_lm

# Locally-constrained decoding
python cli.py --model-type lcd --output-dir $RESULTS_DIR/lcd


# Grammar-only Importance Sampling
python cli.py --model-type grammar-only-is \
    --spider-data-dir data \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/grammar_only_is

# Grammar-only SMC
python cli.py --model-type grammar-only-smc \
    --spider-data-dir data \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/grammar_only_smc

# Sample Rerank
python cli.py --model-type sample-rerank \
    --spider-data-dir data \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/sample_rerank

# Full Importance Sampling
python cli.py --model-type full-is \
    --spider-data-dir data \
    --n-particles 10 \
    --ess-threshold 0.0 \
    --output-dir $RESULTS_DIR/full_is

# Full SMC
python cli.py --model-type full-smc \
    --spider-data-dir data \
    --n-particles 10 \
    --ess-threshold 0.9 \
    --resampling-method multinomial \
    --output-dir $RESULTS_DIR/full_smc
