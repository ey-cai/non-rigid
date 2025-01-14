#!/bin/bash

# This should take in 2 arguments:
# 1. the index of which GPU to use
# 2. model checkpoint

# Example usage:
# ./vis.sh 0 `CHECKPOINT`

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

echo "Visualizing model at checkpoint $CHECKPOINT with command: $COMMAND."
python vis.py \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.run_id=${CHECKPOINT} \
    $COMMAND