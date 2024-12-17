#!/bin/bash

# This should take in 3 arguments:
# 1. the index of which GPU to use
# 2. model checkpoint
# 3. the rest of the arguments for the eval script

# Example usage:
# ./eval.sh 0 `CHECKPOINT` coverage=True

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

echo "Evaluating model at checkpoint $CHECKPOINT with command: $COMMAND."
python eval.py \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.run_id=${CHECKPOINT} \
    $COMMAND