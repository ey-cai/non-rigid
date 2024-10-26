#!/bin/bash

# This should take in 2 arguments:
# 1. DEDO task
# 2. the rest of the arguments for the gen_demonstration_proccloth script

DEDO_TASK=$1
shift
COMMAND=$@

# selecting script based on DEDO task
SCRIPT="gen_demonstration_$DEDO_TASK.py"

python ../third_party/3D-Diffusion-Policy/third_party/dedo_scripts/$SCRIPT \
    $COMMAND