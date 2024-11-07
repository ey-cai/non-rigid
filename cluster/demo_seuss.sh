#!/bin/bash

# Load singularity.
module load singularity

# Run a simple singularity command.

singularity exec \
    --pwd /opt/rpad/code/non-rigid \
    -B /home/baeisner/code/non-rigid:/opt/rpad/code/non-rigid \
    --nv \
    ~/singularity_images/non_rigid_articulated-ben-scratch.sif \
    python third_party/3D-Diffusion-Policy/third_party/dedo_scripts/gen_demonstration_proccloth.py --root_dir /home/baeisner/cloth_data --num_episodes 1 --random_anchor_pose --split train
