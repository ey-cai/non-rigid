#!/bin/bash

# Really simple helper script to elevate policy evaluation code to primary TAX3D repo.
# This should take in 5 arguments:
# 1. the algorithm name
# 2. the task name
# 3. the additional information (e.g. checkpoint name)
# 4. the seed
# 5. the index of which GPU to use

# Example usage:
# ./eval_policy.sh dp3 adroit_hammer 0322 0 0

cd ../third_party/3D-Diffusion-Policy
bash scripts/eval_policy.sh $@