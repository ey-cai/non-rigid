#!/bin/bash

# We want an option to take GPU_TYPE and NUM_GPUS as arguments. They should be flags.

# Parse the arguments
while getopts ":t:n:" opt; do
  case $opt in
    t) GPU_TYPE=$OPTARG ;;
    n) NUM_GPUS=$OPTARG ;;
    e) CONDA_ENV=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Shift the arguments so that $@ contains only the command to run
shift $((OPTIND-1))

# Default values
GPU_TYPE=${GPU_TYPE:-rtx3090}
NUM_GPUS=${NUM_GPUS:-1}
CONDA_ENV=${CONDA_ENV:-tax3d}

OUTPUT=$(python -m rpad.core.autobot available ${GPU_TYPE} ${NUM_GPUS} --quiet --local)

# If the lastr command failed, exit
if [ $? -ne 0 ]; then
    # Print the error message
    echo -e "Error: ${OUTPUT}"
    exit 1
fi

# Split the output on the colon
NODE=${OUTPUT%%:*}
GPU_INDICES=${OUTPUT#*:}
COMMAND=$@
CWD=$(pwd)

# Print the results (optional)
echo -e "######################################"
echo -e "Launching job:"
echo -e "  Node:\t\t${NODE}"
echo -e "  GPU type:\t${GPU_TYPE}"
echo -e "  GPU indices:\t${GPU_INDICES}"
echo -e "  Conda env:\t${CONDA_ENV}"
echo -e "  Command:\t${COMMAND}"
echo -e "  CWD:\t\t${CWD}"
echo -e "######################################"

# For some reason, the -c option causes an error "option requires an argument", but it still works.
ssh -t ${NODE} bash -ic "
# Setup env.
cd $CWD

conda activate $CONDA_ENV

# Now, treat the rest of the command line arguments as the command to run.
CUDA_VISIBLE_DEVICES=$GPU_INDICES $COMMAND

echo '######################################'
echo 'Job finished.'
echo '######################################'

# Deactivate the conda environment.
conda deactivate
"
