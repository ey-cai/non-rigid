#!/bin/bash

# Script to run multiple training processes on different GPUs with hyperparameter tuning
## Check the progress of each session:
### screen -ls

## Attach to a specific session to view its output in real-time:
### screen -r train_lr0.001_bs32_gpu0 (Replace train_lr0.001_bs32_gpu0 with the name of the session you want to view.)

## Detach from a session by pressing:
### Ctrl + A, then D

## Close a session when done by attaching to it and typing exit, or by running:
### screen -S train_lr0.001_bs32_gpu0 -X quit

# Define log directory and create it if it doesn't exist
log_dir="./logs/multi_processing"
mkdir -p "$log_dir"  # -p flag ensures it creates the directory if it doesn’t exist

# Define hyperparameters
sweep_gpu=(0 1)
sweep_wandb_name=(cfr_cfg_0.05_3 cfr_cfg_0.05_2.5)
sweep_p_uncondition=(0.05 0.05)
sweep_guidance_strength=(3 2.5)

# Loop over hyperparameters and create a new `screen` session for each process
for i in "${!sweep_gpu[@]}"; do
    gpu="${sweep_gpu[$i]}"
    wandb_name="${sweep_wandb_name[$i]}"
    p_uncondition="${sweep_p_uncondition[$i]}"
    guidance_strength="${sweep_guidance_strength[$i]}"
    
    # Define a unique session name and log file based on the hyperparameters
    session_name="train_${wandb_name}_gpu${gpu}"
    log_file="$log_dir/log_${session_name}.log"

    # Start a new `screen` session in detached mode and run the training script with output redirected to a log file
    screen -dmS "$session_name" bash -c "./scripts/ndf_train_rigid.sh $gpu cross_flow_relative_cfg online wandb.name=$wandb_name model.p_uncondition=$p_uncondition model.guidance_strength=$guidance_strength dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/ > $log_file 2>&1"

    # Notify user
    echo "Started session $session_name on GPU $gpu with log file at $log_file"
done

# Wait for all background jobs (tail commands) to finish
wait
