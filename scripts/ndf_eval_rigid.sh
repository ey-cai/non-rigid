#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. model checkpoint 
# 4. the rest of the arguments for the eval script

# Example usage:
# ./multi_cloth_eval.sh 0 cross_point_relative `CHECKPOINT`
# ./multi_cloth_eval.sh 1 scene_flow `CHECKPOINT` dataset.multi_cloth.hole=single dataset.multi_cloth.size=100

GPU_INDEX=$1
MODEL_TYPE=$2
CHECKPOINT=$3
shift
shift
shift
COMMAND=$@

# relative frame cross point
if [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Evaluating relative point model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=ndf dataset.type=point dataset.scene=False dataset.world_frame=False"
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Evaluating relative flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=ndf dataset.type=flow dataset.scene=False dataset.world_frame=False"
# relative frame cross flow cfg
elif [ $MODEL_TYPE == "cross_flow_relative_cfg" ]; then
  echo "Evaluating relative flow model with classifier-free guidance with command: $COMMAND."

  MODEL_PARAMS="model=df_cross_cfg model.type=flow"
  DATASET_PARAMS="dataset=ndf dataset.type=flow dataset.scene=False dataset.world_frame=False"
fi

python scripts/eval_rigid.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  resources.gpus=[${GPU_INDEX}] \
  checkpoint.reference='/home/lyuxinghe/non-rigid/logs/train_ndf_df_cross/2024-11-08/18-13-12/checkpoints/epoch_19999.ckpt' \
  checkpoint.run_id=${CHECKPOINT} \
  $COMMAND

<<COMMENT

python scripts/eval_rigid.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  resources.gpus=[${GPU_INDEX}] \
  checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
  checkpoint.run_id=${CHECKPOINT} \
  $COMMAND

# example: 
./scripts/ndf_eval_rigid.sh 0 cross_flow_relative kxl3gt4o wandb.online=True wandb.name=cfr_eval dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/

# scene flow model - no object-centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Evaluating scene flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=flow"
  DATASET_PARAMS="dataset=ndf_point dataset.type=flow dataset.scene=True dataset.world_frame=True"
# scene point model - no object centric processing
elif [ $MODEL_TYPE == "scene_point" ]; then
  echo "Evaluating scene point model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_base model.type=point"
  DATASET_PARAMS="dataset=ndf_point dataset.type=point dataset.scene=True dataset.world_frame=True"
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Evaluting absolute flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=ndf_point dataset.type=flow dataset.scene=False dataset.world_frame=True"
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Evaluating relative flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"
  DATASET_PARAMS="dataset=ndf_point dataset.type=flow dataset.scene=False dataset.world_frame=False"
# world frame cross point
elif [ $MODEL_TYPE == "cross_point_absolute" ]; then
  echo "Evaluating absolute point model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=ndf_point dataset.type=point dataset.scene=False dataset.world_frame=True"
# relative frame cross point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Evaluating relative point model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"
  DATASET_PARAMS="dataset=ndf_point dataset.type=point dataset.scene=False dataset.world_frame=False"
# flow regression baseline
elif [ $MODEL_TYPE == "regression_flow" ]; then
  echo "Evaluating flow regression model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=flow"
  DATASET_PARAMS="dataset=ndf_point dataset.type=flow dataset.scene=False dataset.world_frame=False"
#  point regression baseline
elif [ $MODEL_TYPE == "regression_point" ]; then
  echo "Evaluating linear regression model at checkpoint $CHECKPOINT with command: $COMMAND."

  MODEL_PARAMS="model=regression model.type=point"
  DATASET_PARAMS="dataset=ndf_point dataset.type=point dataset.scene=False dataset.world_frame=False"
else
  echo "Invalid model type."
fi

python scripts/eval_rigid.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  resources.gpus=[${GPU_INDEX}] \
  checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
  $COMMAND

COMMENT