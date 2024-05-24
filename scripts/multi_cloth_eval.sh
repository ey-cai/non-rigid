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

# scene flow model - no object-centric processing
if [ $MODEL_TYPE == "scene_flow" ]; then
  echo "Evaluating scene flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  python eval_cloth.py \
    model=df_base \
    dataset=pc_multi_cloth \
    dataset.scene=True \
    dataset.world_frame=True \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND
# world frame cross flow
elif [ $MODEL_TYPE == "cross_flow_absolute" ]; then
  echo "Evaluting absolute flow model at checkpoint $CHECKPOINT with command: $COMMAND."

    python eval_cloth.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=True \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND
# relative frame cross flow
elif [ $MODEL_TYPE == "cross_flow_relative" ]; then
  echo "Evaluating relative flow model at checkpoint $CHECKPOINT with command: $COMMAND."

  python eval_cloth.py \
    model=df_flow_cross \
    dataset=pc_multi_cloth \
    dataset.scene=False \
    dataset.world_frame=False \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
    $COMMAND
# relative frame ghost point
elif [ $MODEL_TYPE == "cross_point_relative" ]; then
  echo "Evaluating relative point model at checkpoint $CHECKPOINT with command: $COMMAND."

    python eval_cloth.py \
        model=df_point_cross \
        dataset=pc_multi_cloth_point \
        resources.gpus=[${GPU_INDEX}] \
        checkpoint.reference=r-pad/non_rigid/model-${CHECKPOINT}:v0 \
        $COMMAND

else
  echo "Invalid model type."
fi