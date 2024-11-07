# Docker
docker run -it --gpus all -v /home/lyuxinghe/projects/taxpose/:/home/lyuxinghe/taxpose/ --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp.X11-unix -e DISPLAY=$DISPLAY --net=host --name taxpose taxpose:latest

# Train
./scripts/ndf_train_rigid.sh 0 cross_flow_relative offline wandb.name=cfr dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/

# Eval
./scripts/ndf_eval_rigid.sh 0 cross_flow_relative kxl3gt4o wandb.online=True wandb.name=cfr_eval dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/
