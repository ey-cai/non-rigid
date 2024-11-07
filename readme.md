# Docker
docker run -it --gpus all -v /home/lyuxing/non-rigid/:/home/lyuxinghe/non-rigid/ --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp.X11-unix -e DISPLAY=$DISPLAY --net=host --name tax3d lyuxinghe/tax3d:latest

# Train
./scripts/ndf_train_rigid.sh 0 cross_flow_relative_cfg online wandb.name=cfr_cfg_0.15_3.5 model.p_uncondition=0.15 model.guidance_strength=3.5 dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/

# Eval
./scripts/ndf_eval_rigid.sh 0 cross_flow_relative kxl3gt4o wandb.online=True wandb.name=cfr_eval dataset.data_dir=/home/lyuxinghe/non-rigid/datasets/ndf/mugplace/
