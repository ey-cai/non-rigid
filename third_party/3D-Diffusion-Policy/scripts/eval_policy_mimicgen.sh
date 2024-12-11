# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
# run_dir="data/outputs/${exp_name}_seed${seed}"
run_dir="/home/yingyuan/non-rigid/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/${exp_name}_seed${seed}"

gpu_id=${5}
use_goal_pc=${6}
pointnet_type=${7}
version=${8}
use_onehot=${9}
use_flow=${10}
extractor_mode=${11}
use_tax3d_pred=${12}


cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval_mimicgen.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=offline \
                            checkpoint.save_ckpt=${save_ckpt} \
                            policy.pointcloud_encoder_cfg.version=${version} \
                            policy.pointcloud_encoder_cfg.extractor_mode=${extractor_mode} \
                            policy.pointcloud_encoder_cfg.use_goal_pc=${use_goal_pc} \
                            policy.pointnet_type=${pointnet_type} \
                            policy.pointcloud_encoder_cfg.use_onehot=${use_onehot} \
                            policy.pointcloud_encoder_cfg.use_flow=${use_flow} \
                            task.env_runner.tax3d_pred=${use_tax3d_pred}