# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy_goalPC.sh dp3 dedo_proccloth example 1 0 0 tax3d ground_truth pointnet 0



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
# run_dir="data/outputs/${exp_name}_seed${seed}"
run_dir="/home/ktsim/Projects/non-rigid/data/outputs/${exp_name}_seed${seed}"

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
enable_wandb=${6}
algo_version=${7}
goal_pc_version=${8}
pointnet_type=${9}
use_onehot=${10}

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python trainGoalPC.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}\
                            enable_wandb=${enable_wandb}\
                            policy.pointcloud_encoder_cfg.version=${algo_version} \
                            policy.pointcloud_encoder_cfg.extractor_mode=simple \
                            policy.pointcloud_encoder_cfg.goal_pc_version=${goal_pc_version} \
                            policy.pointnet_type=${pointnet_type} \
                            policy.pointcloud_encoder_cfg.use_onehot=${use_onehot}


                                