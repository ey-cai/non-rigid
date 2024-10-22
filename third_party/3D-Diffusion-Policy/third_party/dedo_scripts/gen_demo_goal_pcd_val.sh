# use the same command as training except the script
# for example:
# bash gen_demo_goal_pcd.sh tax3d dedo_proccloth_goalPC jfqahima 1 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
# run_dir="data/outputs/${exp_name}_seed${seed}"
run_dir="/home/ktsim/Projects/non-rigid/data/dataset/${exp_name}"

gpu_id=${5}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python gen_demonstration_proccloth_with_goalPCD_val.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                