# use the same command as training except the script
# for example:
# bash scripts/maniskill_eval.sh r3d_maniskill StackCube 7200 0 0



DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/maniskill_${exp_name}_seed${seed}"

gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}


ROOT_DIR=$(pwd)
cd R3D
PROJECT_DIR=$(pwd)
export PYTHONPATH="${PROJECT_DIR}:${ROOT_DIR}/ManiSkill2:${ROOT_DIR}/ManiSkill2/warp_maniskill:${ROOT_DIR}/third_party/gym-0.21.0:${PYTHONPATH}"
run_dir="$(pwd)/data/outputs/maniskill_${exp_name}_seed${seed}"

python eval.py --config-name=${config_name}.yaml \
                            task=maniskill_${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                