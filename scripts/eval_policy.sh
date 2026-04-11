# use the same command as training except the script
# for example:
# bash scripts/eval_policy.sh dp3 adroit_hammer 0322 0 0
# For robotwin2:
# bash scripts/eval_policy.sh dp3_robotwin2 beat_block_hammer 9999 0 2


# export CUROBO_TORCH_COMPILE=1

DEBUG=False
save_ckpt=False
wandb_mode=offline

alg_name=${1}
task_name=${2}
config_name=${alg_name}
setting="demo_clean"
# expert_data_num="100"
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${setting}-${addition_info}
run_dir="data/outputs/robotwin2_${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}


cd R3D
run_dir="$(pwd)/data/outputs/robotwin2_${exp_name}_seed${seed}"


python eval.py --config-name=${config_name}.yaml \
                            task="robotwin2_demo_task" \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            setting=${setting}
