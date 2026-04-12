#!/bin/bash
# Real robot data training script (without simulation evaluation)
# Can reuse all existing algorithm configurations (dp3_robotwin2, dp3_uni3d_pretrained_robotwin2, etc.)
#
# Examples:
# bash scripts/train_real_robot.sh dp3_uni3d_pretrained_robotwin2 task1 9999 0 0
# bash scripts/train_real_robot.sh dp3_uni3d_scratch_robotwin2 task2 9999 0 0,1
# bash scripts/train_real_robot.sh dp3_robotwin2 task1 9999 0 0

alg_name=${1}          # Algorithm config: dp3_robotwin2, dp3_uni3d_pretrained_robotwin2, etc.
task_num=${2}          # task1 or task2
addition_info=${3:-"9999"}
seed=${4:-"0"}
gpu_id=${5:-"0"}

DEBUG=False
save_ckpt=True

# Check parameters
if [ -z "$alg_name" ] || [ -z "$task_num" ]; then
    echo -e "\033[31mError: Missing arguments!\033[0m"
    echo "Usage: bash scripts/train_real_robot.sh <alg_name> <task_num> <addition_info> <seed> <gpu_id>"
    echo "Example: bash scripts/train_real_robot.sh dp3_uni3d_pretrained_robotwin2 task1 9999 0 0"
    exit 1
fi

# Set task configuration
if [ "$task_num" == "task1" ]; then
    task_name="real_task_1"
    task_config="real_robot_task1"
elif [ "$task_num" == "task2" ]; then
    task_name="real_task_2"
    task_config="real_robot_task2"
elif [ "$task_num" == "task3" ]; then
    task_name="real_task_3"
    task_config="real_robot_task3"
else
    echo -e "\033[31mError: task_num must be 'task1', 'task2', or 'task3'\033[0m"
    exit 1
fi

config_name="${alg_name}"
exp_name="${task_name}-${alg_name}-${addition_info}"

echo -e "\033[33m================================\033[0m"
echo -e "\033[33mTraining Real Robot Data\033[0m"
echo -e "\033[33mTask: ${task_name}\033[0m"
echo -e "\033[33mGPU: ${gpu_id}\033[0m"
echo -e "\033[33m================================\033[0m"

# Function to automatically find an available port
find_free_port() {
    python -c "
import socket
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port
print(find_free_port())
"
}

# Check if gpu_id contains comma (multi-GPU mode)
if [[ $gpu_id == *","* ]]; then
    echo -e "\033[32mMulti-GPU DDP mode detected!\033[0m"
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_id"
    num_gpus=${#GPU_ARRAY[@]}
    echo -e "\033[32mUsing ${num_gpus} GPUs: ${gpu_id}\033[0m"
    
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export WORLD_SIZE=${num_gpus}
    export MASTER_ADDR="localhost"
    
    # Auto-select available port
    MASTER_PORT=$(find_free_port)
    export MASTER_PORT=${MASTER_PORT}
    echo -e "\033[32mAuto-selected master port: ${MASTER_PORT}\033[0m"
    
    USE_DDP=true
else
    echo -e "\033[32mSingle GPU mode\033[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    USE_DDP=false
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy

run_dir="$(pwd)/data/outputs/real_robot_${exp_name}_seed${seed}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

if [ $USE_DDP = true ]; then
    echo -e "\033[32mStarting DDP training...\033[0m"
    torchrun \
        --nproc_per_node=${num_gpus} \
        --master_port=${MASTER_PORT} \
        train.py --config-name=${config_name}.yaml \
                            task=${task_config} \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda" \
                            training.use_ddp=true \
                            training.rollout_every=50000 \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.resume=false \
                            checkpoint.save_ckpt=${save_ckpt}
else
    echo -e "\033[32mStarting single GPU training...\033[0m"
    python train.py --config-name=${config_name}.yaml \
                            task=${task_config} \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            training.use_ddp=false \
                            training.rollout_every=50000 \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.resume=false \
                            checkpoint.save_ckpt=${save_ckpt}
fi