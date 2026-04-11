# Examples:
# bash scripts/train_maniskill_single.sh r3d_maniskill PegInsertionSide 9999 0 2,3
# bash scripts/train_maniskill_single.sh r3d_maniskill PickCube 9999 0 2,3
# bash scripts/train_maniskill_single.sh r3d_maniskill StackCube 9999 0 2,3

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Check if gpu_id contains comma (multi-GPU mode)
if [[ $gpu_id == *","* ]]; then
    # Multi-GPU DDP mode
    echo -e "\033[32mMulti-GPU DDP mode detected!\033[0m"
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_id"
    num_gpus=${#GPU_ARRAY[@]}
    echo -e "\033[32mUsing ${num_gpus} GPUs: ${gpu_id}\033[0m"

    # Set environment variables for DDP
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export WORLD_SIZE=${num_gpus}
    export MASTER_ADDR="localhost"
    export MASTER_PORT="12361"

    USE_DDP=true
else
    # Single GPU mode
    echo -e "\033[32mSingle GPU mode detected!\033[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    USE_DDP=false
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd R3D

run_dir="$(pwd)/data/outputs/maniskill_${exp_name}_seed${seed}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# export MUJOCO_GL=osmesa

# unset LD_PRELOAD

if [ $USE_DDP = true ]; then
    # Multi-GPU DDP training
    echo -e "\033[32mStarting DDP training with ${num_gpus} GPUs...\033[0m"
    torchrun \
        --nproc_per_node=${num_gpus} \
        --master_port=12361 \
        train.py --config-name=${config_name}.yaml \
                            task="maniskill_${task_name}" \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda" \
                            training.use_ddp=true \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} 
else
    # Single GPU training
    echo -e "\033[32mStarting single GPU training...\033[0m"
    python train.py --config-name=${config_name}.yaml \
                            task="maniskill_${task_name}" \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            training.use_ddp=false \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} 
fi