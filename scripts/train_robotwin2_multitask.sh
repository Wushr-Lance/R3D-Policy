# Examples:
# bash scripts/train_robotwin2_multitask.sh r3d_robotwin2_multitask 9999 0 2,3
# bash scripts/train_robotwin2_multitask.sh r3d_robotwin2_multitask 9999 0 2,3 open_microwave

DEBUG=False
save_ckpt=True

alg_name=${1:-r3d_robotwin2_multitask}
config_name=${alg_name}
addition_info=${2:-default}
seed=${3:-0}
gpu_id=${4:-0}
eval_task_name=${5:-}
exp_name="robotwin2_multitask-${alg_name}-${addition_info}-${seed}"

echo -e "[33mgpu id (to use): ${gpu_id}[0m"

if [[ $gpu_id == *,* ]]; then
    echo -e "[32mMulti-GPU DDP mode detected![0m"
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_id"
    num_gpus=${#GPU_ARRAY[@]}
    echo -e "[32mUsing ${num_gpus} GPUs: ${gpu_id}[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export WORLD_SIZE=${num_gpus}
    export MASTER_ADDR="localhost"
    export MASTER_PORT="12358"
    USE_DDP=true
else
    echo -e "[32mSingle GPU mode detected![0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    USE_DDP=false
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "[33mDebug mode![0m"
else
    wandb_mode=online
    echo -e "[33mTrain mode[0m"
fi

cd R3D

run_dir="$(pwd)/data/outputs/robotwin2_${exp_name}_seed${seed}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

common_args=(
    train.py --config-name=${config_name}.yaml
    task="robotwin2_multitask"
    task_name="robotwin2_multitask"
    hydra.run.dir=${run_dir}
    training.debug=$DEBUG
    training.seed=${seed}
    exp_name=${exp_name}
    logging.mode=${wandb_mode}
    checkpoint.save_ckpt=${save_ckpt}
)

if [ -n "$eval_task_name" ]; then
    common_args+=(eval_task_name=${eval_task_name})
fi

if [ $USE_DDP = true ]; then
    echo -e "[32mStarting DDP multi-task training with ${num_gpus} GPUs...[0m"
    torchrun         --nproc_per_node=${num_gpus}         --master_port=12358         "${common_args[@]}"         training.device="cuda"         training.use_ddp=true
else
    echo -e "[32mStarting single GPU multi-task training...[0m"
    python "${common_args[@]}"         training.device="cuda:0"         training.use_ddp=false
fi
