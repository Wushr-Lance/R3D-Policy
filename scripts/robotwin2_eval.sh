#!/bin/bash

# RoboTwin2.0 single-task DP3 policy evaluation script
# Examples:
# Single epoch evaluation:
#   bash scripts/robotwin2_eval.sh dp3_robotwin2 beat_block_hammer 9999 0 0 [epoch] [action_space]
# Batch epoch evaluation:
#   bash scripts/robotwin2_eval.sh dp3_robotwin2 beat_block_hammer 9999 0 0 [start_epoch] [action_space] [end_epoch] [epoch_interval] [wandb_mode]
# bash scripts/robotwin2_eval.sh r3d_robotwin2 beat_block_hammer 7100 0 0
# bash scripts/robotwin2_eval.sh 7200_pretrained beat_block_hammer 7200 0 3

# Default parameters
task_name="beat_block_hammer"
EPOCH=500
NUM_EPISODES=100
EVAL_SEED=0
HEAD_CAMERA_TYPE="D435"
MAX_STEPS=1000
TRAIN_CONFIG="demo_randomized"
TASK_CONFIG="demo_randomized"
INSTRUCTION_TYPE="unseen"
ACTION_SPACE="joint"        # Default: joint space, alternative: 'ee'
START_EPOCH="50"            # Batch eval start epoch
END_EPOCH="3000"            # Batch eval end epoch
EPOCH_INTERVAL=100          # Batch eval epoch interval
WANDB_MODE="online"         # wandb mode (online/offline/disabled)
WANDB_PROJECT="RoboTwin2.0-Evaluation"
WANDB_RUN_ID=""             # wandb run id for resume

alg_name=${1}
alg="r3d_robotwin2"
task_name=${2}
addition_info=${3}
training_seed=${4}

gpu_id=${5}

# Parse arguments
if [ ! -z "${6}" ]; then
    EPOCH=${6}
fi

# Arg 7: action_space (joint or ee)
if [ ! -z "${7}" ]; then
    ACTION_SPACE=${7}
fi

# Arg 8: if provided, sets end_epoch for batch evaluation
if [ ! -z "${8}" ]; then
    START_EPOCH=${EPOCH}
    END_EPOCH=${8}
fi

# Arg 9: epoch_interval
if [ ! -z "${9}" ]; then
    EPOCH_INTERVAL=${9}
fi

# Arg 10: wandb_mode
if [ ! -z "${10}" ]; then
    WANDB_MODE=${10}
fi

# Arg 11: wandb_run_id (for resume)
if [ ! -z "${11}" ]; then
    WANDB_RUN_ID=${11}
fi

exp_name=${task_name}-${alg}-${TRAIN_CONFIG}-${addition_info}
RUN_DIR="robotwin2_${exp_name}_seed${training_seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Display configuration
echo "=================================="
echo "RoboTwin2.0 Single-Task DP3 Evaluation"
echo "=================================="
echo "Task: $task_name"
echo "Action Space: $ACTION_SPACE"
if [ ! -z "$START_EPOCH" ] && [ ! -z "$END_EPOCH" ]; then
    echo "Batch evaluation mode"
    echo "Epoch range: $START_EPOCH to $END_EPOCH"
    echo "Epoch interval: $EPOCH_INTERVAL"
    echo "Wandb mode: $WANDB_MODE"
else
    echo "Single evaluation mode"
    echo "Checkpoint epoch: $EPOCH"
fi
echo "Eval episodes: $NUM_EPISODES"
echo "EVAL_SEED: $EVAL_SEED"
echo "Head camera: $HEAD_CAMERA_TYPE"
echo "Max steps: $MAX_STEPS"
echo "Task config: $TASK_CONFIG"
echo "Instruction type: $INSTRUCTION_TYPE"
echo "=================================="
echo

# Run evaluation
echo "Starting evaluation..."
echo

# Build base command
CMD="python RoboTwin2.0_3D_policy/test.py \
    --task_name $task_name \
    --num_episodes $NUM_EPISODES \
    --seed $EVAL_SEED \
    --head_camera_type $HEAD_CAMERA_TYPE \
    --max_steps $MAX_STEPS \
    --run_dir $RUN_DIR \
    --alg_name $alg_name \
    --task_config $TASK_CONFIG \
    --instruction_type $INSTRUCTION_TYPE \
    --action_space $ACTION_SPACE"

# Add batch or single evaluation parameters
if [ ! -z "$START_EPOCH" ] && [ ! -z "$END_EPOCH" ]; then
    # Batch evaluation mode
    CMD="$CMD --start_epoch $START_EPOCH --end_epoch $END_EPOCH --epoch_interval $EPOCH_INTERVAL --wandb_mode $WANDB_MODE --wandb_project $WANDB_PROJECT"
    # Add wandb_run_id if provided
    if [ ! -z "$WANDB_RUN_ID" ]; then
        CMD="$CMD --wandb_run_id $WANDB_RUN_ID"
    fi
else
    # Single evaluation mode
    CMD="$CMD --epoch $EPOCH"
fi

# Execute
eval $CMD
