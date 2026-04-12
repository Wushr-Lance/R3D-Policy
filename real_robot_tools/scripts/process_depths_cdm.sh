#!/bin/bash
# Batch processing depth images convenient script
# Usage:
#   bash process_depths_cdm.sh /path/to/data /path/to/cdm_model.pth

DATA_DIR=${1}
MODEL_PATH=${2}

# Default parameters
ENCODER=${3:-"vitl"}
INPUT_SIZE=${4:-518}
DEPTH_SCALE=${5:-1000.0}
MAX_DEPTH=${6:-6.0}
DEVICE=${7:-"cuda"}

if [ -z "$DATA_DIR" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash process_depths_cdm.sh <data_dir> <cdm_model_path> [encoder_type] [input_size] [depth_scale] [max_depth] [device]"
    echo ""
    echo "Example:"
    echo "  bash process_depths_cdm.sh ../01 /path/to/cdm_d435.pth"
    echo "  bash process_depths_cdm.sh ../01 /path/to/cdm_d435.pth vitl 518 1000.0 6.0 cuda"
    echo ""
    echo "Parameter description:"
    echo "  data_dir: Data root directory containing images/ meta/ subdirectories"
    echo "  cdm_model_path: CDM pre-trained model .pth file path"
    echo "  encoder_type: vits/vitb/vitl/vitg (Default: vitl)"
    echo "  input_size: Model input size (Default: 518)"
    echo "  depth_scale: Depth unit scaling factor (Default: 1000.0 for mm)"
    echo "  max_depth: Maximum valid depth in meters (Default: 6.0)"
    echo "  device: cuda/cpu (Default: cuda)"
    exit 1
fi

# Check paths
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file does not exist: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "CDM Depth Image Batch Processing"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Model path: $MODEL_PATH"
echo "Encoder: $ENCODER"
echo "Input size: $INPUT_SIZE"
echo "Depth scale: $DEPTH_SCALE"
echo "Max depth: $MAX_DEPTH m"
echo "Device: $DEVICE"
echo "=========================================="
echo ""

# Run Python script
python process_depth_with_cdm.py \
    --data-dir "$DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --encoder "$ENCODER" \
    --input-size "$INPUT_SIZE" \
    --depth-scale "$DEPTH_SCALE" \
    --max-depth "$MAX_DEPTH" \
    --device "$DEVICE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Processing complete!"
    echo "  Processed depth maps saved in: $DATA_DIR/images/observation.depths_after_cdm.*/"
else
    echo ""
    echo "✗ Processing failed, exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
