#!/bin/bash

# Test training script for Wan2.1 with VideoAlign multi-reward GRPO
# This script runs a small test training with video_ocr, aesthetic, and videoalign rewards

echo "=========================================="
echo "Wan2.1 VideoAlign Multi-Reward Test Training"
echo "=========================================="
echo ""
echo "Reward Configuration:"
echo "  - video_ocr: 0.5 (OCR accuracy)"
echo "  - aesthetic: 0.2 (Visual quality)"
echo "  - videoalign: 0.3 (Video quality - VQ+MQ+TA)"
echo ""
echo "Training Settings:"
echo "  - Batch size: 4"
echo "  - Images per prompt: 2"
echo "  - Batches per epoch: 1"
echo ""

# Set number of GPUs (adjust as needed)
NUM_GPUS=${NUM_GPUS:-1}
MAIN_PORT=${MAIN_PORT:-29503}

# Check if VideoAlign checkpoint exists
VIDEOALIGN_CHECKPOINT="hf_cache/VideoReward"
if [ ! -d "$VIDEOALIGN_CHECKPOINT" ]; then
    echo "WARNING: VideoAlign checkpoint not found at $VIDEOALIGN_CHECKPOINT"
    echo "Please download the VideoAlign model:"
    echo ""
    echo "  mkdir -p hf_cache"
    echo "  cd hf_cache"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/KwaiVGI/VideoReward"
    echo "  cd .."
    echo ""
    echo "For more information, see: https://github.com/KlingTeam/VideoAlign"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if VideoAlign is cloned
if [ ! -d "VideoAlign" ]; then
    echo "WARNING: VideoAlign repository not found"
    echo "Cloning VideoAlign repository..."
    git clone https://github.com/KlingTeam/VideoAlign
    echo ""
    echo "Please install VideoAlign dependencies:"
    echo "  cd VideoAlign"
    echo "  conda env update -f environment.yaml"
    echo "  pip install flash-attn==2.5.8 --no-build-isolation"
    echo "  cd .."
    echo ""
fi

echo "Starting training..."
echo ""

# Run training
accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=$NUM_GPUS \
  --main_process_port $MAIN_PORT \
  scripts/train_wan2_1.py \
  --config config/grpo.py:test_wan2_1_videoalign

echo ""
echo "Training complete!"
echo "Check logs at: logs/video_ocr/wan_videoalign_test/"
