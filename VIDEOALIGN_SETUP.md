# VideoAlign Multi-Reward GRPO Setup Guide

This guide explains how to use VideoAlign reward model with the Flow GRPO framework for video generation training.

## Overview

VideoAlign (VideoReward) is a VLM-based reward model that evaluates generated videos across three dimensions:
- **VQ (Visual Quality)**: Overall visual aesthetics and clarity
- **MQ (Motion Quality)**: Smoothness and naturalness of motion
- **TA (Text Alignment)**: Alignment between video content and text prompt

The model is based on Qwen2-VL-2B-Instruct and provides normalized scores for each dimension.

## Installation

### Quick Install (Recommended)

Use the automated installation script:

```bash
cd /home/user/flow_grpo
./scripts/install_videoalign.sh
```

This script will:
1. Clone VideoAlign repository (if not already cloned)
2. Upgrade transformers to support Qwen2-VL (>= 4.45.0)
3. Install all required dependencies
4. Install VideoAlign in editable mode
5. Verify the installation

### Manual Installation

If you prefer to install manually:

#### 1. Clone VideoAlign Repository

```bash
cd /home/user/flow_grpo
git clone https://github.com/KlingTeam/VideoAlign
```

#### 2. Upgrade transformers (CRITICAL!)

VideoAlign uses Qwen2-VL which requires transformers >= 4.45.0:

```bash
pip install --upgrade "transformers>=4.45.0"
```

**Common Error:** If you see `cannot import name 'Qwen2VLForConditionalGeneration'`, your transformers is too old!

#### 3. Install VideoAlign Dependencies

```bash
cd VideoAlign
pip install -e .
pip install qwen-vl-utils einops torchvision Pillow
pip install flash-attn==2.5.8 --no-build-isolation  # Optional but recommended
pip install opencv-python  # For video saving in flow_grpo
cd ..
```

#### 4. Download VideoAlign Model Checkpoint

```bash
mkdir -p hf_cache
cd hf_cache
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
cd ..
```

The checkpoint should be located at: `hf_cache/VideoReward/`

## Usage

### Test Training Script

A test training script is provided that demonstrates multi-reward GRPO with VideoAlign:

```bash
./scripts/test_wan2_1_videoalign.sh
```

This script trains Wan2.1 with three rewards:
- `video_ocr`: 0.5 weight (OCR accuracy)
- `aesthetic`: 0.2 weight (Visual aesthetics)
- `videoalign`: 0.3 weight (Video quality)

### Custom Configuration

To create your own configuration, add a function to `config/grpo.py`:

```python
def my_videoalign_config():
    config = general_ocr_wan2_1()

    config.run_name = "my_videoalign_experiment"

    # Configure rewards and weights
    config.reward_fn = {
        "video_ocr": 0.4,
        "aesthetic": 0.3,
        "videoalign": 0.3,
    }

    # Adjust other training parameters as needed
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 4

    return config
```

Then run training:
```bash
accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes=1 \
  --main_process_port 29503 \
  scripts/train_wan2_1.py \
  --config config/grpo.py:my_videoalign_config
```

## Implementation Details

### VideoAlign Scorer (`flow_grpo/videoalign_scorer.py`)

The `VideoAlignScorer` class handles:
- Loading the VideoAlign model from checkpoint
- Converting video tensors (B, F, C, H, W) to temporary MP4 files
- Running inference and returning normalized scores
- Cleanup of temporary files

### Reward Function (`flow_grpo/rewards.py`)

The `videoalign_score()` function:
- Accepts both image (4D) and video (5D) tensors
- Converts images to single-frame videos if needed
- Returns the Overall score (sum of VQ + MQ + TA)

### Tensor Format Support

VideoAlign handles various input formats:
- **Video tensors**: (B, F, C, H, W) - batch, frames, channels, height, width
- **Image tensors**: (B, C, H, W) - converted to single-frame videos
- **Numpy arrays**: Automatically converted to tensors

## Configuration Parameters

### VideoAlign-Specific Parameters

```python
# In videoalign_scorer.py
checkpoint_path = "hf_cache/VideoReward"  # Path to model checkpoint
device = "cuda"                            # Device for inference
dtype = torch.bfloat16                     # Data type (bfloat16 recommended)
fps = 8                                    # Frames per second for video saving
```

### Reward Weights

Weights should sum to 1.0 for proper normalization. Example configurations:

**Balanced:**
```python
config.reward_fn = {
    "video_ocr": 0.33,
    "aesthetic": 0.33,
    "videoalign": 0.34,
}
```

**OCR-focused:**
```python
config.reward_fn = {
    "video_ocr": 0.6,
    "aesthetic": 0.1,
    "videoalign": 0.3,
}
```

**Quality-focused:**
```python
config.reward_fn = {
    "video_ocr": 0.3,
    "aesthetic": 0.3,
    "videoalign": 0.4,
}
```

## Troubleshooting

### Qwen2VL Import Error (Most Common)

**Error:** `cannot import name 'Qwen2VLForConditionalGeneration' from 'transformers'`

**Cause:** Your transformers library is too old (< 4.45.0)

**Solution:**
```bash
# Quick fix - run the install script
./scripts/install_videoalign.sh

# Or manually upgrade transformers
pip install --upgrade "transformers>=4.45.0"

# Then install VideoAlign
cd VideoAlign
pip install -e .
cd ..
```

After upgrading, verify:
```bash
python -c "from transformers import Qwen2VLForConditionalGeneration; print('✓ Success!')"
```

### Model Not Loading

If you see "VideoAlign model not available, returning dummy scores":

1. Check that VideoAlign is cloned: `ls VideoAlign/`
2. Check that checkpoint exists: `ls hf_cache/VideoReward/`
3. Verify the checkpoint path in the config matches your setup
4. Make sure you ran `pip install -e .` in the VideoAlign directory

### Out of Memory Errors

VideoAlign uses Qwen2-VL-2B which requires ~4-6GB VRAM. If you encounter OOM:

1. Reduce batch size:
   ```python
   config.sample.train_batch_size = 2
   config.sample.num_image_per_prompt = 2
   ```

2. Use gradient checkpointing (already enabled in default config)

3. Consider using a machine with more VRAM

### Video Saving Issues

If video saving fails, ensure opencv-python is installed:
```bash
pip install opencv-python
```

## Performance Notes

- **Frame sampling**: VideoAlign samples every 4th frame by default to reduce computation
- **Inference time**: ~1-2 seconds per video depending on resolution and frame count
- **VRAM usage**: ~4-6GB for the VideoAlign model + video generation model

## References

- **VideoAlign Paper**: arXiv:2501.13918
- **GitHub**: https://github.com/KlingTeam/VideoAlign
- **Model**: https://huggingface.co/KwaiVGI/VideoReward
- **Dataset**: https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench

## Example Output

During training, you'll see reward scores in the logs:

```
Step 10: avg_reward=1.234, video_ocr=0.89, aesthetic=0.65, videoalign=1.45
```

The videoalign score is the sum of VQ + MQ + TA (typically ranging from 0-3 unnormalized, or around 0-2 when normalized).

## Citation

If you use VideoAlign in your research, please cite:

```bibtex
@article{videoalign2025,
  title={VideoAlign: Vision Language Model-based Reward for Video Generation},
  author={...},
  journal={arXiv preprint arXiv:2501.13918},
  year={2025}
}
```
