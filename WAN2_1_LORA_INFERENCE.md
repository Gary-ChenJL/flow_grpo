# Wan 2.1 LoRA Inference Guide

This guide explains the LoRA architecture for Wan 2.1 T2V video generation and how to use LoRA-finetuned models for inference.

## Table of Contents
1. [LoRA Architecture Overview](#lora-architecture-overview)
2. [Wan 2.1 LoRA Configuration](#wan-21-lora-configuration)
3. [Training vs Inference](#training-vs-inference)
4. [Inference Scripts](#inference-scripts)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## LoRA Architecture Overview

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that:
- Freezes the base model weights
- Adds trainable low-rank decomposition matrices to specific layers
- Requires only 1-5% of parameters compared to full fine-tuning
- Enables fast adaptation with minimal memory overhead

### How LoRA Works

For a pre-trained weight matrix **W** of shape (d, k):

```
Original:        Y = W · X                    (full weight matrix)
With LoRA:       Y = W · X + (B · A) · X      (W frozen, A & B trainable)
```

Where:
- **W**: Frozen pre-trained weights (d × k)
- **A**: Trainable down-projection (r × k), initialized randomly
- **B**: Trainable up-projection (d × r), initialized to zero
- **r**: LoRA rank (r << min(d, k)), typically 8-64

The rank **r** controls the capacity:
- Lower r → Fewer parameters, faster training, less flexibility
- Higher r → More parameters, more flexibility, risk of overfitting

---

## Wan 2.1 LoRA Configuration

### Architecture Details

Based on the training configuration in `scripts/train_wan2_1.py`:

```python
# LoRA hyperparameters
lora_r = 32              # LoRA rank (dimension of low-rank matrices)
lora_alpha = 64          # Scaling factor (alpha/r = 2.0)
init_lora_weights = "gaussian"  # Weight initialization strategy

# Target modules (attention layers in transformer)
lora_target_modules = [
    "add_k_proj",        # Additional key projection (dual attention)
    "add_q_proj",        # Additional query projection (dual attention)
    "add_v_proj",        # Additional value projection (dual attention)
    "to_add_out",        # Additional attention output projection
    "to_k",              # Standard key projection
    "to_out.0",          # Standard attention output projection (first layer)
    "to_q",              # Standard query projection
    "to_v",              # Standard value projection
]
```

### Why These Modules?

Wan 2.1 uses a **dual attention mechanism**:

1. **Standard Attention** (`to_q`, `to_k`, `to_v`, `to_out.0`):
   - Handles spatial-temporal relationships within the video
   - Q/K/V projections for self-attention

2. **Additional Attention** (`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`):
   - Cross-attention with text embeddings
   - Enables text-to-video conditioning

**LoRA is applied to ALL attention projections** to:
- Adapt both self-attention and cross-attention
- Fine-tune text-video alignment (critical for OCR/text rendering tasks)
- Preserve pre-trained knowledge while specializing for new tasks

### Parameter Efficiency

For Wan 2.1-T2V-1.3B with LoRA r=32:
- **Base model parameters**: ~1.3 billion
- **LoRA trainable parameters**: ~20-50 million (1.5-3.8% of base)
- **Memory savings**: ~8-10x compared to full fine-tuning

---

## Training vs Inference

### During Training (`scripts/train_wan2_1.py`)

```python
# 1. Load base model
pipeline = WanPipeline.from_pretrained(base_model_path)

# 2. Apply LoRA configuration
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules)
pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)

# 3. Train only LoRA parameters (base model frozen)
optimizer = AdamW(filter(lambda p: p.requires_grad, transformer.parameters()))

# 4. Save LoRA weights only
torch.save(get_peft_model_state_dict(pipeline.transformer), checkpoint_path)
```

**Key points:**
- Base model weights are frozen (`requires_grad=False`)
- Only LoRA matrices (A, B) are updated during training
- Checkpoints contain only LoRA parameters (~100-200 MB vs ~5 GB for full model)

### During Inference (`scripts/demo/wan2_1_lora_inference_demo.py`)

```python
# 1. Load base model
pipeline = WanPipeline.from_pretrained(base_model_path)

# 2. Apply LoRA configuration (same as training)
lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules)
pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)

# 3. Load trained LoRA weights
lora_state_dict = load_file(lora_checkpoint_path)
set_peft_model_state_dict(pipeline.transformer, lora_state_dict)

# 4. Merge LoRA into base model for faster inference
pipeline.transformer = pipeline.transformer.merge_and_unload()

# 5. Generate videos
videos = pipeline(prompt, num_frames=33, height=240, width=416)
```

**Key differences:**
- **Merge and unload**: Combines W + (B·A) into a single matrix for faster inference
- **No gradients needed**: All operations in `torch.no_grad()` context
- **Same architecture required**: LoRA config must match training exactly

---

## Inference Scripts

### 1. Base Model Inference (No LoRA)

**Script**: `scripts/demo/wan2_1_base_inference_demo.py`

**Purpose**: Generate videos with the vanilla Wan 2.1 model for baseline comparison

**Usage**:
```bash
python scripts/demo/wan2_1_base_inference_demo.py
```

**Configuration**:
```python
base_model_path = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"
prompts = ["Your prompt here"]
num_frames = 33       # Video length
height = 240          # Video height
width = 416           # Video width
num_inference_steps = 50  # Quality vs speed trade-off
guidance_scale = 4.5  # Text adherence strength
```

**Output**: Videos saved to `scripts/demo/wan_base_outputs/`

---

### 2. LoRA Model Inference

**Script**: `scripts/demo/wan2_1_lora_inference_demo.py`

**Purpose**: Generate videos with GRPO-finetuned LoRA weights

**Usage**:
```bash
python scripts/demo/wan2_1_lora_inference_demo.py
```

**Configuration**:
```python
base_model_path = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"
lora_checkpoint_path = "logs/video_ocr/wan_flow_grpo/checkpoint-60/model.safetensors"
prompts = ["Your prompt here"]

# LoRA config (must match training)
lora_r = 32
lora_alpha = 64
lora_target_modules = [
    "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
    "to_k", "to_out.0", "to_q", "to_v",
]
```

**Output**: Videos saved to `scripts/demo/wan_lora_outputs/`

---

## Usage Examples

### Example 1: OCR Video Generation

The GRPO-trained LoRA model specializes in text rendering (via video_ocr + aesthetic rewards):

```python
prompts = [
    'New York Skyline with "Hello World" written with fireworks on the sky',
    'A neon sign saying "OPEN 24/7" in a cyberpunk city',
    'Beach scene with "SUMMER 2024" written in the sand',
]
```

**Expected improvements over base model:**
- ✅ More legible text rendering
- ✅ Better text-image composition
- ✅ Higher aesthetic quality

### Example 2: Comparing Base vs LoRA

Generate the same prompt with both scripts:

```bash
# 1. Generate with base model
python scripts/demo/wan2_1_base_inference_demo.py

# 2. Generate with LoRA model
python scripts/demo/wan2_1_lora_inference_demo.py
```

Compare outputs:
- **Text legibility**: Can you read the text clearly?
- **Aesthetic quality**: Does it look visually appealing?
- **Prompt adherence**: Does it match the prompt accurately?

### Example 3: Batch Generation

Modify the script for large-scale generation:

```python
# Load prompts from file
with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines()]

# Process in batches to avoid OOM
batch_size = 4
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    # ... generate videos ...
```

---

## Troubleshooting

### Issue 1: LoRA Checkpoint Not Found

**Error**: `FileNotFoundError: lora_checkpoint_path not found`

**Solution**:
```bash
# List available checkpoints
ls logs/video_ocr/wan_flow_grpo/

# Update path in script
lora_checkpoint_path = "logs/video_ocr/wan_flow_grpo/checkpoint-XXX/model.safetensors"
```

### Issue 2: LoRA Config Mismatch

**Error**: `RuntimeError: Error(s) in loading state_dict`

**Cause**: LoRA configuration in inference doesn't match training

**Solution**: Ensure exact match:
```python
# Training config (scripts/train_wan2_1.py:454-469)
r=32, lora_alpha=64, target_modules=[...]

# Inference config (must be identical)
lora_r = 32
lora_alpha = 64
lora_target_modules = [...]  # Same list
```

### Issue 3: CUDA Out of Memory

**Error**: `torch.cuda.OutOfMemoryError`

**Solutions**:
1. **Reduce batch size**: Generate 1 video at a time
   ```python
   prompts = prompts[:1]  # Process one at a time
   ```

2. **Reduce video dimensions**:
   ```python
   num_frames = 17  # Instead of 33
   height = 120     # Instead of 240
   width = 208      # Instead of 416
   ```

3. **Lower inference steps**:
   ```python
   num_inference_steps = 30  # Instead of 50
   ```

### Issue 4: Low Quality Output

**Problem**: Generated videos are blurry or lack detail

**Solutions**:
1. **Increase inference steps**:
   ```python
   num_inference_steps = 100  # More denoising iterations
   ```

2. **Adjust guidance scale**:
   ```python
   guidance_scale = 6.0  # Higher = stronger prompt adherence
   ```

3. **Check checkpoint quality**: Use later training checkpoints
   ```bash
   # Compare checkpoint-60 vs checkpoint-120
   lora_checkpoint_path = "logs/.../checkpoint-120/model.safetensors"
   ```

### Issue 5: Text Still Not Rendering Well

**Problem**: Even with LoRA, text in videos is illegible

**Possible causes**:
1. **Insufficient training**: Model needs more GRPO iterations
2. **Prompt format**: Wrap text in quotes
   ```python
   # Good: Explicit quotes
   'A sign with "HELLO" written on it'

   # Bad: Ambiguous
   'A sign with HELLO written on it'
   ```
3. **OCR reward weight too low**: Increase video_ocr reward weight during training

---

## Comparison with QwenVL LoRA

The Wan 2.1 LoRA implementation follows the same pattern as QwenVL (`scripts/demo/qwenimage_edit_lora_inference_demo.py`):

| Aspect | QwenVL (Image Edit) | Wan 2.1 (Video) |
|--------|---------------------|-----------------|
| **Base Pipeline** | `QwenImageEditPipeline` | `WanPipeline` |
| **LoRA Rank** | r=64 | r=32 |
| **LoRA Alpha** | 128 | 64 |
| **Scaling Factor** | 128/64 = 2.0 | 64/32 = 2.0 |
| **Target Modules** | 12 modules (attn + mlp) | 8 modules (attention only) |
| **Output Format** | Images (H, W, C) | Videos (F, H, W, C) |
| **Prompt Encoding** | CLIP text encoder | T5 text encoder |

**Key similarity**: Both merge and unload LoRA weights for inference

```python
# Both scripts use this pattern
pipe.transformer = pipe.transformer.merge_and_unload()
```

---

## Advanced Usage

### Custom LoRA Configuration

To experiment with different LoRA configurations:

```python
# Higher rank for more flexibility (costs more memory)
lora_r = 64
lora_alpha = 128

# Target fewer modules for efficiency
lora_target_modules = ["to_q", "to_v"]  # Only Q/V projections
```

**Note**: Custom configs require retraining. Cannot be used with existing checkpoints.

### Multiple LoRA Adapters

Load different LoRA adapters for different tasks:

```python
# Load OCR-specialized LoRA
lora_ocr = load_file("checkpoint-ocr/model.safetensors")

# Load aesthetic-specialized LoRA
lora_aesthetic = load_file("checkpoint-aesthetic/model.safetensors")

# Blend LoRA weights
blended_lora = {k: 0.5*lora_ocr[k] + 0.5*lora_aesthetic[k] for k in lora_ocr}
```

### Extract LoRA for Distribution

Share only LoRA weights (not full model):

```bash
# LoRA checkpoint is ~100-200 MB (vs ~5 GB for full model)
tar -czf wan2.1_ocr_lora.tar.gz logs/video_ocr/wan_flow_grpo/checkpoint-60/

# Users only need base model + your LoRA weights
```

---

## Summary

**Wan 2.1 LoRA Architecture:**
- Rank: r=32, Alpha: 64, Scaling: 2.0
- Targets: All 8 attention projection layers (Q/K/V for standard + additional attention)
- Parameter efficiency: ~2-4% of base model size
- Training: GRPO with multi-reward (video_ocr + aesthetic)

**Inference Scripts:**
- `wan2_1_base_inference_demo.py`: Vanilla model baseline
- `wan2_1_lora_inference_demo.py`: GRPO-finetuned LoRA model

**Best Practices:**
- ✅ Always match LoRA config between training and inference
- ✅ Use merge_and_unload() for faster inference
- ✅ Compare base vs LoRA outputs to validate improvements
- ✅ Wrap text prompts in quotes for better text rendering

For questions or issues, refer to the training script: `scripts/train_wan2_1.py`
