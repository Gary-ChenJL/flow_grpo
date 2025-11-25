"""
Wan 2.1 T2V LoRA Inference Demo

This script demonstrates how to:
1. Load the Wan 2.1 base model
2. Apply LoRA weights trained via GRPO
3. Generate videos with the fine-tuned model

Based on the QwenVL LoRA inference pattern adapted for Wan 2.1 video generation.
"""

import torch
from pathlib import Path
import imageio
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

# ==================== Configuration ====================
base_model_path = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"  # Path to base Wan 2.1 model
lora_checkpoint_path = "logs/video_ocr/wan_flow_grpo/checkpoint-60/model.safetensors"  # Path to trained LoRA weights
output_dir = "scripts/demo/wan_lora_outputs/"  # Output directory for generated videos
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inference parameters
prompts = [
    'New York Skyline with "Hello World" written with fireworks on the sky',
    'A beautiful sunset over the ocean with "GRPO" text appearing in the clouds',
]
negative_prompt = ""  # Can add negative prompts if needed
num_frames = 33  # Number of frames in video (must match training config)
height = 240  # Video height (must match training config)
width = 416  # Video width (must match training config)
num_inference_steps = 50  # Number of denoising steps (higher = better quality, slower)
guidance_scale = 4.5  # CFG scale (higher = more prompt adherence)
seed = 42  # Random seed for reproducibility

# LoRA configuration (must match training config)
lora_r = 32
lora_alpha = 64
lora_target_modules = [
    "add_k_proj",
    "add_q_proj",
    "add_v_proj",
    "to_add_out",
    "to_k",
    "to_out.0",
    "to_q",
    "to_v",
]

# ==================== Model Loading ====================
print(f"Loading base model from: {base_model_path}")
pipe = WanPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
)

# ==================== Apply LoRA ====================
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    init_lora_weights="gaussian",
    target_modules=lora_target_modules,
)

# Apply PEFT to the transformer
pipe.transformer = get_peft_model(pipe.transformer, lora_config)
print(f"LoRA applied to transformer with r={lora_r}, alpha={lora_alpha}")

# Load LoRA weights
print(f"Loading LoRA weights from: {lora_checkpoint_path}")
lora_state_dict = load_file(lora_checkpoint_path, device="cpu")

# Filter and rename keys (remove "base_model.model." prefix if present)
filtered_lora_state_dict = {}
for key, value in lora_state_dict.items():
    if "lora_" in key:
        # Remove prefix if it exists
        new_key = key.replace("base_model.model.", "")
        filtered_lora_state_dict[new_key] = value

if not filtered_lora_state_dict:
    raise ValueError(f"No LoRA weights found in {lora_checkpoint_path} (keys containing 'lora_')")

print(f"Found {len(filtered_lora_state_dict)} LoRA parameters")

# Set the LoRA weights
set_peft_model_state_dict(pipe.transformer, filtered_lora_state_dict)
print("LoRA weights loaded successfully")

# Merge LoRA weights into the base model for faster inference
print("Merging LoRA weights into base model...")
pipe.transformer = pipe.transformer.merge_and_unload()

# Move pipeline to device
pipe = pipe.to(device)
print(f"Pipeline moved to {device}")

# ==================== Generate Videos ====================
print(f"\nGenerating {len(prompts)} videos...")
print(f"Parameters: frames={num_frames}, size={height}x{width}, steps={num_inference_steps}, cfg={guidance_scale}")

# Create output directory
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Set random seed
generator = torch.Generator(device=device).manual_seed(seed)

# Encode prompts
with torch.no_grad():
    prompt_embeds = encode_prompt(
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        prompt=prompts,
        num_videos_per_prompt=1,
        device=device,
        dtype=torch.bfloat16,
    )

    # Encode negative prompt
    if negative_prompt:
        negative_prompt_list = [negative_prompt] * len(prompts)
        negative_prompt_embeds = encode_prompt(
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            prompt=negative_prompt_list,
            num_videos_per_prompt=1,
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        negative_prompt_embeds = None

# Generate videos
with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device=="cuda")):
    videos, latents, log_probs, _ = wan_pipeline_with_logprob(
        pipe,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_type="pt",  # Return as torch tensor
        return_dict=False,
        num_frames=num_frames,
        height=height,
        width=width,
        generator=generator,
        determistic=True,
    )

# ==================== Save Videos ====================
print("\nSaving generated videos...")

# Convert videos from (B, F, C, H, W) to (B, F, H, W, C) and to uint8
videos_np = videos.cpu().numpy()
videos_np = (videos_np * 255).clip(0, 255).astype('uint8')
videos_np = videos_np.transpose(0, 1, 3, 4, 2)  # (B, F, C, H, W) -> (B, F, H, W, C)

for i, (video, prompt) in enumerate(zip(videos_np, prompts)):
    # Save as MP4
    video_filename = f"wan_lora_video_{i:03d}.mp4"
    video_path = output_path / video_filename

    # Use imageio to save video
    imageio.mimwrite(video_path, video, fps=8, codec='libx264', quality=8)

    print(f"  [{i+1}/{len(prompts)}] Saved: {video_path}")
    print(f"       Prompt: {prompt}")

    # Also save first frame as preview image
    preview_filename = f"wan_lora_video_{i:03d}_frame0.png"
    preview_path = output_path / preview_filename
    imageio.imwrite(preview_path, video[0])

print(f"\n✓ All videos saved to: {output_path}")
print(f"✓ Generated {len(prompts)} videos with {num_frames} frames each")
