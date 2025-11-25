"""
Wan 2.1 T2V Base Model Inference Demo

This script demonstrates basic video generation with the Wan 2.1 base model (no LoRA).
Use this as a baseline to compare with LoRA-finetuned models.
"""

import torch
from pathlib import Path
import imageio
from diffusers import WanPipeline
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

# ==================== Configuration ====================
base_model_path = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"  # Path to base Wan 2.1 model
output_dir = "scripts/demo/wan_base_outputs/"  # Output directory for generated videos
device = "cuda" if torch.cuda.is_available() else "cpu"

# Inference parameters
prompts = [
    'New York Skyline with "Hello World" written with fireworks on the sky',
    'A beautiful sunset over the ocean with "GRPO" text appearing in the clouds',
]
negative_prompt = ""  # Can add negative prompts if needed
num_frames = 33  # Number of frames in video
height = 240  # Video height
width = 416  # Video width
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 4.5  # CFG scale
seed = 42  # Random seed for reproducibility

# ==================== Model Loading ====================
print(f"Loading base model from: {base_model_path}")
pipe = WanPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
)
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
    video_filename = f"wan_base_video_{i:03d}.mp4"
    video_path = output_path / video_filename

    # Use imageio to save video
    imageio.mimwrite(video_path, video, fps=8, codec='libx264', quality=8)

    print(f"  [{i+1}/{len(prompts)}] Saved: {video_path}")
    print(f"       Prompt: {prompt}")

    # Also save first frame as preview image
    preview_filename = f"wan_base_video_{i:03d}_frame0.png"
    preview_path = output_path / preview_filename
    imageio.imwrite(preview_path, video[0])

print(f"\n✓ All videos saved to: {output_path}")
print(f"✓ Generated {len(prompts)} videos with {num_frames} frames each")
