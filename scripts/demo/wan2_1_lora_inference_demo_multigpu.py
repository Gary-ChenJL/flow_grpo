"""
Wan 2.1 T2V LoRA Inference Demo - Multi-GPU Optimized

This script is optimized for memory efficiency:
1. Multi-GPU support via device_map or manual sharding
2. Sequential video generation (one at a time)
3. CPU offloading for VAE and text encoder
4. CUDA cache clearing between generations
5. Reduced memory footprint

Use this when you encounter CUDA OOM errors.
"""

import torch
import gc
from pathlib import Path
import imageio
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt

# ==================== Configuration ====================
base_model_path = "Wan2.1-T2V-1.3B"  # Path to base Wan 2.1 model
lora_checkpoint_path = "logs/video_ocr/wan_flow_grpo/checkpoints/checkpoint-476/lora/adapter_model.safetensors"  # Path to trained LoRA weights
output_dir = "logs/demo/wan_lora_outputs/"  # Output directory for generated videos

# Memory optimization settings
enable_cpu_offload = True  # Offload VAE and text encoder to CPU when not in use
enable_model_cpu_offload = False  # Offload entire pipeline (slower but more memory efficient)
use_sequential_cpu_offload = False  # Most aggressive offloading (slowest, least memory)
enable_attention_slicing = True  # Reduce memory during attention computation
enable_vae_slicing = True  # Process VAE in slices
enable_vae_tiling = False  # Process VAE in tiles (for very large resolutions)

# Multi-GPU settings
num_gpus = torch.cuda.device_count()
primary_device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🔧 Found {num_gpus} GPU(s)")

# Inference parameters
prompts = [
    'A vibrant scene of Kenyan golfers at a lush green golf course on a sunny day. The golfers, dressed in casual yet stylish attire, are teeing off with animated expressions, showcasing their enthusiasm for the game. Rolling hills and pristine greens stretch out behind them, creating a picturesque backdrop. In the foreground, a golf buggy and a caddy stand ready, adding to the serene atmosphere. The camera captures the action from a mid-shot angle, focusing on the golfers dynamic motions as they swing their clubs.',
    'A historical reenactment of the Communist Revolution in Moscow, featuring a bustling crowd of diverse Russian people, including workers, soldiers, and civilians, all united in revolutionary fervor. They are waving red flags and banners with hammer and sickle symbols, marching through the snow-covered streets of early 20th century Moscow. The camera captures the intense emotions on their faces, the cold weather conditions, and the iconic architecture of the city in the background. Wide shots and close-ups convey the scale and passion of the revolution.',
    'A quaint A-frame cottage made of wood and glass, nestled off the coast of Mexico. The cottage sits on a small peninsula surrounded by crystal-clear turquoise waters and sandy beaches. The exterior is crafted from weathered wooden planks and large glass windows, offering panoramic views of the ocean. Palm trees sway gently in the background, and colorful Mexican flowers adorn the front of the cottage. The sun sets in the distance, casting a warm golden glow over the scene. Wide shot, capturing the serene coastal landscape and the charming cottage.'
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

# ==================== Helper Functions ====================
def print_memory_usage(prefix=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved {prefix}")

def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ==================== Model Loading ====================
print(f"Loading base model from: {base_model_path}")
print_memory_usage("(before loading)")

# Load pipeline with memory optimizations
pipe = WanPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    # device_map="auto",  # Uncomment for automatic multi-GPU distribution
)

print_memory_usage("(after loading base model)")

# ==================== Apply LoRA ====================
print("\nConfiguring LoRA...")
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

print_memory_usage("(after LoRA merge)")

# ==================== Memory Optimizations ====================
print("\n🚀 Applying memory optimizations...")

# Move pipeline to primary device
pipe = pipe.to(primary_device)
print(f"✓ Pipeline moved to {primary_device}")

# Enable attention slicing (reduces memory during attention computation)
if enable_attention_slicing:
    try:
        pipe.enable_attention_slicing(slice_size="auto")
        print("✓ Attention slicing enabled")
    except:
        print("⚠ Attention slicing not supported")

# Enable VAE slicing (processes VAE in slices)
if enable_vae_slicing:
    try:
        pipe.enable_vae_slicing()
        print("✓ VAE slicing enabled")
    except:
        print("⚠ VAE slicing not supported")

# Enable VAE tiling (for very large resolutions)
if enable_vae_tiling:
    try:
        pipe.enable_vae_tiling()
        print("✓ VAE tiling enabled")
    except:
        print("⚠ VAE tiling not supported")

# CPU offloading strategies (choose one)
if use_sequential_cpu_offload:
    # Most aggressive: Each component loaded only when needed
    try:
        pipe.enable_sequential_cpu_offload()
        print("✓ Sequential CPU offload enabled (most memory efficient)")
    except:
        print("⚠ Sequential CPU offload not supported")
elif enable_model_cpu_offload:
    # Moderate: Offload entire model when not in use
    try:
        pipe.enable_model_cpu_offload()
        print("✓ Model CPU offload enabled")
    except:
        print("⚠ Model CPU offload not supported")
elif enable_cpu_offload:
    # Light: Offload VAE and text encoder only
    try:
        # Move VAE and text encoder to CPU
        pipe.vae.to("cpu")
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            if isinstance(pipe.text_encoder, list):
                for encoder in pipe.text_encoder:
                    encoder.to("cpu")
            else:
                pipe.text_encoder.to("cpu")
        print("✓ VAE and text encoder offloaded to CPU")
    except Exception as e:
        print(f"⚠ CPU offload failed: {e}")

print_memory_usage("(after optimizations)")

# ==================== Generate Videos ====================
print(f"\n📹 Generating {len(prompts)} videos sequentially...")
print(f"Parameters: frames={num_frames}, size={height}x{width}, steps={num_inference_steps}, cfg={guidance_scale}")

# Create output directory
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Prepare text encoders and tokenizers as lists
text_encoders = [pipe.text_encoder] if not isinstance(pipe.text_encoder, list) else pipe.text_encoder
tokenizers = [pipe.tokenizer] if not isinstance(pipe.tokenizer, list) else pipe.tokenizer

# Process each video sequentially to avoid OOM
for video_idx, prompt in enumerate(prompts):
    print(f"\n{'='*80}")
    print(f"Generating video {video_idx+1}/{len(prompts)}")
    print(f"Prompt: {prompt[:100]}...")
    print_memory_usage("(before generation)")

    # Set random seed (add video_idx for variation)
    generator = torch.Generator(device=primary_device).manual_seed(seed + video_idx)

    # Encode prompt for single video
    with torch.no_grad():
        # Move text encoder to GPU temporarily if offloaded
        if enable_cpu_offload:
            for encoder in text_encoders:
                encoder.to(primary_device)

        prompt_embeds = encode_prompt(
            text_encoder=text_encoders,
            tokenizer=tokenizers,
            prompt=[prompt],  # Single prompt
            num_videos_per_prompt=1,
            device=primary_device,
            dtype=torch.bfloat16,
        )

        # Encode negative prompt
        if negative_prompt:
            negative_prompt_embeds = encode_prompt(
                text_encoder=text_encoders,
                tokenizer=tokenizers,
                prompt=[negative_prompt],
                num_videos_per_prompt=1,
                device=primary_device,
                dtype=torch.bfloat16,
            )
        else:
            negative_prompt_embeds = None

        # Move text encoder back to CPU if offloaded
        if enable_cpu_offload:
            for encoder in text_encoders:
                encoder.to("cpu")

        # Clear cache before generation
        clear_cuda_cache()
        print_memory_usage("(after encoding, before diffusion)")

        # Generate video
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
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

        print_memory_usage("(after generation)")

    # ==================== Save Video ====================
    # Convert video from (1, F, C, H, W) to (F, H, W, C) and to uint8
    video_np = videos[0].cpu().numpy()  # Take first (and only) video
    video_np = (video_np * 255).clip(0, 255).astype('uint8')
    video_np = video_np.transpose(0, 2, 3, 1)  # (F, C, H, W) -> (F, H, W, C)

    # Save as MP4
    video_filename = f"wan_lora_video_{video_idx:03d}.mp4"
    video_path = output_path / video_filename

    imageio.mimwrite(video_path, video_np, fps=8, codec='libx264', quality=8)

    print(f"✓ Saved: {video_path}")

    # Also save first frame as preview image
    preview_filename = f"wan_lora_video_{video_idx:03d}_frame0.png"
    preview_path = output_path / preview_filename
    imageio.imwrite(preview_path, video_np[0])

    # Clean up
    del videos, latents, log_probs, prompt_embeds
    if negative_prompt_embeds is not None:
        del negative_prompt_embeds
    del video_np

    # Clear CUDA cache
    clear_cuda_cache()
    print_memory_usage("(after cleanup)")

print(f"\n{'='*80}")
print(f"✓ All videos saved to: {output_path}")
print(f"✓ Generated {len(prompts)} videos with {num_frames} frames each")
print(f"\nFinal memory usage:")
print_memory_usage()
