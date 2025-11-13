# XFormers Version Mismatch Fix

## Problem

You're encountering this error:
```
ImportError: cannot import name '_flash_attention_backward_flop' from 'torch.utils.flop_counter'
```

This happens because xformers was built for PyTorch 2.6.0+cu124, but your environment has PyTorch 2.3.1+cu121. The versions are incompatible.

## Quick Fix (Recommended)

Run the automated fix script:

```bash
cd /home/user/flow_grpo
./scripts/fix_dependencies.sh
```

This will automatically:
1. Detect your PyTorch version
2. Uninstall incompatible xformers
3. Install the correct xformers version
4. Verify the installation

## Manual Fix Options

### Option 1: Reinstall xformers for PyTorch 2.3.x

```bash
# Uninstall current xformers
pip uninstall -y xformers

# Install compatible version for PyTorch 2.3.x + CUDA 12.1
pip install xformers==0.0.26.post1
```

### Option 2: Disable xformers (Training still works)

If you continue to have issues, you can disable xformers entirely:

```bash
export XFORMERS_DISABLED=1
```

Then run your training as normal. The warning will appear but training will proceed using PyTorch's native attention implementation.

### Option 3: Upgrade PyTorch (Advanced)

If you want to use the newer xformers, upgrade PyTorch instead:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xformers --no-build-isolation
```

**Note:** This may require reinstalling other dependencies and could break existing code.

## Verification

After applying the fix, verify it works:

```bash
python -c "from diffusers import StableDiffusion3Pipeline, WanPipeline; print('✓ Success!')"
```

If you see "✓ Success!", you're good to go!

## Understanding the Warning

The xformers warning:
```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions
```

This warning is **usually safe to ignore**. It means:
- xformers will use Python fallback implementations instead of optimized CUDA kernels
- Training will be slightly slower but still functional
- Memory-efficient attention features won't be available

## Training Impact

| Scenario | Impact |
|----------|--------|
| xformers fully working | Best performance, memory-efficient attention |
| xformers warnings | Slightly slower, but training works fine |
| xformers disabled | Uses PyTorch native attention, minimal impact |

## Recommended Action

1. **Run the fix script first**: `./scripts/fix_dependencies.sh`
2. **If still see warnings**: Ignore them - training will work
3. **If import errors persist**: Set `export XFORMERS_DISABLED=1`

## Testing Your Fix

Run a simple test:

```bash
cd /home/user/flow_grpo
python -c "
import sys
sys.path.insert(0, '.')
from scripts.train_wan2_1 import *
print('✓ All imports successful!')
"
```

## Still Having Issues?

If you still encounter problems after trying these fixes:

1. Check your Python version: `python --version` (should be 3.10.x)
2. Check your CUDA version: `nvidia-smi`
3. Ensure you're in the correct conda environment
4. Try a fresh environment:
   ```bash
   conda create -n flow_grpo_fresh python=3.10
   conda activate flow_grpo_fresh
   # Reinstall requirements
   ```

## For VideoAlign Specifically

If you've installed VideoAlign dependencies and they conflict:

```bash
# VideoAlign requires flash-attn 2.5.8
pip install flash-attn==2.5.8 --no-build-isolation

# Make sure it's compatible with your PyTorch
pip install xformers==0.0.26.post1
```

## Summary

The **easiest solution** is to run:
```bash
./scripts/fix_dependencies.sh
```

This will automatically fix the version mismatch. If you still see warnings after this, they can be safely ignored.
