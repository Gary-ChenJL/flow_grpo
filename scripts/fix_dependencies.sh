#!/bin/bash

# Fix xformers and PyTorch compatibility issues for flow_grpo

echo "=========================================="
echo "Flow GRPO Dependency Fix Script"
echo "=========================================="
echo ""

# Check current versions
echo "Current environment:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: NOT INSTALLED"
python -c "import xformers; print(f'xformers: {xformers.__version__}')" 2>/dev/null || echo "xformers: NOT INSTALLED"
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')" 2>/dev/null || echo "diffusers: NOT INSTALLED"
echo ""

# Detect PyTorch CUDA version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)

echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"
echo ""

# Uninstall incompatible xformers
echo "Uninstalling incompatible xformers..."
pip uninstall -y xformers

echo ""
echo "Installing compatible xformers for PyTorch $TORCH_VERSION..."

# Install xformers compatible with PyTorch 2.3.x + cu121
if [[ $TORCH_VERSION == *"2.3"* ]] && [[ $CUDA_VERSION == *"12.1"* ]]; then
    echo "Installing xformers for PyTorch 2.3.x + CUDA 12.1..."
    pip install xformers==0.0.26.post1 --no-deps
elif [[ $TORCH_VERSION == *"2.3"* ]]; then
    echo "Installing xformers for PyTorch 2.3.x (auto-detect CUDA)..."
    pip install xformers==0.0.26.post1
elif [[ $TORCH_VERSION == *"2.4"* ]]; then
    echo "Installing xformers for PyTorch 2.4.x..."
    pip install xformers==0.0.27
else
    echo "Installing xformers (auto-detect version)..."
    pip install xformers --no-build-isolation
fi

echo ""
echo "Verifying installation..."
python -c "import xformers; print(f'✓ xformers {xformers.__version__} installed successfully')" || echo "✗ xformers installation failed"

echo ""
echo "Testing imports..."
python -c "
try:
    from diffusers import StableDiffusion3Pipeline, WanPipeline
    print('✓ diffusers imports successfully')
except Exception as e:
    print(f'✗ diffusers import failed: {e}')
"

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="
echo ""
echo "If you still see xformers warnings, they can be safely ignored."
echo "The training should work even without xformers C++/CUDA extensions."
echo ""
echo "To suppress xformers warnings, set:"
echo "  export XFORMERS_DISABLED=1"
echo ""
