#!/bin/bash

# VideoAlign Installation and Setup Script
# Installs all required dependencies for VideoAlign reward model

echo "=========================================="
echo "VideoAlign Installation Script"
echo "=========================================="
echo ""

# Check if VideoAlign directory exists
if [ ! -d "VideoAlign" ]; then
    echo "Cloning VideoAlign repository..."
    git clone https://github.com/KlingTeam/VideoAlign
    echo ""
fi

cd VideoAlign

echo "Step 1: Upgrading transformers for Qwen2-VL support..."
echo "----------------------------------------"
# Qwen2VL requires transformers >= 4.44.0
pip install --upgrade "transformers>=4.45.0"

echo ""
echo "Step 2: Installing VideoAlign dependencies..."
echo "----------------------------------------"
# Install other required packages
pip install qwen-vl-utils
pip install einops
pip install torchvision
pip install Pillow

echo ""
echo "Step 3: Installing flash-attention..."
echo "----------------------------------------"
# Flash attention is optional but recommended
pip install flash-attn==2.5.8 --no-build-isolation || {
    echo "Warning: flash-attn installation failed (this is optional)"
    echo "VideoAlign will work without it, but may be slower"
}

echo ""
echo "Step 4: Installing VideoAlign in editable mode..."
echo "----------------------------------------"
pip install -e .

cd ..

echo ""
echo "Step 5: Verifying installation..."
echo "----------------------------------------"
python -c "
import sys
try:
    from transformers import Qwen2VLForConditionalGeneration
    print('✓ Qwen2VL model available in transformers')
except ImportError as e:
    print(f'✗ Qwen2VL import failed: {e}')
    sys.exit(1)

try:
    sys.path.insert(0, 'VideoAlign')
    from inference import VideoVLMRewardInference
    print('✓ VideoAlign inference module available')
except ImportError as e:
    print(f'✗ VideoAlign import failed: {e}')
    sys.exit(1)

print('')
print('✓ All VideoAlign dependencies installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "VideoAlign Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Download the VideoAlign model checkpoint:"
    echo "   mkdir -p hf_cache"
    echo "   cd hf_cache"
    echo "   git lfs install"
    echo "   git clone https://huggingface.co/KwaiVGI/VideoReward"
    echo "   cd .."
    echo ""
    echo "2. Test your training:"
    echo "   ./scripts/test_wan2_1_videoalign.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Installation encountered errors"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above and try again."
    exit 1
fi
