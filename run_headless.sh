#!/bin/bash
# Run script for headless environments
# This script sets up the environment variables needed to run without display

echo "Setting up headless environment..."

# Set matplotlib backend
export MPLBACKEND=Agg

# Set Qt platform to offscreen
export QT_QPA_PLATFORM=offscreen

# Disable OpenCV GUI features
export OPENCV_IO_ENABLE_OPENEXR=1

# Optional: Set TensorFlow to not allocate all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Print environment info
echo "Environment configured:"
echo "  MPLBACKEND=$MPLBACKEND"
echo "  QT_QPA_PLATFORM=$QT_QPA_PLATFORM"
echo ""

# Run the command with all arguments passed to this script
echo "Running: python main.py $@"
python main.py "$@"