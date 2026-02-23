#!/usr/bin/env python3
"""
Quick script to check GPU availability and PyTorch setup.
Run this before training to verify your environment.
"""

import sys
import torch

print("=" * 60)
print("PyTorch GPU Check")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  CUDA not available - training will use CPU (much slower)")
    print("   If you expected GPU, check:")
    print("   1. GPU requested in sbatch script (--gres=gpu:1)")
    print("   2. PyTorch installed with CUDA support")
    print("   3. CUDA drivers available on compute node")

print("\n" + "=" * 60)
print("Test tensor operations:")
print("=" * 60)

# Test CPU
x_cpu = torch.randn(1000, 1000)
print(f"CPU tensor creation: ✓")

# Test GPU if available
if torch.cuda.is_available():
    x_gpu = torch.randn(1000, 1000).cuda()
    print(f"GPU tensor creation: ✓")
    
    # Test computation
    y_gpu = torch.matmul(x_gpu, x_gpu)
    print(f"GPU matrix multiplication: ✓")
    print(f"GPU device: {x_gpu.device}")
else:
    print("GPU tensor operations: ✗ (CUDA not available)")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)
if torch.cuda.is_available():
    print("✅ GPU is available - training will be fast!")
    print("   Make sure to uncomment GPU lines in sbatch scripts:")
    print("   #SBATCH --gres=gpu:1")
else:
    print("⚠️  No GPU - training will be slower but still workable")
    print("   For faster training, request GPU in sbatch script")

sys.exit(0)
