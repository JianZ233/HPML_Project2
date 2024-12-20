"""
FP8 Model Loading and Quantization Package

This package provides utilities for loading and quantizing large models
using FP8 format with memory-efficient sharded loading support.
"""

from fp8_format import FP8Format
from verification import ByteVerification
from gpu_utils import GPUMemoryTracker
from shard_loader import AsyncShardLoader
from quantization import FP8QuantizationHandler
from model_loader import ShardedFP8ModelLoader
from gpu_verification import verify_gpu_availability  # Add this line

__version__ = '1.0.0'

__all__ = [
    "FP8Format",
    "ByteVerification",
    "GPUMemoryTracker",
    "AsyncShardLoader",
    "FP8QuantizationHandler",
    "ShardedFP8ModelLoader",
    "verify_gpu_availability"  # Add this line
]