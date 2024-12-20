#!/usr/bin/env python3
"""
FP8 Model Loading and Quantization Tool

This script provides a command-line interface for loading and quantizing large
models using FP8 format with memory-efficient sharded loading support.
"""

import sys
import argparse
import logging
from typing import List, Optional
from .model_loader import ShardedFP8ModelLoader
from .gpu_verification import verify_gpu_availability  # Add this line

def setup_logging(verbose: bool = False) -> None:
    """Configure logging settings."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args() -> argparse.Namespace:
    # ... (rest of the function remains the same)
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Validate shard size
    if args.shard_size_gb <= 0:
        raise ValueError("Shard size must be greater than 0 GB")
    
    # Remove the device ID validation since it will be handled by verify_gpu_availability
    
def main() -> int:
    """
    Main entry point for the FP8 model loading tool.
    
    Returns:
        0 on success, non-zero on error
    """
    try:
        # Parse and validate arguments
        args = parse_args()
        validate_args(args)
        
        # Setup logging
        setup_logging(args.verbose)
        logging.info("Starting model loading process")
        
        # Verify GPU availability
        available_devices, warning = verify_gpu_availability(args.device_ids)
        if warning:
            logging.warning(warning)
        
        # Initialize model loader with verified devices
        loader = ShardedFP8ModelLoader(
            model_dir=args.model_dir,
            device_ids=available_devices,  # Use verified devices
            use_ddp=args.use_ddp,
            memory_efficient=args.memory_efficient,
            mixed_precision=args.mixed_precision,
            shard_size_gb=args.shard_size_gb
        )
        
        # Load model
        logging.info("Loading model from checkpoint")
        model = loader.load_model(args.checkpoint_path)
        logging.info("Model loaded successfully")
        
        return 0
        
    except ValueError as e:
        logging.error(f"Invalid argument: {str(e)}")
        return 1
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())