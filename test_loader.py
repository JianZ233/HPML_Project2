import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Union
from quantize_loader import QuantizedModelLoader, QuantizationFormat

def create_test_data(save_dir: Path):
    """Create a small test model with known values"""
    save_dir.mkdir(exist_ok=True)
    
    # Create a small tensor with known values
    tensor = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ], dtype=torch.float32).cuda()
    
    # Save original tensor for reference
    torch.save(tensor, save_dir / "reference.pt")
    
    # Save raw bytes (simulating quantized format)
    with open(save_dir / "layer.0.weight.bin", "wb") as f:
        f.write(tensor.cpu().numpy().tobytes())
    
    # Create config with tensor shapes
    config = {
        "tensor_shapes": {
            "layer.0.weight": [2, 3]
        }
    }
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f)
        
    return tensor

def run_loader_test():
    """Test the quantized loader with a small tensor"""
    test_dir = Path("./test_model")
    
    # Create test data and get reference tensor
    print("Creating test data...")
    reference_tensor = create_test_data(test_dir)
    
    # Initialize loader
    print("Initializing loader...")
    loader = QuantizedModelLoader(
        model_path=test_dir,
        quant_format=QuantizationFormat.FP8,
        max_memory={0: "20GiB"}  # L4 setting
    )
    
    # Load and validate tensor
    print("Loading tensor...")
    loaded_tensor = loader.load_tensor("layer.0.weight")
    
    # Compare with reference
    print("Validating loaded tensor...")
    is_valid = loader.validate_tensor("layer.0.weight", reference_tensor)
    
    print(f"\nResults:")
    print(f"Reference tensor:\n{reference_tensor}")
    print(f"\nLoaded tensor:\n{loaded_tensor}")
    print(f"\nByte-level match: {is_valid}")
    
    if is_valid:
        print("\nSuccess! Loader preserved exact byte representation")
    else:
        print("\nWarning: Loaded tensor does not match reference")
        
        # Print byte-level comparison for debugging
        ref_bytes = reference_tensor.cpu().numpy().tobytes()
        loaded_bytes = loaded_tensor.cpu().numpy().tobytes()
        print("\nFirst 20 bytes comparison:")
        print(f"Reference bytes: {list(ref_bytes[:20])}")
        print(f"Loaded bytes:    {list(loaded_bytes[:20])}")

if __name__ == "__main__":
    run_loader_test()