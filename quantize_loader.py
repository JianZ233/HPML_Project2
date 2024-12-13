import torch
import cupy as cp
from pathlib import Path
from typing import Dict, Optional, Union
from enum import Enum
import json

class QuantizationFormat(Enum):
    FP8 = "fp8"
    GPTQ = "gptq"
    FP16 = "fp16"  # for testing
    FP32 = "fp32"  # for testing

class QuantizedModelLoader:
    def __init__(
        self, 
        model_path: Path,
        quant_format: QuantizationFormat,
        device: str = 'cuda',
        max_memory: Optional[Dict[int, str]] = None  # memory limit per GPU device
    ):
        self.model_path = Path(model_path)
        self.quant_format = quant_format
        self.device = device
        self.max_memory = max_memory or {0: "24GiB"}  # default to 24GB on first GPU
        
        # Load model configuration
        self.config = self._load_config()
        
        # Initialize empty weight cache
        self.weight_cache: Dict[str, torch.Tensor] = {}
        
    def _load_config(self) -> dict:
        """Load model configuration from json"""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config file found at {config_path}")
        return json.load(open(config_path))
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """
        Load a single tensor while preserving its quantized format
        """
        if tensor_name in self.weight_cache:
            return self.weight_cache[tensor_name]
            
        # Construct path to tensor file
        tensor_path = self.model_path / f"{tensor_name}.bin"
        
        # Memory-efficient loading depending on quantization format
        if self.quant_format == QuantizationFormat.FP8:
            tensor = self._load_fp8_tensor(tensor_path)
        elif self.quant_format == QuantizationFormat.GPTQ:
            tensor = self._load_gptq_tensor(tensor_path)
        else:
            # Default loading for testing with regular formats
            tensor = torch.load(tensor_path, map_location=self.device)
            
        self.weight_cache[tensor_name] = tensor
        return tensor
    
    def _load_fp8_tensor(self, path: Path) -> torch.Tensor:
        """
        Load FP8 tensor preserving exact byte representation
        """
        # Load raw bytes
        with open(path, 'rb') as f:
            raw_bytes = f.read()
                
        # Convert to uint8 tensor to preserve exact bytes and reshape
        uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8).reshape(-1)
        
        # Get tensor shape from metadata
        shape = self._get_tensor_shape(path.stem)
        
        # Just move to GPU and reshape
        return uint8_tensor.cuda().reshape(shape)

    
    def _load_gptq_tensor(self, path: Path) -> torch.Tensor:
        """
        Load GPTQ quantized tensor preserving exact byte representation
        """
        # Similar to FP8 but with GPTQ-specific handling
        # Implementation would depend on exact GPTQ format being used
        raise NotImplementedError("GPTQ loading to be implemented")
    
    def _get_tensor_shape(self, tensor_name: str) -> tuple:
        """Get expected tensor shape from config"""
        # Implementation depends on your config format
        # This is a placeholder
        return self.config["tensor_shapes"][tensor_name]
    
    def validate_tensor(self, tensor_name: str, reference_tensor: torch.Tensor) -> bool:
        """
        Validate that loaded tensor matches reference byte-for-byte
        """
        loaded_tensor = self.load_tensor(tensor_name)
        
        # Convert both to CuPy arrays
        ref_array = cp.asarray(reference_tensor.cuda())
        loaded_array = cp.asarray(loaded_tensor)
        
        # View them as bytes
        ref_bytes = ref_array.view(dtype=cp.uint8)
        loaded_bytes = loaded_array.view(dtype=cp.uint8)
        
        return cp.array_equal(ref_bytes, loaded_bytes)

# Example usage
def test_loader():
    model_path = Path("./quantized_model")
    loader = QuantizedModelLoader(
    model_path=model_path,
    quant_format=QuantizationFormat.FP8,
    max_memory={0: "20GiB"}  
)
    
    # Load a tensor
    tensor = loader.load_tensor("model.layer.0.weight")
    
    # Validate against reference
    reference = torch.load("reference_weight.pt")
    is_valid = loader.validate_tensor("model.layer.0.weight", reference)
    print(f"Loaded tensor matches reference: {is_valid}")