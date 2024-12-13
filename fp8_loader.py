import torch
import cupy as cp
from pathlib import Path
from typing import Dict, Optional
import json

class FP8ModelLoader:
    def __init__(
        self, 
        model_path: Path,
        device: str = 'cuda',
        max_memory: Optional[Dict[int, str]] = None
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.max_memory = max_memory or {0: "24GiB"}
        
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
        """Load a single FP8 tensor while preserving its byte representation"""
        if tensor_name in self.weight_cache:
            return self.weight_cache[tensor_name]
            
        tensor_path = self.model_path / f"{tensor_name}.bin"
        
        # Load raw bytes
        with open(tensor_path, 'rb') as f:
            raw_bytes = f.read()
                
        # Convert to uint8 tensor to preserve exact bytes
        uint8_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8).reshape(-1)
        
        # Get tensor shape from metadata
        shape = self._get_tensor_shape(tensor_name)
        
        # Move to GPU and reshape
        tensor = uint8_tensor.cuda().reshape(shape)
        self.weight_cache[tensor_name] = tensor
        
        return tensor
    
    def _get_tensor_shape(self, tensor_name: str) -> tuple:
        """Get expected tensor shape from config"""
        return self.config["tensor_shapes"][tensor_name]
    
    def validate_tensor(self, tensor_name: str, reference_tensor: torch.Tensor) -> bool:
        """Validate loaded tensor matches reference byte-for-byte using CuPy"""
        loaded_tensor = self.load_tensor(tensor_name)
        
        # Convert both to CuPy arrays and view as bytes
        ref_array = cp.asarray(reference_tensor.cuda())
        loaded_array = cp.asarray(loaded_tensor)
        
        ref_bytes = ref_array.view(dtype=cp.uint8)
        loaded_bytes = loaded_array.view(dtype=cp.uint8)
        
        return cp.array_equal(ref_bytes, loaded_bytes)