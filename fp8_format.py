from dataclasses import dataclass
from typing import Optional

@dataclass
class FP8Format:
    """Configuration for FP8 format with optimized settings."""
    e4m3: bool = True  # True for e4m3, False for e5m2
    scale: float = 1.0
    bias: int = 7  # 7 for e4m3, 15 for e5m2
    max_value: float = 448.0  # 448.0 for e4m3, 57344.0 for e5m2
    dynamic_scaling: bool = True  # Enable dynamic scaling for better precision
    
    def __post_init__(self):
        # Adjust settings based on format
        if not self.e4m3:
            self.bias = 15
            self.max_value = 57344.0
    
    @property
    def dtype_name(self) -> str:
        """Get the name of the FP8 format."""
        return "e4m3" if self.e4m3 else "e5m2"
    
    def get_optimal_settings(self, model_size_gb: Optional[float] = None) -> dict:
        """Get optimal settings based on model size."""
        settings = {
            'use_dynamic_scaling': self.dynamic_scaling,
            'precision_bits': 7 if self.e4m3 else 8,
            'overflow_threshold': self.max_value * 0.9,
            'underflow_threshold': 1e-4
        }
        
        # Adjust settings based on model size if provided
        if model_size_gb is not None:
            if model_size_gb > 10:  # Large models
                settings['scale'] = 2.0
                settings['dynamic_scaling'] = True
            elif model_size_gb > 5:  # Medium models
                settings['scale'] = 1.5
                settings['dynamic_scaling'] = self.dynamic_scaling
        
        return settings