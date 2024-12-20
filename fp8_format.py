from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class FP8Format:
    """
    Configuration class for FP8 format settings.
    
    This class manages the configuration for FP8 quantization, supporting both e4m3
    and e5m2 formats with automatic parameter adjustment. It provides optimal settings
    based on model size and format type.
    
    Attributes:
        e4m3: If True, use e4m3 format; if False, use e5m2 format
        scale: Scaling factor for quantization
        bias: Format-specific bias (7 for e4m3, 15 for e5m2)
        max_value: Maximum representable value (448.0 for e4m3, 57344.0 for e5m2)
        dynamic_scaling: Enable dynamic scaling for improved precision
    """
    e4m3: bool = True
    scale: float = 1.0
    bias: int = 7
    max_value: float = 448.0
    dynamic_scaling: bool = True
    
    def __post_init__(self) -> None:
        """
        Adjust format-specific parameters after initialization.
        
        Updates bias and max_value if e5m2 format is selected.
        """
        if not self.e4m3:
            self.bias = 15
            self.max_value = 57344.0
    
    @property
    def dtype_name(self) -> str:
        """
        Get the string name of the current FP8 format.
        
        Returns:
            str: Format name ('e4m3' or 'e5m2')
        """
        return "e4m3" if self.e4m3 else "e5m2"
    
    def get_optimal_settings(self, model_size_gb: Optional[float] = None) -> Dict:
        """
        Get optimal quantization settings based on model size.
        
        This method provides optimized settings for quantization, adjusting parameters
        based on the model size when provided. Larger models receive more aggressive
        optimization settings.
        
        Args:
            model_size_gb: Model size in gigabytes (optional)
            
        Returns:
            Dict: Optimized settings including scaling, precision, and thresholds
        """
        settings = {
            'use_dynamic_scaling': self.dynamic_scaling,
            'precision_bits': 7 if self.e4m3 else 8,
            'overflow_threshold': self.max_value * 0.9,
            'underflow_threshold': 1e-4
        }
        
        if model_size_gb is not None:
            if model_size_gb > 10:  # Large models (>10GB)
                settings['scale'] = 2.0
                settings['dynamic_scaling'] = True
            elif model_size_gb > 5:  # Medium models (5-10GB)
                settings['scale'] = 1.5
                settings['dynamic_scaling'] = self.dynamic_scaling
        
        return settings