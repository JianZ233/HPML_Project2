import torch
from typing import Optional, Dict
from fp8_format import FP8Format
from verification import ByteVerification
from verification import ExtendedVerificationPlotter

class FP8QuantizationHandler:
    """Handles FP8 quantization while preserving precision."""
    
    def __init__(self, fp8_format: Optional[FP8Format] = None):
        self.fp8_format = fp8_format or FP8Format()
        self.verifier = ByteVerification()

    def process_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process weights with proper FP8 handling and extended visualization."""
        print("\n========= Starting process_weights =========")
        plotter = ExtendedVerificationPlotter()  # Create plotter instance
        processed = {}

        # Determine target FP8 dtype upfront
        if self.fp8_format.e4m3 and hasattr(torch, 'float8_e4m3fn'):
            target_dtype = torch.float8_e4m3fn
            print("Using e4m3 format")
        elif not self.fp8_format.e4m3 and hasattr(torch, 'float8_e5m2'):
            target_dtype = torch.float8_e5m2
            print("Using e5m2 format")
        else:
            target_dtype = torch.float16  # Fallback if FP8 is not supported
            print("FP8 not supported; falling back to FP16")
        
        for name, tensor in weights.items():
            try:
                original_tensor = tensor.clone()
                if target_dtype:
                    tensor = tensor.cpu().to(torch.float32)
                    tensor = self._apply_fp8_constraints(tensor)
                    tensor = tensor.to(target_dtype)
                processed[name] = tensor

                # Verify the conversion
                is_identical, num_diff, first_diff_idx = self.verifier.verify_tensors(
                    original_tensor.cpu().to(torch.float32), tensor.cpu().to(torch.float32)
                )
                print(f"Verification for {name}: Identical={is_identical}, Differences={num_diff}")
                
            except Exception as e:
                print(f"Error processing weight {name}: {e}")
                if isinstance(tensor, torch.Tensor):
                    processed[name] = tensor.to(torch.float16)
        
        print("\n========= Completed process_weights =========")
        return processed

    def _apply_fp8_constraints(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply FP8 format constraints to tensor."""
        tensor = tensor * self.fp8_format.scale
        tensor = torch.clamp(tensor, -self.fp8_format.max_value, self.fp8_format.max_value)
        return tensor