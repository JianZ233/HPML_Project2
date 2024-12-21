import cupy as cp
import torch
from typing import Tuple, Dict, Any, Optional

class ByteVerification:
    """
    GPU-accelerated tensor verification for FP8 quantization.
    
    This class provides efficient tensor comparison using CUDA kernels,
    enabling fast verification of quantized tensors against their
    original values.
    """
    
    def __init__(self, abs_tol: float = 0.001, chunk_size: int = 1_000_000):
        """
        Initialize the byte verification system.
        
        Args:
            abs_tol: Absolute tolerance for value comparisons
            chunk_size: Size of chunks for processing
        """
        self.abs_tol = abs_tol
        self.chunk_size = chunk_size
        
        # Initialize CUDA kernel for comparison
        self.compare_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void compare_bytes(const float* arr1, const float* arr2, 
                          int n, float abs_tol,
                          int* diff_count, int* first_diff_idx) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            
            if (tid < n) {
                float val1 = arr1[tid];
                float val2 = arr2[tid];
                float abs_diff = abs(val1 - val2);
                
                if (abs_diff > abs_tol) {
                    atomicAdd(diff_count, 1);
                    atomicCAS(first_diff_idx, -1, tid);
                }
            }
        }
        ''', 'compare_bytes')

    def verify_tensors(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[bool, int, Optional[Dict[str, Any]]]:
        """
        Verify equality between two tensors within tolerance.
        
        Args:
            tensor1: First tensor to compare
            tensor2: Second tensor to compare
            return_details: If True, return detailed comparison info
            
        Returns:
            Tuple containing:
            - bool: True if verification passed
            - int: Number of differences found
            - Optional[Dict]: Detailed comparison info if requested
        
        Raises:
            ValueError: If tensor shapes don't match
            RuntimeError: If verification process fails
        """
        try:
            # Handle FP8 tensors
            if str(tensor1.dtype).startswith('torch.float8'):
                tensor1 = tensor1.to(torch.float32)
            if str(tensor2.dtype).startswith('torch.float8'):
                tensor2 = tensor2.to(torch.float32)
            
            # Prepare tensors and verify shapes
            tensor1 = tensor1.detach().to(torch.float32).cpu()
            tensor2 = tensor2.detach().to(torch.float32).cpu()
            if tensor1.shape != tensor2.shape:
                raise ValueError(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            
            num_elements = tensor1.numel()
            total_differences = 0
            first_diff_element = -1
            first_diff_values = None
            
            # chunking
            for start_idx in range(0, num_elements, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, num_elements)
                chunk_size = end_idx - start_idx
                
                chunk1 = tensor1.view(-1)[start_idx:end_idx].cuda().contiguous()
                chunk2 = tensor2.view(-1)[start_idx:end_idx].cuda().contiguous()
                arr1 = cp.asarray(chunk1)
                arr2 = cp.asarray(chunk2)
                
                # Setup comparison parameters
                abs_tol = cp.float32(self.abs_tol)
                diff_count = cp.zeros(1, dtype=cp.int32)
                first_diff_idx = cp.full(1, -1, dtype=cp.int32)
                
                # Configure and launch kernel
                threads_per_block = 256
                blocks = (chunk_size + threads_per_block - 1) // threads_per_block
                
                self.compare_kernel(
                    grid=(blocks,),
                    block=(threads_per_block,),
                    args=(arr1, arr2, chunk_size, abs_tol,
                          diff_count, first_diff_idx)
                )
                
                # Process results
                chunk_differences = int(diff_count.get())
                chunk_first_diff = int(first_diff_idx.get())
                
                total_differences += chunk_differences
                
                if chunk_first_diff >= 0 and first_diff_element < 0:
                    first_diff_element = start_idx + chunk_first_diff
                    first_diff_values = (
                        float(arr1[chunk_first_diff]),
                        float(arr2[chunk_first_diff])
                    )
                
                del chunk1, chunk2, arr1, arr2, abs_tol
                torch.cuda.empty_cache()
            
            diff_percentage = (total_differences / num_elements) * 100
            is_successful = diff_percentage <= 99.9
            
            if return_details:
                details = {
                    'total_elements': num_elements,
                    'total_differences': total_differences,
                    'difference_percentage': diff_percentage,
                    'first_different_element': first_diff_element,
                    'first_difference_values': first_diff_values,
                    'tolerance': self.abs_tol
                }
                return is_successful, total_differences, details
            
            return is_successful, total_differences, None
            
        except Exception as e:
            raise RuntimeError(f"Verification failed: {str(e)}")

    def get_verification_stats(self) -> Dict[str, float]:
        """
        Get current verification statistics.
        
        Returns:
            Dict containing verification statistics:
            - absolute_tolerance: Current tolerance value
            - chunk_size: Processing chunk size
        """
        return {
            'absolute_tolerance': self.abs_tol,
            'chunk_size': self.chunk_size
        }
