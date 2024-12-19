import cupy as cp
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple, Dict, Any

class ByteVerification:
    def __init__(self):
        print("Initializing ByteVerification...")
        self.abs_tol = 0.001
        self.chunk_size = 1_000_000
        
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

    def verify_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[bool, int, int]:
        print("\n=== Starting Tensor Verification ===")
        print(f"Using absolute tolerance: {self.abs_tol}")
        
        try:
            # Convert FP8 to float32 first if needed
            if str(tensor1.dtype).startswith('torch.float8'):
                tensor1 = tensor1.to(torch.float32)
            if str(tensor2.dtype).startswith('torch.float8'):
                tensor2 = tensor2.to(torch.float32)
            
            # Convert both tensors to float32 and detach
            tensor1 = tensor1.detach().to(torch.float32)
            tensor2 = tensor2.detach().to(torch.float32)
            
            # Move to CPU
            tensor1 = tensor1.cpu()
            tensor2 = tensor2.cpu()
            
            num_elements = tensor1.numel()
            
            if tensor1.shape != tensor2.shape:
                print(f"WARNING: Shape mismatch! {tensor1.shape} vs {tensor2.shape}")
                return False, abs(tensor1.numel() - tensor2.numel()), 0
            
            # Initialize counters
            total_differences = 0
            first_diff_element = -1
            first_diff_val1 = None
            first_diff_val2 = None
            
            # Process in chunks
            for start_idx in range(0, num_elements, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, num_elements)
                chunk_size = end_idx - start_idx
                
                # Move chunk to GPU
                chunk1 = tensor1.view(-1)[start_idx:end_idx].cuda().contiguous()
                chunk2 = tensor2.view(-1)[start_idx:end_idx].cuda().contiguous()
                
                # Create arrays from chunks
                arr1 = cp.asarray(chunk1)
                arr2 = cp.asarray(chunk2)
                
                # Create GPU parameters
                abs_tol = cp.float32(self.abs_tol)
                diff_count = cp.zeros(1, dtype=cp.int32)
                first_diff_idx = cp.full(1, -1, dtype=cp.int32)
                
                # Configure kernel grid
                threads_per_block = 256
                blocks = (chunk_size + threads_per_block - 1) // threads_per_block
                
                # Launch kernel
                self.compare_kernel(
                    grid=(blocks,),
                    block=(threads_per_block,),
                    args=(arr1, arr2, chunk_size, abs_tol,
                          diff_count, first_diff_idx)
                )
                
                # Get results for this chunk
                chunk_differences = int(diff_count.get())
                chunk_first_diff = int(first_diff_idx.get())
                
                # Update total differences
                total_differences += chunk_differences
                
                # Track first difference across all chunks
                if chunk_first_diff >= 0 and first_diff_element < 0:
                    first_diff_element = start_idx + chunk_first_diff
                    first_diff_val1 = float(arr1[chunk_first_diff])
                    first_diff_val2 = float(arr2[chunk_first_diff])
                
                # Clean up GPU memory
                del chunk1, chunk2, arr1, arr2, abs_tol
                torch.cuda.empty_cache()
            
            if first_diff_element >= 0:
                print(f"First difference at element {first_diff_element}")
                print(f"Values at difference: {first_diff_val1} vs {first_diff_val2}")
                abs_diff = abs(first_diff_val1 - first_diff_val2)
                print(f"Absolute difference: {abs_diff}")
            
            # Calculate percentage that exceed tolerance
            diff_percentage = (total_differences / num_elements) * 100
            is_successful = diff_percentage <= 99.9
            
            if is_successful:
                print(f"Verification passed: {diff_percentage:.4f}% values exceed tolerance")
            else:
                print(f"Verification failed: {diff_percentage:.4f}% values exceed tolerance")
            
            return is_successful, total_differences, first_diff_element
            
        except Exception as e:
            print(f"ERROR during verification: {e}")
            raise


class ExtendedVerificationPlotter:
    def __init__(self):
        self.layer_results = {}  # Store basic percentage differences
        self.detailed_results = {}  # Store more detailed metrics
        
    def add_result(self, layer_name: str, num_diffs: int, total_elements: int, 
                   first_diff_val1: float = None, first_diff_val2: float = None,
                   abs_diff: float = None):
        """Add verification result with detailed metrics for a layer."""
        diff_percentage = (num_diffs / total_elements) * 100
        self.layer_results[layer_name] = diff_percentage
        
        self.detailed_results[layer_name] = {
            'num_diffs': num_diffs,
            'total_elements': total_elements,
            'diff_percentage': diff_percentage,
            'first_diff_val1': first_diff_val1,
            'first_diff_val2': first_diff_val2,
            'abs_diff': abs_diff
        }

    def plot(self, title: str = "FP8 Verification Results", figsize=(15, 8)):
        """Create a bar plot of difference percentages."""
        if not self.layer_results:
            print("No data available for plotting.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort layers by name for consistent ordering
        layers = sorted(self.layer_results.keys())
        percentages = [self.layer_results[layer] for layer in layers]
        
        # Create bars
        bars = ax.bar(range(len(layers)), percentages)
        
        # Customize plot
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Percentage of Values Exceeding Tolerance')
        ax.set_title(title)
        
        # Set y-axis limits with some padding
        if len(percentages) > 0:
            max_percentage = max(percentages)
            ax.set_ylim([0, max_percentage * 1.1])  # Add 10% padding at the top
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap(self, figsize=(12, 8)):
        """Create a heatmap showing the distribution of differences across layers."""
        if not self.layer_results:
            print("No data available for plotting.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        # Reshape data for heatmap
        layer_names = list(self.layer_results.keys())
        percentages = np.array([self.layer_results[layer] for layer in layer_names])
        
        # Create heatmap matrix (1 row, multiple columns)
        heatmap_data = percentages.reshape(1, -1)
        
        # Plot heatmap
        sns.heatmap(heatmap_data, 
                    cmap='YlOrRd',
                    annot=True, 
                    fmt='.2f',
                    xticklabels=layer_names,
                    yticklabels=['Difference %'],
                    cbar_kws={'label': 'Percentage of Different Values'})
        
        plt.title('Distribution of Differences Across Layers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def plot_difference_distribution(self, figsize=(15, 6)):
        """Create a violin plot showing the distribution of absolute differences."""
        # Gather absolute differences
        abs_diffs = [result['abs_diff'] for result in self.detailed_results.values() 
                     if result['abs_diff'] is not None and not np.isnan(result['abs_diff'])]

        if not abs_diffs:
            print("No absolute difference data to plot.")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Violin plot of absolute differences
        sns.violinplot(data=abs_diffs, ax=ax1)
        ax1.set_title('Distribution of Absolute Differences')
        ax1.set_ylabel('Absolute Difference')

        # Right plot: Log-scale distribution
        # Filter out zero or negative values before taking log10
        log_abs_diffs = [d for d in abs_diffs if d > 0]
        if log_abs_diffs:
            sns.violinplot(data=np.log10(log_abs_diffs), ax=ax2)
            ax2.set_title('Log-Scale Distribution of Differences')
            ax2.set_ylabel('Log10(Absolute Difference)')
        else:
            ax2.text(0.5, 0.5, 'No positive differences for log scale', ha='center', va='center')
            ax2.set_title('Log-Scale Distribution of Differences (N/A)')

        plt.tight_layout()
        return fig
    
    def plot_layer_comparison(self, figsize=(12, 6)):
        """Create a comparative visualization of original vs converted values for the first difference."""
        layers = []
        orig_vals = []
        conv_vals = []

        for layer, data in self.detailed_results.items():
            if data['first_diff_val1'] is not None and data['first_diff_val2'] is not None:
                layers.append(layer)
                orig_vals.append(data['first_diff_val1'])
                conv_vals.append(data['first_diff_val2'])

        if not layers:
            print("No per-layer difference data to compare.")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(layers))
        width = 0.35

        ax.bar(x - width/2, orig_vals, width, label='Original Value')
        ax.bar(x + width/2, conv_vals, width, label='Converted Value')
        
        ax.set_ylabel('Value')
        ax.set_title('Comparison of First Different Values in Each Layer')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        return fig
    
    def plot_summary_statistics(self, figsize=(10, 6)):
        """Create a summary statistics visualization."""
        if not self.layer_results:
            print("No data available for summary statistics.")
            return None

        diff_values = list(self.layer_results.values())
        stats = {
            'Mean Diff %': np.mean(diff_values),
            'Median Diff %': np.median(diff_values),
            'Max Diff %': np.max(diff_values),
            'Min Diff %': np.min(diff_values),
            'Std Dev %': np.std(diff_values)
        }
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(stats.keys(), stats.values())
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        ax.set_title('Summary Statistics of Differences')
        ax.set_ylabel('Percentage')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
