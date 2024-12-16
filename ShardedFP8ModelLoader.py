import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Tuple, Union
import os
import queue
import threading
from dataclasses import dataclass
import numpy as np
from safetensors.torch import load_file
from enum import Enum
from transformers import AutoConfig, AutoModelForCausalLM
import gc

@dataclass
class FP8Format:
    """Configuration for FP8 format."""
    e4m3: bool = True  # True for e4m3, False for e5m2
    scale: float = 1.0
    bias: int = 7  # 7 for e4m3, 15 for e5m2
    max_value: float = 448.0  # 448.0 for e4m3, 57344.0 for e5m2

class GPUMemoryTracker:
    """Tracks GPU memory usage and allocation."""
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.memory_stats = {device: self._get_memory_stats(device) 
                           for device in device_ids}
    
    def _get_memory_stats(self, device: int) -> Dict[str, int]:
        """Get memory statistics for a GPU."""
        return {
            'total': torch.cuda.get_device_properties(device).total_memory,
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device)
        }
    
    def get_optimal_device(self, tensor_size: int) -> int:
        """Get the GPU with most available memory."""
        self.update_stats()
        available_memory = {
            device: stats['total'] - stats['allocated']
            for device, stats in self.memory_stats.items()
        }
        return max(available_memory.items(), key=lambda x: x[1])[0]
    
    def update_stats(self):
        """Update memory statistics for all GPUs."""
        for device in self.device_ids:
            self.memory_stats[device] = self._get_memory_stats(device)

class AsyncShardLoader:
    """Handles asynchronous loading of weight shards."""
    
    def __init__(self, num_prefetch: int = 2):
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.current_shard = None
        self.thread = None
        self.error = None

    def start_prefetch(self, shard_mapping: Dict, fp8_handler, device: torch.device):
        """Start prefetch thread for shards."""
        def _prefetch_worker():
            try:
                for shard_id, shard_data in shard_mapping.items():
                    if self.stop_event.is_set():
                        break
                    # Process weights with proper FP8 handling
                    processed = fp8_handler.process_weights(shard_data)
                    self.queue.put((shard_id, processed))
                self.queue.put(None)  # Signal completion
            except Exception as e:
                self.error = e
                self.queue.put(None)

        self.thread = threading.Thread(target=_prefetch_worker)
        self.thread.start()

    def get_next_shard(self) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """Get next processed shard."""
        if self.error:
            raise RuntimeError(f"Error in prefetch worker: {self.error}")
        
        if self.current_shard is None:
            self.current_shard = self.queue.get()
        return self.current_shard

    def advance(self):
        """Move to next shard."""
        self.current_shard = None

    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()

class FP8QuantizationHandler:
    """Handles FP8 quantization while preserving precision."""
    
    def __init__(self, fp8_format: Optional[FP8Format] = None):
        self.fp8_format = fp8_format or FP8Format()

    def process_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process weights with proper FP8 handling."""
        processed = {}
        
        for name, tensor in weights.items():
            # Handle different weight formats
            if isinstance(tensor, dict):
                weight = tensor.get('weight', tensor.get('quantized', None))
                scale = tensor.get('scale', self.fp8_format.scale)
                if weight is not None:
                    tensor = weight * scale
                else:
                    continue
            
            # Determine target FP8 dtype
            target_dtype = None
            if self.fp8_format.e4m3 and hasattr(torch, 'float8_e4m3fn'):
                target_dtype = torch.float8_e4m3fn
            elif not self.fp8_format.e4m3 and hasattr(torch, 'float8_e5m2'):
                target_dtype = torch.float8_e5m2
            
            try:
                # Handle tensor conversion
                if target_dtype:
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(torch.float32)
                        tensor = self._apply_fp8_constraints(tensor)
                        tensor = tensor.to(target_dtype)
                else:
                    tensor = tensor.to(torch.float16)
                
                processed[name] = tensor
                
            except Exception as e:
                print(f"Warning: Error processing weight {name}: {e}")
                processed[name] = tensor.to(torch.float16)
        
        return processed

    def _apply_fp8_constraints(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply FP8 format constraints to tensor."""
        # Apply scaling
        tensor = tensor * self.fp8_format.scale
        
        # Apply bias
        if self.fp8_format.bias != 0:
            tensor = tensor + self.fp8_format.bias
        
        # Clamp values
        tensor = torch.clamp(tensor, -self.fp8_format.max_value, self.fp8_format.max_value)
        
        return tensor

    def optimize_memory_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize memory layout for FP8 tensors."""
        if not str(tensor.dtype).startswith('torch.float8'):
            return tensor
            
        # Ensure proper alignment
        if tensor.storage_offset() % 8 != 0:
            tensor = tensor.clone()
            
        # Ensure proper padding
        if tensor.numel() % 8 != 0:
            pad_size = 8 - (tensor.numel() % 8)
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
            
        return tensor

class ShardedFP8ModelLoader:
    """Main loader class for handling sharded FP8 models."""
    
    def __init__(
        self,
        model_dir: str,
        device_ids: Optional[List[int]] = None,
        use_ddp: bool = False,
        memory_efficient: bool = True,
        mixed_precision: bool = True,
        shard_size_gb: float = 4.0,
        fp8_format: Optional[FP8Format] = None
    ):
        """
        Initialize the FP8 model loader.
        
        Args:
            model_dir: Directory containing model files
            device_ids: List of GPU device IDs to use
            use_ddp: Whether to use DistributedDataParallel
            memory_efficient: Whether to use memory-efficient loading
            mixed_precision: Whether to use mixed precision
            shard_size_gb: Size of each shard in gigabytes
            fp8_format: FP8 format configuration
        """
        self.model_dir = model_dir
        self.device_ids = device_ids or [0]
        self.use_ddp = use_ddp
        self.memory_efficient = memory_efficient
        self.mixed_precision = mixed_precision
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.fp8_handler = FP8QuantizationHandler(fp8_format=fp8_format)
        self.async_loader = AsyncShardLoader()
        self.memory_tracker = GPUMemoryTracker(self.device_ids)

    def _create_shard_mapping(self, weights: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Create optimal mapping of weights to shards.
        
        Args:
            weights: Dictionary of weight tensors
            
        Returns:
            Dictionary mapping shard IDs to weight dictionaries
        """
        shard_mapping = {}
        current_shard = {}
        current_shard_size = 0
        shard_id = 0
        
        # Sort weights by size for better distribution
        sorted_weights = sorted(
            weights.items(),
            key=lambda x: x[1].numel() * x[1].element_size() if isinstance(x[1], torch.Tensor) 
            else x[1]['weight'].numel() * x[1]['weight'].element_size() if isinstance(x[1], dict) and 'weight' in x[1]
            else 0,
            reverse=True
        )
        
        for name, tensor in sorted_weights:
            tensor_size = (tensor.numel() * tensor.element_size() if isinstance(tensor, torch.Tensor)
                         else tensor['weight'].numel() * tensor['weight'].element_size() if isinstance(tensor, dict) and 'weight' in tensor
                         else 0)
            
            if current_shard_size + tensor_size > self.shard_size_bytes and current_shard:
                shard_mapping[shard_id] = current_shard
                current_shard = {}
                current_shard_size = 0
                shard_id += 1
            
            current_shard[name] = tensor
            current_shard_size += tensor_size
        
        if current_shard:
            shard_mapping[shard_id] = current_shard
        
        return shard_mapping

    def _convert_to_meta(self, model: nn.Module):
        """
        Convert model parameters to meta tensors with proper type handling.
        Replaces the parameters entirely rather than assigning .data.
        """
        # We store a list of (parent, param_name, new_param) so we can set them after iteration
        updates = []
        with torch.no_grad():
            for full_name, param in list(model.named_parameters(recurse=True)):
                # If already meta, skip
                if param.is_meta:
                    continue
    
                # Determine appropriate meta dtype
                orig_dtype = param.dtype
                # If FP8, revert to float16 for meta representation
                if str(orig_dtype).startswith('torch.float8'):
                    meta_dtype = torch.float16
                else:
                    meta_dtype = orig_dtype
    
                # Create a new parameter on meta device
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=meta_dtype, device='meta'),
                    requires_grad=param.requires_grad
                )
    
                # Find the parent module and the attribute name
                parent = model
                attrs = full_name.split('.')
                for attr in attrs[:-1]:
                    parent = getattr(parent, attr)
                param_name = attrs[-1]
    
                # Schedule the update
                updates.append((parent, param_name, new_param))
    
        # Apply all updates
        for parent, param_name, new_param in updates:
            setattr(parent, param_name, new_param)

    def _ensure_no_meta_left(self, model: nn.Module, device: torch.device):
        """
        Ensure no parameters remain on meta device.
        Replace meta parameters with new nn.Parameters on the target device.
        """
        with torch.no_grad():
            named_params = list(model.named_parameters())
        for full_name, param in named_params:
            if param.is_meta:
                # Determine appropriate dtype
                dtype = getattr(param, '_original_dtype', param.dtype)
                if str(dtype).startswith('torch.float8'):
                    dtype = torch.float16
    
                # Create a real tensor on the target device
                real_tensor = torch.empty(param.shape, dtype=dtype, device=device)
                new_param = nn.Parameter(real_tensor, requires_grad=param.requires_grad)
    
                # Locate parent module
                parent = model
                parts = full_name.split('.')
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, parts[-1], new_param)

    def _distribute_model_across_gpus(self, model: nn.Module) -> nn.Module:
        """
        Distribute model layers across available GPUs.
        
        Args:
            model: Model to distribute
            
        Returns:
            Distributed model
        """
        if len(self.device_ids) <= 1:
            return model.to(f'cuda:{self.device_ids[0]}')

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            layers_per_gpu = max(1, num_layers // len(self.device_ids))
            
            # Move base model components to first GPU
            primary_device = f'cuda:{self.device_ids[0]}'
            model.model.embed_tokens = model.model.embed_tokens.to(primary_device)
            if hasattr(model.model, 'norm'):
                model.model.norm = model.model.norm.to(primary_device)
            if hasattr(model, 'lm_head'):
                model.lm_head = model.lm_head.to(primary_device)
            
            # Distribute transformer layers
            for i in range(0, num_layers, layers_per_gpu):
                device_idx = (i // layers_per_gpu) % len(self.device_ids)
                device = f'cuda:{self.device_ids[device_idx]}'
                
                end_idx = min(i + layers_per_gpu, num_layers)
                for j in range(i, end_idx):
                    model.model.layers[j] = model.model.layers[j].to(device)
                    
            # Print distribution info
            print(f"Model distributed across {len(self.device_ids)} GPUs")

        return model

    def _handle_fp8_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Handle FP8 weight loading with scales.
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            Processed weights
        """
        processed = {}
        for name, tensor in weights.items():
            try:
                if isinstance(tensor, dict):
                    # Handle FP8 weight with scales
                    if all(k in tensor for k in ['weight', 'scale']):
                        weight = tensor['weight']
                        scale = tensor['scale']
                        processed[name] = weight * scale
                    else:
                        processed[name] = tensor.get('weight', tensor)
                else:
                    processed[name] = tensor
            except Exception as e:
                print(f"Warning: Error processing weight {name}: {e}")
                processed[name] = tensor
        return processed

    def _load_shard_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor], device: torch.device):
        """
        Load weights from a shard into the model.
        
        Args:
            model: Model to load weights into
            weights: Dictionary of weights to load
            device: Target device for weights
        """
        unexpected_keys = []
        missing_keys = []
        
        # Process FP8 weights
        processed_weights = self._handle_fp8_weights(weights)
        
        for name, tensor in processed_weights.items():
            try:
                # Navigate through the model attributes to find the parameter
                param = None
                obj = model
                parts = name.split('.')
                for attr in parts[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        # If any part of the path doesn't exist, it's unexpected
                        param = None
                        break
                # Now obj should be the parent module, and parts[-1] should be the parameter name
                if param is None:
                    if hasattr(obj, parts[-1]) and isinstance(getattr(obj, parts[-1]), nn.Parameter):
                        param = getattr(obj, parts[-1])
                    else:
                        # If the final attribute doesn't match a parameter, it's unexpected
                        unexpected_keys.append(name)
                        continue
    
                # Optimize memory format if using FP8
                tensor = self.fp8_handler.optimize_memory_format(tensor)
    
                # Move the tensor to the correct device and ensure matching dtype
                # If the parameter is meta, we must create a new Parameter
                if param.is_meta:
                    # Determine correct dtype for final parameter
                    target_dtype = param.dtype
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)
    
                    tensor = tensor.to(device)
                    new_param = nn.Parameter(tensor, requires_grad=param.requires_grad)
    
                    # Replace the parameter in the parent module
                    parent_module = model
                    for attr in parts[:-1]:
                        parent_module = getattr(parent_module, attr)
                    setattr(parent_module, parts[-1], new_param)
    
                else:
                    # If not meta, just copy the data
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        param.copy_(tensor)
    
                # Memory cleanup if memory efficient
                if self.memory_efficient:
                    del tensor
                    torch.cuda.empty_cache()
    
            except Exception as e:
                print(f"Error loading weight {name}: {e}")
                continue
    
        # Check if any parameter in the model was not matched by the weights
        model_state_dict = model.state_dict()
        model_keys = set(model_state_dict.keys())
        weight_keys = set(processed_weights.keys())
        missing = model_keys - weight_keys
        if missing:
            missing_keys.extend(list(missing))
        
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")

    def load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load and initialize the model with proper error handling.
        """
        try:
            # Load config and create model
            config = AutoConfig.from_pretrained(self.model_dir)
            model = AutoModelForCausalLM.from_config(config)
            model = model.cpu()
    
            if self.memory_efficient:
                # Convert parameters to meta tensors
                self._convert_to_meta(model)
    
            # Load and process weights
            weights = load_file(checkpoint_path)
    
            # Optional: Filter out unwanted scale keys if they're not used
            # For example, keys ending with "input_scale" or "weight_scale"
            weights = {k: v for k, v in weights.items() 
                       if not (k.endswith('input_scale') or k.endswith('weight_scale'))}
    
            # Handle key prefix adjustments if needed
            processed_weights = {}
            for key, value in weights.items():
                if not key.startswith('model.'):
                    key = f'model.{key}'
                processed_weights[key] = value
    
            # Create shard mapping
            shard_mapping = self._create_shard_mapping(processed_weights)
    
            # Load on CPU first
            load_device = torch.device('cpu')
            self.async_loader.start_prefetch(shard_mapping, self.fp8_handler, load_device)
    
            while True:
                shard = self.async_loader.get_next_shard()
                if shard is None:
                    break
    
                shard_id, shard_weights = shard
                self._load_shard_weights(model, shard_weights, device=load_device)
                self.async_loader.advance()
    
                if self.memory_efficient:
                    torch.cuda.empty_cache()
                    gc.collect()
    
            # Ensure no meta tensors remain and move to GPU
            self._ensure_no_meta_left(model, load_device)
            model = self._distribute_model_across_gpus(model)
    
            if self.use_ddp and len(self.device_ids) > 1:
                model = DDP(model, device_ids=[self.device_ids[0]])
    
            return model
    
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        finally:
            self.async_loader.stop()