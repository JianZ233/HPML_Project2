import os
import gc
import json
import torch
import numpy as np
from torch import nn
from typing import Dict, List, Optional
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors import safe_open
from safetensors.torch import load_file
from fp8_format import FP8Format
from quantization import FP8QuantizationHandler
from gpu_utils import GPUMemoryTracker
from shard_loader import AsyncShardLoader
from gpu_verification import verify_gpu_availability

class ShardedFP8ModelLoader:
    """
    A memory-efficient model loader that supports FP8 quantization and sharded loading.
    
    This loader implements both synchronous and asynchronous loading strategies for large
    models, with support for FP8 quantization, multi-GPU distribution, and memory optimization.
    """
    
    def __init__(
        self,
        model_dir: str,
        device_ids: Optional[List[int]] = None,
        use_ddp: bool = False,
        memory_efficient: bool = True,
        mixed_precision: bool = True,
        shard_size_gb: float = 4.0,
        fp8_format: Optional[FP8Format] = None,
        chunk_size_mb: int = 512,
        num_prefetch: int = 2,
        use_async: bool = False
    ):
        """
        Initialize the model loader with specified configuration.
        
        Args:
            model_dir: Path to the model directory
            device_ids: List of GPU device IDs to use (default: [0])
            use_ddp: Enable DistributedDataParallel
            memory_efficient: Enable memory optimization
            mixed_precision: Enable mixed precision training
            shard_size_gb: Size of each shard in gigabytes
            fp8_format: FP8 format configuration
            chunk_size_mb: Processing chunk size in megabytes
            num_prefetch: Number of shards to prefetch in async mode
            use_async: Enable asynchronous loading
        """
        self.model_dir = model_dir
        self.device_ids = device_ids or [0]
        self.use_ddp = use_ddp
        self.memory_efficient = memory_efficient
        self.mixed_precision = mixed_precision
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.chunk_size = int(chunk_size_mb * 1024**2)
        self.use_async = use_async
        self.callbacks = []
        
        # Initialize components
        self.fp8_handler = FP8QuantizationHandler(fp8_format=fp8_format)
        self.async_loader = AsyncShardLoader(num_prefetch=num_prefetch) if use_async else None
        self.memory_tracker = GPUMemoryTracker(self.device_ids)
        self.primary_device = f"cuda:{self.device_ids[0]}"
        self.device_ids, warning = verify_gpu_availability(self.device_ids)
        if warning:
            print(warning)
        self.use_ddp = use_ddp

        
        self._configure_memory_settings()

    def register_callback(self, callback):
        """Register a callback for loading progress monitoring."""
        self.callbacks.append(callback)

    def _notify_callbacks(self, event: str, *args, **kwargs):
        """Notify registered callbacks of loading events."""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(*args, **kwargs)

    def _configure_memory_settings(self):
        """Configure GPU memory settings for optimal performance."""
        for device_id in self.device_ids:
            torch.cuda.set_per_process_memory_fraction(0.95, device_id)
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.reset_peak_memory_stats(device_id)

    def _prepare_gpu_environment(self):
        """Prepare GPU environment before model loading."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _get_shard_files(self):
        """Get sorted list of model shard files."""
        return sorted([
            f for f in os.listdir(self.model_dir) 
            if f.startswith("model-") and f.endswith(".safetensors")
        ])

    def _convert_to_meta(self, model: nn.Module):
        """Convert model parameters to meta tensors for efficient loading."""
        for name, param in model.named_parameters():
            if not param.is_meta:
                meta_param = nn.Parameter(
                    torch.empty_like(param, device='meta'),
                    requires_grad=param.requires_grad
                )
                parent_module = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name_parts[-1], meta_param)

    def _move_to_device(self, module: nn.Module, device: str):
        """Move a module to specified device with initialization."""
        for name, param in module.named_parameters(recurse=False):
            if param.is_meta:
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad
                )
                setattr(module, name, new_param)
        module.to(device)

    def _cleanup_memory(self):
        """Perform memory cleanup and optimization."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if hasattr(torch.cuda, 'memory_stats'):
            for device_id in self.device_ids:
                torch.cuda.reset_peak_memory_stats(device_id)

    def _ensure_model_integrity(self, model: nn.Module):
        """Verify and ensure model parameter initialization."""
        meta_params_found = 0
        
        try:
            for name, param in model.named_parameters():
                if param.is_meta:
                    meta_params_found += 1
                    parent = model
                    parts = name.split('.')
                    
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    
                    new_param = nn.Parameter(
                        torch.empty(param.shape, 
                                  dtype=param.dtype,
                                  device=self.primary_device),
                        requires_grad=param.requires_grad
                    )
                    setattr(parent, parts[-1], new_param)
                    
                    if self.memory_efficient and meta_params_found % 1000 == 0:
                        torch.cuda.empty_cache()
                        
        except Exception as e:
            raise RuntimeError(f"Error converting meta parameters: {str(e)}")

    def _process_weight_batch(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """Process and update model weights with FP8 quantization."""
        try:
            processed_weights = self.fp8_handler.process_weights(weights)
            
            for name, tensor in processed_weights.items():
                try:
                    model_key = f"model.{name}" if not name.startswith("model.") else name
                    param = model
                    key_parts = model_key.split('.')
                    
                    for part in key_parts[:-1]:
                        if not hasattr(param, part):
                            continue
                        param = getattr(param, part)
                    
                    if hasattr(param, key_parts[-1]):
                        param_tensor = getattr(param, key_parts[-1])
                        if isinstance(param_tensor, nn.Parameter):
                            new_param = nn.Parameter(
                                tensor.to(device=self.primary_device),
                                requires_grad=param_tensor.requires_grad
                            )
                            setattr(param, key_parts[-1], new_param)
                
                except Exception:
                    continue
            
        except Exception as e:
            raise RuntimeError(f"Error processing weight batch: {str(e)}")

    def _distribute_layers(self, model):
        """Distribute model layers across available GPUs."""
        if len(self.device_ids) <= 1:
            return

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            
            # Calculate memory-based layer distribution
            memory_per_gpu = {}
            total_memory = 0
            for device in self.device_ids:
                free_memory = torch.cuda.get_device_properties(device).total_memory - \
                             torch.cuda.memory_allocated(device)
                memory_per_gpu[device] = free_memory
                total_memory += free_memory
            
            layers_per_gpu = {
                device: max(1, int((memory / total_memory) * num_layers))
                for device, memory in memory_per_gpu.items()
            }
            
            # Adjust distribution for remaining layers
            remaining = num_layers - sum(layers_per_gpu.values())
            if remaining > 0:
                sorted_devices = sorted(
                    self.device_ids,
                    key=lambda d: memory_per_gpu[d],
                    reverse=True
                )
                for i in range(remaining):
                    layers_per_gpu[sorted_devices[i % len(sorted_devices)]] += 1
            
            # Move layers to devices
            current_layer = 0
            for device, num_layers in layers_per_gpu.items():
                device_str = f"cuda:{device}"
                for _ in range(num_layers):
                    if current_layer < len(model.model.layers):
                        layer = model.model.layers[current_layer]
                        self._move_to_device(layer, device_str)
                        current_layer += 1

    def load_model(self) -> nn.Module:
        """
        Load and initialize the model with specified configuration.
        
        Returns:
            nn.Module: Loaded and initialized model
        """
        try:
            self._prepare_gpu_environment()
            
            # Initialize model configuration
            config = AutoConfig.from_pretrained(self.model_dir)
            config.use_fp8 = True
            if hasattr(self.fp8_handler, 'fp8_format'):
                config.fp8_e4m3 = self.fp8_handler.fp8_format.e4m3
            
            if self.mixed_precision:
                config.use_cache = False
            
            # Create and initialize model
            model = AutoModelForCausalLM.from_config(config)
            self._convert_to_meta(model)
            
            return self._load_model_async(model) if self.use_async else self._load_model_sync(model)
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
        finally:
            if self.use_async and self.async_loader:
                self.async_loader.stop()
            if self.memory_efficient:
                self._cleanup_memory()

    def _load_model_async(self, model: nn.Module) -> nn.Module:
        """Load model using asynchronous shard loading."""
        shard_files = self._get_shard_files()
        
        # Create shard mapping
        shard_mapping = {}
        for shard_file in shard_files:
            file_path = os.path.join(self.model_dir, shard_file)
            self._notify_callbacks('on_shard_load', shard_file)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                shard_weights = {name: f.get_tensor(name) for name in f.keys()}
                shard_mapping[shard_file] = shard_weights
        
        # Initialize async loading
        self.async_loader.start_prefetch(
            shard_mapping=shard_mapping,
            fp8_handler=self.fp8_handler,
            device=self.primary_device
        )
        
        # Process shards
        while True:
            shard = self.async_loader.get_next_shard()
            if shard is None:
                break
                
            shard_id, processed_weights = shard
            self._notify_callbacks('on_shard_process', shard_id)
            self._process_weight_batch(model, processed_weights)
            
            if self.memory_efficient:
                self._cleanup_memory()
            
            self.async_loader.advance()
        
        self._ensure_model_integrity(model)
        self._distribute_layers(model)
        
        return model

    def _load_model_sync(self, model: nn.Module) -> nn.Module:
        """Load model synchronously."""
        try:
            shard_files = self._get_shard_files()
            
            # Process each shard
            for shard_file in shard_files:
                file_path = os.path.join(self.model_dir, shard_file)
                self._notify_callbacks('on_shard_load', shard_file)
                
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    weights = {name: f.get_tensor(name) for name in f.keys()}
                
                self._notify_callbacks('on_shard_process', shard_file)
                self._process_weight_batch(model, weights)
                
                if self.memory_efficient:
                    self._cleanup_memory()
            
            self._ensure_model_integrity(model)
            self._distribute_layers(model)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error in synchronous loading: {str(e)}")

    def save_model(self, model: nn.Module, output_dir: str) -> None:
        """
        Save the processed model to disk.
        
        Args:
            model: Model to save
            output_dir: Output directory path
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save configuration
            if hasattr(model, 'config'):
                model.config.save_pretrained(output_dir)
            
            # Save weights in shards
            shard_size = self.shard_size_bytes
            current_shard = {}
            current_size = 0
            shard_idx = 0
            
            for name, param in model.named_parameters():
                if param.is_meta:
                    continue
                
                param_size = param.nelement() * param.element_size()
                
                if current_size + param_size > shard_size and current_shard:
                    shard_path = os.path.join(
                        output_dir,
                        f"model-{str(shard_idx + 1).zfill(5)}-of-{str(len(self.device_ids)).zfill(5)}.safetensors"
                    )
                    save_file(current_shard, shard_path)
                    
                    current_shard = {}
                    current_size = 0
                    shard_idx += 1
                
                current_shard[name] = param.detach().cpu()
                current_size += param_size
            
            # Save final shard
            if current_shard:
                shard_path = os.path.join(
                    output_dir,
                    f"model-{str(shard_idx + 1).zfill(5)}-of-{str(len(self.device_ids)).zfill(5)}.safetensors"
                )
                save_file(current_shard, shard_path)
            
        except Exception as e:
            raise RuntimeError(f"Error saving model: {str(e)}")
            
    def _optimize_memory_layout(self, model: nn.Module) -> None:
        """
        Optimize model memory layout for improved performance.
        
        Args:
            model: Model to optimize
        """
        try:
            # Make tensors contiguous for better memory access
            for param in model.parameters():
                if param.is_contiguous():
                    continue
                param.data = param.data.contiguous()
            
            # Pin memory for CPU->GPU transfers if memory efficient mode is enabled
            if self.memory_efficient:
                for param in model.parameters():
                    if not param.is_cuda:
                        param.pin_memory()
                        
        except Exception as e:
            raise RuntimeError(f"Error optimizing memory layout: {str(e)}")

    def _verify_loaded_model(self, model: nn.Module) -> bool:
        """
        Verify model loading integrity.
        
        Args:
            model: Model to verify
            
        Returns:
            bool: True if verification passes, False otherwise
        """
        try:
            # Check for remaining meta tensors
            meta_tensors = [
                name for name, param in model.named_parameters()
                if param.is_meta
            ]
            if meta_tensors:
                return False
            
            # Verify parameter devices
            misplaced_params = [
                name for name, param in model.named_parameters()
                if not param.is_cuda and not param.is_meta
            ]
            if misplaced_params:
                return False
            
            # Verify FP8 conversion
            if hasattr(self.fp8_handler, 'fp8_format'):
                target_dtype = torch.float8_e4m3fn if self.fp8_handler.fp8_format.e4m3 else torch.float8_e5m2
                incorrect_dtype = [
                    name for name, param in model.named_parameters()
                    if param.dtype != target_dtype and not param.is_meta
                ]
                if incorrect_dtype:
                    return False
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Error verifying model: {str(e)}")

    def _convert_layers_to_fp8(self, model: nn.Module) -> None:
        """
        Convert model layers to FP8 format.
        
        Args:
            model: Model to convert
        """
        try:
            for name, module in model.named_modules():
                # Skip non-parameter modules
                if not any(isinstance(p, nn.Parameter) for p in module.parameters(recurse=False)):
                    continue
                
                # Convert parameters
                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        processed = self.fp8_handler.process_weights({f"{name}.{param_name}": param})
                        new_param = nn.Parameter(
                            processed[f"{name}.{param_name}"],
                            requires_grad=param.requires_grad
                        )
                        setattr(module, param_name, new_param)
                
                # Convert buffers
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    processed = self.fp8_handler.process_weights({f"{name}.{buffer_name}": buffer})
                    module.register_buffer(buffer_name, processed[f"{name}.{buffer_name}"])
                    
        except Exception as e:
            raise RuntimeError(f"Error converting layers to FP8: {str(e)}")