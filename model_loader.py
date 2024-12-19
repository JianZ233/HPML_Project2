import torch
import gc
import os
import time
from torch import nn
from typing import Dict, List, Optional
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file
from fp8_format import FP8Format
from quantization import FP8QuantizationHandler
from gpu_utils import GPUMemoryTracker
from shard_loader import AsyncShardLoader

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
            model_dir: Directory containing model files.
            device_ids: List of GPU device IDs to use.
            use_ddp: Whether to use DistributedDataParallel.
            memory_efficient: Whether to use memory-efficient loading.
            mixed_precision: Whether to use mixed precision.
            shard_size_gb: Size of each shard in gigabytes.
            fp8_format: FP8 format configuration.
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
        self.primary_device = f"cuda:{self.device_ids[0]}"

    def load_model(self) -> nn.Module:
        """
        Load and initialize the model from safetensor shards.
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            config = AutoConfig.from_pretrained(self.model_dir)
            config.use_fp8 = True
            config.fp8_e4m3 = self.fp8_handler.fp8_format.e4m3 if self.fp8_handler.fp8_format else True
            
            # Initialize model on CPU with meta tensors
            model = AutoModelForCausalLM.from_config(config)
            self._convert_to_meta(model)
            
            shard_files = sorted([
                f for f in os.listdir(self.model_dir) 
                if f.startswith("model-") and f.endswith(".safetensors")
            ])
            
            print(f"Found {len(shard_files)} shards: {shard_files}")
            
            start_time = time.time()
            for shard_file in shard_files:
                shard_path = os.path.join(self.model_dir, shard_file)
                print(f"\nLoading shard: {shard_path}")
                
                # Load shard weights
                shard_weights = load_file(shard_path)
                self._process_shard(model, shard_weights)
                
                del shard_weights
                if self.memory_efficient:
                    gc.collect()
                    torch.cuda.empty_cache()

            # Ensure no meta parameters remain before distributing layers
            self._ensure_no_meta_left(model, self.primary_device)

            # Distribute model across GPUs
            self._distribute_layers(model)
            
            end_time = time.time()
            print(f"\nModel loaded in {end_time - start_time:.2f} seconds")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        finally:
            self.async_loader.stop()

    def _convert_to_meta(self, model: nn.Module):
        """
        Convert model parameters to meta tensors.
        """
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

    def _ensure_no_meta_left(self, model: nn.Module, device: str):
        """
        Ensure no parameters remain on meta device.
        Replace meta parameters with empty parameters on the target device.
        """
        for full_name, param in list(model.named_parameters()):
            if param.is_meta:
                real_param = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad
                )
                parent = model
                parts = full_name.split('.')
                for attr in parts[:-1]:
                    parent = getattr(parent, attr)
                setattr(parent, parts[-1], real_param)

    def _process_shard(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """
        Process each shard: load weights, apply scales in FP32, then convert to FP8 if needed.
        """
        print(f"\nProcessing {len(weights)} weights...")
        
        unexpected_keys = []
        missing_keys = []
        device = f"cuda:{self.device_ids[0]}"
    
        for key, tensor in weights.items():
            # If the key ends with _scale and is not a dict that includes a 'weight', skip it
            # because it's a standalone scale entry with no direct parameter.
            if (key.endswith("input_scale") or key.endswith("weight_scale")) and not (isinstance(tensor, dict) and 'weight' in tensor):
                # This is just a scale key with no direct parameter. We skip it.
                continue
    
            try:
                # Ensure model keys have "model." prefix
                model_key = f"model.{key}" if not key.startswith("model.") else key
    
                # Navigate model structure to find the parameter
                param = model
                key_parts = model_key.split('.')
                for part in key_parts[:-1]:
                    if not hasattr(param, part):
                        missing_keys.append(model_key)
                        param = None
                        break
                    param = getattr(param, part)
    
                if param is None:
                    continue
    
                if not hasattr(param, key_parts[-1]):
                    unexpected_keys.append(key)
                    continue
    
                param = getattr(param, key_parts[-1])
                if not isinstance(param, nn.Parameter):
                    # Not a parameter, skip
                    continue
    
                # Process weights in FP32 first on CPU
                if isinstance(tensor, dict):
                    weight = tensor.get('weight', None)
                    if weight is None:
                        # No actual weight tensor found
                        continue
    
                    # Move to CPU and float32 for scaling
                    weight = weight.float().cpu()
    
                    # Apply scales in FP32 on CPU
                    if 'weight_scale' in tensor:
                        weight_scale = tensor['weight_scale'].float().cpu()
                        weight = weight * weight_scale
                    if 'input_scale' in tensor:
                        input_scale = tensor['input_scale'].float().cpu()
                        weight = weight * input_scale
    
                    # Move scaled weight to GPU as FP32
                    weight = weight.to(device=device, dtype=torch.float32)
                    processed_tensor = weight
                else:
                    # Plain tensor, just ensure FP32 on GPU
                    processed_tensor = tensor.to(device=device, dtype=torch.float32)
    
                # Convert to FP8 after all arithmetic is done
                if self.fp8_handler and 'weight' in key:
                    processed_tensor = self.fp8_handler._apply_fp8_constraints(processed_tensor)
                    if hasattr(torch, 'float8_e4m3fn'):
                        fp8_dtype = (torch.float8_e4m3fn 
                                     if self.fp8_handler.fp8_format.e4m3 else torch.float8_e5m2)
                        processed_tensor = processed_tensor.to(fp8_dtype)
    
                # Replace parameter with the new FP8 parameter (or FP16/FP32 if FP8 not available)
                parent = model
                for part in key_parts[:-1]:
                    parent = getattr(parent, part)
                new_param = nn.Parameter(processed_tensor, requires_grad=param.requires_grad)
                setattr(parent, key_parts[-1], new_param)
    
            except Exception as e:
                print(f"Error processing tensor for {key}: {e}")
                continue
    
        if unexpected_keys:
            print(f"\nUnexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"\nMissing keys: {missing_keys}")

    def _distribute_layers(self, model: nn.Module):
        """
        Distribute model layers across available GPUs.
        """
        if len(self.device_ids) <= 1:
            return

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            layers_per_gpu = max(1, num_layers // len(self.device_ids))
            
            # Move embeddings to first GPU
            if hasattr(model.model, 'embed_tokens'):
                device = f"cuda:{self.device_ids[0]}"
                if model.model.embed_tokens.weight.is_meta:
                    w = nn.Parameter(
                        torch.empty(model.model.embed_tokens.weight.shape,
                                    dtype=model.model.embed_tokens.weight.dtype, 
                                    device=device),
                        requires_grad=model.model.embed_tokens.weight.requires_grad
                    )
                    model.model.embed_tokens.weight = w
                model.model.embed_tokens = model.model.embed_tokens.to(device)
            
            # Distribute layers
            for i in range(num_layers):
                device_idx = min(i // layers_per_gpu, len(self.device_ids) - 1)
                device = f"cuda:{self.device_ids[device_idx]}"
                
                layer = model.model.layers[i]
                # Ensure no meta params remain
                for name, param in layer.named_parameters(recurse=False):
                    if param.is_meta:
                        new_param = nn.Parameter(
                            torch.empty(param.shape, dtype=param.dtype, device=device),
                            requires_grad=param.requires_grad
                        )
                        setattr(layer, name, new_param)

                layer.to(device)
            
            # Move head to first GPU
            if hasattr(model, 'lm_head'):
                device = f"cuda:{self.device_ids[0]}"
                for name, param in model.lm_head.named_parameters(recurse=False):
                    if param.is_meta:
                        new_param = nn.Parameter(
                            torch.empty(param.shape, dtype=param.dtype, device=device),
                            requires_grad=param.requires_grad
                        )
                        setattr(model.lm_head, name, new_param)
                model.lm_head = model.lm_head.to(device)
            
            # Move norm to first GPU if it exists
            if hasattr(model.model, 'norm'):
                device = f"cuda:{self.device_ids[0]}"
                for name, param in model.model.norm.named_parameters(recurse=False):
                    if param.is_meta:
                        new_param = nn.Parameter(
                            torch.empty(param.shape, dtype=param.dtype, device=device),
                            requires_grad=param.requires_grad
                        )
                        setattr(model.model.norm, name, new_param)
                model.model.norm = model.model.norm.to(device)
            
            if self.memory_efficient:
                gc.collect()
                torch.cuda.empty_cache()

    def _create_shard_mapping(self, weights: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Create mapping of weights to shards.
        """
        shard_mapping = {}
        current_shard = {}
        current_size = 0
        shard_id = 0

        # Sort weights by size
        sorted_weights = sorted(
            weights.items(),
            key=lambda x: (x[1]['weight'] if isinstance(x[1], dict) else x[1]).numel(),
            reverse=True
        )

        for name, tensor in sorted_weights:
            tensor_size = (tensor['weight'] if isinstance(tensor, dict) else tensor).numel() * 4  # assume FP32 size

            if current_size + tensor_size > self.shard_size_bytes and current_shard:
                shard_mapping[shard_id] = current_shard
                current_shard = {}
                current_size = 0
                shard_id += 1

            current_shard[name] = tensor
            current_size += tensor_size

        if current_shard:
            shard_mapping[shard_id] = current_shard

        return shard_mapping
