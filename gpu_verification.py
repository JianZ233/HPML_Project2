def verify_gpu_availability(device_ids):
    """
    Verify GPU availability and CUDA capability for specified devices.
    
    Args:
        device_ids: List of GPU device IDs to verify
        
    Returns:
        tuple: (available_devices, warning_message)
        - available_devices: List of verified available GPU devices
        - warning_message: String with any warnings or None if no warnings
    """
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation and GPU drivers.")
    
    total_devices = torch.cuda.device_count()
    if total_devices == 0:
        raise RuntimeError("No CUDA devices found despite CUDA being available.")
        
    warning_msg = None
    available_devices = []
    
    for device_id in device_ids:
        if device_id >= total_devices:
            if warning_msg is None:
                warning_msg = f"Warning: Some requested GPU devices are not available. "
                warning_msg += f"Only {total_devices} GPU(s) found. Using available devices: "
            continue
            
        try:
            # Test device properties
            props = torch.cuda.get_device_properties(device_id)
            if props.major < 7:  # Minimum compute capability for FP8
                if warning_msg is None:
                    warning_msg = "Warning: Some GPUs may not support FP8. "
                    warning_msg += "Using available devices: "
                continue
                
            # Test basic CUDA operations
            torch.cuda.set_device(device_id)
            test_tensor = torch.zeros((1,), device=f"cuda:{device_id}")
            del test_tensor
            torch.cuda.empty_cache()
            
            available_devices.append(device_id)
            
        except Exception as e:
            if warning_msg is None:
                warning_msg = f"Warning: Error accessing some GPU devices ({str(e)}). "
                warning_msg += "Using available devices: "
            continue
    
    if not available_devices:
        raise RuntimeError("No usable GPU devices found among the requested devices.")
        
    if warning_msg:
        warning_msg += f"{available_devices}"
        
    return available_devices, warning_msg