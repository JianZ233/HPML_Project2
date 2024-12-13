import torch

def basic_model_loader(file_path):
    """
    Basic loader for standard PyTorch models.
    """
    if file_path.endswith(".pt"):
        model = torch.load(file_path)
    elif file_path.endswith(".npy"):
        import numpy as np
        data = np.load(file_path)
        model = torch.tensor(data, dtype=torch.float32)
    else:
        raise ValueError("Unsupported file format!")
    return model

model = basic_model_loader("model.pth")
print("Loaded model:", model)

def _is_sm89_or_later():
    """
    Check if the current GPU supports SM89+ architecture required for FP8.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)

def load_fp8_model(file_path):
    """
    Load an FP8 model, ensuring compatibility with hardware and proper byte alignment.
    """
    if not _is_sm89_or_later():
        raise RuntimeError("FP8 requires SM89+ GPU architecture (e.g., H100, L40).")

    try:
        from torchao.float8 import CastConfig, Float8LinearConfig, ScalingType
    except ImportError as e:
        raise ImportError(
            "torchao is not installed. Please install it to use FP8 model handling."
        ) from e

    # Lazy loading: load the model checkpoint on CPU first
    checkpoint = torch.load(file_path, map_location='cpu')

    # Ensure weights are properly aligned for FP8
    aligned_weights = align_weights_for_fp8(checkpoint['weights'])
    print("Model successfully loaded and aligned for FP8.")
    return aligned_weights

def align_weights_for_fp8(weights):
    """
    Align model weights for FP8 format to ensure byte-level compatibility,
    adding necessary padding for proper memory alignment.
    """
    aligned_weights = {}
    alignment_requirement = 16  # Assume 16-byte alignment for optimal GPU performance

    for key, tensor in weights.items():
        if tensor.dtype == torch.float8:
            # Calculate padding if tensor is not aligned
            size_in_bytes = tensor.numel() * tensor.element_size()
            padding_needed = (alignment_requirement - (size_in_bytes % alignment_requirement)) % alignment_requirement

            if padding_needed > 0:
                # Create a padded tensor
                padded_size = tensor.numel() + (padding_needed // tensor.element_size())
                padded_tensor = torch.zeros(padded_size, dtype=torch.float8, device=tensor.device)
                padded_tensor[:tensor.numel()] = tensor
                aligned_weights[key] = padded_tensor
            else:
                aligned_weights[key] = tensor
        else:
            aligned_weights[key] = tensor  # Non-FP8 tensors are left unchanged

    return aligned_weights

# Example usage
if __name__ == "__main__":
    model_path = "path_to_fp8_model.pt"
    try:
        fp8_model = load_fp8_model(model_path)
        print("FP8 model loaded successfully:", fp8_model)
    except Exception as e:
        print(f"Failed to load FP8 model: {e}")


