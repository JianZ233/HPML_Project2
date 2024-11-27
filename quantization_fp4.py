def apply_fp4_quantization(model):
    """Apply FP4 quantization to the model."""
    print("Applying FP4 quantization...")
    for name, param in model.named_parameters():
        param.data = _to_fp4(param.data)
    print("FP4 quantization applied.")


def _to_fp4(tensor):
    """Convert tensor to FP4 format."""
    #https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/quantization
    # Placeholder: Simulate FP4 with a scaled-down representation
    scale = torch.max(torch.abs(tensor)) / 15.0  # FP4 range [-15, 15]
    fp4_tensor = torch.clamp(torch.round(tensor / scale), -15, 15).to(torch.int8)
    return fp4_tensor
