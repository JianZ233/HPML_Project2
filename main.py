import argparse
from .model_loader import ShardedFP8ModelLoader

def main():
    parser = argparse.ArgumentParser(description="Sharded FP8 Model Loader")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help="List of GPU device IDs to use")
    parser.add_argument("--use_ddp", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--memory_efficient", action="store_true", help="Enable memory-efficient mode")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision mode")
    parser.add_argument("--shard_size_gb", type=float, default=4.0, help="Shard size in gigabytes")

    args = parser.parse_args()

    # Initialize the model loader
    loader = ShardedFP8ModelLoader(
        model_dir=args.model_dir,
        device_ids=args.device_ids,
        use_ddp=args.use_ddp,
        memory_efficient=args.memory_efficient,
        mixed_precision=args.mixed_precision,
        shard_size_gb=args.shard_size_gb,
    )

    # Load the model
    model = loader.load_model(args.checkpoint_path)
    print("Model loaded successfully.")

if __name__ == "__main__":
    main()
