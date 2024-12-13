# Initialize loader with async loading
loader = ShardedFP8ModelLoader(
    model_cls=YourModel,
    device_ids=[0, 1, 2, 3],
    use_ddp=True,
    memory_efficient=True,
    mixed_precision=True,
    shard_size_gb=4.0,
    num_prefetch_shards=2
)

# Load model
model = loader.load_model("path/to/checkpoint.pt")