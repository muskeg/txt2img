import os

# Without this, offloaded models fragment VRAM and spill past the limit (33.4 vs 30.2 GiB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
