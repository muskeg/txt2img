import os
import warnings

# Without this, offloaded models fragment VRAM and spill past the limit (33.4 vs 30.2 GiB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# diffusers' AutoPipeline imports transformers' VITS, which uses torch.jit.script at import time.
warnings.filterwarnings(
    "ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning
)
