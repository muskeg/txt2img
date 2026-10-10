import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

GiB = 2**30

# (width, height) presets from the Qwen-Image-2.1 model card.
QWEN_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
# ~1 megapixel buckets (multiples of 64) for SDXL, SD 3.5, FLUX.1 and Z-Image.
MEGAPIXEL_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}
ASPECTS = list(QWEN_RESOLUTIONS)
DEFAULT_ASPECT = "1:1"


@dataclass(frozen=True)
class ModelPreset:
    repo_id: str
    license: str
    resolutions: dict[str, tuple[int, int]]
    # None: the checkpoint ships its own sampling schedule, so steps can't be changed.
    default_steps: int | None
    # Big components that --quantize applies to; the VAE always stays in bf16.
    quantizable: tuple[str, ...]
    call_kwargs: dict[str, object] = field(default_factory=dict)
    variant: str | None = None
    gated: bool = False
    # Native RGBA output, prompted with TRANSPARENT_TEMPLATE.
    transparency: bool = False
    # Accepts reference images (`image=`), which animation frames are edited from.
    editing: bool = False


# Recommended prompt format from the Qwen-Image-2.1 README.
TRANSPARENT_TEMPLATE = (
    "This is an RGBA image with transparency. {prompt}. "
    "The image has alpha channel and the background is transparent."
)

# Steps and guidance come from each model card, or the pipeline defaults when the card has none.
MODELS: dict[str, ModelPreset] = {
    "z-image-turbo": ModelPreset(
        "Tongyi-MAI/Z-Image-Turbo",
        "Apache-2.0",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=9,
        quantizable=("transformer", "text_encoder"),
        call_kwargs={"guidance_scale": 0.0},
    ),
    "z-image": ModelPreset(
        "Tongyi-MAI/Z-Image",
        "Apache-2.0",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=50,
        quantizable=("transformer", "text_encoder"),
        call_kwargs={"guidance_scale": 4.0},
    ),
    "qwen-image-2.1": ModelPreset(
        "Qwen/Qwen-Image-2.1",
        "Qwen Research License",
        QWEN_RESOLUTIONS,
        default_steps=40,
        quantizable=("transformer", "text_encoder"),
        transparency=True,
        editing=True,
    ),
    "qwen-image-2.1-turbo": ModelPreset(
        "Qwen/Qwen-Image-2.1-Turbo",
        "Qwen Research License",
        QWEN_RESOLUTIONS,
        default_steps=None,
        quantizable=("transformer", "text_encoder"),
        transparency=True,
        editing=True,
    ),
    "flux.1-schnell": ModelPreset(
        "black-forest-labs/FLUX.1-schnell",
        "Apache-2.0",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=4,
        quantizable=("transformer", "text_encoder_2"),
        call_kwargs={"guidance_scale": 0.0, "max_sequence_length": 256},
        gated=True,
    ),
    "sd3.5-large": ModelPreset(
        "stabilityai/stable-diffusion-3.5-large",
        "Stability Community License",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=28,
        quantizable=("transformer", "text_encoder_3"),
        call_kwargs={"guidance_scale": 3.5},
        gated=True,
    ),
    "sd3.5-large-turbo": ModelPreset(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        "Stability Community License",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=4,
        quantizable=("transformer", "text_encoder_3"),
        call_kwargs={"guidance_scale": 0.0},
        gated=True,
    ),
    "sd3.5-medium": ModelPreset(
        "stabilityai/stable-diffusion-3.5-medium",
        "Stability Community License",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=40,
        quantizable=("transformer", "text_encoder_3"),
        call_kwargs={"guidance_scale": 4.5},
        gated=True,
    ),
    "sdxl": ModelPreset(
        "stabilityai/stable-diffusion-xl-base-1.0",
        "CreativeML Open RAIL++-M",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=50,
        quantizable=("unet",),
        call_kwargs={"guidance_scale": 5.0},
        variant="fp16",
    ),
    "realvisxl-v5": ModelPreset(
        "SG161222/RealVisXL_V5.0",
        "CreativeML Open RAIL++-M",
        MEGAPIXEL_RESOLUTIONS,
        default_steps=50,
        quantizable=("unet",),
        call_kwargs={"guidance_scale": 5.0},
        variant="fp16",
    ),
}
DEFAULT_MODEL = "z-image-turbo"
DEFAULT_EDIT_MODEL = "qwen-image-2.1-turbo"


class Placement(StrEnum):
    AUTO = "auto"
    GPU = "gpu"
    OFFLOAD = "offload"


class Quantization(StrEnum):
    NONE = "none"
    NF4 = "4bit"


# Room left for activations and the CUDA context when auto placement checks whether weights fit.
VRAM_HEADROOM = 4 * GiB

MODELS_DIR_ENV = "TXT2IMG_MODELS_DIR"


def default_models_dir() -> Path | None:
    """Model weights location; None means the Hugging Face default cache."""
    value = os.environ.get(MODELS_DIR_ENV)
    return Path(value).expanduser() if value else None


OUTPUT_DIR = Path("outputs")
