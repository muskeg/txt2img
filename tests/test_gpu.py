import pytest

from txt2img.config import default_models_dir
from txt2img.pipeline import ImageGenerator


@pytest.mark.gpu
def test_sdxl_generates_on_gpu_without_vram_overflow() -> None:
    """Downloads ~7 GB of weights on first run."""
    generator = ImageGenerator("sdxl", models_dir=default_models_dir())
    generation = generator.generate("a red apple on a wooden table", seed=0, steps=10)

    assert generation.images[0].image.size == (1024, 1024)
    assert generation.memory.gpu_percent == 100
    assert generation.memory.overflow_bytes == 0
