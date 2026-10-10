from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from txt2img import pipeline
from txt2img.config import GiB, Placement, Quantization
from txt2img.pipeline import (
    GeneratedImage,
    GPUUnavailableError,
    ImageGenerator,
    MemoryReport,
    ModelAccessError,
    save_image,
)


def from_pretrained() -> MagicMock:
    return pipeline.DiffusionPipeline.from_pretrained


def test_loads_preset_in_bf16_into_models_dir(fake_pipe: MagicMock, tmp_path: Path) -> None:
    ImageGenerator("sdxl", models_dir=tmp_path)

    from_pretrained().assert_called_once_with(
        "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.bfloat16,
        cache_dir=tmp_path,
        variant="fp16",
        quantization_config=None,
    )


def test_auto_placement_keeps_everything_on_gpu_when_it_fits(fake_pipe: MagicMock) -> None:
    generator = ImageGenerator("z-image-turbo")

    assert generator.placement == Placement.GPU
    fake_pipe.to.assert_called_once_with("cuda")
    fake_pipe.enable_model_cpu_offload.assert_not_called()


def test_auto_placement_offloads_when_weights_do_not_fit(
    fake_pipe: MagicMock, fake_cuda: SimpleNamespace
) -> None:
    fake_cuda.total = fake_cuda.free = 4 * GiB  # no room left after the headroom

    generator = ImageGenerator("z-image-turbo")

    assert generator.placement == Placement.OFFLOAD
    assert generator.requested_placement == Placement.AUTO
    fake_pipe.enable_model_cpu_offload.assert_called_once()
    fake_pipe.to.assert_not_called()


def test_forced_gpu_placement_is_respected_even_if_it_does_not_fit(
    fake_pipe: MagicMock, fake_cuda: SimpleNamespace
) -> None:
    fake_cuda.total = fake_cuda.free = 4 * GiB

    generator = ImageGenerator("z-image-turbo", placement=Placement.GPU)

    assert generator.placement == Placement.GPU
    fake_pipe.to.assert_called_once_with("cuda")


def test_forced_offload(fake_pipe: MagicMock) -> None:
    generator = ImageGenerator("sdxl", placement=Placement.OFFLOAD)

    assert generator.placement == Placement.OFFLOAD
    fake_pipe.enable_model_cpu_offload.assert_called_once()


def test_4bit_quantizes_the_preset_components(fake_pipe: MagicMock) -> None:
    ImageGenerator("flux.1-schnell", quantization=Quantization.NF4)

    config = from_pretrained().call_args.kwargs["quantization_config"]
    assert config.quant_backend == "bitsandbytes_4bit"
    assert config.components_to_quantize == ["transformer", "text_encoder_2"]


def test_gated_model_raises_access_error(fake_pipe: MagicMock) -> None:
    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
    from_pretrained().side_effect = GatedRepoError("gated", response=response)

    with pytest.raises(ModelAccessError, match="hf auth login"):
        ImageGenerator("sd3.5-large")


def test_unknown_model_is_rejected(fake_pipe: MagicMock) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        ImageGenerator("nope")


def test_missing_gpu_raises(fake_pipe: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)

    with pytest.raises(GPUUnavailableError):
        ImageGenerator("sdxl")


def test_load_report_measures_where_weights_actually_are(fake_pipe: MagicMock) -> None:
    report = ImageGenerator("sdxl").load_memory

    # The fake modules never leave the CPU, and the report must reflect that.
    fp32_linear_bytes = (64 * 64 + 64) * 4 + (8 * 8 + 8) * 4
    assert (report.gpu_bytes, report.cpu_bytes) == (0, fp32_linear_bytes)
    assert report.vram_other == 2 * GiB


def test_uses_preset_resolution_steps_and_call_kwargs(fake_pipe: MagicMock) -> None:
    [result] = ImageGenerator("flux.1-schnell").generate("a cat", aspect="16:9", seed=7).images

    kwargs = fake_pipe.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (1344, 768)
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["guidance_scale"] == 0.0
    assert kwargs["max_sequence_length"] == 256
    assert (result.seed, result.steps, result.model) == (7, 4, "flux.1-schnell")


def test_qwen_uses_its_own_resolutions(fake_pipe: MagicMock) -> None:
    ImageGenerator("qwen-image-2.1").generate("a cat", aspect="16:9")

    kwargs = fake_pipe.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (2752, 1536)
    assert kwargs["num_inference_steps"] == 40


def test_custom_steps(fake_pipe: MagicMock) -> None:
    ImageGenerator("sdxl").generate("a cat", steps=12)

    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 12


def test_qwen_turbo_keeps_checkpoint_schedule(fake_pipe: MagicMock) -> None:
    [result] = ImageGenerator("qwen-image-2.1-turbo").generate("a cat").images

    assert "num_inference_steps" not in fake_pipe.call_args.kwargs
    assert "sigmas" not in fake_pipe.call_args.kwargs
    assert result.steps is None


def test_qwen_turbo_rejects_steps(fake_pipe: MagicMock) -> None:
    with pytest.raises(ValueError, match="schedule stored in its checkpoint"):
        ImageGenerator("qwen-image-2.1-turbo").generate("a cat", steps=20)


def test_each_image_gets_consecutive_reproducible_seed(fake_pipe: MagicMock) -> None:
    images = ImageGenerator("sdxl").generate("a cat", seed=100, count=3).images

    assert [r.seed for r in images] == [100, 101, 102]
    used_seeds = [c.kwargs["generator"].initial_seed() for c in fake_pipe.call_args_list]
    assert used_seeds == [100, 101, 102]


def test_random_seed_when_none_given(fake_pipe: MagicMock) -> None:
    [result] = ImageGenerator("sdxl").generate("a cat").images

    assert 0 <= result.seed < 2**32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"aspect": "5:4"}, "Unknown aspect"), ({"count": 0}, "count"), ({"steps": 0}, "steps")],
)
def test_invalid_generate_arguments(fake_pipe: MagicMock, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ImageGenerator("sdxl").generate("a cat", **kwargs)


def test_generation_reports_peak_vram_and_overflow(
    fake_pipe: MagicMock, fake_cuda: SimpleNamespace
) -> None:
    generator = ImageGenerator("sdxl")
    fake_cuda.peak = 31 * GiB  # plus 2 GiB used by other apps > 32 GiB

    memory = generator.generate("a cat").memory

    assert memory.vram_used == 31 * GiB
    assert memory.overflow_bytes == 1 * GiB
    assert "WARNING: VRAM limit exceeded by 1.0 GiB" in memory.summary()


def test_memory_report_summary() -> None:
    report = MemoryReport(
        gpu_bytes=3 * GiB,
        cpu_bytes=1 * GiB,
        offloaded=False,
        vram_used=6 * GiB,
        vram_other=2 * GiB,
        vram_total=32 * GiB,
    )

    assert report.gpu_percent == 75
    assert report.overflow_bytes == 0
    assert report.summary() == (
        "Weights: 3.0 GiB in VRAM (75%), 1.0 GiB in RAM (25%)\n"
        "VRAM: 8.0 GiB of 32.0 GiB (25%), including 2.0 GiB used by other apps"
    )


def test_offloaded_report_explains_placement() -> None:
    report = MemoryReport(0, 20 * GiB, True, 14 * GiB, 2 * GiB, 32 * GiB)

    assert "0.0 GiB in VRAM (0%), 20.0 GiB in RAM (100%), offloaded" in report.summary()


def test_save_image_writes_png_with_metadata(tmp_path: Path) -> None:
    result = GeneratedImage(Image.new("RGB", (8, 8)), "a cat", "sdxl", 42, 50, Quantization.NF4)

    path = save_image(result, tmp_path / "out")

    assert path.parent == tmp_path / "out"
    assert path.name.endswith("_sdxl_42.png")
    with Image.open(path) as saved:
        assert saved.text == {
            "prompt": "a cat",
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "seed": "42",
            "steps": "50",
            "quantization": "4bit",
        }
