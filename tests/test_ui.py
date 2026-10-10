from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import gradio as gr
import pytest
from PIL import Image

from txt2img import pipeline, ui
from txt2img.config import GiB


@pytest.fixture(autouse=True)
def isolated_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "_generator", None)
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path)


def generate(model: str = "sdxl", **overrides):
    args = {
        "prompt": "a cat",
        "model": model,
        "aspect": "1:1",
        "seed": 3,
        "steps": 20,
        "count": 1,
        "transparent": False,
        "placement": "auto",
        "quantization": "none",
    }
    return ui.generate_images(**(args | overrides))


def test_generates_saves_and_returns_captioned_images(fake_pipe: MagicMock, tmp_path: Path) -> None:
    gallery, memory = generate(count=2)

    assert [caption for _, caption in gallery] == ["seed 3", "seed 4"]
    assert len(list(tmp_path.glob("*.png"))) == 2
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 20
    assert "**Loaded sdxl (gpu)**" in memory
    assert "**Peak during generation**" in memory


def test_vram_overflow_warning_is_bold(fake_pipe: MagicMock, fake_cuda: SimpleNamespace) -> None:
    fake_cuda.peak = 31 * GiB

    _, memory = generate()

    assert "**WARNING: VRAM limit exceeded by 1.0 GiB" in memory


def test_negative_seed_means_random(fake_pipe: MagicMock) -> None:
    [(_, caption)], _ = generate(seed=-1)

    assert 0 <= int(caption.removeprefix("seed ")) < 2**32


def test_steps_ignored_for_qwen_turbo(fake_pipe: MagicMock) -> None:
    generate("qwen-image-2.1-turbo", steps=40)

    assert "num_inference_steps" not in fake_pipe.call_args.kwargs


def test_reuses_loaded_model_and_reloads_when_settings_change(fake_pipe: MagicMock) -> None:
    generate("sdxl")
    generate("sdxl")
    generate("sdxl", quantization="4bit")
    generate("sdxl", quantization="4bit", placement="offload")
    generate("z-image-turbo", quantization="4bit", placement="offload")

    loaded = [c.args[0] for c in pipeline.DiffusionPipeline.from_pretrained.call_args_list]
    assert loaded == [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "Tongyi-MAI/Z-Image-Turbo",
    ]


def test_uses_models_dir_from_environment(
    fake_pipe: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TXT2IMG_MODELS_DIR", str(tmp_path / "models"))

    generate()

    assert pipeline.DiffusionPipeline.from_pretrained.call_args.kwargs["cache_dir"] == (
        tmp_path / "models"
    )


def test_empty_prompt_is_rejected(fake_pipe: MagicMock) -> None:
    with pytest.raises(gr.Error):
        generate(prompt="  ")


def test_transparent_only_for_supporting_models(fake_pipe: MagicMock) -> None:
    with pytest.raises(gr.Error, match="transparent"):
        generate("sdxl", transparent=True)

    generate("qwen-image-2.1-turbo", transparent=True)

    assert fake_pipe.call_args.kwargs["prompt"].startswith("This is an RGBA image")


def test_controls_follow_model() -> None:
    steps, transparent = ui._model_controls("sdxl")
    assert (steps.interactive, steps.value, transparent.interactive) == (True, 50, False)

    steps, transparent = ui._model_controls("qwen-image-2.1-turbo")
    assert (steps.interactive, transparent.interactive) == (False, True)


def test_mask_from_editor_merges_painted_layers() -> None:
    first = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    first.putpixel((1, 1), (255, 255, 255, 120))
    second = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    second.putpixel((8, 8), (255, 255, 255, 255))

    mask = ui.mask_from_editor({"background": None, "layers": [first, second]})

    assert (mask.getpixel((1, 1)), mask.getpixel((8, 8)), mask.getpixel((5, 5))) == (255, 255, 0)


@pytest.mark.parametrize(
    "value", [None, {"layers": []}, {"layers": [Image.new("RGBA", (4, 4), (0, 0, 0, 0))]}]
)
def test_mask_from_editor_without_paint_is_none(value: dict | None) -> None:
    assert ui.mask_from_editor(value) is None


def animate(editor: dict | None, **overrides):
    args = {
        "editor": editor,
        "model": "qwen-image-2.1-turbo",
        "frame_text": "",
        "motion": "the flame flickers",
        "frame_count": 3,
        "source": "base",
        "feather": 0,
        "fps": 8,
        "formats": ["webp", "gif"],
        "seed": 5,
        "steps": 40,
        "transparent": False,
        "placement": "auto",
        "quantization": "none",
    }
    return ui.animate_images(**(args | overrides))


def test_animate_returns_preview_frames_and_files(
    fake_pipe: MagicMock, small_qwen: None, tmp_path: Path
) -> None:
    layer = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    layer.paste((255, 255, 255, 255), (0, 0, 32, 32))
    editor = {"background": Image.new("RGBA", (64, 64), "red"), "layers": [layer]}

    preview, frames, files, memory = animate(editor)

    assert preview.endswith("animation.webp") and Path(preview).exists()
    assert [caption for _, caption in frames] == [
        "frame 1 - seed 5",
        "frame 2 - seed 6",
        "frame 3 - seed 7",
    ]
    assert [Path(f).name for f in files] == ["animation.webp", "animation.gif"]
    assert frames[0][0].getpixel((60, 60)) == (255, 0, 0, 255)  # outside the mask: base
    assert "Peak during generation" in memory


def test_animate_needs_a_base_image(fake_pipe: MagicMock) -> None:
    with pytest.raises(gr.Error, match="base image"):
        animate({"background": None, "layers": []})


def test_generate_base_fills_the_editor(fake_pipe: MagicMock, small_qwen: None) -> None:
    value, _ = ui.generate_base(
        "a candle", "qwen-image-2.1-turbo", "1:1", 1, 40, False, "auto", "none"
    )

    assert value["background"] is value["composite"]
    assert value["layers"] == []


def test_ui_builds() -> None:
    assert isinstance(ui.build_ui(), gr.Blocks)
