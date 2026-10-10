from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from typer.testing import CliRunner

from txt2img import pipeline
from txt2img.cli import animate_app, app

runner = CliRunner()


def test_generates_and_saves_images(fake_pipe: MagicMock, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["a cat", "-m", "qwen-image-2.1-turbo", "-a", "3:4", "--seed", "5", "-n", "2"]
        + ["-o", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name.split("_", 1)[1] for p in tmp_path.iterdir()) == [
        "qwen-image-2.1-turbo_5.png",
        "qwen-image-2.1-turbo_6.png",
    ]
    assert "(seed 5)" in result.stdout
    kwargs = fake_pipe.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (1792, 2400)


def test_prints_memory_reports_to_stderr(fake_pipe: MagicMock, tmp_path: Path) -> None:
    result = runner.invoke(app, ["a cat", "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Loaded z-image-turbo (gpu)" in result.stderr
    assert "Peak during generation" in result.stderr
    assert "Weights:" not in result.stdout


def test_passes_placement_quantization_and_models_dir(
    fake_pipe: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TXT2IMG_MODELS_DIR", str(tmp_path / "models"))

    result = runner.invoke(
        app, ["a cat", "--placement", "offload", "--quantize", "4bit", "-o", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    kwargs = pipeline.DiffusionPipeline.from_pretrained.call_args.kwargs
    assert kwargs["cache_dir"] == tmp_path / "models"
    assert kwargs["quantization_config"].quant_backend == "bitsandbytes_4bit"
    fake_pipe.enable_model_cpu_offload.assert_called_once()


def test_turbo_steps_fail_before_loading_model(fake_pipe: MagicMock, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["a cat", "-m", "qwen-image-2.1-turbo", "--steps", "20", "-o", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "--steps" in result.output
    pipeline.DiffusionPipeline.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["-m", "nope"],
        ["-a", "5:4"],
        ["-n", "0"],
        ["--seed", "-1"],
        ["--placement", "cpu"],
        ["--quantize", "8bit"],
    ],
)
def test_rejects_invalid_options(fake_pipe: MagicMock, args: list[str]) -> None:
    result = runner.invoke(app, ["a cat", *args])

    assert result.exit_code == 2


def test_missing_gpu_exits_with_message(
    fake_pipe: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)

    result = runner.invoke(app, ["a cat"])

    assert result.exit_code == 1
    assert "No CUDA GPU available" in result.stderr


def test_gated_model_exits_with_instructions(fake_pipe: MagicMock) -> None:
    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co"))
    pipeline.DiffusionPipeline.from_pretrained.side_effect = GatedRepoError(
        "gated", response=response
    )

    result = runner.invoke(app, ["a cat", "-m", "flux.1-schnell"])

    assert result.exit_code == 1
    assert "hf auth login" in result.stderr


def test_transparent_uses_rgba_prompt(fake_pipe: MagicMock, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["a sticker", "-m", "qwen-image-2.1-turbo", "--transparent", "-o", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert fake_pipe.call_args.kwargs["prompt"].startswith("This is an RGBA image")


def test_transparent_fails_before_loading_unsupported_model(fake_pipe: MagicMock) -> None:
    result = runner.invoke(app, ["a sticker", "-m", "sdxl", "--transparent"])

    assert result.exit_code == 2
    assert "qwen-image-2.1" in result.output
    pipeline.DiffusionPipeline.from_pretrained.assert_not_called()


def test_animate_existing_base_with_mask(
    fake_pipe: MagicMock, small_qwen: None, tmp_path: Path
) -> None:
    Image.new("RGBA", (64, 64), "red").save(base := tmp_path / "base.png")
    Image.new("L", (64, 64), 255).save(mask := tmp_path / "mask.png")

    result = runner.invoke(
        animate_app,
        ["--base", str(base), "--mask", str(mask), "-f", "flame left", "-f", "flame right"]
        + ["--seed", "7", "-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    [webp] = [Path(line) for line in result.stdout.splitlines()]
    assert webp.name == "animation.webp"
    assert sorted(p.name for p in webp.parent.glob("frame_*.png")) == [
        "frame_01.png",
        "frame_02.png",
    ]
    assert [c.kwargs["prompt"] for c in fake_pipe.call_args_list] == ["flame left", "flame right"]


def test_animate_generates_base_then_motion_frames(
    fake_pipe: MagicMock, small_qwen: None, tmp_path: Path
) -> None:
    result = runner.invoke(
        animate_app,
        ["--base-prompt", "a candle", "--motion", "the flame flickers", "--frames", "3"]
        + ["--seed", "1", "--format", "gif", "--format", "apng", "-o", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    calls = fake_pipe.call_args_list
    assert [c.kwargs["prompt"] for c in calls] == ["a candle"] + ["the flame flickers"] * 3
    assert "image" not in calls[0].kwargs
    assert [c.kwargs["generator"].initial_seed() for c in calls] == [1, 2, 3, 4]
    assert result.stdout.startswith("Base: ")
    assert [Path(p).name for p in result.stdout.splitlines()[1:]] == [
        "animation.gif",
        "animation.apng",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["-f", "a"],
        ["--base-prompt", "x", "--base", "pyproject.toml", "-f", "a"],
        ["--base-prompt", "x"],
        ["--base-prompt", "x", "-f", "a", "--motion", "b"],
        ["--base-prompt", "x", "-f", "a", "-m", "sdxl"],
        ["--base-prompt", "x", "-f", "a", "--format", "mp4"],
    ],
)
def test_animate_rejects_invalid_options(fake_pipe: MagicMock, args: list[str]) -> None:
    result = runner.invoke(animate_app, args)

    assert result.exit_code == 2
    pipeline.DiffusionPipeline.from_pretrained.assert_not_called()
