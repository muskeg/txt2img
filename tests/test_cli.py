from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from typer.testing import CliRunner

from txt2img import pipeline
from txt2img.cli import app

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
