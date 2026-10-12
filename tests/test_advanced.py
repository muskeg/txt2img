from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from txt2img import pipeline, ui
from txt2img.cli import app
from txt2img.config import PONY_NEGATIVE, PONY_PREFIX
from txt2img.local import list_local, parse_lora, resolve_local
from txt2img.pipeline import GeneratedImage, ImageGenerator, save_image

runner = CliRunner()


def from_single_file() -> MagicMock:
    return pipeline.diffusers.StableDiffusionXLPipeline.from_single_file


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A models dir with one checkpoint and two LoRAs, exposed through the env var."""
    root = tmp_path / "models"
    (root / "checkpoints").mkdir(parents=True)
    (root / "loras").mkdir()
    for path in ("checkpoints/my_pony.safetensors", "loras/a.safetensors", "loras/b.safetensors"):
        (root / path).write_bytes(b"")
    (root / "loras" / "notes.txt").write_text("not a LoRA")
    monkeypatch.setenv("TXT2IMG_MODELS_DIR", str(root))
    return root


# Local files


def test_list_local_lists_safetensors_only(models_dir: Path) -> None:
    assert list_local("loras", models_dir) == ["a.safetensors", "b.safetensors"]
    assert list_local("loras", None) == []
    assert list_local("missing", models_dir) == []


@pytest.mark.parametrize("value", ["my_pony", "my_pony.safetensors"])
def test_resolve_local_by_name(models_dir: Path, value: str) -> None:
    assert resolve_local(value, "checkpoints", models_dir) == (
        models_dir / "checkpoints" / "my_pony.safetensors"
    )


def test_resolve_local_by_path(models_dir: Path) -> None:
    path = models_dir / "loras" / "a.safetensors"

    assert resolve_local(str(path), "loras", None) == path


def test_resolve_local_rejects_pickle_formats(tmp_path: Path) -> None:
    (unsafe := tmp_path / "model.ckpt").write_bytes(b"")

    with pytest.raises(ValueError, match="safetensors"):
        resolve_local(str(unsafe), "checkpoints", None)


def test_resolve_local_missing_file_says_where_it_looked(models_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoints"):
        resolve_local("nope", "checkpoints", models_dir)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("detail", ("detail", 1.0)),
        ("detail:0.6", ("detail", 0.6)),
        ("/x/detail.safetensors:1.2", ("/x/detail.safetensors", 1.2)),
        ("odd:name", ("odd:name", 1.0)),
    ],
)
def test_parse_lora(spec: str, expected: tuple[str, float]) -> None:
    assert parse_lora(spec) == expected


# Negative prompt, CFG and prompt prefix


def test_model_default_negative_prompt_is_used(fake_pipe: MagicMock) -> None:
    ImageGenerator("realvisxl-v5").generate("a portrait")

    assert fake_pipe.call_args.kwargs["negative_prompt"].startswith("bad hands")


def test_negative_prompt_replaces_default_and_empty_disables_it(fake_pipe: MagicMock) -> None:
    generator = ImageGenerator("realvisxl-v5")

    generator.generate("a portrait", negative_prompt="clothes")
    assert fake_pipe.call_args.kwargs["negative_prompt"] == "clothes"

    generator.generate("a portrait", negative_prompt="")
    assert "negative_prompt" not in fake_pipe.call_args.kwargs


def test_cfg_overrides_guidance(fake_pipe: MagicMock) -> None:
    [result] = ImageGenerator("sdxl").generate("a cat", guidance=8.5).images

    assert fake_pipe.call_args.kwargs["guidance_scale"] == 8.5
    assert result.guidance == 8.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"negative_prompt": "x"}, "negative prompt"), ({"guidance": 3.0}, "guidance is fixed")],
)
def test_distilled_models_reject_negative_and_cfg(
    fake_pipe: MagicMock, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageGenerator("z-image-turbo").generate("a cat", **kwargs)


def test_pony_prefix_and_negative(fake_pipe: MagicMock) -> None:
    [result] = ImageGenerator("pony-realism-v23").generate("a woman on a beach").images

    kwargs = fake_pipe.call_args.kwargs
    assert kwargs["prompt"] == f"{PONY_PREFIX}, a woman on a beach"
    assert kwargs["negative_prompt"] == PONY_NEGATIVE
    assert result.prompt == "a woman on a beach"


# Single-file weights and LoRAs


def test_single_file_preset_loads_from_its_repo_file(fake_pipe: MagicMock, tmp_path: Path) -> None:
    ImageGenerator("pornworks-pony-v04", models_dir=tmp_path)

    from_single_file().assert_called_once_with(
        "https://huggingface.co/Manjushri/pornworks-real-porn-photo-realistic-nsfw-sdxl-and-pony-chekpoint"
        "/blob/main/pornworksRealPornPhoto_ponyV04.safetensors",
        torch_dtype=torch.bfloat16,
        cache_dir=tmp_path,
    )
    pipeline.DiffusionPipeline.from_pretrained.assert_not_called()


def test_local_checkpoint_replaces_preset_weights(fake_pipe: MagicMock, models_dir: Path) -> None:
    checkpoint = models_dir / "checkpoints" / "my_pony.safetensors"

    generator = ImageGenerator("pony-realism-v23", checkpoint=checkpoint)
    [result] = generator.generate("a cat").images

    assert from_single_file().call_args.args == (str(checkpoint),)
    assert result.checkpoint == "my_pony.safetensors"
    assert fake_pipe.call_args.kwargs["prompt"].startswith(PONY_PREFIX)


@pytest.mark.parametrize(
    ("model", "kwargs", "message"),
    [
        ("flux.1-schnell", {"checkpoint": Path("x.safetensors")}, "SDXL"),
        ("sdxl", {"checkpoint": Path("x.safetensors"), "quantization": "4bit"}, "quantization"),
        ("pornworks-pony-v04", {"quantization": "4bit"}, "quantization"),
    ],
)
def test_invalid_checkpoint_combinations(
    fake_pipe: MagicMock, model: str, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageGenerator(model, **kwargs)


def test_loras_are_loaded_with_their_strengths(fake_pipe: MagicMock, models_dir: Path) -> None:
    loras = [
        (models_dir / "loras" / "a.safetensors", 0.7),
        (models_dir / "loras" / "b.safetensors", 1.0),
    ]

    generator = ImageGenerator("sdxl", loras=loras)
    [result] = generator.generate("a cat").images

    load_calls = fake_pipe.load_lora_weights.call_args_list
    assert [c.kwargs["adapter_name"] for c in load_calls] == ["lora_0", "lora_1"]
    assert [c.kwargs["weight_name"] for c in load_calls] == ["a.safetensors", "b.safetensors"]
    fake_pipe.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.7, 1.0])
    assert result.loras == ("a:0.7", "b:1")


def test_save_image_records_advanced_settings(tmp_path: Path) -> None:
    result = GeneratedImage(
        Image.new("RGB", (8, 8)),
        "a cat",
        "sdxl",
        1,
        30,
        negative_prompt="blurry",
        guidance=6.5,
        checkpoint="my_pony.safetensors",
        loras=("a:0.7",),
    )

    with Image.open(save_image(result, tmp_path)) as saved:
        assert {
            k: saved.text[k] for k in ("negative_prompt", "guidance", "checkpoint", "loras")
        } == {
            "negative_prompt": "blurry",
            "guidance": "6.5",
            "checkpoint": "my_pony.safetensors",
            "loras": "a:0.7",
        }


# CLI


def test_cli_checkpoint_defaults_to_sdxl_and_finds_files(
    fake_pipe: MagicMock, models_dir: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        ["a cat", "--checkpoint", "my_pony", "--lora", "a:0.5", "-N", "blurry", "--cfg", "7"]
        + ["-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 0, result.output
    assert from_single_file().call_args.args == (
        str(models_dir / "checkpoints" / "my_pony.safetensors"),
    )
    fake_pipe.set_adapters.assert_called_once_with(["lora_0"], adapter_weights=[0.5])
    kwargs = fake_pipe.call_args.kwargs
    assert (kwargs["negative_prompt"], kwargs["guidance_scale"]) == ("blurry", 7.0)
    assert "Loaded sdxl (gpu, checkpoint my_pony.safetensors, LoRA a x0.5)" in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["-m", "z-image-turbo", "-N", "blurry"],
        ["-m", "z-image-turbo", "--cfg", "5"],
        ["-m", "flux.1-schnell", "--checkpoint", "my_pony"],
        ["--checkpoint", "my_pony", "--quantize", "4bit"],
        ["-m", "sdxl", "--checkpoint", "missing"],
        ["-m", "sdxl", "--lora", "missing"],
    ],
)
def test_cli_rejects_bad_advanced_options_before_loading(
    fake_pipe: MagicMock, models_dir: Path, args: list[str]
) -> None:
    result = runner.invoke(app, ["a cat", *args])

    assert result.exit_code == 2, result.output
    pipeline.DiffusionPipeline.from_pretrained.assert_not_called()
    from_single_file().assert_not_called()


# UI


@pytest.fixture
def isolated_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui, "_generator", None)
    monkeypatch.setattr(ui, "_generator_key", None)
    monkeypatch.setattr(ui, "OUTPUT_DIR", tmp_path / "out")


def ui_generate(model: str = "sdxl", **advanced):
    return ui.generate_images("a cat", model, "1:1", 1, 30, 1, False, "auto", "none", **advanced)


def test_ui_advanced_options_reach_the_model(
    fake_pipe: MagicMock, models_dir: Path, isolated_ui: None
) -> None:
    _, memory = ui_generate(
        "pony-realism-v23",
        negative="blurry",
        cfg=7.5,
        checkpoint="my_pony.safetensors",
        loras=["a.safetensors"],
        lora_scale=0.8,
    )

    kwargs = fake_pipe.call_args.kwargs
    assert (kwargs["negative_prompt"], kwargs["guidance_scale"]) == ("blurry", 7.5)
    fake_pipe.set_adapters.assert_called_once_with(["lora_0"], adapter_weights=[0.8])
    assert "checkpoint my_pony.safetensors, LoRA a x0.8" in memory


def test_ui_reloads_when_loras_change(
    fake_pipe: MagicMock, models_dir: Path, isolated_ui: None
) -> None:
    ui_generate(loras=["a.safetensors"])
    ui_generate(loras=["a.safetensors"])
    ui_generate(loras=["a.safetensors", "b.safetensors"])

    assert pipeline.DiffusionPipeline.from_pretrained.call_count == 2


def test_ui_preset_weights_choice_means_no_checkpoint(
    fake_pipe: MagicMock, isolated_ui: None
) -> None:
    ui_generate(checkpoint=ui.PRESET_WEIGHTS)

    from_single_file().assert_not_called()


def test_ui_checkpoint_needs_sdxl_model(fake_pipe: MagicMock, isolated_ui: None) -> None:
    with pytest.raises(gr.Error, match="SDXL"):
        ui_generate("z-image-turbo", checkpoint="my_pony.safetensors")


def test_ui_guidance_controls_follow_model() -> None:
    negative, cfg = ui._guidance_controls("realvisxl-v5")
    assert (negative.interactive, cfg.interactive, cfg.value) == (True, True, 5.0)
    assert negative.value.startswith("bad hands")

    negative, cfg = ui._guidance_controls("z-image-turbo")
    assert (negative.interactive, cfg.interactive) == (False, False)


def test_ui_local_choices_scan_models_dir(models_dir: Path) -> None:
    checkpoints, loras = ui._local_choices()

    assert [c[0] for c in checkpoints.choices] == [ui.PRESET_WEIGHTS, "my_pony.safetensors"]
    assert [c[0] for c in loras.choices] == ["a.safetensors", "b.safetensors"]
