import gc
import os
import sys
from dataclasses import replace
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageChops, UnidentifiedImageError

from txt2img.animate import AnimationFormat, FrameSource, animate, frame_prompts, save_animation
from txt2img.config import (
    ASPECTS,
    DEFAULT_ASPECT,
    DEFAULT_EDIT_MODEL,
    DEFAULT_MODEL,
    DEFAULT_STRENGTH,
    FOLLOW_ASPECTS,
    MAX_REFERENCES,
    MODELS,
    OUTPUT_DIR,
    Placement,
    Quantization,
    default_models_dir,
)
from txt2img.local import CHECKPOINTS, LORAS, list_local, resolve_local
from txt2img.pipeline import (
    GPUUnavailableError,
    ImageGenerator,
    MemoryReport,
    ModelAccessError,
    load_image,
    save_image,
)

_generator: ImageGenerator | None = None
_generator_key: tuple | None = None
PRESET_WEIGHTS = "(model's own weights)"
AUTO_ASPECT = "auto"
EDITOR_IMAGE = "(editor)"


def _get_generator(
    model: str,
    placement: str,
    quantization: str,
    checkpoint: str | None = None,
    loras: tuple[tuple[str, float], ...] = (),
) -> ImageGenerator:
    global _generator, _generator_key
    wanted = (model, placement, quantization, checkpoint, loras)
    if _generator is None or _generator_key != wanted:
        # Only one model fits in RAM at a time; free the old one before loading another.
        _generator = _generator_key = None
        gc.collect()
        torch.cuda.empty_cache()
        models_dir = default_models_dir()
        try:
            _generator = ImageGenerator(
                model,
                placement=Placement(placement),
                quantization=Quantization(quantization),
                models_dir=models_dir,
                checkpoint=resolve_local(checkpoint, CHECKPOINTS, models_dir)
                if checkpoint
                else None,
                loras=[(resolve_local(name, LORAS, models_dir), scale) for name, scale in loras],
            )
        except (GPUUnavailableError, ModelAccessError, FileNotFoundError, ValueError) as e:
            raise gr.Error(str(e)) from e
        _generator_key = wanted
    return _generator


def _memory_markdown(generator: ImageGenerator, peak: MemoryReport) -> str:
    sections = [
        (f"Loaded {generator.model} ({_load_details(generator)})", generator.load_memory),
        ("Peak during generation", peak),
    ]
    blocks = []
    for title, report in sections:
        lines = [f"**{title}**", *report.summary().splitlines()]
        blocks.append(
            "  \n".join(f"**{line}**" if line.startswith("WARNING") else line for line in lines)
        )
    return "\n\n".join(blocks)


def _load_details(generator: ImageGenerator) -> str:
    details = [str(generator.placement)]
    if generator.checkpoint:
        details.append(f"checkpoint {generator.checkpoint.name}")
    details += [f"LoRA {path.stem} x{scale:g}" for path, scale in generator.loras]
    return ", ".join(details)


def _seed(value: float | None) -> int | None:
    return None if value is None or value < 0 else int(value)


def _steps(model: str, value: float) -> int | None:
    return int(value) if MODELS[model].default_steps is not None else None


def _check_transparent(model: str, transparent: bool) -> None:
    if transparent and not MODELS[model].transparency:
        raise gr.Error(f"{model} can't generate transparent images.")


def generate_images(
    prompt: str,
    model: str,
    aspect: str,
    seed: float | None,
    steps: float,
    count: float,
    transparent: bool,
    placement: str,
    quantization: str,
    negative: str = "",
    cfg: float | None = None,
    checkpoint: str | None = None,
    loras: list[str] | None = None,
    lora_scale: float = 1.0,
    editor: dict | None = None,
    references: list[tuple[str, str | None]] | None = None,
    strength: float = DEFAULT_STRENGTH,
) -> tuple[list[tuple[Image.Image, str]], str]:
    """`editor` holds image 1 and the painted edit area; `references` are images 2, 3..."""
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    _check_transparent(model, transparent)
    preset = MODELS[model]
    checkpoint = None if not checkpoint or checkpoint == PRESET_WEIGHTS else checkpoint
    if checkpoint and not preset.checkpoint_pipeline:
        raise gr.Error(f"{model} isn't SDXL-based; pick sdxl or a Pony/SDXL model for checkpoints.")
    images, names = [], []
    base = (editor or {}).get("background")
    if base is not None:
        images.append(base)
        names.append(EDITOR_IMAGE)
    try:
        for path, _ in references or []:
            images.append(load_image(Path(path)))
            names.append(Path(path).name)
    except UnidentifiedImageError as e:
        raise gr.Error(str(e)) from e
    if len(images) > 1 and not preset.editing:
        raise gr.Error(f"{model} takes one input image; use a Qwen-Image-2.1 model to combine.")
    mask = mask_from_editor(editor) if base is not None else None

    lora_specs = tuple((name, float(lora_scale)) for name in loras or [])
    generator = _get_generator(model, placement, quantization, checkpoint, lora_specs)
    try:
        generation = generator.generate(
            prompt,
            aspect=None if aspect == AUTO_ASPECT else aspect,
            seed=_seed(seed),
            steps=_steps(model, steps),
            count=int(count),
            transparent=transparent,
            images=images,
            mask=mask,
            strength=float(strength) if images and not preset.editing else None,
            negative_prompt=negative if preset.supports_negative else None,
            guidance=float(cfg) if preset.adjustable_guidance and cfg is not None else None,
        )
    except ValueError as e:
        raise gr.Error(str(e)) from e
    for r in generation.images:
        save_image(
            replace(r, references=tuple(names), mask="painted" if mask else None), OUTPUT_DIR
        )
    gallery = [(r.image, f"seed {r.seed}") for r in generation.images]
    return gallery, _memory_markdown(generator, generation.memory)


def generate_base(
    prompt: str,
    model: str,
    aspect: str,
    seed: float | None,
    steps: float,
    transparent: bool,
    placement: str,
    quantization: str,
) -> tuple[dict, str]:
    [(image, _)], memory = generate_images(
        prompt, model, aspect, seed, steps, 1, transparent, placement, quantization
    )
    return {"background": image, "layers": [], "composite": image}, memory


def use_as_input(results: list | None) -> tuple[dict, gr.Accordion]:
    """Put the first result in the image 1 editor, to keep editing it."""
    if not results:
        raise gr.Error("Generate an image first.")
    image = load_image(Path(results[0][0]))
    return {"background": image, "layers": [], "composite": image}, gr.Accordion(open=True)


def mask_from_editor(value: dict | None) -> Image.Image | None:
    """Union of everything painted on the editor layers, or None if nothing was painted."""
    layers = (value or {}).get("layers") or []
    mask = None
    for layer in layers:
        alpha = layer.convert("RGBA").getchannel("A")
        mask = alpha if mask is None else ImageChops.lighter(mask, alpha)
    if mask is None or mask.getbbox() is None:
        return None
    return mask.point(lambda a: 255 if a else 0)


def animate_images(
    editor: dict | None,
    model: str,
    frame_text: str,
    motion: str,
    frame_count: float,
    source: str,
    feather: float,
    fps: float,
    formats: list[str],
    seed: float | None,
    steps: float,
    transparent: bool,
    placement: str,
    quantization: str,
) -> tuple[str, list[tuple[Image.Image, str]], list[str], str]:
    base = (editor or {}).get("background")
    if base is None:
        raise gr.Error("Upload or generate a base image first.")
    if not formats:
        raise gr.Error("Pick at least one animation format.")
    try:
        prompts = frame_prompts(frame_text.splitlines(), motion, int(frame_count))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    _check_transparent(model, transparent)

    generator = _get_generator(model, placement, quantization)
    animation = animate(
        generator,
        base,
        prompts,
        mask=mask_from_editor(editor),
        source=FrameSource(source),
        seed=_seed(seed),
        steps=_steps(model, steps),
        feather=int(feather),
        transparent=transparent,
    )
    paths = save_animation(
        animation, OUTPUT_DIR, fps=int(fps), formats=[AnimationFormat(f) for f in formats]
    )
    frames = [(f.image, f"frame {k} - seed {f.seed}") for k, f in enumerate(animation.frames, 1)]
    files = [str(p) for p in paths]
    return files[0], frames, files, _memory_markdown(generator, animation.memory)


def _model_controls(model: str) -> tuple[gr.Slider, gr.Checkbox]:
    preset = MODELS[model]
    if preset.default_steps is None:
        steps = gr.Slider(interactive=False, label="Steps (fixed by this model)")
    else:
        steps = gr.Slider(value=preset.default_steps, interactive=True, label="Steps")
    transparent = gr.Checkbox(
        value=False,
        interactive=preset.transparency,
        label="Transparent background" if preset.transparency else "Transparent (not supported)",
    )
    return steps, transparent


def _guidance_controls(model: str) -> tuple[gr.Textbox, gr.Slider]:
    preset = MODELS[model]
    negative = gr.Textbox(
        value=preset.default_negative,
        interactive=preset.supports_negative,
        label="Negative prompt" if preset.supports_negative else "Negative prompt (not supported)",
    )
    adjustable = preset.adjustable_guidance
    cfg = gr.Slider(
        value=float(preset.call_kwargs.get("guidance_scale", 0)),
        interactive=adjustable,
        label="CFG / guidance" if adjustable else "CFG / guidance (fixed by this model)",
    )
    return negative, cfg


def _strength_control(model: str) -> gr.Slider:
    editing = MODELS[model].editing
    label = "Strength (not used: this model follows the prompt)" if editing else "Strength"
    return gr.Slider(interactive=not editing, label=label)


def _local_choices() -> tuple[gr.Dropdown, gr.Dropdown]:
    models_dir = default_models_dir()
    return (
        gr.Dropdown(choices=[PRESET_WEIGHTS, *list_local(CHECKPOINTS, models_dir)]),
        gr.Dropdown(choices=list_local(LORAS, models_dir)),
    )


def build_ui() -> gr.Blocks:
    model_choices = [(f"{name} (gated)" if p.gated else name, name) for name, p in MODELS.items()]
    edit_models = [name for name, p in MODELS.items() if p.editing]
    with gr.Blocks(title="txt2img") as demo:
        with gr.Row():
            placement = gr.Radio(
                [p.value for p in Placement],
                value=Placement.AUTO.value,
                label="Placement",
                info="auto: all on GPU if it fits, else CPU offload",
            )
            quantization = gr.Radio(
                [q.value for q in Quantization],
                value=Quantization.NONE.value,
                label="Quantization",
                info="4bit: NF4 weights, ~3.5x smaller, slightly lower quality",
            )

        with gr.Tabs():
            with gr.Tab("Generate"):
                prompt = gr.Textbox(label="Prompt", lines=3)
                with gr.Accordion(
                    "Input images (optional): edit, combine or redraw images", open=False
                ) as inputs:
                    gr.Markdown(
                        "**Qwen-Image-2.1** models follow the prompt as an edit instruction: "
                        "*make it night*, *put the dog from &lt;image2&gt; on the sofa in "
                        "&lt;image1&gt;*. Other models redraw image 1 from the prompt, by "
                        "**Strength**. With any model, paint over part of image 1 to change only "
                        "that area."
                    )
                    with gr.Row():
                        input_editor = gr.ImageEditor(
                            type="pil",
                            image_mode="RGBA",
                            label="Image 1",
                            brush=gr.Brush(colors=["#ff0000"], color_mode="fixed"),
                            transforms=(),
                        )
                        references = gr.Gallery(
                            label=f"Images 2 to {MAX_REFERENCES} (Qwen-Image-2.1 only)",
                            type="filepath",
                            file_types=["image"],
                            interactive=True,
                            columns=3,
                            format="png",
                        )
                    strength = gr.Slider(
                        0.05,
                        1,
                        value=DEFAULT_STRENGTH,
                        step=0.05,
                        info="How much to redraw image 1: low keeps it, 1 replaces it",
                    )
                with gr.Row():
                    model = gr.Dropdown(model_choices, value=DEFAULT_MODEL, label="Model")
                    aspect = gr.Dropdown(
                        [("Auto (image 1, else 1:1)", AUTO_ASPECT), *ASPECTS]
                        + [(f"Follow image {n[5:]}", n) for n in FOLLOW_ASPECTS],
                        value=AUTO_ASPECT,
                        label="Aspect ratio",
                    )
                with gr.Row():
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    steps = gr.Slider(1, 100, step=1)
                    count = gr.Slider(1, 4, value=1, step=1, label="Images")
                    transparent = gr.Checkbox()
                with gr.Accordion("Advanced: negative prompt, CFG, checkpoint, LoRAs", open=False):
                    negative = gr.Textbox(lines=2)
                    cfg = gr.Slider(0, 15, step=0.5)
                    with gr.Row():
                        checkpoint = gr.Dropdown(
                            [PRESET_WEIGHTS],
                            value=PRESET_WEIGHTS,
                            allow_custom_value=True,
                            label="Checkpoint (SDXL models only, from <models>/checkpoints)",
                        )
                        loras = gr.Dropdown(
                            [],
                            multiselect=True,
                            allow_custom_value=True,
                            label="LoRAs (from <models>/loras)",
                        )
                        lora_scale = gr.Slider(0, 2, value=1.0, step=0.05, label="LoRA strength")
                    rescan = gr.Button("Rescan checkpoint and LoRA folders", size="sm")
                button = gr.Button("Generate", variant="primary")
                gallery = gr.Gallery(label="Results", columns=2, format="png", interactive=False)
                reuse = gr.Button("Use first result as image 1", size="sm")
                memory = gr.Markdown()

            with gr.Tab("Animate"):
                with gr.Row():
                    a_model = gr.Dropdown(edit_models, value=DEFAULT_EDIT_MODEL, label="Model")
                    a_aspect = gr.Dropdown(ASPECTS, value=DEFAULT_ASPECT, label="Base aspect ratio")
                    a_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    a_steps = gr.Slider(1, 100, step=1)
                    a_transparent = gr.Checkbox()
                with gr.Row():
                    base_prompt = gr.Textbox(label="Base prompt", lines=2, scale=4)
                    base_button = gr.Button("Generate base", scale=1)
                editor = gr.ImageEditor(
                    type="pil",
                    image_mode="RGBA",
                    label="Base image: upload or generate it, then paint the area to animate",
                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"),
                    transforms=(),
                )
                frame_text = gr.Textbox(
                    label="Frame prompts, one per line (or leave empty and use a motion prompt)",
                    lines=5,
                )
                with gr.Row():
                    motion = gr.Textbox(label="Motion prompt for every frame", scale=3)
                    frame_count = gr.Slider(1, 24, value=8, step=1, label="Frames", scale=1)
                with gr.Row():
                    source = gr.Radio(
                        [s.value for s in FrameSource],
                        value=FrameSource.BASE.value,
                        label="Edit each frame from",
                    )
                    feather = gr.Slider(0, 64, value=8, step=1, label="Mask feather (px)")
                    fps = gr.Slider(1, 30, value=8, step=1, label="FPS")
                    formats = gr.CheckboxGroup(
                        [f.value for f in AnimationFormat],
                        value=[AnimationFormat.WEBP.value],
                        label="Formats",
                    )
                animate_button = gr.Button("Animate", variant="primary")
                with gr.Row():
                    preview = gr.Image(type="filepath", label="Animation")
                    frames = gr.Gallery(label="Frames", columns=4)
                files = gr.File(label="Animation files", file_count="multiple")
                a_memory = gr.Markdown()

        for m, s, t in [(model, steps, transparent), (a_model, a_steps, a_transparent)]:
            demo.load(_model_controls, m, [s, t])
            m.change(_model_controls, m, [s, t])
        demo.load(_guidance_controls, model, [negative, cfg])
        model.change(_guidance_controls, model, [negative, cfg])
        demo.load(_strength_control, model, strength)
        model.change(_strength_control, model, strength)
        demo.load(_local_choices, None, [checkpoint, loras])
        rescan.click(_local_choices, None, [checkpoint, loras])
        button.click(
            generate_images,
            [prompt, model, aspect, seed, steps, count, transparent, placement, quantization]
            + [negative, cfg, checkpoint, loras, lora_scale, input_editor, references, strength],
            [gallery, memory],
        )
        reuse.click(use_as_input, gallery, [input_editor, inputs])
        base_button.click(
            generate_base,
            [base_prompt, a_model, a_aspect, a_seed, a_steps, a_transparent]
            + [placement, quantization],
            [editor, a_memory],
        )
        animate_button.click(
            animate_images,
            [editor, a_model, frame_text, motion, frame_count, source, feather, fps, formats]
            + [a_seed, a_steps, a_transparent, placement, quantization],
            [preview, frames, files, a_memory],
        )
    return demo


def main() -> None:
    # Localhost only; WSL forwards it to the Windows browser.
    build_ui().launch(server_name="127.0.0.1", allowed_paths=[str(Path(OUTPUT_DIR).resolve())])
    # Ctrl+C: Python's teardown of a loaded model takes from seconds to minutes, and waits
    # for a running generation; nothing is left to save, so skip it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
