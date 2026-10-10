import gc
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageChops

from txt2img.animate import AnimationFormat, FrameSource, animate, frame_prompts, save_animation
from txt2img.config import (
    ASPECTS,
    DEFAULT_ASPECT,
    DEFAULT_EDIT_MODEL,
    DEFAULT_MODEL,
    MODELS,
    OUTPUT_DIR,
    Placement,
    Quantization,
    default_models_dir,
)
from txt2img.pipeline import (
    GPUUnavailableError,
    ImageGenerator,
    MemoryReport,
    ModelAccessError,
    save_image,
)

_generator: ImageGenerator | None = None


def _get_generator(model: str, placement: str, quantization: str) -> ImageGenerator:
    global _generator
    wanted = (model, placement, quantization)
    current = _generator and (
        _generator.model,
        _generator.requested_placement,
        _generator.quantization,
    )
    if current != wanted:
        # Only one model fits in RAM at a time; free the old one before loading another.
        _generator = None
        gc.collect()
        torch.cuda.empty_cache()
        try:
            _generator = ImageGenerator(
                model,
                placement=Placement(placement),
                quantization=Quantization(quantization),
                models_dir=default_models_dir(),
            )
        except (GPUUnavailableError, ModelAccessError) as e:
            raise gr.Error(str(e)) from e
    return _generator


def _memory_markdown(generator: ImageGenerator, peak: MemoryReport) -> str:
    sections = [
        (f"Loaded {generator.model} ({generator.placement})", generator.load_memory),
        ("Peak during generation", peak),
    ]
    blocks = []
    for title, report in sections:
        lines = [f"**{title}**", *report.summary().splitlines()]
        blocks.append(
            "  \n".join(f"**{line}**" if line.startswith("WARNING") else line for line in lines)
        )
    return "\n\n".join(blocks)


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
) -> tuple[list[tuple[Image.Image, str]], str]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    _check_transparent(model, transparent)
    generator = _get_generator(model, placement, quantization)
    generation = generator.generate(
        prompt,
        aspect=aspect,
        seed=_seed(seed),
        steps=_steps(model, steps),
        count=int(count),
        transparent=transparent,
    )
    for r in generation.images:
        save_image(r, OUTPUT_DIR)
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

        with gr.Tab("Generate"):
            prompt = gr.Textbox(label="Prompt", lines=3)
            with gr.Row():
                model = gr.Dropdown(model_choices, value=DEFAULT_MODEL, label="Model")
                aspect = gr.Dropdown(ASPECTS, value=DEFAULT_ASPECT, label="Aspect ratio")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                steps = gr.Slider(1, 100, step=1)
                count = gr.Slider(1, 4, value=1, step=1, label="Images")
                transparent = gr.Checkbox()
            button = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Results", columns=2)
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
        button.click(
            generate_images,
            [prompt, model, aspect, seed, steps, count, transparent, placement, quantization],
            [gallery, memory],
        )
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
