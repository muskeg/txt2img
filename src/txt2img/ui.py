import gc

import gradio as gr
import torch
from PIL import Image

from txt2img.config import (
    ASPECTS,
    DEFAULT_ASPECT,
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
        _generator = ImageGenerator(
            model,
            placement=Placement(placement),
            quantization=Quantization(quantization),
            models_dir=default_models_dir(),
        )
    return _generator


def _memory_markdown(title: str, report: MemoryReport) -> str:
    lines = [f"**{title}**", *report.summary().splitlines()]
    return "  \n".join(f"**{line}**" if line.startswith("WARNING") else line for line in lines)


def generate_images(
    prompt: str,
    model: str,
    aspect: str,
    seed: float | None,
    steps: float,
    count: float,
    placement: str,
    quantization: str,
) -> tuple[list[tuple[Image.Image, str]], str]:
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")
    try:
        generator = _get_generator(model, placement, quantization)
    except (GPUUnavailableError, ModelAccessError) as e:
        raise gr.Error(str(e)) from e

    generation = generator.generate(
        prompt,
        aspect=aspect,
        seed=None if seed is None or seed < 0 else int(seed),
        steps=int(steps) if MODELS[model].default_steps is not None else None,
        count=int(count),
    )
    for r in generation.images:
        save_image(r, OUTPUT_DIR)
    memory = "\n\n".join(
        [
            _memory_markdown(f"Loaded {model} ({generator.placement})", generator.load_memory),
            _memory_markdown("Peak during generation", generation.memory),
        ]
    )
    return [(r.image, f"seed {r.seed}") for r in generation.images], memory


def _steps_slider(model: str) -> gr.Slider:
    default = MODELS[model].default_steps
    if default is None:
        return gr.Slider(interactive=False, label="Steps (fixed by this model)")
    return gr.Slider(value=default, interactive=True, label="Steps")


def build_ui() -> gr.Blocks:
    model_choices = [(f"{name} (gated)" if p.gated else name, name) for name, p in MODELS.items()]
    with gr.Blocks(title="txt2img") as demo:
        prompt = gr.Textbox(label="Prompt", lines=3)
        with gr.Row():
            model = gr.Dropdown(model_choices, value=DEFAULT_MODEL, label="Model")
            aspect = gr.Dropdown(ASPECTS, value=DEFAULT_ASPECT, label="Aspect ratio")
        with gr.Row():
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            steps = gr.Slider(1, 100, value=MODELS[DEFAULT_MODEL].default_steps, step=1)
            count = gr.Slider(1, 4, value=1, step=1, label="Images")
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
        button = gr.Button("Generate", variant="primary")
        gallery = gr.Gallery(label="Results", columns=2)
        memory = gr.Markdown()

        demo.load(_steps_slider, model, steps)
        model.change(_steps_slider, model, steps)
        button.click(
            generate_images,
            [prompt, model, aspect, seed, steps, count, placement, quantization],
            [gallery, memory],
        )
    return demo


def main() -> None:
    # Localhost only; WSL forwards it to the Windows browser.
    build_ui().launch(server_name="127.0.0.1")
