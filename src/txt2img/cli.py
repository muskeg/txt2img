from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image

from txt2img.animate import AnimationFormat, FrameSource, animate, frame_prompts, save_animation
from txt2img.config import (
    ASPECTS,
    DEFAULT_ASPECT,
    DEFAULT_EDIT_MODEL,
    DEFAULT_MODEL,
    MODELS,
    MODELS_DIR_ENV,
    OUTPUT_DIR,
    Placement,
    Quantization,
)
from txt2img.pipeline import GPUUnavailableError, ImageGenerator, ModelAccessError, save_image

ModelChoice = StrEnum("ModelChoice", {name: name for name in MODELS})
EditModelChoice = StrEnum(
    "EditModelChoice", {name: name for name, p in MODELS.items() if p.editing}
)
AspectChoice = StrEnum("AspectChoice", {name: name for name in ASPECTS})
_DEFAULT_MODEL = ModelChoice(DEFAULT_MODEL)
_DEFAULT_EDIT_MODEL = EditModelChoice(DEFAULT_EDIT_MODEL)
_DEFAULT_ASPECT = AspectChoice(DEFAULT_ASPECT)

AspectOption = Annotated[AspectChoice, typer.Option("--aspect", "-a", help="Aspect ratio.")]
SeedOption = Annotated[int | None, typer.Option(min=0, help="Seed. Random if omitted.")]
StepsOption = Annotated[
    int | None, typer.Option(min=1, help="Denoising steps. Defaults to the model's.")
]
TransparentOption = Annotated[
    bool,
    typer.Option(help="RGBA output with a transparent background (Qwen-Image-2.1 models only)."),
]
OutputDirOption = Annotated[
    Path, typer.Option("--output-dir", "-o", file_okay=False, help="Where to save images.")
]
PlacementOption = Annotated[
    Placement,
    typer.Option(help="auto: all on GPU if it fits, else CPU offload. gpu/offload force either."),
]
QuantizeOption = Annotated[
    Quantization,
    typer.Option(help="4bit: NF4 weights, ~3.5x smaller, slightly lower quality."),
]
ModelsDirOption = Annotated[
    Path | None,
    typer.Option(
        envvar=MODELS_DIR_ENV,
        file_okay=False,
        help="Where model weights are stored. Defaults to the Hugging Face cache.",
    ),
]

app = typer.Typer(add_completion=False)
animate_app = typer.Typer(add_completion=False)


def _check_model_options(model: str, steps: int | None, transparent: bool) -> None:
    """Fail on incompatible flags before the slow model load."""
    preset = MODELS[model]
    if steps is not None and preset.default_steps is None:
        raise typer.BadParameter(
            f"the {model} model uses the schedule stored in its checkpoint.", param_hint="--steps"
        )
    if transparent and not preset.transparency:
        supported = ", ".join(n for n, p in MODELS.items() if p.transparency)
        raise typer.BadParameter(
            f"the {model} model can't generate transparency (use {supported}).",
            param_hint="--transparent",
        )


def _load(
    model: str, placement: Placement, quantize: Quantization, models_dir: Path | None
) -> ImageGenerator:
    try:
        generator = ImageGenerator(
            model, placement=placement, quantization=quantize, models_dir=models_dir
        )
    except (GPUUnavailableError, ModelAccessError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Loaded {model} ({generator.placement})", err=True)
    typer.echo(generator.load_memory.summary(), err=True)
    return generator


@app.command()
def main(
    prompt: Annotated[str, typer.Argument(help="Text prompt describing the image.")],
    model: Annotated[
        ModelChoice, typer.Option("--model", "-m", help="Model preset.")
    ] = _DEFAULT_MODEL,
    aspect: AspectOption = _DEFAULT_ASPECT,
    seed: SeedOption = None,
    steps: StepsOption = None,
    count: Annotated[int, typer.Option("--count", "-n", min=1, help="Number of images.")] = 1,
    transparent: TransparentOption = False,
    output_dir: OutputDirOption = OUTPUT_DIR,
    placement: PlacementOption = Placement.AUTO,
    quantize: QuantizeOption = Quantization.NONE,
    models_dir: ModelsDirOption = None,
) -> None:
    """Generate images from a text prompt."""
    _check_model_options(model.value, steps, transparent)
    generator = _load(model.value, placement, quantize, models_dir)

    generation = generator.generate(
        prompt,
        aspect=aspect.value,
        seed=seed,
        steps=steps,
        count=count,
        transparent=transparent,
    )
    for result in generation.images:
        path = save_image(result, output_dir)
        typer.echo(f"{path} (seed {result.seed})")
    typer.echo(f"Peak during generation:\n{generation.memory.summary()}", err=True)


@animate_app.command()
def animate_main(
    base: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Existing image to animate."),
    ] = None,
    base_prompt: Annotated[
        str | None, typer.Option(help="Generate the base image from this prompt instead.")
    ] = None,
    frame: Annotated[
        list[str] | None,
        typer.Option("--frame", "-f", help="Prompt for one frame. Repeat once per frame."),
    ] = None,
    motion: Annotated[
        str | None, typer.Option(help="One prompt for every frame, with --frames.")
    ] = None,
    frames: Annotated[int, typer.Option(min=1, help="Number of frames with --motion.")] = 8,
    mask: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="White = animated area. Everything else stays identical to the base.",
        ),
    ] = None,
    source: Annotated[
        FrameSource,
        typer.Option(help="What each frame is edited from: the base, the previous frame, or both."),
    ] = FrameSource.BASE,
    feather: Annotated[int, typer.Option(min=0, help="Mask edge softness in pixels.")] = 8,
    fps: Annotated[int, typer.Option(min=1, help="Animation frames per second.")] = 8,
    format: Annotated[
        list[AnimationFormat] | None,
        typer.Option("--format", help="Animation format. Repeat for several. Default: webp."),
    ] = None,
    model: Annotated[
        EditModelChoice, typer.Option("--model", "-m", help="Model preset (editing models).")
    ] = _DEFAULT_EDIT_MODEL,
    aspect: AspectOption = _DEFAULT_ASPECT,
    seed: SeedOption = None,
    steps: StepsOption = None,
    transparent: TransparentOption = False,
    output_dir: OutputDirOption = OUTPUT_DIR,
    placement: PlacementOption = Placement.AUTO,
    quantize: QuantizeOption = Quantization.NONE,
    models_dir: ModelsDirOption = None,
) -> None:
    """Animate parts of an image: each frame is an edit, pasted over the base through the mask."""
    if (base is None) == (base_prompt is None):
        raise typer.BadParameter("give exactly one of --base or --base-prompt.")
    try:
        prompts = frame_prompts(frame, motion, frames)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--frame / --motion") from e
    _check_model_options(model.value, steps, transparent)
    generator = _load(model.value, placement, quantize, models_dir)

    if base_prompt is not None:
        result = generator.generate(
            base_prompt, aspect=aspect.value, seed=seed, steps=steps, transparent=transparent
        ).images[0]
        typer.echo(f"Base: {save_image(result, output_dir)} (seed {result.seed})")
        base_image = result.image
        seed = result.seed + 1
    else:
        base_image = Image.open(base)
    mask_image = Image.open(mask) if mask else None

    animation = animate(
        generator,
        base_image,
        prompts,
        mask=mask_image,
        source=source,
        seed=seed,
        steps=steps,
        feather=feather,
        transparent=transparent,
    )
    for path in save_animation(animation, output_dir, fps=fps, formats=format):
        typer.echo(path)
    typer.echo(f"Peak during generation:\n{animation.memory.summary()}", err=True)
