from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from txt2img.config import (
    ASPECTS,
    DEFAULT_ASPECT,
    DEFAULT_MODEL,
    MODELS,
    MODELS_DIR_ENV,
    OUTPUT_DIR,
    Placement,
    Quantization,
)
from txt2img.pipeline import GPUUnavailableError, ImageGenerator, ModelAccessError, save_image

ModelChoice = StrEnum("ModelChoice", {name: name for name in MODELS})
AspectChoice = StrEnum("AspectChoice", {name: name for name in ASPECTS})
_DEFAULT_MODEL = ModelChoice(DEFAULT_MODEL)
_DEFAULT_ASPECT = AspectChoice(DEFAULT_ASPECT)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    prompt: Annotated[str, typer.Argument(help="Text prompt describing the image.")],
    model: Annotated[
        ModelChoice, typer.Option("--model", "-m", help="Model preset.")
    ] = _DEFAULT_MODEL,
    aspect: Annotated[
        AspectChoice, typer.Option("--aspect", "-a", help="Aspect ratio.")
    ] = _DEFAULT_ASPECT,
    seed: Annotated[int | None, typer.Option(min=0, help="Seed. Random if omitted.")] = None,
    steps: Annotated[
        int | None, typer.Option(min=1, help="Denoising steps. Defaults to the model's.")
    ] = None,
    count: Annotated[int, typer.Option("--count", "-n", min=1, help="Number of images.")] = 1,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", file_okay=False, help="Where to save images.")
    ] = OUTPUT_DIR,
    placement: Annotated[
        Placement,
        typer.Option(
            help="auto: all on GPU if it fits, else CPU offload. gpu/offload force either."
        ),
    ] = Placement.AUTO,
    quantize: Annotated[
        Quantization,
        typer.Option(help="4bit: NF4 weights, ~3.5x smaller, slightly lower quality."),
    ] = Quantization.NONE,
    models_dir: Annotated[
        Path | None,
        typer.Option(
            envvar=MODELS_DIR_ENV,
            file_okay=False,
            help="Where model weights are stored. Defaults to the Hugging Face cache.",
        ),
    ] = None,
) -> None:
    """Generate images from a text prompt."""
    # Checked here too so a bad flag fails before the slow model load.
    if steps is not None and MODELS[model].default_steps is None:
        raise typer.BadParameter(
            f"the {model} model uses the schedule stored in its checkpoint.", param_hint="--steps"
        )
    try:
        generator = ImageGenerator(
            model.value, placement=placement, quantization=quantize, models_dir=models_dir
        )
    except (GPUUnavailableError, ModelAccessError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Loaded {model} ({generator.placement})", err=True)
    typer.echo(generator.load_memory.summary(), err=True)

    generation = generator.generate(
        prompt, aspect=aspect.value, seed=seed, steps=steps, count=count
    )
    for result in generation.images:
        path = save_image(result, output_dir)
        typer.echo(f"{path} (seed {result.seed})")
    typer.echo(f"Peak during generation:\n{generation.memory.summary()}", err=True)
