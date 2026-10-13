import secrets
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from PIL import Image, ImageOps

from txt2img.pipeline import (
    GeneratedImage,
    ImageGenerator,
    MemoryReport,
    closest_aspect,
    composite,
    save_image,
)


class FrameSource(StrEnum):
    BASE = "base"
    PREVIOUS = "previous"
    BOTH = "both"


class AnimationFormat(StrEnum):
    WEBP = "webp"
    APNG = "apng"
    GIF = "gif"


_SAVE_OPTIONS: dict[AnimationFormat, dict] = {
    AnimationFormat.WEBP: {"quality": 90},
    AnimationFormat.APNG: {"format": "PNG"},
    AnimationFormat.GIF: {"disposal": 2},
}


@dataclass(frozen=True)
class Animation:
    base: Image.Image
    frames: list[GeneratedImage]
    memory: MemoryReport


def frame_prompts(
    prompts: list[str] | None = None, motion: str | None = None, count: int = 0
) -> list[str]:
    """Explicit per-frame prompts, or one motion prompt repeated `count` times."""
    prompts = [p.strip() for p in prompts or [] if p.strip()]
    motion = (motion or "").strip()
    if bool(prompts) == bool(motion):
        raise ValueError("Give either one prompt per frame or a motion prompt, not both.")
    if prompts:
        return prompts
    if count < 1:
        raise ValueError("frame count must be at least 1")
    return [motion] * count


def fit_to_preset(
    image: Image.Image, resolutions: dict[str, tuple[int, int]]
) -> tuple[str, Image.Image]:
    """Center-crop and resize to the preset resolution with the closest aspect ratio."""
    aspect = closest_aspect(image.size, resolutions)
    size = resolutions[aspect]
    if image.size != size:
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    return aspect, image


def animate(
    generator: ImageGenerator,
    base: Image.Image,
    prompts: list[str],
    *,
    mask: Image.Image | None = None,
    source: FrameSource = FrameSource.BASE,
    seed: int | None = None,
    steps: int | None = None,
    feather: int = 8,
    transparent: bool = False,
) -> Animation:
    """Edit `base` once per prompt; frame k uses seed + k."""
    if not prompts:
        raise ValueError("at least one frame prompt is required")
    if not generator.preset.editing:
        raise ValueError(f"The {generator.model} model can't follow edit instructions.")
    aspect, base = fit_to_preset(base.convert("RGBA"), generator.preset.resolutions)
    if mask is not None:
        mask = ImageOps.fit(mask.convert("L"), base.size, Image.Resampling.LANCZOS)
    first_seed = secrets.randbelow(2**32) if seed is None else seed

    frames, reports = [], []
    previous = base
    for k, prompt in enumerate(prompts):
        if source == FrameSource.BASE or previous is base:
            references = [base]
        elif source == FrameSource.PREVIOUS:
            references = [previous]
        else:
            references = [base, previous]
        generation = generator.generate(
            prompt,
            aspect=aspect,
            seed=first_seed + k,
            steps=steps,
            transparent=transparent,
            images=references,
        )
        result = generation.images[0]
        previous = composite(base, result.image, mask, feather)
        frames.append(replace(result, image=previous))
        reports.append(generation.memory)
    return Animation(base, frames, max(reports, key=lambda r: r.vram_used))


def save_animation(
    animation: Animation,
    output_dir: Path,
    *,
    fps: int = 8,
    formats: list[AnimationFormat] | None = None,
) -> list[Path]:
    """Save base and frames as PNGs plus one animation per format; the base is frame 0."""
    first = animation.frames[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = output_dir / f"{timestamp}_{first.model}_anim_{first.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    animation.base.save(folder / "base.png")
    for k, frame in enumerate(animation.frames, start=1):
        save_image(frame, folder, name=f"frame_{k:02d}.png")

    sequence = [animation.base, *(f.image for f in animation.frames)]
    paths = []
    for fmt in formats or [AnimationFormat.WEBP]:
        path = folder / f"animation.{fmt.value}"
        sequence[0].save(
            path,
            save_all=True,
            append_images=sequence[1:],
            duration=round(1000 / fps),
            loop=0,
            **_SAVE_OPTIONS[fmt],
        )
        paths.append(path)
    return paths
