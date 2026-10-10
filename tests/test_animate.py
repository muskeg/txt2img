from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from txt2img.animate import (
    AnimationFormat,
    FrameSource,
    animate,
    composite,
    fit_to_preset,
    frame_prompts,
    save_animation,
)
from txt2img.config import GiB
from txt2img.pipeline import ImageGenerator

RED, BLUE = (255, 0, 0, 255), (0, 0, 255, 255)


@pytest.fixture
def blue_frames(fake_pipe: MagicMock) -> MagicMock:
    """The fake model always returns a solid blue image."""
    fake_pipe.side_effect = lambda **kwargs: SimpleNamespace(
        images=[Image.new("RGBA", (8, 8), BLUE)]
    )
    return fake_pipe


def left_half_mask(size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, size[0] // 2, size[1]))
    return mask


def test_frame_prompts_per_frame_skips_blank_lines() -> None:
    assert frame_prompts(["flame left", "  ", "flame right "]) == ["flame left", "flame right"]


def test_frame_prompts_motion_repeats() -> None:
    assert frame_prompts(motion="flicker", count=3) == ["flicker"] * 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompts": ["a"], "motion": "b", "count": 2},
        {},
        {"motion": "b", "count": 0},
    ],
)
def test_frame_prompts_rejects_bad_combinations(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        frame_prompts(**kwargs)


def test_fit_to_preset_picks_closest_aspect() -> None:
    aspect, image = fit_to_preset(
        Image.new("RGBA", (200, 110)), {"1:1": (64, 64), "16:9": (128, 72)}
    )

    assert (aspect, image.size) == ("16:9", (128, 72))


def test_composite_keeps_base_outside_mask() -> None:
    base = Image.new("RGBA", (64, 64), RED)
    frame = Image.new("RGB", (8, 8), BLUE[:3])

    result = composite(base, frame, left_half_mask(base.size), feather=0)

    assert (result.getpixel((5, 30)), result.getpixel((60, 30))) == (BLUE, RED)


def test_composite_without_mask_uses_whole_frame() -> None:
    result = composite(Image.new("RGBA", (16, 16), RED), Image.new("RGBA", (4, 4), BLUE), None, 0)

    assert result.size == (16, 16)
    assert result.getpixel((15, 15)) == BLUE


def test_animate_edits_base_each_frame_with_consecutive_seeds(
    blue_frames: MagicMock, small_qwen: None
) -> None:
    base = Image.new("RGB", (64, 64), RED[:3])

    animation = animate(
        ImageGenerator("qwen-image-2.1-turbo"),
        base,
        ["a", "b", "c"],
        mask=left_half_mask((64, 64)),
        seed=10,
        feather=0,
    )

    assert [f.seed for f in animation.frames] == [10, 11, 12]
    assert all(c.kwargs["image"] == [animation.base] for c in blue_frames.call_args_list)
    for frame in animation.frames:
        assert (frame.image.getpixel((5, 30)), frame.image.getpixel((60, 30))) == (BLUE, RED)


@pytest.mark.parametrize(
    ("source", "second_call_refs"),
    [(FrameSource.PREVIOUS, ["frame1"]), (FrameSource.BOTH, ["base", "frame1"])],
)
def test_animate_frame_sources(
    blue_frames: MagicMock, small_qwen: None, source: FrameSource, second_call_refs: list[str]
) -> None:
    animation = animate(
        ImageGenerator("qwen-image-2.1-turbo"),
        Image.new("RGB", (64, 64)),
        ["a", "b"],
        source=source,
    )

    named = {"base": animation.base, "frame1": animation.frames[0].image}
    first, second = (c.kwargs["image"] for c in blue_frames.call_args_list)
    assert first == [animation.base]
    assert all(a is b for a, b in zip(second, [named[n] for n in second_call_refs], strict=True))


def test_animate_fits_base_and_mask_to_model_resolution(
    blue_frames: MagicMock, small_qwen: None
) -> None:
    base = Image.new("RGB", (256, 140), RED[:3])
    mask = left_half_mask(base.size)

    animation = animate(ImageGenerator("qwen-image-2.1-turbo"), base, ["a"], mask=mask, feather=0)

    assert animation.base.size == (128, 72)
    assert blue_frames.call_args.kwargs["width"] == 128
    assert animation.frames[0].image.getpixel((120, 36)) == RED


def test_animate_transparent_wraps_every_frame_prompt(
    blue_frames: MagicMock, small_qwen: None
) -> None:
    animate(
        ImageGenerator("qwen-image-2.1-turbo"), Image.new("RGBA", (64, 64)), ["a"], transparent=True
    )

    assert blue_frames.call_args.kwargs["prompt"].startswith("This is an RGBA image")


def test_animate_reports_the_highest_peak(
    blue_frames: MagicMock, small_qwen: None, fake_cuda: SimpleNamespace
) -> None:
    peaks = iter([5 * GiB, 9 * GiB, 7 * GiB])

    def pipe(**kwargs):
        fake_cuda.peak = next(peaks)
        return SimpleNamespace(images=[Image.new("RGBA", (8, 8), BLUE)])

    blue_frames.side_effect = pipe

    animation = animate(
        ImageGenerator("qwen-image-2.1-turbo"), Image.new("RGB", (64, 64)), ["a"] * 3
    )

    assert animation.memory.vram_used == 9 * GiB


def test_animate_requires_an_editing_model(fake_pipe: MagicMock) -> None:
    with pytest.raises(ValueError, match="reference images"):
        animate(ImageGenerator("sdxl"), Image.new("RGB", (64, 64)), ["a"])


def test_save_animation_writes_frames_and_all_formats(
    blue_frames: MagicMock, small_qwen: None, tmp_path: Path
) -> None:
    # Encoders merge identical consecutive frames, so make each frame a different blue.
    shades = iter([(0, 0, 250, 255), (0, 0, 220, 255)])
    blue_frames.side_effect = lambda **kwargs: SimpleNamespace(
        images=[Image.new("RGBA", (8, 8), next(shades))]
    )
    base = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    animation = animate(
        ImageGenerator("qwen-image-2.1-turbo"),
        base,
        ["flame left", "flame right"],
        mask=left_half_mask((64, 64)),
        seed=3,
        feather=0,
    )

    paths = save_animation(animation, tmp_path, fps=10, formats=list(AnimationFormat))

    folder = paths[0].parent
    assert folder.name.endswith("_qwen-image-2.1-turbo_anim_3")
    assert sorted(p.name for p in folder.glob("*.png")) == [
        "base.png",
        "frame_01.png",
        "frame_02.png",
    ]
    with Image.open(folder / "frame_02.png") as frame:
        assert frame.text["prompt"] == "flame right"
    for path in paths:
        with Image.open(path) as anim:
            assert anim.n_frames == 3
            anim.seek(1)
            frame = anim.convert("RGBA")
            # Left half animated, right half keeps the transparent base.
            assert frame.getpixel((5, 30))[2] > 200
            assert frame.getpixel((60, 30))[3] == 0
