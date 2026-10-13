import itertools
import math
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import diffusers
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    DiffusionPipeline,
    PipelineQuantizationConfig,
)
from huggingface_hub.errors import GatedRepoError
from PIL import Image, ImageChops, ImageFilter, ImageOps
from PIL.PngImagePlugin import PngInfo

from txt2img.config import (
    DEFAULT_ASPECT,
    DEFAULT_STRENGTH,
    EDIT_AREA_HINT,
    FOLLOW_ASPECTS,
    MASK_FEATHER,
    MAX_REFERENCES,
    MODELS,
    TRANSPARENT_TEMPLATE,
    VRAM_HEADROOM,
    GiB,
    ModelPreset,
    Placement,
    Quantization,
)


class GPUUnavailableError(RuntimeError):
    pass


class ModelAccessError(RuntimeError):
    pass


def gib(n: int) -> str:
    return f"{n / GiB:.1f} GiB"


def closest_aspect(size: tuple[int, int], resolutions: dict[str, tuple[int, int]]) -> str:
    ratio = size[0] / size[1]

    def distance(aspect: str) -> float:
        width, height = resolutions[aspect]
        return abs(math.log(width / height / ratio))

    return min(resolutions, key=distance)


def load_image(path: Path) -> Image.Image:
    """Open an image upright: phone photos store their rotation in EXIF."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image)


def composite(
    base: Image.Image, frame: Image.Image, mask: Image.Image | None, feather: int
) -> Image.Image:
    """Keep `base` outside the mask (white = changed), blending over `feather` pixels."""
    frame = frame.convert("RGBA")
    if frame.size != base.size:
        frame = frame.resize(base.size, Image.Resampling.LANCZOS)
    if mask is None:
        return frame
    soft = mask.convert("L").resize(base.size)
    if feather:
        soft = soft.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(frame, base.convert("RGBA"), soft)


def outline_area(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Draw a red outline just outside the mask's white area."""
    width = max(4, min(image.size) // 160)
    grown = mask.filter(ImageFilter.GaussianBlur(width)).point(lambda v: 255 if v > 16 else 0)
    marked = image.convert("RGBA")
    marked.paste((255, 0, 0, 255), mask=ImageChops.subtract(grown, mask))
    return marked


@dataclass(frozen=True)
class MemoryReport:
    gpu_bytes: int  # weights that stay in VRAM
    cpu_bytes: int  # weights that stay in RAM
    offloaded: bool
    vram_used: int  # reserved by this process: current after loading, peak after generating
    vram_other: int  # used by other processes when the model was loaded
    vram_total: int

    @property
    def gpu_percent(self) -> float:
        total = self.gpu_bytes + self.cpu_bytes
        return 100 * self.gpu_bytes / total if total else 0.0

    @property
    def overflow_bytes(self) -> int:
        return max(0, self.vram_used + self.vram_other - self.vram_total)

    def summary(self) -> str:
        used = self.vram_used + self.vram_other
        weights = (
            f"Weights: {gib(self.gpu_bytes)} in VRAM ({self.gpu_percent:.0f}%), "
            f"{gib(self.cpu_bytes)} in RAM ({100 - self.gpu_percent:.0f}%)"
        )
        if self.offloaded:
            weights += ", offloaded: each component moves to the GPU while it runs"
        lines = [
            weights,
            f"VRAM: {gib(used)} of {gib(self.vram_total)} ({100 * used / self.vram_total:.0f}%), "
            f"including {gib(self.vram_other)} used by other apps",
        ]
        if self.overflow_bytes:
            lines.append(
                f"WARNING: VRAM limit exceeded by {gib(self.overflow_bytes)}. The driver spills "
                "into system RAM, which is ~70x slower. "
                "Use 4-bit quantization or offload placement."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class GeneratedImage:
    image: Image.Image
    prompt: str
    model: str
    seed: int
    steps: int | None
    quantization: Quantization = Quantization.NONE
    transparent: bool = False
    negative_prompt: str = ""
    guidance: float | None = None
    # Local checkpoint file name and "name:strength" LoRAs, when used.
    checkpoint: str | None = None
    loras: tuple[str, ...] = ()
    # File names of the reference images; set by the front end, which knows them.
    references: tuple[str, ...] = ()
    mask: str | None = None
    # How much an input image was redrawn, for models without edit instructions.
    strength: float | None = None


@dataclass(frozen=True)
class Generation:
    images: list[GeneratedImage]
    memory: MemoryReport


def _weight_bytes(pipe: DiffusionPipeline) -> tuple[int, int]:
    """(bytes in VRAM, bytes in RAM) of all weights in the pipeline."""
    gpu = cpu = 0
    for component in pipe.components.values():
        if not isinstance(component, torch.nn.Module):
            continue
        for tensor in itertools.chain(component.parameters(), component.buffers()):
            size = tensor.numel() * tensor.element_size()
            if tensor.device.type == "cuda":
                gpu += size
            else:
                cpu += size
    return gpu, cpu


def _quantization_config(
    preset: ModelPreset, quantization: Quantization
) -> PipelineQuantizationConfig | None:
    if quantization == Quantization.NONE:
        return None
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=list(preset.quantizable),
    )


class ImageGenerator:
    """Loads one model once and generates images with it."""

    def __init__(
        self,
        model: str,
        *,
        placement: Placement = Placement.AUTO,
        quantization: Quantization = Quantization.NONE,
        models_dir: Path | None = None,
        checkpoint: Path | None = None,
        loras: list[tuple[Path, float]] | None = None,
    ) -> None:
        """`checkpoint` replaces the preset's weights with a local single-file model;
        `loras` are (file, strength) pairs applied on top."""
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}, expected one of {sorted(MODELS)}")
        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "No CUDA GPU available. Check `nvidia-smi` and that torch was built for CUDA."
            )
        self.model = model
        self.preset = MODELS[model]
        self.quantization = Quantization(quantization)
        if (checkpoint or self.preset.single_file) and not self.preset.checkpoint_pipeline:
            raise ValueError(f"The {model} model can't load checkpoints; use an SDXL-based model.")
        if (checkpoint or self.preset.single_file) and self.quantization != Quantization.NONE:
            raise ValueError("4-bit quantization isn't supported with single-file checkpoints.")
        self.checkpoint = checkpoint
        self.loras = list(loras or [])
        free, self._vram_total = torch.cuda.mem_get_info()
        self._vram_other = max(0, self._vram_total - free - torch.cuda.memory_reserved())
        try:
            self.pipe = self._load(models_dir)
        except GatedRepoError as e:
            raise ModelAccessError(
                f"{self.preset.repo_id} is gated. Accept its license at "
                f"https://huggingface.co/{self.preset.repo_id}, then run `hf auth login`."
            ) from e

        self.requested_placement = Placement(placement)
        if self.preset.scheduler:
            name, overrides = self.preset.scheduler
            scheduler_cls = getattr(diffusers, name)
            self.pipe.scheduler = scheduler_cls.from_config(self.pipe.scheduler.config, **overrides)
        if self.loras:
            names = [f"lora_{i}" for i in range(len(self.loras))]
            for name, (path, _) in zip(names, self.loras, strict=True):
                self.pipe.load_lora_weights(
                    str(path.parent), weight_name=path.name, adapter_name=name
                )
            self.pipe.set_adapters(names, adapter_weights=[scale for _, scale in self.loras])
        if tile := self.preset.vae_tile:
            # At 2K the full decode peaks above 30 GiB; tiles overlap by 1/8 to hide seams.
            self.pipe.vae.enable_tiling(
                tile_sample_min_height=tile,
                tile_sample_min_width=tile,
                tile_sample_stride_height=tile * 7 // 8,
                tile_sample_stride_width=tile * 7 // 8,
            )
        self.placement = self._resolve_placement(self.requested_placement)
        if self.placement == Placement.GPU:
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()
        self._derived: dict[type, DiffusionPipeline] = {}
        self.load_memory = self._memory_report(torch.cuda.memory_reserved())

    def _load(self, models_dir: Path | None) -> DiffusionPipeline:
        if self.checkpoint or self.preset.single_file:
            pipeline_cls = getattr(diffusers, self.preset.checkpoint_pipeline)
            source = (
                str(self.checkpoint)
                if self.checkpoint
                else (
                    f"https://huggingface.co/{self.preset.repo_id}/blob/main/{self.preset.single_file}"
                )
            )
            # from_single_file still takes the older torch_dtype argument.
            return pipeline_cls.from_single_file(
                source, torch_dtype=torch.bfloat16, cache_dir=models_dir
            )
        return DiffusionPipeline.from_pretrained(
            self.preset.repo_id,
            dtype=torch.bfloat16,
            cache_dir=models_dir,
            variant=self.preset.variant,
            quantization_config=_quantization_config(self.preset, self.quantization),
        )

    def _resolve_placement(self, placement: Placement) -> Placement:
        if placement != Placement.AUTO:
            return placement
        weights = sum(_weight_bytes(self.pipe))
        fits = weights + VRAM_HEADROOM <= self._vram_total - self._vram_other
        return Placement.GPU if fits else Placement.OFFLOAD

    def _memory_report(self, vram_used: int) -> MemoryReport:
        gpu, cpu = _weight_bytes(self.pipe)
        offloaded = self.placement == Placement.OFFLOAD
        return MemoryReport(gpu, cpu, offloaded, vram_used, self._vram_other, self._vram_total)

    def _derived_pipe(self, auto_cls: type) -> DiffusionPipeline:
        """Img2img or inpainting pipeline sharing the loaded weights."""
        if auto_cls not in self._derived:
            pipe = auto_cls.from_pipe(self.pipe)
            if self.placement == Placement.OFFLOAD:
                pipe.enable_model_cpu_offload()
            self._derived[auto_cls] = pipe
        return self._derived[auto_cls]

    def _resolve_aspect(self, aspect: str | None, images: list[Image.Image]) -> str:
        if aspect is None:
            aspect = FOLLOW_ASPECTS[0] if images else DEFAULT_ASPECT
        if aspect in FOLLOW_ASPECTS:
            index = FOLLOW_ASPECTS.index(aspect)
            if index >= len(images):
                raise ValueError(f"Aspect {aspect} needs at least {index + 1} input images.")
            return closest_aspect(images[index].size, self.preset.resolutions)
        if aspect not in self.preset.resolutions:
            raise ValueError(
                f"Unknown aspect {aspect!r}, expected one of {list(self.preset.resolutions)} "
                "or imageN"
            )
        return aspect

    def generate(
        self,
        prompt: str,
        *,
        aspect: str | None = None,
        seed: int | None = None,
        steps: int | None = None,
        count: int = 1,
        transparent: bool = False,
        images: list[Image.Image] | None = None,
        mask: Image.Image | None = None,
        strength: float | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
    ) -> Generation:
        """Generate `count` images; image i uses seed + i so each one can be reproduced alone.

        `images` are input images, called <image1>, <image2>... in the prompt. Editing models
        follow the prompt as an instruction; other models redraw the single input image by
        `strength` (0-1). `mask` (white = change) limits changes to that area of image 1.
        `aspect` is a ratio such as "16:9", or "imageN" to follow input image N; None follows
        image 1, or uses the default without images.
        `negative_prompt=None` uses the preset's default; "" uses none.
        """
        images = list(images or [])
        editing = self.preset.editing
        if count < 1:
            raise ValueError("count must be at least 1")
        if transparent and not self.preset.transparency:
            raise ValueError(f"The {self.model} model can't generate transparent images.")
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} input images are supported.")
        if len(images) > 1 and not editing:
            raise ValueError(
                f"The {self.model} model takes one input image; "
                "use a Qwen-Image-2.1 model to combine several."
            )
        if mask is not None and (not images or mask.convert("L").getbbox() is None):
            raise ValueError("The mask needs an input image and some white area.")
        if strength is not None:
            if editing or not images:
                raise ValueError(
                    "strength only applies to models without edit instructions, with an image."
                )
            if not 0 < strength <= 1:
                raise ValueError("strength must be between 0 and 1")
        if negative_prompt and not self.preset.supports_negative:
            raise ValueError(f"The {self.model} model doesn't take a negative prompt.")
        if guidance is not None and not self.preset.adjustable_guidance:
            raise ValueError(f"The {self.model} model's guidance is fixed.")
        if steps is not None:
            if self.preset.default_steps is None:
                raise ValueError(
                    f"The {self.model} model uses the schedule stored in its checkpoint; "
                    "steps can't be changed."
                )
            if steps < 1:
                raise ValueError("steps must be at least 1")
        steps = steps or self.preset.default_steps
        width, height = self.preset.resolutions[self._resolve_aspect(aspect, images)]
        base_seed = secrets.randbelow(2**32) if seed is None else seed
        call_kwargs = dict(self.preset.call_kwargs)
        if steps is not None:
            call_kwargs["num_inference_steps"] = steps
        if guidance is not None:
            call_kwargs["guidance_scale"] = guidance
        negative = self.preset.default_negative if negative_prompt is None else negative_prompt
        if negative and self.preset.supports_negative:
            call_kwargs["negative_prompt"] = negative
        prefix = self.preset.prompt_prefix
        model_prompt = f"{prefix}, {prompt}" if prefix else prompt

        pipe, canvas = self.pipe, None
        if images and (mask is not None or not editing):
            # The output is pasted over image 1 or redrawn from it, so it must be the same size.
            canvas = ImageOps.fit(images[0], (width, height), Image.Resampling.LANCZOS)
        if mask is not None:
            mask = mask.convert("L").resize(images[0].size)
            mask = ImageOps.fit(mask, (width, height), Image.Resampling.LANCZOS)
            mask = mask.point(lambda v: 255 if v >= 128 else 0)
        if images and editing:
            if mask is not None:
                images[0] = outline_area(canvas, mask)
                model_prompt = f"{model_prompt.strip()} {EDIT_AREA_HINT}"
            call_kwargs["image"] = images
        elif images:
            pipe = self._derived_pipe(
                AutoPipelineForImage2Image if mask is None else AutoPipelineForInpainting
            )
            strength = DEFAULT_STRENGTH if strength is None else strength
            call_kwargs.update(image=canvas.convert("RGB"), strength=strength)
            if mask is not None:
                call_kwargs["mask_image"] = mask
            if steps is not None:
                # Img2img runs only the last `strength` share of the steps; keep the model's count.
                call_kwargs["num_inference_steps"] = math.ceil(steps / strength)
        if transparent:
            model_prompt = TRANSPARENT_TEMPLATE.format(prompt=model_prompt.strip().rstrip("."))

        torch.cuda.reset_peak_memory_stats()
        images_out = []
        for i in range(count):
            image_seed = base_seed + i
            output = pipe(
                prompt=model_prompt,
                width=width,
                height=height,
                generator=torch.Generator("cpu").manual_seed(image_seed),
                **call_kwargs,
            )
            image = output.images[0]
            if mask is not None:
                image = composite(canvas, image, mask, MASK_FEATHER)
            images_out.append(
                GeneratedImage(
                    image=image,
                    prompt=prompt,
                    model=self.model,
                    seed=image_seed,
                    steps=steps,
                    quantization=self.quantization,
                    transparent=transparent,
                    negative_prompt=call_kwargs.get("negative_prompt", ""),
                    guidance=call_kwargs.get("guidance_scale"),
                    checkpoint=self.checkpoint.name if self.checkpoint else None,
                    loras=tuple(f"{path.stem}:{scale:g}" for path, scale in self.loras),
                    strength=call_kwargs.get("strength"),
                )
            )
        return Generation(images_out, self._memory_report(torch.cuda.max_memory_reserved()))


def save_image(result: GeneratedImage, output_dir: Path, name: str | None = None) -> Path:
    """Save as PNG with the generation parameters embedded as text metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / (name or f"{timestamp}_{result.model}_{result.seed}.png")
    info = PngInfo()
    info.add_text("prompt", result.prompt)
    info.add_text("model", MODELS[result.model].repo_id)
    info.add_text("seed", str(result.seed))
    if result.steps is not None:
        info.add_text("steps", str(result.steps))
    if result.quantization != Quantization.NONE:
        info.add_text("quantization", result.quantization.value)
    if result.transparent:
        info.add_text("transparent", "true")
    if result.negative_prompt:
        info.add_text("negative_prompt", result.negative_prompt)
    if result.guidance is not None:
        info.add_text("guidance", f"{result.guidance:g}")
    if result.checkpoint:
        info.add_text("checkpoint", result.checkpoint)
    if result.loras:
        info.add_text("loras", ", ".join(result.loras))
    if result.references:
        info.add_text("references", ", ".join(result.references))
    if result.mask:
        info.add_text("mask", result.mask)
    if result.strength is not None:
        info.add_text("strength", f"{result.strength:g}")
    result.image.save(path, pnginfo=info)
    return path
