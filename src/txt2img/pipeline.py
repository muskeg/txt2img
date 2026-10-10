import itertools
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, PipelineQuantizationConfig
from huggingface_hub.errors import GatedRepoError
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from txt2img.config import (
    DEFAULT_ASPECT,
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
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}, expected one of {sorted(MODELS)}")
        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "No CUDA GPU available. Check `nvidia-smi` and that torch was built for CUDA."
            )
        self.model = model
        self.preset = MODELS[model]
        self.quantization = Quantization(quantization)
        free, self._vram_total = torch.cuda.mem_get_info()
        self._vram_other = max(0, self._vram_total - free - torch.cuda.memory_reserved())
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.preset.repo_id,
                dtype=torch.bfloat16,
                cache_dir=models_dir,
                variant=self.preset.variant,
                quantization_config=_quantization_config(self.preset, self.quantization),
            )
        except GatedRepoError as e:
            raise ModelAccessError(
                f"{self.preset.repo_id} is gated. Accept its license at "
                f"https://huggingface.co/{self.preset.repo_id}, then run `hf auth login`."
            ) from e

        self.requested_placement = Placement(placement)
        self.placement = self._resolve_placement(self.requested_placement)
        if self.placement == Placement.GPU:
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()
        self.load_memory = self._memory_report(torch.cuda.memory_reserved())

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

    def generate(
        self,
        prompt: str,
        *,
        aspect: str = DEFAULT_ASPECT,
        seed: int | None = None,
        steps: int | None = None,
        count: int = 1,
        transparent: bool = False,
        images: list[Image.Image] | None = None,
    ) -> Generation:
        """Generate `count` images; image i uses seed + i so each one can be reproduced alone.

        `images` are reference images the model edits from (editing models only).
        """
        resolutions = self.preset.resolutions
        if aspect not in resolutions:
            raise ValueError(f"Unknown aspect {aspect!r}, expected one of {list(resolutions)}")
        if count < 1:
            raise ValueError("count must be at least 1")
        if transparent and not self.preset.transparency:
            raise ValueError(f"The {self.model} model can't generate transparent images.")
        if images and not self.preset.editing:
            raise ValueError(f"The {self.model} model doesn't accept reference images.")
        if steps is not None:
            if self.preset.default_steps is None:
                raise ValueError(
                    f"The {self.model} model uses the schedule stored in its checkpoint; "
                    "steps can't be changed."
                )
            if steps < 1:
                raise ValueError("steps must be at least 1")
        steps = steps or self.preset.default_steps
        width, height = resolutions[aspect]
        base_seed = secrets.randbelow(2**32) if seed is None else seed
        call_kwargs = dict(self.preset.call_kwargs)
        if steps is not None:
            call_kwargs["num_inference_steps"] = steps
        if images:
            call_kwargs["image"] = images
        model_prompt = (
            TRANSPARENT_TEMPLATE.format(prompt=prompt.strip().rstrip("."))
            if transparent
            else prompt
        )

        torch.cuda.reset_peak_memory_stats()
        images_out = []
        for i in range(count):
            image_seed = base_seed + i
            output = self.pipe(
                prompt=model_prompt,
                width=width,
                height=height,
                generator=torch.Generator("cpu").manual_seed(image_seed),
                **call_kwargs,
            )
            images_out.append(
                GeneratedImage(
                    image=output.images[0],
                    prompt=prompt,
                    model=self.model,
                    seed=image_seed,
                    steps=steps,
                    quantization=self.quantization,
                    transparent=transparent,
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
    result.image.save(path, pnginfo=info)
    return path
