# txt2img

Generate images from text prompts with open-weight models, locally on an NVIDIA GPU, using Hugging Face [diffusers](https://github.com/huggingface/diffusers). It comes with command-line tools and a local web UI.

## Features

- **14 model presets**: Z-Image and Z-Image-Turbo, Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, FLUX.1-schnell, Stable Diffusion 3.5 (Large, Large Turbo, Medium), SDXL, RealVisXL V5, and four uncensored community fine-tunes (three SDXL, one FLUX.1-dev). Each preset uses the steps, guidance and resolutions from its model card.
- **Automatic VRAM placement**: the whole model goes on the GPU when it fits; otherwise components are offloaded to system RAM and moved to the GPU only while they run. Optional 4-bit (NF4) quantization shrinks big models for smaller GPUs.
- **Memory report**: shows what share of the weights sits in VRAM vs RAM and the peak VRAM use, and warns when VRAM overflows into system RAM. On WSL that overflow is silent and roughly 70x slower.
- **Transparent images**: native RGBA output with the Qwen-Image-2.1 models.
- **Masked animation**: animate part of an image (a flame, an eye, a flag) while everything outside the mask stays identical to the base. Output is animated WebP, APNG or GIF.
- **Reproducible**: image `i` of a batch uses seed `seed + i`, and every PNG stores its prompt, model, seed and settings as metadata.

## Requirements

- Linux or WSL2 with an NVIDIA GPU, and a driver that supports **CUDA 13.2** (`nvidia-smi` shows the maximum CUDA version). Tested on an RTX 5090 (32 GB) under WSL2.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and every dependency into a virtual environment.
- Disk space for the weights: about 7 GB (SDXL) to 33 GB (Qwen-Image-2.1, FLUX.1-schnell) per model.
- System RAM at least as large as the model when it's offloaded (about 32 GB for Qwen-Image-2.1).
- A [Hugging Face](https://huggingface.co) account for the gated models (FLUX.1-schnell and SD 3.5).

### Dependencies

The exact versions are pinned in `uv.lock`.

| Package | Role |
|---|---|
| `torch`, `torchvision` | PyTorch from the CUDA 13.2 wheel index, which Blackwell GPUs (sm_120) need |
| `diffusers` | Model pipelines. Installed from git, because Qwen-Image-2.1 and Z-Image need unreleased code |
| `transformers`, `accelerate` | Text encoders and CPU offload |
| `bitsandbytes` | 4-bit NF4 quantization |
| `typer` | Command line |
| `gradio` | Web UI |
| `pillow` | Image I/O and animations |
| `pytest`, `ruff` | Tests and lint/format (dev only) |

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd txt2img
uv sync
```

`uv sync` creates the virtual environment and installs everything (several GB, mostly PyTorch and CUDA libraries).

### Configuration

| Variable | Purpose |
|---|---|
| `TXT2IMG_MODELS_DIR` | Where model weights are downloaded. Defaults to the Hugging Face cache (`~/.cache/huggingface`). Can be overridden per command with `--models-dir`. |
| `UV_PROJECT_ENVIRONMENT` | Optional: put the virtual environment outside the repo (but see the WSL tip below). |

```bash
# Example for ~/.bashrc
export TXT2IMG_MODELS_DIR="$HOME/models"
```

To use gated models, accept their license on their Hugging Face page, then log in once:

```bash
uv run hf auth login
```

On WSL, keep the virtual environment and model weights on the Linux filesystem (`~`), not under `/mnt/c` or `/mnt/d`. Reading files there is about 4x slower, which you'll notice every time a model loads. If the repo itself is under `/mnt/c`, point `.venv` at a Linux-side environment. Then every terminal finds it, without any environment variable:

```bash
uv venv ~/.venvs/txt2img && ln -s ~/.venvs/txt2img .venv && uv sync
```

## Usage

Weights are downloaded the first time a model is used. After that, a model loads in seconds.

### Generate images

```bash
uv run txt2img "a red fox in the snow, photo"
uv run txt2img "a lighthouse at dusk" -m qwen-image-2.1 -a 16:9 --seed 42 -n 4
uv run txt2img "a castle on a cliff" -m sdxl --steps 30 --quantize 4bit
```

| Option | Default | Description |
|---|---|---|
| `-m, --model` | `z-image-turbo` | Model preset (see [Models](#models)) |
| `-a, --aspect` | `1:1` | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16` |
| `--seed` | random | Image `i` uses `seed + i` |
| `--steps` | model's | Denoising steps (Qwen-Image-2.1-Turbo has a fixed schedule) |
| `-n, --count` | 1 | Number of images |
| `--transparent` | off | Transparent background (Qwen-Image-2.1 models only) |
| `--placement` | `auto` | `auto`, `gpu` or `offload` |
| `--quantize` | `none` | `none` or `4bit` |
| `-o, --output-dir` | `outputs` | Where images are saved |
| `--models-dir` | `$TXT2IMG_MODELS_DIR` | Where weights are stored |

Image paths are printed on stdout. The memory report goes to stderr:

```text
Loaded sdxl (gpu)
Weights: 6.5 GiB in VRAM (100%), 0.0 GiB in RAM (0%)
VRAM: 8.5 GiB of 31.8 GiB (27%), including 1.6 GiB used by other apps
outputs/20261010-134613_sdxl_1.png (seed 1)
Peak during generation:
Weights: 6.5 GiB in VRAM (100%), 0.0 GiB in RAM (0%)
VRAM: 13.0 GiB of 31.8 GiB (41%), including 1.6 GiB used by other apps
```

If the peak goes over the limit, the report adds a `WARNING: VRAM limit exceeded` line. Switch to `--quantize 4bit` or `--placement offload` when that happens.

### Transparent images

```bash
uv run txt2img "a cute cartoon candle sticker" -m qwen-image-2.1-turbo --transparent
```

The prompt is wrapped in the format recommended by the Qwen team, and the PNG is saved with an alpha channel.

### Animation

Each frame is an edit of the base image, made with a Qwen-Image-2.1 model. It's then pasted over the base through the mask, so only the masked area moves.

```bash
# Generate a base image, then 3 frames described one by one
uv run txt2img-animate --base-prompt "a candle on a wooden table" \
  -f "the flame leans to the left" \
  -f "the flame is tall and bright" \
  -f "the flame leans to the right" \
  --format webp --format gif

# Animate an existing image, only where the mask is white, with one motion prompt
uv run txt2img-animate --base photo.png --mask mask.png \
  --motion "the flag waves in the wind" --frames 8 --fps 10
```

| Option | Default | Description |
|---|---|---|
| `--base` / `--base-prompt` | | An existing image, or a prompt to generate the base (exactly one is required) |
| `-f, --frame` | | Prompt for one frame. Repeat it once per frame |
| `--motion`, `--frames` | `8` frames | One prompt used for every frame (instead of `-f`) |
| `--mask` | whole image | Black-and-white image: white = animated area |
| `--source` | `base` | Edit each frame from the `base`, the `previous` frame, or `both`. `base` doesn't drift; `previous` gives smoother but drifting motion |
| `--feather` | 8 | Mask edge softness, in pixels |
| `--fps` | 8 | Animation speed |
| `--format` | `webp` | `webp`, `apng` or `gif`. Repeat for several |

`--model`, `--aspect`, `--seed`, `--steps`, `--transparent`, `--placement`, `--quantize` and `--models-dir` work as for `txt2img`. Each animation goes into its own folder under `outputs/`, which holds `base.png`, `frame_01.png`… and `animation.webp`/`.apng`/`.gif`. The base is the first frame of the animation.

### Web UI

```bash
uv run txt2img-ui
```

Then open http://127.0.0.1:7860. The UI listens on localhost only. On WSL, the Windows browser can reach it at the same address. It has two tabs:

- **Generate**: everything `txt2img` does, with the memory report under the results.
- **Animate**: upload a base image or generate one, paint the area to animate with the white brush, enter frame prompts or a motion prompt, then click **Animate**.

The UI keeps one model loaded between generations. It unloads it when you switch models or change placement or quantization.

## Models

| Preset | Size (bf16) | Fits in 32 GB | Default steps | Notes |
|---|---|---|---|---|
| `z-image-turbo` (default) | 20.5 GB | yes | 9 | Apache-2.0, fast |
| `z-image` | 20.5 GB | yes | 50 | Apache-2.0 |
| `qwen-image-2.1` | 32.4 GB | no (offloads) | 40 | Transparency, animation. Qwen Research License |
| `qwen-image-2.1-turbo` | 32.4 GB | no (offloads) | fixed (8) | Transparency, animation. Qwen Research License |
| `flux.1-schnell` | ~33.7 GB | no (offloads) | 4 | Gated, Apache-2.0 |
| `sd3.5-large` | ~27.5 GB | yes | 28 | Gated, Stability Community License |
| `sd3.5-large-turbo` | ~27.5 GB | yes | 4 | Gated, Stability Community License |
| `sd3.5-medium` | ~16 GB | yes | 40 | Gated, Stability Community License |
| `sdxl` | 7.1 GB | yes | 50 | CreativeML Open RAIL++-M |
| `realvisxl-v5` | 6.9 GB | yes | 50 | Photorealistic SDXL fine-tune, CreativeML Open RAIL++-M |
| `heartsync-nsfw` | ~7 GB | yes | 30 | Uncensored SDXL fine-tune, MIT |
| `unfiltered-nsfw-v2` | ~7 GB | yes | 50 | Uncensored SDXL fine-tune, custom license (see model card) |
| `wai-nsfw-illustrious-v8` | 6.9 GB | yes | 50 | Uncensored anime SDXL (Illustrious), FAIPL-1.0-SD |
| `fhdr-uncensored` | ~33.7 GB | no (offloads) | 40 | Uncensored FLUX.1-dev fine-tune. Gated, FLUX.1-dev non-commercial |

Qwen models render at about 4 megapixels (2048×2048 for 1:1). The other presets render at about 1 megapixel (1024×1024 for 1:1). Check each model's license before any commercial use.

## Development

```bash
uv run pytest                 # tests with a fake pipeline: no GPU, no downloads
uv run pytest -m gpu          # real SDXL run (downloads ~7 GB)
uv run ruff check --fix . && uv run ruff format .
```

The code lives in `src/txt2img/`:
- [config.py](src/txt2img/config.py): model presets.
- [pipeline.py](src/txt2img/pipeline.py): loading, placement and generation.
- [animate.py](src/txt2img/animate.py): animation.
- [cli.py](src/txt2img/cli.py) and [ui.py](src/txt2img/ui.py): thin front ends.

Adding a model only takes a new preset in `config.py`. [AGENTS.md](AGENTS.md) lists the conventions and environment pitfalls.
