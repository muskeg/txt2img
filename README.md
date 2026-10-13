# txt2img

Generate images from text prompts with open-weight models, locally on an NVIDIA GPU, using Hugging Face [diffusers](https://github.com/huggingface/diffusers). It comes with command-line tools and a local web UI.

## Features

- **18 model presets**: Z-Image and Z-Image-Turbo, Qwen-Image-2.1 and Qwen-Image-2.1-Turbo, FLUX.1-schnell, Stable Diffusion 3.5 (Large, Large Turbo, Medium), SDXL, RealVisXL V5, and eight uncensored community fine-tunes (photorealistic and anime SDXL, two Pony-based, one FLUX.1-dev). Each preset uses the steps, guidance, sampler, negative prompt and resolutions from its model card.
- **Your own checkpoints and LoRAs** (optional): load any SDXL or Pony `.safetensors` checkpoint (e.g. from Civitai) and stack LoRAs at chosen strengths, plus negative prompts and CFG control.
- **Automatic VRAM placement**: the whole model goes on the GPU when it fits; otherwise components are offloaded to system RAM and moved to the GPU only while they run. Optional 4-bit (NF4) quantization shrinks big models for smaller GPUs.
- **Memory report**: shows what share of the weights sits in VRAM vs RAM and the peak VRAM use, and warns when VRAM overflows into system RAM. On WSL that overflow is silent and roughly 70x slower.
- **Transparent images**: native RGBA output with the Qwen-Image-2.1 models.
- **Image editing**: start from your own images. The Qwen-Image-2.1 models follow edit instructions ("make it night"), combine up to 10 input images ("the woman from &lt;image1&gt; on the bench from &lt;image2&gt;") and cut subjects out onto a transparent background. Every other model can redraw an image by a chosen strength (image-to-image) or repaint a masked area (inpainting).
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
| `peft` | Loading LoRAs |
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
| `-m, --model` | `z-image-turbo` (`qwen-image-2.1-turbo` with `--image`, `sdxl` with `--checkpoint`) | Model preset (see [Models](#models)) |
| `-i, --image` | | Input image. Repeatable, up to 10 with Qwen-Image-2.1 (see [Editing](#editing-images)) |
| `--mask` | | Black-and-white image: only the white area of the first `--image` changes |
| `--strength` | 0.6 | How much other models redraw the `--image` (0-1) |
| `-a, --aspect` | `1:1` (`image1` with `--image`) | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`, or `imageN` to follow input image N |
| `--seed` | random | Image `i` uses `seed + i` |
| `--steps` | model's | Denoising steps (Qwen-Image-2.1-Turbo has a fixed schedule) |
| `-n, --count` | 1 | Number of images |
| `-N, --negative-prompt` | model's | What to avoid. `""` for none |
| `--cfg` | model's | Guidance (CFG) scale. Higher follows the prompt more closely |
| `--checkpoint` | | SDXL/Pony `.safetensors` file: a path, or a name in `<models-dir>/checkpoints` |
| `--lora` | | `name[:strength]` from `<models-dir>/loras` (or a path). Repeatable |
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

### Negative prompt, CFG, checkpoints and LoRAs

All optional. Distilled models (Turbo, schnell, Qwen) have fixed guidance and no negative prompt, and the CLI says so before loading anything.

```bash
# Steer away from things, and follow the prompt more strictly
uv run txt2img "a portrait photo, natural light" -m lustify-v2 -N "blurry, cartoon, watermark" --cfg 7

# Any SDXL/Pony checkpoint dropped in ~/models/checkpoints (sdxl settings by default)
uv run txt2img "a portrait photo" --checkpoint myModel_v10

# Same checkpoint with Pony prompt handling (adds the score tags and Pony negative prompt)
uv run txt2img "a portrait photo" -m pony-realism-v23 --checkpoint myPonyModel_v10

# LoRAs from ~/models/loras, each with its own strength
uv run txt2img "a portrait photo" -m lustify-v2 --lora detail_tweaker:0.6 --lora my_style:0.9
```

Put files in `<models-dir>/checkpoints` and `<models-dir>/loras` (`~/models/checkpoints` and `~/models/loras` with the example config). Only `.safetensors` files are accepted: `.ckpt`, `.pt` and `.bin` files can run arbitrary code when loaded. Pony-based models are trained on `score_9, score_8_up, …` tags, which the Pony presets add for you; add `rating_explicit` and similar tags to your prompt yourself. Each PNG records the negative prompt, CFG, checkpoint and LoRAs used.

### Transparent images

```bash
uv run txt2img "a cute cartoon candle sticker" -m qwen-image-2.1-turbo --transparent
```

The prompt is wrapped in the format recommended by the Qwen team, and the PNG is saved with an alpha channel.

### Editing images

With `--image`, every model starts from your images, in one of two ways.

**Qwen-Image-2.1 models** (`qwen-image-2.1-turbo` is picked by default with `--image`) follow the prompt as an edit instruction. Refer to the inputs as `<image1>`, `<image2>`… in the order you pass them.

```bash
# Edit a photo
uv run txt2img "change the sky to a stormy sunset" -i photo.jpg

# Put people or objects from several images into one picture, shaped like the beach photo
uv run txt2img "the man from <image1> and the dog from <image2> walking on the beach in <image3>" \
  -i man.jpg -i dog.png -i beach.jpg -a image3 -n 4

# Cut a subject out onto a transparent background
uv run txt2img "Extract the cup from <image1>." -i photo.jpg --transparent

# Change only the white area of the mask; the rest stays pixel-identical
uv run txt2img "Replace the cup with a glass of orange juice." -i photo.jpg --mask mask.png
```

With `--mask`, the area is outlined in red on the input image, the way Qwen-Image-2.1 marks local edits, and the result is pasted back over the original through the mask. Paint the mask a little larger than the object: anything outside it is restored from the original.

**Other models** don't understand instructions, so describe the picture you want. They redraw the single input image, keeping its composition and colors. Raise `--strength` to change more: 1 replaces it entirely. With `--mask`, they repaint only the white area (inpainting).

```bash
uv run txt2img "a portrait photo, natural light, film grain" -m realvisxl-v5 -i photo.jpg --strength 0.5
uv run txt2img "a tall glass of orange juice on a wooden table" -m realvisxl-v5 -i photo.jpg --mask mask.png --strength 0.9
```

Checkpoints, LoRAs, negative prompts and CFG work as for text-to-image. Input images can be any size: the output follows the first image's aspect ratio unless you pass `-a`. When it has to match image 1 (with a mask, or with other models), image 1 is center-cropped to that aspect ratio. Each PNG records the input file names, the mask and the strength in its metadata.


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

- **Generate**: everything `txt2img` does, with the memory report under the results. **Input images** takes image 1 (paint over it to change only that area), images 2 to 10 for Qwen-Image-2.1, and the strength for other models. **Use first result as image 1** lets you keep editing step by step. The **Advanced** section has the negative prompt (prefilled with the model's), CFG, a checkpoint dropdown and a LoRA multi-select with a strength slider. Click **Rescan** after adding files.
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
| `lustify-v2` | ~7 GB | yes | 30 | Photorealistic uncensored SDXL, DPM++ 2M SDE Karras sampler. License not stated |
| `epicrealism-xl-v8` | ~7 GB | yes | 50 | Photorealistic SDXL (unofficial re-upload). License not stated |
| `pony-realism-v23` | ~7 GB | yes | 30 | Photorealistic Pony SDXL, adds score tags. FAIPL-1.0-SD |
| `pornworks-pony-v04` | ~7 GB | yes | 30 | Photorealistic explicit Pony SDXL (single file), adds score tags. CreativeML Open RAIL++-M |

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
