# txt2img

CLI and local web UI that generate images from text prompts using open-weight models through Hugging Face `diffusers`.
Update this file when a convention changes or a new pitfall turns up.

## Stack
- Python 3.12, managed with **uv** (`pyproject.toml` + `uv.lock`, no `requirements.txt`). If `uv` is missing, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Always run code inside the project virtual environment, through `uv run` or `uv add`. Never install packages with `pip` or into the system Python.
- The venv lives at `~/.venvs/txt2img` (ext4). The repo's `.venv` is a **symlink** to it, so uv finds it from any terminal; [.vscode/settings.json](.vscode/settings.json) also sets `UV_PROJECT_ENVIRONMENT`. If `.venv` is ever a real directory (uv creating packages on `/mnt/c`, "Failed to hardlink files" warning), delete it and run `ln -s ~/.venvs/txt2img .venv`.
- `diffusers`, `transformers`, `accelerate`, PyTorch (CUDA).
- CLI: Typer. Web UI: Gradio, bound to `127.0.0.1` only, as a thin layer over `pipeline.py`.
- For Typer choice options, use `StrEnum` types. `click_type=click.Choice` prints a traceback on bad input instead of a usage error.
- Lint/format: ruff. Tests: pytest.

## Commands
- `uv sync`: install dependencies
- `uv run txt2img "a prompt"`: generate an image (`--help` lists the options)
- `uv run txt2img-animate --base-prompt "..." -f "frame 1" -f "frame 2"`: masked animation (`--help` lists the options)
- `uv run txt2img-ui`: web UI at http://127.0.0.1:7860 (Generate and Animate tabs)
- `uv run pytest`: run the mocked tests. `uv run pytest -m gpu` runs the real-model test (SDXL, downloads ~7 GB)
- `uv run ruff check --fix . && uv run ruff format .`: lint and format

## Layout
- `src/txt2img/` (src layout):
  - [config.py](src/txt2img/config.py): model presets (repo, resolutions, steps, fixed call kwargs, quantizable components), placement/quantization enums and the models dir. This is the only place model IDs appear.
  - [pipeline.py](src/txt2img/pipeline.py): `ImageGenerator` (loads any preset via `DiffusionPipeline`, places it, generates, optionally with reference images), `MemoryReport` and `save_image`.
  - [animate.py](src/txt2img/animate.py): frame prompts, fitting the base to a preset size, mask compositing and saving WebP/APNG/GIF.
  - [cli.py](src/txt2img/cli.py) and [ui.py](src/txt2img/ui.py): thin front ends with no generation logic.
- Image `i` of a batch uses `seed + i`, so any single image can be reproduced. PNGs embed prompt, model, seed, steps, quantization and transparency as text metadata.
- Write generated images to `outputs/`, which is gitignored. Animations go to their own folder there (`base.png`, `frame_NN.png`, `animation.*`).

## Transparency and animation
- Only presets with `transparency=True` / `editing=True` (Qwen-Image-2.1 and Turbo) support them. Front ends check the flags before loading the model.
- Transparency wraps the prompt in `TRANSPARENT_TEMPLATE` (the official Qwen format). The VAE outputs RGBA natively, so don't post-process alpha.
- Each animation frame is a Qwen edit of the base, the previous frame, or both (`FrameSource`), pasted over the base through the feathered mask. The paste is what guarantees the area outside the mask stays unchanged, not the model.
- Animated WebP/APNG/GIF encoders merge identical consecutive frames, so use frames that differ in tests.

## Models
- Presets in [config.py](src/txt2img/config.py): Z-Image (+Turbo, default), Qwen-Image-2.1 (+Turbo), FLUX.1-schnell, SD 3.5 Large/Large Turbo/Medium, SDXL, RealVisXL V5, plus uncensored community fine-tunes (3 SDXL, 1 FLUX.1-dev). Take steps, guidance and resolutions from each model card, or the pipeline defaults when the card has none; a new model needs a preset, nothing else.
- `variant="fp16"` is fine when only some components ship fp16 files: diffusers loads the rest from the regular files.
- Qwen-Image-2.1 and Z-Image need **diffusers from git** (`[tool.uv.sources]`; `uv.lock` pins the commit) and `transformers>=5.17`. Qwen's Qwen3-VL text encoder also needs `torchvision`, which comes from the same CUDA index as `torch`.
- Qwen-Image-2.1-Turbo stores its sampling schedule in the checkpoint, so `num_inference_steps` does **not** change it (`default_steps=None`). Distilled models (Turbo, schnell) need guidance 0.
- Qwen uses 2048-class resolutions; the others use ~1 MP buckets. Don't use arbitrary sizes.
- FLUX.1-schnell and SD 3.5 are gated: `GatedRepoError` becomes `ModelAccessError` asking for `hf auth login`. Check each license (Qwen Research, Stability Community) before commercial use.
- Weights go to `TXT2IMG_MODELS_DIR` (`~/models` here, set in `~/.bashrc`) via `cache_dir`, or the HF cache when unset.

## VRAM placement
- `--placement auto` puts everything on the GPU if weights + `VRAM_HEADROOM` fit, else model CPU offload (whole components move per image at ~17 GB/s, cheap). Never use sequential offload.
- `--quantize 4bit` is bitsandbytes NF4 on the preset's `quantizable` components. bnb 8-bit failed with a cuBLAS error inside pipelines here, so it isn't offered.
- **WSL never raises CUDA OOM:** past the limit the driver silently spills VRAM into system RAM at ~3 GB/s (vs ~200 GB/s). `MemoryReport` detects it from `max_memory_reserved() + other apps > total` and warns. Keep that check when changing memory code.
- [`__init__.py`](src/txt2img/__init__.py) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch loads. Without it, offloaded Qwen fragmented to 33.4 GiB reserved for 26.9 GiB used and spilled. With it: 30.2 GiB, no spill.
- WSL has 45 GB of RAM, enough for one large model at a time, so the UI unloads the old model before loading another.

## Environment pitfalls
- **GPU is an RTX 5090 (Blackwell, sm_120, 32 GB).** It needs PyTorch wheels built for CUDA 12.8+. torch is pinned to the `https://download.pytorch.org/whl/cu132` index (`[[tool.uv.index]]` + `[tool.uv.sources]` in `pyproject.toml`). Don't switch to `cu126`, which has no sm_120 support. The cu128 index stopped at torch 2.11. Before you change the index, check that the driver supports the CUDA version (`nvidia-smi`, currently up to 13.4). If you see `no kernel image is available`, torch was built for the wrong CUDA version.
- After you change the torch CUDA index, run `uv sync --reinstall-package` for every `nvidia-*` and `cuda-*` package (or recreate the venv). Old and new NVIDIA wheels install to the same paths, so removing the old ones deletes the new ones' files (`libcudnn.so.9: cannot open shared object file`).
- **WSL:** the distro's disk lives on `D:\WSL\Ubuntu` (NVMe). The repo stays on `/mnt/c` so Windows can reach it, but file I/O there is slow (9P, ~0.3 GB/s vs ~1.3 GB/s on ext4). Keep the venv and model weights on ext4 (`~`). Don't move either one to `/mnt/*`.
- **Stale editors on `/mnt/c`:** VS Code doesn't notice files rewritten by terminal tools (`ruff format`, `uv add`), so open tabs keep old content. Saving them then fails ("content of the file is newer") or overwrites the tool's changes (this once dropped a dependency from `pyproject.toml`). Agents: re-read such files from disk (`cat`/`git diff`) before editing, verify with `git diff` after, and ask the user to close the tabs while you work.
- Gated models require `hf auth login`. Never commit tokens or `.env` files.
- On CUDA, load with `dtype=torch.bfloat16`. `torch_dtype=` is deprecated in this diffusers version.
- Tests must not download models or require a GPU. Use the `fake_pipe` and `fake_cuda` fixtures in [tests/conftest.py](tests/conftest.py). Put real-model tests under `@pytest.mark.gpu`, which `addopts` skips by default.
