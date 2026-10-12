from pathlib import Path

CHECKPOINTS = "checkpoints"
LORAS = "loras"


def list_local(kind: str, models_dir: Path | None) -> list[str]:
    """Names of the .safetensors files in `models_dir/kind`."""
    folder = models_dir / kind if models_dir else None
    if folder is None or not folder.is_dir():
        return []
    return sorted(p.name for p in folder.glob("*.safetensors"))


def resolve_local(value: str, kind: str, models_dir: Path | None) -> Path:
    """A file path, or a file name (with or without extension) in `models_dir/kind`."""
    candidates = [Path(value).expanduser()]
    if models_dir is not None:
        candidates += [models_dir / kind / value, models_dir / kind / f"{value}.safetensors"]
    for path in candidates:
        if path.is_file():
            if path.suffix != ".safetensors":
                # .ckpt/.bin/.pt files are pickles, which can run arbitrary code when loaded.
                raise ValueError(f"{path} isn't a .safetensors file.")
            return path
    where = models_dir / kind if models_dir else "the current directory"
    raise FileNotFoundError(f"{value!r} not found (looked in {where}).")


def parse_lora(spec: str) -> tuple[str, float]:
    """'name:0.8' -> ('name', 0.8). The strength defaults to 1.0."""
    name, sep, scale = spec.rpartition(":")
    if sep:
        try:
            return name, float(scale)
        except ValueError:
            pass
    return spec, 1.0
