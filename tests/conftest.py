from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from txt2img import pipeline
from txt2img.config import GiB


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Pretend there is a 32 GiB GPU with 2 GiB used by other apps. Tests can change the numbers."""
    state = SimpleNamespace(total=32 * GiB, free=30 * GiB, reserved=0, peak=0)
    cuda = pipeline.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "mem_get_info", lambda: (state.free, state.total))
    monkeypatch.setattr(cuda, "memory_reserved", lambda: state.reserved)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: state.peak)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: None)
    return state


@pytest.fixture
def fake_pipe(monkeypatch: pytest.MonkeyPatch, fake_cuda: SimpleNamespace) -> MagicMock:
    """Fake model loading: small real modules for memory accounting, tiny images as output."""
    pipe = MagicMock()
    pipe.components = {
        "transformer": torch.nn.Linear(64, 64),
        "vae": torch.nn.Linear(8, 8),
        "tokenizer": object(),
    }
    pipe.side_effect = lambda **kwargs: SimpleNamespace(images=[Image.new("RGB", (8, 8))])
    monkeypatch.setattr(pipeline.DiffusionPipeline, "from_pretrained", MagicMock(return_value=pipe))
    return pipe
