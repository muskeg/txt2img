from unittest.mock import MagicMock

import diffusers
import pytest
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

from txt2img.config import MODELS
from txt2img.pipeline import ImageGenerator


@pytest.mark.parametrize("name", [n for n, p in MODELS.items() if p.scheduler])
def test_preset_schedulers_exist_in_diffusers(name: str) -> None:
    scheduler_cls, overrides = MODELS[name].scheduler
    scheduler = getattr(diffusers, scheduler_cls).from_config(
        EulerDiscreteScheduler().config, **overrides
    )

    assert all(scheduler.config[k] == v for k, v in overrides.items())


def test_lustify_uses_dpm_2m_sde_karras(fake_pipe: MagicMock) -> None:
    generator = ImageGenerator("lustify-v2")

    scheduler = generator.pipe.scheduler
    assert isinstance(scheduler, DPMSolverMultistepScheduler)
    assert scheduler.config.algorithm_type == "sde-dpmsolver++"
    assert scheduler.config.use_karras_sigmas is True


def test_presets_without_scheduler_keep_the_repo_one(fake_pipe: MagicMock) -> None:
    generator = ImageGenerator("epicrealism-xl-v8")

    assert isinstance(generator.pipe.scheduler, EulerDiscreteScheduler)
