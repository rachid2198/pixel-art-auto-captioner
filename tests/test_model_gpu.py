"""GPU tests for the HF backend (``JoyCaptionHFModel``) with the real model.

Marked ``@pytest.mark.gpu`` and protected by the dual guard (AGENTS.md §6.3):
they run only when CUDA is available **and** JoyCaption weights exist at
``$JOYCAPTION_MODEL_PATH`` (default ``./Models``). Everywhere else (the CPU
suite, the Linux sandbox) they are skipped without importing ``torch``.

Local run (RTX 3060, NF4)::

    ./.env/Scripts/python.exe -m pytest tests/test_model_gpu.py -q -m gpu

Per AGENTS.md §6.7 no test asserts bit-identical generated text.
"""

from __future__ import annotations

import os
from pathlib import Path

import PIL.Image
import pytest

from pixel_art_auto_captioner.common.types import CaptionRequest


def _cuda_available() -> bool:
    """``True`` if torch is installed and sees a CUDA device (torch imported lazily)."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_MODEL_PATH = Path(os.environ.get("JOYCAPTION_MODEL_PATH", "./Models"))
_MODEL_AVAILABLE = (_MODEL_PATH / "config.json").exists()
requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
requires_model = pytest.mark.skipif(not _MODEL_AVAILABLE, reason="JoyCaption weights not found")

pytestmark = [pytest.mark.gpu, requires_cuda, requires_model]


def _request(i: int = 0, *, max_new_tokens: int = 96, seed: int = 1234) -> CaptionRequest:
    image = PIL.Image.new("RGB", (512, 512), color=(40, 120, 200))
    for x in range(128, 384):  # a simple "sprite": a red square on blue
        for y in range(128, 384):
            image.putpixel((x, y), (220, 30, 30))
    return CaptionRequest(
        image_rel_path=Path(f"gpu-test/0001/{i:06d}.jpg"),
        caption_type="short",
        image=image,
        prompt="Write a short description for this image in one or two sentences.",
        gen_params={"max_new_tokens": max_new_tokens, "temperature": 0.6, "top_p": 0.9, "top_k": None},
        seed=seed,
    )


@pytest.fixture(scope="module")
def hf_model():
    """The real model, NF4 (fits a 12 GB card), loaded once for the module."""
    from pixel_art_auto_captioner.captioning.joycaption_hf import JoyCaptionHFModel

    model = JoyCaptionHFModel()
    try:
        model.load({"model_path": str(_MODEL_PATH), "dtype": "bfloat16", "hf": {"quantization": "nf4", "device_map": 0}})
        yield model
    finally:
        model.unload()


def test_loads_through_get_model() -> None:
    from pixel_art_auto_captioner.captioning import get_model
    from pixel_art_auto_captioner.captioning.joycaption_hf import JoyCaptionHFModel

    assert type(get_model("hf")) is JoyCaptionHFModel


def test_vision_tower_and_projector_not_quantized(hf_model) -> None:
    import torch

    model = hf_model._model
    quantized = {"Linear4bit", "Linear8bitLt"}
    vision_modules = [
        (name, type(module).__name__)
        for name, module in model.named_modules()
        if "vision_tower" in name or "multi_modal_projector" in name or name.endswith("lm_head")
    ]
    assert vision_modules, "vision tower / projector modules not found"
    assert not [n for n, t in vision_modules if t in quantized]
    language = [type(m).__name__ for n, m in model.named_modules() if "language_model" in n]
    assert any(t == "Linear4bit" for t in language), "language model should be NF4-quantized"
    for name, param in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            assert param.dtype.is_floating_point and param.dtype != torch.uint8, (name, param.dtype)


def test_caption_returns_text_and_stop(hf_model) -> None:
    result = hf_model.caption(_request())
    assert result.error is None
    assert result.finish_reason in ("stop", "length")
    assert result.text and result.text == result.text.strip()
    assert 0 < result.num_output_tokens <= 96


def test_short_limit_reports_length(hf_model) -> None:
    result = hf_model.caption(_request(max_new_tokens=4))
    assert result.finish_reason == "length"
    assert result.num_output_tokens == 4


def test_batch_returns_one_result_per_request_in_order(hf_model) -> None:
    requests = [_request(i, max_new_tokens=8, seed=100 + i) for i in range(3)]
    results = hf_model.caption_batch(requests)
    assert len(results) == 3
    assert all(r.finish_reason in ("stop", "length") and r.error is None for r in results)


def test_describe_reports_versions(hf_model) -> None:
    info = hf_model.describe()
    assert info["backend"] == "hf" and info["quantization"] == "nf4"
    assert info["model_path"] == str(_MODEL_PATH)
    assert info["torch_version"] and info["transformers_version"] and info["bitsandbytes_version"]


def test_unload_frees_memory_and_is_idempotent(hf_model) -> None:
    """Last test of the module: unloads the shared model (the fixture's own unload is then a no-op)."""
    import torch

    loaded_bytes = torch.cuda.memory_allocated()
    hf_model.unload()
    hf_model.unload()
    assert not hf_model.loaded
    assert torch.cuda.memory_allocated() < loaded_bytes
