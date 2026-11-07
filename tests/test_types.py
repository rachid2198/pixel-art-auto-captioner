"""Tests for ``common/types.py`` (SPEC v1.1 §3).

Covers construction, defaults, field order (the JSONL key contract, SPEC §8.3),
required fields, mutable-state isolation, the value-set constants, the
no-heavy-imports rule (SPEC §2.2) and POSIX rendering of relative paths.
"""

import copy
import dataclasses
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pytest
from PIL import Image

from pixel_art_auto_captioner.common.types import (
    CAPTION_FLAGS,
    CAPTION_STATUSES,
    FINISH_REASONS,
    CaptionRecord,
    CaptionRequest,
    CaptionResult,
    ImageRecord,
    LoadFailure,
)

# ---------------------------------------------------------------------------
# Expected field orders (SPEC §3). Order is part of the contract.
# ---------------------------------------------------------------------------

IMAGE_RECORD_FIELDS = ["path", "rel_path", "source", "stem", "image", "width", "height"]
LOAD_FAILURE_FIELDS = ["path", "rel_path", "source", "error"]
CAPTION_REQUEST_FIELDS = [
    "image_rel_path", "caption_type", "image", "prompt", "gen_params", "seed", "attempt",
]
CAPTION_RESULT_FIELDS = ["text", "finish_reason", "num_output_tokens", "error"]
CAPTION_RECORD_FIELDS = [
    "image_rel_path", "image_path", "source", "category", "caption_type",
    "caption_text", "status", "flags", "model_name", "model_path",
    "model_revision", "backend", "prompt", "generation_params", "seed",
    "attempt", "finish_reason", "num_output_tokens", "image_width",
    "image_height", "run_id", "shard", "timestamp_utc", "extra",
]

REL = Path("celeste/0001/000001.jpg")
ABS = Path("/data/pixel-art/celeste/0001/000001.jpg")
GEN_PARAMS = {"max_new_tokens": 512, "temperature": 0.6, "top_p": 0.9, "top_k": None}


def _image() -> Image.Image:
    return Image.new("RGB", (64, 48), color=(10, 20, 30))


def _caption_record_kwargs(**overrides) -> dict:
    """All ``CaptionRecord`` fields with realistic values (fresh mutables)."""
    kwargs = {
        "image_rel_path": REL,
        "image_path": ABS,
        "source": "celeste",
        "category": "game_crop",
        "caption_type": "long",
        "caption_text": "A pixel-art character stands on a snowy ledge.",
        "status": "ok",
        "flags": [],
        "model_name": "joycaption-beta-one",
        "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "model_revision": None,
        "backend": "hf",
        "prompt": "Write a long descriptive caption for this image.",
        "generation_params": dict(GEN_PARAMS),
        "seed": 987654321,
        "attempt": 1,
        "finish_reason": "stop",
        "num_output_tokens": 12,
        "image_width": 64,
        "image_height": 48,
        "run_id": "20261107-120000-abc123",
        "shard": "0/1",
        "timestamp_utc": "2026-11-07T12:00:00Z",
        "extra": {"source_name": "Celeste"},
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 1. Construction with all fields
# ---------------------------------------------------------------------------


def test_image_record_construction():
    img = _image()
    rec = ImageRecord(
        path=ABS, rel_path=REL, source="celeste", stem="000001",
        image=img, width=64, height=48,
    )
    assert rec.path == ABS
    assert rec.rel_path == REL
    assert rec.source == "celeste"
    assert rec.stem == "000001"
    assert rec.image is img
    assert rec.image.mode == "RGB"
    assert (rec.width, rec.height) == (64, 48)


def test_image_record_root_level_source_is_empty_string():
    """Files directly in ``input_dir`` carry ``source == ""`` (SPEC §3)."""
    rec = ImageRecord(
        path=Path("/data/pixel-art/loose.png"), rel_path=Path("loose.png"),
        source="", stem="loose", image=_image(), width=64, height=48,
    )
    assert rec.source == ""


def test_load_failure_construction():
    rec = LoadFailure(
        path=ABS, rel_path=REL, source="celeste",
        error="UnidentifiedImageError: cannot identify image file",
    )
    assert rec.path == ABS
    assert rec.rel_path == REL
    assert rec.source == "celeste"
    assert rec.error == "UnidentifiedImageError: cannot identify image file"


def test_caption_request_construction():
    img = _image()
    req = CaptionRequest(
        image_rel_path=REL, caption_type="long", image=img,
        prompt="Describe this image.", gen_params=dict(GEN_PARAMS),
        seed=42, attempt=2,
    )
    assert req.image_rel_path == REL
    assert req.caption_type == "long"
    assert req.image is img
    assert req.prompt == "Describe this image."
    assert req.gen_params == GEN_PARAMS
    assert req.seed == 42
    assert req.attempt == 2


def test_caption_result_construction():
    res = CaptionResult(
        text="", finish_reason="error", num_output_tokens=0,
        error="RuntimeError: CUDA out of memory",
    )
    assert res.text == ""
    assert res.finish_reason == "error"
    assert res.num_output_tokens == 0
    assert res.error == "RuntimeError: CUDA out of memory"


def test_caption_record_construction():
    kwargs = _caption_record_kwargs()
    rec = CaptionRecord(**kwargs)
    for name, value in kwargs.items():
        assert getattr(rec, name) == value, name


def test_caption_record_accepts_revision_and_flags():
    rec = CaptionRecord(**_caption_record_kwargs(
        model_revision="a1b2c3d4", status="flagged", flags=["truncated"],
        finish_reason="length", attempt=2,
    ))
    assert rec.model_revision == "a1b2c3d4"
    assert rec.status == "flagged"
    assert rec.flags == ["truncated"]


# ---------------------------------------------------------------------------
# 2. Defaults
# ---------------------------------------------------------------------------


def test_caption_request_attempt_defaults_to_1():
    req = CaptionRequest(
        image_rel_path=REL, caption_type="long", image=_image(),
        prompt="p", gen_params={}, seed=1,
    )
    assert req.attempt == 1


def test_caption_result_error_defaults_to_none():
    res = CaptionResult(text="A caption.", finish_reason="stop", num_output_tokens=3)
    assert res.error is None


def test_caption_record_has_no_defaults():
    """Every ``CaptionRecord`` field is required (SPEC §3)."""
    for f in dataclasses.fields(CaptionRecord):
        assert f.default is dataclasses.MISSING, f.name
        assert f.default_factory is dataclasses.MISSING, f.name


@pytest.mark.parametrize("cls", [ImageRecord, LoadFailure])
def test_image_and_failure_records_have_no_defaults(cls):
    for f in dataclasses.fields(cls):
        assert f.default is dataclasses.MISSING, f.name
        assert f.default_factory is dataclasses.MISSING, f.name


# ---------------------------------------------------------------------------
# 3. Field order (JSONL key order, SPEC §8.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (ImageRecord, IMAGE_RECORD_FIELDS),
        (LoadFailure, LOAD_FAILURE_FIELDS),
        (CaptionRequest, CAPTION_REQUEST_FIELDS),
        (CaptionResult, CAPTION_RESULT_FIELDS),
        (CaptionRecord, CAPTION_RECORD_FIELDS),
    ],
)
def test_field_order(cls, expected):
    assert [f.name for f in dataclasses.fields(cls)] == expected


def test_caption_record_asdict_key_order():
    """``asdict`` keys follow field order, which the JSONL writer relies on."""
    d = dataclasses.asdict(CaptionRecord(**_caption_record_kwargs()))
    assert list(d) == CAPTION_RECORD_FIELDS


# ---------------------------------------------------------------------------
# 4. image_stem removed from CaptionRecord
# ---------------------------------------------------------------------------


def test_caption_record_has_no_image_stem():
    names = {f.name for f in dataclasses.fields(CaptionRecord)}
    assert "image_stem" not in names
    assert "prompt_template" not in names  # renamed to ``prompt`` in v1.1


def test_caption_record_rejects_image_stem_kwarg():
    with pytest.raises(TypeError):
        CaptionRecord(**_caption_record_kwargs(image_stem="000001"))


def test_image_record_keeps_stem():
    assert "stem" in {f.name for f in dataclasses.fields(ImageRecord)}


# ---------------------------------------------------------------------------
# 5. Missing required fields raise TypeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["flags", "extra", "status", "model_revision", "shard"])
def test_caption_record_missing_field_raises(missing):
    kwargs = _caption_record_kwargs()
    del kwargs[missing]
    with pytest.raises(TypeError, match=missing):
        CaptionRecord(**kwargs)


def test_image_record_missing_source_raises():
    with pytest.raises(TypeError, match="source"):
        ImageRecord(path=ABS, rel_path=REL, stem="000001", image=_image(), width=64, height=48)


def test_load_failure_missing_error_raises():
    with pytest.raises(TypeError, match="error"):
        LoadFailure(path=ABS, rel_path=REL, source="celeste")


def test_caption_request_missing_seed_raises():
    with pytest.raises(TypeError, match="seed"):
        CaptionRequest(
            image_rel_path=REL, caption_type="long", image=_image(),
            prompt="p", gen_params={},
        )


# ---------------------------------------------------------------------------
# 6. No shared mutable state
# ---------------------------------------------------------------------------


def test_caption_records_do_not_share_flags_or_extra():
    a = CaptionRecord(**_caption_record_kwargs(flags=[], extra={}))
    b = CaptionRecord(**_caption_record_kwargs(flags=[], extra={}))
    a.flags.append("truncated")
    a.extra["latency_s"] = 1.9
    assert b.flags == []
    assert b.extra == {}


def test_caption_record_copy_is_independent():
    a = CaptionRecord(**_caption_record_kwargs())
    b = copy.deepcopy(a)
    b.flags.append("repetitive")
    assert a.flags == []


# ---------------------------------------------------------------------------
# 7. Constants
# ---------------------------------------------------------------------------


def test_caption_statuses():
    assert CAPTION_STATUSES == frozenset({"ok", "flagged", "failed"})
    assert isinstance(CAPTION_STATUSES, frozenset)


def test_caption_flags():
    assert CAPTION_FLAGS == frozenset({"truncated", "repetitive", "empty", "error"})
    assert isinstance(CAPTION_FLAGS, frozenset)


def test_finish_reasons():
    assert FINISH_REASONS == frozenset({"stop", "length", "error"})
    assert isinstance(FINISH_REASONS, frozenset)


def test_new_names_exported_from_common():
    import pixel_art_auto_captioner.common as common

    for name in (
        "ImageRecord", "LoadFailure", "CaptionRequest", "CaptionResult",
        "CaptionRecord", "CAPTION_STATUSES", "CAPTION_FLAGS", "FINISH_REASONS",
    ):
        assert name in common.__all__
        assert hasattr(common, name)


# ---------------------------------------------------------------------------
# 8. No heavy imports (SPEC §2.2)
# ---------------------------------------------------------------------------


def test_types_import_pulls_in_no_torch_or_vllm():
    """Import ``types`` in a clean interpreter: no torch, no vllm.

    ``PYTHONPATH`` is replaced (not extended) with the repo's ``src`` so that
    any test-only torch stub on the parent's path is not visible.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(src)
    code = (
        "import sys, pixel_art_auto_captioner.common.types; "
        "assert 'torch' not in sys.modules and 'vllm' not in sys.modules, "
        "sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'vllm'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 9. Windows-style relative paths render as POSIX
# ---------------------------------------------------------------------------


def test_windows_style_rel_path_renders_posix():
    win_rel = PureWindowsPath("celeste\\0001\\000001.jpg")
    rec = ImageRecord(
        path=ABS, rel_path=win_rel, source=win_rel.parts[0], stem=win_rel.stem,
        image=_image(), width=64, height=48,
    )
    assert rec.rel_path.as_posix() == "celeste/0001/000001.jpg"
    assert rec.source == "celeste"
    assert Path(*win_rel.parts).as_posix() == "celeste/0001/000001.jpg"


# ---------------------------------------------------------------------------
# Shared fixtures (conftest.py) build valid v1.1 records
# ---------------------------------------------------------------------------


def test_sample_fixtures(tmp_path, sample_image_record, sample_caption_record):
    assert sample_image_record.path == (tmp_path / sample_image_record.rel_path).resolve()
    assert sample_image_record.path.is_file()
    assert sample_image_record.rel_path.as_posix() == "celeste/0001/000001.jpg"
    assert sample_image_record.source == "celeste"
    assert sample_caption_record.image_rel_path == sample_image_record.rel_path
    assert sample_caption_record.status in CAPTION_STATUSES
    assert set(sample_caption_record.flags) <= CAPTION_FLAGS
    assert sample_caption_record.finish_reason in FINISH_REASONS
