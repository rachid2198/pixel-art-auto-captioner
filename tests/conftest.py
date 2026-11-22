"""Shared test fixtures for the pixel-art-auto-captioner test suite.

The tests import the installed package (``pip install -e ".[dev]"`` from the
repo root, SPEC §15 / AGENTS.md §5); nothing here edits ``sys.path``.
"""

from pathlib import Path

import PIL.Image
import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "gpu: tests that require a CUDA-capable GPU")


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_rgb() -> PIL.Image.Image:
    """A small synthetic RGB image (64×64 pixel art)."""
    return PIL.Image.new("RGB", (64, 64), color=(128, 64, 200))


# ---------------------------------------------------------------------------
# ImageRecord / CaptionRecord fixtures for export tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_record(tmp_path: Path) -> "ImageRecord":
    """A pre-built ``ImageRecord`` for export utility tests.

    The synthetic image is a 32×32 red square written under ``tmp_path``
    at its ``rel_path`` (``celeste/0001/000001.jpg``), so ``tmp_path``
    plays the role of ``input_dir`` and ``path`` points to a real file.
    """
    from pixel_art_auto_captioner.common.types import ImageRecord

    rel_path = Path("celeste/0001/000001.jpg")
    img = PIL.Image.new("RGB", (32, 32), color=(255, 0, 0))
    disk_path = tmp_path / rel_path
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(disk_path)

    return ImageRecord(
        path=disk_path.resolve(),
        rel_path=rel_path,
        source="celeste",
        stem="000001",
        image=img,
        width=32,
        height=32,
    )


@pytest.fixture
def sample_caption_record(sample_image_record) -> "CaptionRecord":
    """A pre-built ``CaptionRecord`` for export utility tests."""
    from pixel_art_auto_captioner.common.types import CaptionRecord

    return CaptionRecord(
        image_rel_path=sample_image_record.rel_path,
        image_path=sample_image_record.path,
        source=sample_image_record.source,
        category="game_crop",
        caption_type="long",
        caption_text="A red square in pixel art style.",
        status="ok",
        flags=[],
        model_name="joycaption-beta-one",
        model_path="fancyfeast/llama-joycaption-beta-one-hf-llava",
        model_revision=None,
        backend="hf",
        prompt="Write a long descriptive caption for this image.",
        generation_params={
            "max_new_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": None,
        },
        seed=123456789,
        attempt=1,
        finish_reason="stop",
        num_output_tokens=9,
        image_width=sample_image_record.width,
        image_height=sample_image_record.height,
        run_id="20261107-120000-test",
        shard="0/1",
        timestamp_utc="2026-11-07T12:00:00Z",
        extra={"source_name": "Celeste"},
    )

@pytest.fixture(autouse=True)
def _no_real_hub_cache(monkeypatch):
    """Keep the suite hermetic: ``resolved_revision`` never reads the real HF cache (R4).

    Backend tests that load a repo id would otherwise import ``huggingface_hub``
    and look in this machine's cache. Tests of the lookup itself call the
    original function directly (``tests/test_revision.py``).
    """
    from pixel_art_auto_captioner.captioning import revision

    monkeypatch.setattr(revision, "_hub_cache_lookup", lambda repo_id, rev: None)
