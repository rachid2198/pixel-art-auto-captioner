"""Shared test fixtures for the pixel-art-auto-captioner test suite."""

import sys
from pathlib import Path

import PIL.Image
import pytest

# Ensure the package is importable when running tests from the project root.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


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

    The synthetic image is a 32×32 red square written to a temp
    directory so the ``path`` field points to a real, valid file.
    """
    from pixel_art_auto_captioner.common.types import ImageRecord

    img = PIL.Image.new("RGB", (32, 32), color=(255, 0, 0))
    disk_path = tmp_path / "test_image.png"
    img.save(disk_path)

    return ImageRecord(
        path=disk_path.resolve(),
        stem="test_image",
        rel_path=Path("test_image.png"),
        image=img,
        width=32,
        height=32,
    )


@pytest.fixture
def sample_caption_record(sample_image_record) -> "CaptionRecord":
    """A pre-built ``CaptionRecord`` for export utility tests."""
    from pixel_art_auto_captioner.common.types import CaptionRecord

    return CaptionRecord(
        image_path=sample_image_record.path,
        image_stem=sample_image_record.stem,
        image_rel_path=sample_image_record.rel_path,
        caption_text="A red square in pixel art style.",
        model_name="test-model",
        prompt_template="Describe this image.",
        generation_params={"temperature": 0.6, "max_new_tokens": 100},
        timestamp_utc="2026-06-28T12:00:00+00:00",
        image_width=32,
        image_height=32,
        extra={},
    )