"""Tests for ``common/types.py`` — ImageRecord and CaptionRecord dataclasses."""

import datetime
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from pixel_art_auto_captioner.common.types import CaptionRecord, ImageRecord


# ---------------------------------------------------------------------------
# ImageRecord
# ---------------------------------------------------------------------------


def test_image_record_construction():
    """All required fields accept values and retain them."""
    img = Image.new("RGB", (64, 64), color="red")
    record = ImageRecord(
        path=Path("/images/sprite.png"),
        stem="folder1_sprite",
        image=img,
        width=64,
        height=64,
    )
    assert record.path == Path("/images/sprite.png")
    assert record.stem == "folder1_sprite"
    assert record.image is img
    assert record.width == 64
    assert record.height == 64


def test_image_record_types():
    """Every field has the expected Python type."""
    img = Image.new("RGB", (32, 48), color="blue")
    record = ImageRecord(
        path=Path("/a/b/c.webp"),
        stem="sub_c",
        image=img,
        width=32,
        height=48,
    )
    assert isinstance(record.path, Path)
    assert isinstance(record.stem, str)
    assert isinstance(record.image, Image.Image)
    assert isinstance(record.width, int)
    assert isinstance(record.height, int)


def test_image_record_rgb_mode():
    """The stored image reference preserves RGB mode."""
    img = Image.new("RGB", (16, 16), color="green")
    record = ImageRecord(
        path=Path("/t.png"),
        stem="t",
        image=img,
        width=16,
        height=16,
    )
    assert record.image.mode == "RGB"


def test_image_record_asdict():
    """``dataclasses.asdict`` serializes all fields including the PIL Image."""
    img = Image.new("RGB", (10, 10))
    record = ImageRecord(
        path=Path("/img.png"),
        stem="root_img",
        image=img,
        width=10,
        height=10,
    )
    d = asdict(record)
    assert d["path"] == Path("/img.png")
    assert d["stem"] == "root_img"
    assert d["width"] == 10
    assert d["height"] == 10
    assert isinstance(d["image"], Image.Image)
    assert d["image"].mode == "RGB"
    assert d["image"].size == (10, 10)


# ---------------------------------------------------------------------------
# CaptionRecord
# ---------------------------------------------------------------------------


def test_caption_record_construction():
    """All required fields accept values and retain them."""
    record = CaptionRecord(
        image_path=Path("/images/sprite.png"),
        image_stem="folder1_sprite",
        caption_text="A red pixel art character.",
        model_name="joycaption-beta-one",
        prompt_template="Describe this pixel art image.",
        generation_params={"temperature": 0.6, "max_new_tokens": 512},
        timestamp_utc="2026-06-16T12:00:00Z",
        image_width=64,
        image_height=64,
        extra={"confidence": 0.95},
    )
    assert record.image_path == Path("/images/sprite.png")
    assert record.image_stem == "folder1_sprite"
    assert record.caption_text == "A red pixel art character."
    assert record.model_name == "joycaption-beta-one"
    assert record.prompt_template == "Describe this pixel art image."
    assert record.generation_params == {"temperature": 0.6, "max_new_tokens": 512}
    assert record.timestamp_utc == "2026-06-16T12:00:00Z"
    assert record.image_width == 64
    assert record.image_height == 64
    assert record.extra == {"confidence": 0.95}


def test_caption_record_types():
    """Every field has the expected Python type."""
    record = CaptionRecord(
        image_path=Path("/i.png"),
        image_stem="sub_i",
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="",
        image_width=0,
        image_height=0,
        extra={},
    )
    assert isinstance(record.image_path, Path)
    assert isinstance(record.image_stem, str)
    assert isinstance(record.caption_text, str)
    assert isinstance(record.model_name, str)
    assert isinstance(record.prompt_template, str)
    assert isinstance(record.generation_params, dict)
    assert isinstance(record.timestamp_utc, str)
    assert isinstance(record.image_width, int)
    assert isinstance(record.image_height, int)
    assert isinstance(record.extra, dict)


def test_caption_record_empty_extra():
    """``extra`` defaults to an empty dict when no metadata is provided."""
    record = CaptionRecord(
        image_path=Path("/i.png"),
        image_stem="sub_i",
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="",
        image_width=0,
        image_height=0,
        extra={},
    )
    assert record.extra == {}


def test_caption_record_extra_arbitrary_keys():
    """``extra`` dict accepts arbitrary string-keyed metadata."""
    record = CaptionRecord(
        image_path=Path("/i.png"),
        image_stem="sub_i",
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="",
        image_width=0,
        image_height=0,
        extra={"tags": ["pixel", "character"], "score": 0.88},
    )
    assert record.extra["tags"] == ["pixel", "character"]
    assert record.extra["score"] == 0.88


def test_caption_record_asdict():
    """``dataclasses.asdict`` serializes all fields correctly."""
    record = CaptionRecord(
        image_path=Path("/images/sprite.png"),
        image_stem="folder1_sprite",
        caption_text="A red pixel art character.",
        model_name="joycaption-beta-one",
        prompt_template="Describe this pixel art image.",
        generation_params={"temperature": 0.6},
        timestamp_utc="2026-06-16T12:00:00Z",
        image_width=64,
        image_height=64,
        extra={},
    )
    d = asdict(record)
    assert d["image_path"] == Path("/images/sprite.png")
    assert d["caption_text"] == "A red pixel art character."
    assert d["generation_params"] == {"temperature": 0.6}
    assert d["timestamp_utc"] == "2026-06-16T12:00:00Z"
    assert d["image_width"] == 64
    assert d["image_height"] == 64


def test_caption_record_timestamp_iso8601():
    """``timestamp_utc`` stores a valid ISO 8601 UTC datetime string."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = CaptionRecord(
        image_path=Path("/i.png"),
        image_stem="sub_i",
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc=ts,
        image_width=0,
        image_height=0,
        extra={},
    )
    assert "T" in record.timestamp_utc
    parsed = datetime.datetime.fromisoformat(record.timestamp_utc)
    assert parsed.tzinfo is not None
