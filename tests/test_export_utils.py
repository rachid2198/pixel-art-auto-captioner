"""Unit tests for ``common/export_utils.py``.

Covers the three public functions: ``save_txt_sidecar``,
``save_jsonl_entry``, and ``build_record``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pixel_art_auto_captioner.common.export_utils import (
    build_record,
    save_jsonl_entry,
    save_txt_sidecar,
)
from pixel_art_auto_captioner.common.types import CaptionRecord, ImageRecord


# ============================================================================
# save_txt_sidecar
# ============================================================================


def test_save_txt_sidecar_creates_file(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """The sidecar file exists on disk after saving."""
    out = save_txt_sidecar(sample_caption_record, tmp_path)
    assert out.exists()
    assert out.suffix == ".txt"


def test_save_txt_sidecar_writes_correct_content(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """The file contains exactly the caption text, UTF-8 encoded."""
    out = save_txt_sidecar(sample_caption_record, tmp_path)
    content = out.read_text(encoding="utf-8")
    assert content == sample_caption_record.caption_text


def test_save_txt_sidecar_preserves_directory_structure(tmp_path: Path) -> None:
    """When rel_path includes subdirectories, the output mirrors them."""
    record = CaptionRecord(
        image_path=Path("/fake/img.png"),
        image_stem="img",
        image_rel_path=Path("folder1/subfolder/sprite.png"),
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="2026-06-28T12:00:00+00:00",
        image_width=64,
        image_height=64,
        extra={},
    )
    out = save_txt_sidecar(record, tmp_path)
    expected = tmp_path / "folder1" / "subfolder" / "sprite.txt"
    assert out == expected.resolve()
    assert out.exists()


def test_save_txt_sidecar_creates_parent_dirs(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Parent directories are created automatically."""
    out_dir = tmp_path / "deeply" / "nested" / "output"
    out = save_txt_sidecar(sample_caption_record, out_dir)
    assert out.exists()


def test_save_txt_sidecar_overwrites_existing(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Writing to an existing sidecar path overwrites the old content."""
    out = save_txt_sidecar(sample_caption_record, tmp_path)
    first_mtime = out.stat().st_mtime

    # Modify the caption and save again
    sample_caption_record.caption_text = "Updated caption."
    out = save_txt_sidecar(sample_caption_record, tmp_path)

    assert out.read_text(encoding="utf-8") == "Updated caption."
    assert out.stat().st_mtime >= first_mtime


# ============================================================================
# save_jsonl_entry
# ============================================================================


def test_save_jsonl_appends_line(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """A single entry creates a valid JSONL file with one line."""
    out = save_jsonl_entry(sample_caption_record, tmp_path)
    assert out.exists()
    assert out.name == "captions.jsonl"

    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["caption_text"] == sample_caption_record.caption_text


def test_save_jsonl_entry_multiple_appends(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Calling save_jsonl_entry twice appends a second line."""
    save_jsonl_entry(sample_caption_record, tmp_path)

    # Second record with different text
    record2 = CaptionRecord(
        image_path=Path("/fake/img2.png"),
        image_stem="img2",
        image_rel_path=Path("img2.png"),
        caption_text="Second caption.",
        model_name="test-model",
        prompt_template="Describe this image.",
        generation_params={"temperature": 0.6},
        timestamp_utc="2026-06-28T12:05:00+00:00",
        image_width=16,
        image_height=16,
        extra={},
    )
    save_jsonl_entry(record2, tmp_path)

    jsonl_path = tmp_path / "captions.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    data2 = json.loads(lines[1])
    assert data2["caption_text"] == "Second caption."


def test_save_jsonl_entry_creates_parent_dirs(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Parent directories of the JSONL file are created automatically."""
    out_dir = tmp_path / "a" / "b" / "c"
    out = save_jsonl_entry(sample_caption_record, out_dir)
    assert out.exists()


def test_save_jsonl_entry_schema_keys(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Every record in the JSONL contains the expected top-level keys."""
    out = save_jsonl_entry(sample_caption_record, tmp_path)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    data = json.loads(lines[0])

    expected_keys = {
        "image_path",
        "image_stem",
        "image_rel_path",
        "caption_text",
        "model_name",
        "prompt_template",
        "generation_params",
        "timestamp_utc",
        "image_width",
        "image_height",
        "extra",
    }
    assert set(data.keys()) == expected_keys


def test_save_jsonl_entry_paths_are_posix_strings(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Path values in the JSON are POSIX-style strings, not objects."""
    out = save_jsonl_entry(sample_caption_record, tmp_path)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    data = json.loads(lines[0])
    assert isinstance(data["image_path"], str)
    assert isinstance(data["image_rel_path"], str)


def test_save_jsonl_entry_generation_params_preserved(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """The generation_params dict is serialised faithfully."""
    out = save_jsonl_entry(sample_caption_record, tmp_path)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    data = json.loads(lines[0])
    assert data["generation_params"] == sample_caption_record.generation_params


# ============================================================================
# build_record
# ============================================================================


def test_build_record_includes_timestamp(
    sample_image_record: ImageRecord,
) -> None:
    """The timestamp is present and ISO 8601 UTC."""
    record = build_record(sample_image_record, "A caption.", "test-model", "prompt", {})
    assert isinstance(record.timestamp_utc, str)
    # ISO 8601 UTC: should contain "T" and end with "+00:00" or "Z"
    assert "T" in record.timestamp_utc
    assert record.timestamp_utc.endswith("+00:00") or record.timestamp_utc.endswith(
        "Z"
    )


def test_build_record_maps_image_fields(
    sample_image_record: ImageRecord,
) -> None:
    """build_record transfers image metadata to the CaptionRecord."""
    record = build_record(sample_image_record, "cap", "m", "p", {"t": 0.5})
    assert record.image_path == sample_image_record.path
    assert record.image_stem == sample_image_record.stem
    assert record.image_rel_path == sample_image_record.rel_path
    assert record.image_width == sample_image_record.width
    assert record.image_height == sample_image_record.height


def test_build_record_stores_caption_and_params(
    sample_image_record: ImageRecord,
) -> None:
    """The caption text, model, prompt, and gen_params are stored correctly."""
    gen_params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 300}
    record = build_record(
        sample_image_record,
        "A pixel-art description.",
        "joycaption-beta-one",
        "Write a caption.",
        gen_params,
    )
    assert record.caption_text == "A pixel-art description."
    assert record.model_name == "joycaption-beta-one"
    assert record.prompt_template == "Write a caption."
    assert record.generation_params == gen_params


def test_build_record_extra_is_empty_dict(
    sample_image_record: ImageRecord,
) -> None:
    """The extra field defaults to an empty dict."""
    record = build_record(sample_image_record, "c", "m", "p", {})
    assert record.extra == {}


def test_build_record_timestamp_is_utc_now(
    sample_image_record: ImageRecord,
) -> None:
    """The timestamp is close to current UTC time."""
    before = datetime.now(timezone.utc)
    record = build_record(sample_image_record, "c", "m", "p", {})
    after = datetime.now(timezone.utc)

    ts = datetime.fromisoformat(record.timestamp_utc)
    # Allow a small clock skew margin
    delta_before = (ts - before).total_seconds()
    delta_after = (after - ts).total_seconds()
    assert delta_before >= -1.0
    assert delta_after >= -1.0


# ============================================================================
# Edge case: empty generation_params
# ============================================================================


def test_build_record_empty_gen_params(
    sample_image_record: ImageRecord,
) -> None:
    """Empty generation_params dict is stored as-is."""
    record = build_record(sample_image_record, "c", "m", "p", {})
    assert record.generation_params == {}


def test_save_txt_sidecar_empty_caption(
    tmp_path: Path,
) -> None:
    """An empty caption text produces an empty sidecar file."""
    record = CaptionRecord(
        image_path=Path("/fake/img.png"),
        image_stem="img",
        image_rel_path=Path("img.png"),
        caption_text="",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="2026-06-28T12:00:00+00:00",
        image_width=64,
        image_height=64,
        extra={},
    )
    out = save_txt_sidecar(record, tmp_path)
    assert out.read_text(encoding="utf-8") == ""


# ============================================================================
# Edge case: extra dict with content
# ============================================================================


def test_save_jsonl_entry_nonempty_extra(
    tmp_path: Path,
) -> None:
    """The extra dict is serialised even when non-empty."""
    record = CaptionRecord(
        image_path=Path("/fake/img.png"),
        image_stem="img",
        image_rel_path=Path("img.png"),
        caption_text="caption",
        model_name="m",
        prompt_template="p",
        generation_params={},
        timestamp_utc="2026-06-28T12:00:00+00:00",
        image_width=64,
        image_height=64,
        extra={"foo": "bar", "nested": {"a": 1}},
    )
    out = save_jsonl_entry(record, tmp_path)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    data = json.loads(lines[0])
    assert data["extra"] == {"foo": "bar", "nested": {"a": 1}}


# ============================================================================
# Path traversal containment (Critique 1 fix)
# ============================================================================


def test_save_txt_sidecar_raises_on_absolute_image_rel_path(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """An absolute image_rel_path must not escape the output directory."""
    sample_caption_record.image_rel_path = Path("/etc/passwd")
    with pytest.raises(ValueError, match="escapes output directory"):
        save_txt_sidecar(sample_caption_record, tmp_path)


def test_save_txt_sidecar_raises_on_parent_traversal(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """``..`` segments in image_rel_path must not escape output_dir."""
    sample_caption_record.image_rel_path = Path("../../escape.txt")
    with pytest.raises(ValueError, match="escapes output directory"):
        save_txt_sidecar(sample_caption_record, tmp_path)


# ============================================================================
# Error-path coverage (Critique 3 fix)
# ============================================================================


def test_save_jsonl_entry_raises_typeerror_on_unserializable(
    tmp_path: Path,
) -> None:
    """Non-JSON-serialisable metadata must raise TypeError."""
    record = CaptionRecord(
        image_path=Path("/fake/img.png"),
        image_stem="img",
        image_rel_path=Path("img.png"),
        caption_text="test",
        model_name="m",
        prompt_template="p",
        generation_params={"bad_value": b"bytes are not JSON"},
        timestamp_utc="2026-06-28T12:00:00+00:00",
        image_width=64,
        image_height=64,
        extra={},
    )
    with pytest.raises(TypeError):
        save_jsonl_entry(record, tmp_path)


def test_save_txt_sidecar_raises_on_unwritable_output(
    sample_caption_record: CaptionRecord, tmp_path: Path
) -> None:
    """Writing to a path whose parent is a file (not a directory) must raise."""
    # Create a regular file where a parent directory would need to exist.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")

    # Make image_rel_path try to create a directory tree through 'blocker'.
    sample_caption_record.image_rel_path = Path("blocker") / "sub" / "sprite.png"

    with pytest.raises((OSError, FileExistsError)):
        save_txt_sidecar(sample_caption_record, tmp_path)