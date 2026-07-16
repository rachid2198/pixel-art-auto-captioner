"""Caption serialization utilities.

Provides three public functions for writing caption output:

- ``save_txt_sidecar``: writes a human-readable ``.txt`` file preserving
  the input directory structure.
- ``save_jsonl_entry``: appends one JSON line to a ``captions.jsonl``
  manifest.
- ``build_record``: constructs a ``CaptionRecord`` from an
  ``ImageRecord`` and generated caption data, stamping a UTC timestamp.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pixel_art_auto_captioner.common.types import CaptionRecord, ImageRecord

logger = logging.getLogger(__name__)


def save_txt_sidecar(record: CaptionRecord, output_dir: Path) -> Path:
    """Write the caption text as a ``.txt`` sidecar file.

    The output path is ``output_dir / image_rel_path.with_suffix('.txt')``,
    which reconstructs the original input directory tree under
    *output_dir* (e.g. ``output/folder1/sprite.txt``).

    Parent directories are created automatically if they do not exist.
    Existing files are silently overwritten.

    Args:
        record: The ``CaptionRecord`` whose text will be written.
        output_dir: Root directory for caption output files.

    Returns:
        The absolute ``Path`` of the newly written ``.txt`` file.

    Raises:
        OSError: If the file cannot be written (e.g. permission denied,
            disk full).
        ValueError: If the resolved output path escapes *output_dir*
            (e.g. via an absolute ``image_rel_path`` or ``..`` segments).
    """
    out_path = (output_dir / record.image_rel_path).with_suffix(".txt")
    resolved = out_path.resolve()
    output_root = output_dir.resolve()

    # Defence-in-depth: prevent path traversal via malformed image_rel_path.
    try:
        resolved.relative_to(output_root)
    except ValueError:
        raise ValueError(
            f"Output path {resolved} escapes output directory {output_root}"
        ) from None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(record.caption_text, encoding="utf-8")
    logger.debug("Wrote txt sidecar: %s", out_path)
    return resolved


def save_jsonl_entry(record: CaptionRecord, output_dir: Path) -> Path:
    """Append a single JSON line to ``captions.jsonl`` inside *output_dir*.

    The JSON schema follows the structure defined in SPEC §6.2.  Every
    call opens the file in append mode so multiple records can be
    written sequentially without holding a file handle open.

    Args:
        record: The ``CaptionRecord`` to serialise.
        output_dir: Root directory for caption output files.  The JSONL
            file is always written as ``output_dir / "captions.jsonl"``.

    Returns:
        The absolute ``Path`` of the ``captions.jsonl`` file.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If any field value is not JSON-serialisable.
    """
    jsonl_path = output_dir / "captions.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = _record_to_json_dict(record)
    line = json.dumps(serialisable, ensure_ascii=False)

    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    logger.debug("Appended JSONL entry to: %s", jsonl_path)
    return jsonl_path.resolve()


def build_record(
    image: ImageRecord,
    caption: str,
    model_name: str,
    prompt: str,
    gen_params: dict,
) -> CaptionRecord:
    """Construct a ``CaptionRecord`` from an image and generation results.

    The ``timestamp_utc`` field is set to the current UTC time in ISO
    8601 format at the moment this function is called.

    Args:
        image: The source ``ImageRecord`` that was captioned.
        caption: The generated caption text.
        model_name: Identifier for the VLM that produced the caption.
        prompt: The user prompt text used for generation.
        gen_params: Dictionary of generation parameters (temperature,
            max_new_tokens, top_p, etc.).

    Returns:
        A fully-populated ``CaptionRecord``.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    return CaptionRecord(
        image_path=image.path,
        image_stem=image.stem,
        image_rel_path=image.rel_path,
        caption_text=caption,
        model_name=model_name,
        prompt_template=prompt,
        generation_params=gen_params,
        timestamp_utc=timestamp,
        image_width=image.width,
        image_height=image.height,
        extra={},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _record_to_json_dict(record: CaptionRecord) -> dict:
    """Convert a ``CaptionRecord`` to a JSON-serialisable dictionary.

    ``Path`` fields are converted to POSIX-style strings.  The custom
    ``image_rel_path`` key ``image_relative_path`` follows the
    SPEC §6.2 example schema.
    """
    return {
        "image_path": record.image_path.as_posix(),
        "image_stem": record.image_stem,
        "image_rel_path": record.image_rel_path.as_posix(),
        "caption_text": record.caption_text,
        "model_name": record.model_name,
        "prompt_template": record.prompt_template,
        "generation_params": record.generation_params,
        "timestamp_utc": record.timestamp_utc,
        "image_width": record.image_width,
        "image_height": record.image_height,
        "extra": record.extra,
    }