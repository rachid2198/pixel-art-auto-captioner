"""Shared data types for the pixel-art auto-captioner pipeline (SPEC v1.1 §3).

Records that flow through the pipeline:

- ``ImageRecord``: an image decoded from disk, ready for captioning.
- ``LoadFailure``: an image that was discovered but could not be decoded.
- ``CaptionRequest``: one (image, caption type) generation job for a backend.
- ``CaptionResult``: a backend's raw output for one ``CaptionRequest``.
- ``CaptionRecord``: one persisted caption, i.e. one JSONL line (SPEC §8.3).

All types are plain ``@dataclass`` es with no validation and no serialization
helpers (serialization lives in ``export_utils``). Paths are ``pathlib.Path``
in memory and POSIX strings (``as_posix()``) on disk.

Field order is part of the contract: JSONL keys follow dataclass field order
(SPEC §8.3), so do not reorder fields.

The module-level constants mirror the value sets in SPEC §3 / §5.3 for use by
later modules. The dataclasses do not enforce them.

This module imports nothing from the package and never imports ``torch`` or
``vllm`` (SPEC §2.2).
"""

from dataclasses import dataclass
from pathlib import Path

import PIL.Image

#: Allowed values of ``CaptionRecord.status`` (SPEC §3, §5.3).
CAPTION_STATUSES: frozenset[str] = frozenset({"ok", "flagged", "failed"})

#: Allowed members of ``CaptionRecord.flags`` (SPEC §3, §5.3).
CAPTION_FLAGS: frozenset[str] = frozenset({"truncated", "repetitive", "empty", "error"})

#: Allowed values of ``CaptionResult.finish_reason`` / ``CaptionRecord.finish_reason``
#: (SPEC §3).
FINISH_REASONS: frozenset[str] = frozenset({"stop", "length", "error"})


@dataclass
class ImageRecord:
    """An image decoded from disk, ready for captioning (SPEC §3).

    Attributes:
        path: Absolute path of the image file.
        rel_path: Path relative to ``input_dir``, e.g.
            ``Path("celeste/0001/000001.jpg")``. This is the unique key of an
            image; stems repeat across sources, so never key by stem alone.
        source: First component of ``rel_path`` (e.g. ``"celeste"``), or
            ``""`` when the file sits directly in ``input_dir``.
        stem: File name without extension (e.g. ``"000001"``).
        image: Decoded image in RGB mode, **not** resized (the backend's
            processor resizes).
        width: Original image width in pixels, before any resize.
        height: Original image height in pixels, before any resize.
    """

    path: Path
    rel_path: Path
    source: str
    stem: str
    image: PIL.Image.Image
    width: int
    height: int


@dataclass
class LoadFailure:
    """An image that was discovered but could not be decoded (SPEC §3).

    Counted as *failed* by the runner and logged to ``failures/``.

    Attributes:
        path: Absolute path of the image file.
        rel_path: Path relative to ``input_dir`` (the unique key).
        source: First component of ``rel_path``, or ``""`` for files directly
            in ``input_dir``.
        error: Error description, formatted ``"<ExceptionType>: <message>"``.
    """

    path: Path
    rel_path: Path
    source: str
    error: str


@dataclass
class CaptionRequest:
    """One (image, caption type) generation job sent to a backend (SPEC §3).

    Attributes:
        image_rel_path: Relative path of the image; the routing key.
        caption_type: Caption type name, e.g. ``"long"``.
        image: Decoded RGB image (not resized).
        prompt: Fully rendered user prompt (SPEC §5.1).
        gen_params: Generation parameters: ``max_new_tokens``,
            ``temperature``, ``top_p``, ``top_k`` (a value of ``None`` means
            the backend default).
        seed: Per-request deterministic seed (SPEC §5.2).
        attempt: Attempt number: 1 = first try, 2 = first retry, and so on.
            Defaults to 1.
    """

    image_rel_path: Path
    caption_type: str
    image: PIL.Image.Image
    prompt: str
    gen_params: dict
    seed: int
    attempt: int = 1


@dataclass
class CaptionResult:
    """Backend output for one ``CaptionRequest`` (SPEC §3).

    Backends return results in the same order as the requests.

    Attributes:
        text: Raw decoded text, only ``.strip()`` -ed.
        finish_reason: ``"stop"``, ``"length"`` or ``"error"`` (see
            ``FINISH_REASONS``).
        num_output_tokens: Number of generated tokens.
        error: Error description when ``finish_reason == "error"``,
            otherwise ``None``. Defaults to ``None``.
    """

    text: str
    finish_reason: str
    num_output_tokens: int
    error: str | None = None


@dataclass
class CaptionRecord:
    """One persisted caption, i.e. one JSONL line (SPEC §3, §8.3).

    Every field is required (no defaults), so a missing field fails loudly.
    Fields are in JSONL key order.

    Attributes:
        image_rel_path: Path relative to ``input_dir`` (the unique key,
            with extension).
        image_path: Absolute path of the image file.
        source: First component of ``image_rel_path``, or ``""``.
        category: Category of the source (e.g. ``"game_crop"``), from the
            ``sources`` config (SPEC §5.1).
        caption_type: Caption type name, e.g. ``"long"``.
        caption_text: Raw caption text (never modified after generation;
            may be empty for failed captions).
        status: ``"ok"``, ``"flagged"`` or ``"failed"`` (see
            ``CAPTION_STATUSES``).
        flags: Quality flags, a subset of ``CAPTION_FLAGS``
            (``"truncated"``, ``"repetitive"``, ``"empty"``, ``"error"``).
        model_name: Model identifier, e.g. ``"joycaption-beta-one"``.
        model_path: Hugging Face repo id or local path of the model.
        model_revision: Hugging Face commit hash, or ``None`` if not set.
        backend: ``"vllm"`` or ``"hf"``.
        prompt: Fully rendered user prompt that was sent.
        generation_params: Generation parameters used for this caption.
        seed: Seed used for the final attempt.
        attempt: Attempt number that produced this caption (1 = first try).
        finish_reason: ``"stop"``, ``"length"`` or ``"error"``.
        num_output_tokens: Number of generated tokens.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        run_id: Identifier of the run that produced this record.
        shard: Shard of the run, formatted ``"i/N"``.
        timestamp_utc: ISO 8601 UTC timestamp, second precision, ``"Z"``
            suffix (e.g. ``"2026-11-06T19:14:05Z"``).
        extra: Free-form metadata, e.g.
            ``{"source_name": "Celeste", "latency_s": 1.9}``.
    """

    image_rel_path: Path
    image_path: Path
    source: str
    category: str
    caption_type: str
    caption_text: str
    status: str
    flags: list[str]
    model_name: str
    model_path: str
    model_revision: str | None
    backend: str
    prompt: str
    generation_params: dict
    seed: int
    attempt: int
    finish_reason: str
    num_output_tokens: int
    image_width: int
    image_height: int
    run_id: str
    shard: str
    timestamp_utc: str
    extra: dict
