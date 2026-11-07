"""Caption output: JSONL shard writer, resume index, ``.txt`` export, QA page.

Implements SPEC v1.1 §8 (output formats and resume) and the QA page of §10.3.

Public API:

- ``ShardWriter``: appends ``CaptionRecord`` s to
  ``<output_dir>/captions/shard-<iii>-of-<NNN>.jsonl`` (JSONL first, then the
  optional ``.txt`` file), with ``fsync`` every ``fsync_every`` records.
- ``ResumeIndex``: reads **every** ``captions/*.jsonl`` file (D3) and answers
  ``is_complete(rel_path, caption_type)``; the latest ``timestamp_utc`` wins.
- ``build_record``: builds a ``CaptionRecord`` with a ``"Z"`` UTC timestamp.
- ``save_txt``: writes one ``.txt`` file with the §8.2 naming and collision rule.
- ``read_records``: iterates the JSON objects of JSONL files, skipping a
  malformed final line (crash mid-write) with a WARNING.
- ``generate_qa_page``: writes a self-contained ``qa/<name>.html`` spot-check
  page (§10.3).

Resume is driven only by the JSONL files; ``.txt`` files are never consulted
(AGENTS.md §6.8).

This module imports only the standard library, Pillow and other ``common``
modules; it never imports ``torch``/``vllm`` or ``ingestion``/``captioning``/
``batch`` (SPEC §2.2).
"""

from __future__ import annotations

import base64
import dataclasses
import html
import io
import json
import logging
import os
import random
import sys
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePath, PurePosixPath
from types import TracebackType
from typing import Any

import PIL.Image

from pixel_art_auto_captioner.common.config import _parse_shard
from pixel_art_auto_captioner.common.types import (
    CAPTION_FLAGS,
    CAPTION_STATUSES,
    CaptionRecord,
    CaptionResult,
    ImageRecord,
    LoadFailure,
)

logger = logging.getLogger(__name__)

_CAPTIONS_DIR = "captions"
_TXT_DIR = "txt"
_QA_DIR = "qa"

#: Statuses that count as complete for resume and get a ``.txt`` file (§8.2, §8.4).
_DONE_STATUSES: frozenset[str] = frozenset({"ok", "flagged"})

#: Keys a JSONL record must have for resume (with non-empty ``str`` values).
_RESUME_KEYS: tuple[str, ...] = ("image_rel_path", "caption_type", "status", "timestamp_utc")

#: Max longest side of a QA thumbnail, in pixels.
_QA_THUMB_SIZE = 256

#: Max number of groups listed in the prompt-change WARNING.
_MAX_LISTED = 10


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Append-only JSONL writer for one shard (SPEC §8.4).

    Opens ``<output_dir>/captions/shard-<iii>-of-<NNN>.jsonl`` in append mode
    (creating ``captions/`` if needed). Only one process may write a given
    shard file. If the file ends with a malformed line (a crash mid-write),
    that line is removed before appending, with a WARNING, so it can never end
    up in the middle of the file; a valid but unterminated final line just
    gets its newline.

    ``write(record)`` writes the JSON line and flushes it, then writes the
    ``.txt`` file if ``write_txt`` is set and the status is ``ok`` or
    ``flagged``. The JSONL is written first because it is the source of truth
    for resume. The file is ``fsync`` -ed every ``fsync_every`` records and on
    ``close()``. Usable as a context manager.

    Attributes:
        output_dir: The run's output directory.
        shard: ``(i, N)``.
        write_txt: Whether ``.txt`` files are written.
        fsync_every: ``fsync`` cadence in records.
        path: Path of the shard's JSONL file.
        records_written: Number of records written by this writer.
    """

    def __init__(
        self,
        output_dir: Path,
        shard: str | tuple[int, int],
        write_txt: bool,
        *,
        fsync_every: int = 100,
        collision_set: Collection[Path | str] | None = None,
    ) -> None:
        """Open (or create) the shard file for appending.

        Args:
            output_dir: Output directory (SPEC §8.1).
            shard: ``"i/N"`` or ``(i, N)`` with ``0 <= i < N``.
            write_txt: Also write ``.txt`` files (``output.write_txt``).
            fsync_every: ``fsync`` the JSONL file every this many records
                (``output.fsync_every``, default 100).
            collision_set: Relative paths of images whose stem collides with
                another image in the same directory (§8.2); these get
                ``<stem>.<ext>.<caption_type>.txt``. ``None`` = no collisions.

        Raises:
            ValueError: Invalid ``shard`` or ``fsync_every``.
            OSError: The directory or file cannot be created/opened.
        """
        if isinstance(fsync_every, bool) or not isinstance(fsync_every, int) or fsync_every < 1:
            raise ValueError(f"fsync_every must be a positive int, got {fsync_every!r}")
        self.output_dir = Path(output_dir)
        self.shard = _coerce_shard(shard)
        self.write_txt = bool(write_txt)
        self.fsync_every = fsync_every
        self._collisions: frozenset[str] = frozenset(_posix(p) for p in (collision_set or ()))
        self.path = self.output_dir / _CAPTIONS_DIR / _shard_filename(*self.shard)
        self.records_written = 0
        self._since_fsync = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _repair_tail(self.path)
        self._fh: Any = open(self.path, "ab")  # binary: no newline translation on Windows
        logger.info(
            "Writing shard %d/%d to %s (txt=%s, fsync every %d records)",
            self.shard[0], self.shard[1], self.path.as_posix(), self.write_txt, fsync_every,
        )

    @property
    def closed(self) -> bool:
        """``True`` once ``close()`` has been called."""
        return self._fh is None

    def write(self, record: CaptionRecord) -> None:
        """Append one record: JSON line + ``flush()``, then the ``.txt`` file.

        Args:
            record: The record to persist.

        Raises:
            ValueError: The writer is closed, or the ``.txt`` path is unsafe
                (see ``save_txt``; the JSON line is already written then).
            TypeError: A field is not JSON-serializable (nothing is written).
            OSError: Write failure (e.g. disk full).
        """
        if self._fh is None:
            raise ValueError("ShardWriter is closed")
        data = (_dumps(record) + "\n").encode("utf-8")
        self._fh.write(data)
        self._fh.flush()
        self.records_written += 1
        self._since_fsync += 1
        if self._since_fsync >= self.fsync_every:
            self._fsync()
        if self.write_txt and record.status in _DONE_STATUSES:
            save_txt(record, self.output_dir, self._collisions)

    def close(self) -> None:
        """Flush, ``fsync`` and close the file. Safe to call more than once.

        Raises:
            OSError: Flush/fsync failure (the file is closed anyway).
        """
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        try:
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()
        logger.info("Closed %s (%d records written)", self.path.as_posix(), self.records_written)

    def _fsync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_fsync = 0

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Resume index
# ---------------------------------------------------------------------------


class ResumeIndex:
    """Which (image, caption type) pairs already have a usable caption (SPEC §8.4).

    Built from **all** ``captions/*.jsonl`` files of an output directory (D3),
    never from ``.txt`` files. For each key ``(image_rel_path, caption_type)``
    the record with the latest ``timestamp_utc`` wins (on equal timestamps,
    the one read last: files in name order, lines in file order). A key is
    complete if its winning record has status ``ok`` or ``flagged``;
    ``failed`` records are retried.

    Only the key, timestamp, status and a small group id are kept per record,
    so ~1M records fit in a few hundred MB.

    ``ResumeIndex()`` is an empty index (nothing is complete), e.g. for
    ``resume=False``.

    Attributes:
        files_read: JSONL files read, in reading order.
        records_read: Number of records read (superseded ones included).
    """

    def __init__(self) -> None:
        """Create an empty index."""
        # rel_path (POSIX) -> caption_type -> (epoch, location, status, group id);
        # location = (file index << 32) | line number, increasing in reading order.
        self._entries: dict[str, dict[str, tuple[float, int, str, int]]] = {}
        # group id -> (caption_type, category, source, prompt)
        self._groups: list[tuple[str, str, str, str]] = []
        self._group_ids: dict[tuple[str, str, str, str], int] = {}
        self.files_read: list[Path] = []
        self.records_read = 0

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> ResumeIndex:
        """Build the index from every ``<output_dir>/captions/*.jsonl`` file.

        A missing ``captions/`` directory gives an empty index. A malformed
        final line of a file is skipped with a WARNING. Logs a WARNING if the
        complete records of one (caption type, category, source) were produced
        with different prompts (the prompt config changed between runs); they
        still count as complete.

        Args:
            output_dir: The output directory (SPEC §8.1).

        Returns:
            The populated index.

        Raises:
            ValueError: A malformed line other than a file's final line, or a
                record without non-empty string ``image_rel_path``,
                ``caption_type``, ``status`` and a parseable ``timestamp_utc``.
            OSError: A file cannot be read.
        """
        files = _caption_files(Path(output_dir))
        index = cls._scan(files)
        index._warn_prompt_changes()
        logger.info(
            "Resume index: %d records in %d files, %d of %d (image, caption type) pairs complete",
            index.records_read, len(files), index.num_complete, len(index),
        )
        return index

    @classmethod
    def _scan(cls, files: Sequence[Path]) -> ResumeIndex:
        index = cls()
        for file_idx, path in enumerate(files):
            for lineno, rec in _iter_jsonl_file(path):
                index._add(rec, path, lineno, (file_idx << 32) | lineno)
        index.files_read = list(files)
        return index

    def _add(self, rec: dict[str, Any], path: Path, lineno: int, location: int) -> None:
        values = [rec.get(k) for k in _RESUME_KEYS]
        if not all(isinstance(v, str) and v for v in values):
            raise ValueError(
                f"{path.as_posix()}:{lineno}: record needs non-empty string fields "
                f"{list(_RESUME_KEYS)}, got {dict(zip(_RESUME_KEYS, values))!r}"
            )
        rel, caption_type, status, timestamp = values
        epoch = _parse_timestamp(timestamp)
        if epoch is None:
            raise ValueError(f"{path.as_posix()}:{lineno}: unparseable timestamp_utc {timestamp!r}")
        group = (
            sys.intern(caption_type),
            _str_field(rec, "category"),
            _str_field(rec, "source"),
            _str_field(rec, "prompt"),
        )
        gid = self._group_ids.get(group)
        if gid is None:
            gid = self._group_ids[group] = len(self._groups)
            self._groups.append(group)
        self.records_read += 1
        per_image = self._entries.setdefault(rel, {})
        prev = per_image.get(group[0])
        if prev is None or epoch >= prev[0]:
            per_image[group[0]] = (epoch, location, sys.intern(status), gid)

    def is_complete(self, rel_path: Path | str, caption_type: str) -> bool:
        """Whether ``(rel_path, caption_type)`` has an ``ok``/``flagged`` latest record.

        Args:
            rel_path: Image path relative to ``input_dir`` (with extension);
                a ``Path`` or a POSIX string.
            caption_type: Caption type name.

        Returns:
            ``True`` if the latest record for the key has status ``ok`` or
            ``flagged``; ``False`` if it is ``failed`` or there is none.
        """
        entry = self._entries.get(_posix(rel_path), {}).get(caption_type)
        return entry is not None and entry[2] in _DONE_STATUSES

    @property
    def num_complete(self) -> int:
        """Number of complete (image, caption type) pairs."""
        return sum(
            1 for per in self._entries.values() for e in per.values() if e[2] in _DONE_STATUSES
        )

    def __len__(self) -> int:
        """Number of (image, caption type) pairs with at least one record."""
        return sum(len(per) for per in self._entries.values())

    def _warn_prompt_changes(self) -> None:
        counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for per in self._entries.values():
            for _, _, status, gid in per.values():
                if status in _DONE_STATUSES:
                    caption_type, category, source, prompt = self._groups[gid]
                    counts[(caption_type, category, source)][prompt] += 1
        changed = sorted(k for k, prompts in counts.items() if len(prompts) > 1)
        if not changed:
            return
        listed = "; ".join(
            f"{t}/{c}/{s or '<root>'}: {len(counts[(t, c, s)])} prompts "
            f"({', '.join(str(n) for n in sorted(counts[(t, c, s)].values(), reverse=True))} records)"
            for t, c, s in changed[:_MAX_LISTED]
        )
        more = f"; ... and {len(changed) - _MAX_LISTED} more" if len(changed) > _MAX_LISTED else ""
        logger.warning(
            "Existing complete records were produced with different prompts for %d "
            "(caption type/category/source) groups; the prompt config changed between runs. "
            "They still count as complete; use --no-resume or a new output_dir to regenerate. %s%s",
            len(changed), listed, more,
        )


# ---------------------------------------------------------------------------
# Records, .txt files, JSONL reading
# ---------------------------------------------------------------------------


def build_record(
    image: ImageRecord | LoadFailure,
    result: CaptionResult,
    *,
    caption_type: str,
    category: str,
    prompt: str,
    generation_params: Mapping[str, Any],
    seed: int,
    attempt: int,
    status: str,
    flags: Sequence[str],
    model_name: str,
    model_path: str,
    model_revision: str | None,
    backend: str,
    run_id: str,
    shard: str | tuple[int, int],
    extra: Mapping[str, Any] | None = None,
) -> CaptionRecord:
    """Build a ``CaptionRecord`` stamped with the current UTC time.

    Image fields (``image_rel_path``, ``image_path``, ``source``, original
    ``image_width``/``image_height``) come from ``image``; for a
    ``LoadFailure`` width and height are ``0``. ``caption_text``,
    ``finish_reason`` and ``num_output_tokens`` come from ``result``; if
    ``result.error`` is set and ``extra`` has no ``"error"`` key, it is stored
    as ``extra["error"]``. ``timestamp_utc`` is ISO 8601, second precision,
    ``"Z"`` suffix. Lists and dicts are copied.

    Args:
        image: The captioned image, or the image that failed to load.
        result: Final backend result (the runner builds an error result for a
            ``LoadFailure``).
        caption_type: Caption type name.
        category: Category of the image's source (§5.1).
        prompt: Rendered user prompt.
        generation_params: Generation parameters used.
        seed: Seed of the final attempt.
        attempt: Attempt number of the final attempt.
        status: ``"ok"``, ``"flagged"`` or ``"failed"``.
        flags: Quality flags (subset of ``CAPTION_FLAGS``).
        model_name: Model identifier.
        model_path: Repo id or local path of the model.
        model_revision: Hugging Face commit hash, or ``None``.
        backend: ``"vllm"`` or ``"hf"``.
        run_id: Run identifier.
        shard: ``"i/N"`` or ``(i, N)``; stored as ``"i/N"``.
        extra: Free-form metadata (copied).

    Returns:
        The new record.

    Raises:
        TypeError: ``image`` is not an ``ImageRecord`` or ``LoadFailure``.
        ValueError: Unknown ``status`` or flag, or an invalid ``shard``.
    """
    if not isinstance(image, (ImageRecord, LoadFailure)):
        raise TypeError(f"image must be an ImageRecord or LoadFailure, got {type(image).__name__}")
    if status not in CAPTION_STATUSES:
        raise ValueError(f"status must be one of {sorted(CAPTION_STATUSES)}, got {status!r}")
    unknown = [f for f in flags if f not in CAPTION_FLAGS]
    if unknown:
        raise ValueError(f"unknown flags {unknown!r}; allowed: {sorted(CAPTION_FLAGS)}")
    i, n = _coerce_shard(shard)
    extra_dict = dict(extra or {})
    if result.error is not None and "error" not in extra_dict:
        extra_dict["error"] = result.error
    if isinstance(image, ImageRecord):
        width, height = image.width, image.height
    else:
        width = height = 0
    return CaptionRecord(
        image_rel_path=image.rel_path,
        image_path=image.path,
        source=image.source,
        category=category,
        caption_type=caption_type,
        caption_text=result.text,
        status=status,
        flags=list(flags),
        model_name=model_name,
        model_path=model_path,
        model_revision=model_revision,
        backend=backend,
        prompt=prompt,
        generation_params=dict(generation_params),
        seed=seed,
        attempt=attempt,
        finish_reason=result.finish_reason,
        num_output_tokens=result.num_output_tokens,
        image_width=width,
        image_height=height,
        run_id=run_id,
        shard=f"{i}/{n}",
        timestamp_utc=_utc_now(),
        extra=extra_dict,
    )


def save_txt(
    record: CaptionRecord,
    output_dir: Path,
    collision_set: Collection[Path | str] | None,
) -> Path:
    """Write ``record.caption_text`` to its ``.txt`` file (SPEC §8.2).

    Path: ``<output_dir>/txt/<rel_dir>/<stem>.<caption_type>.txt``, or
    ``<stem>.<ext>.<caption_type>.txt`` when ``image_rel_path`` is in
    ``collision_set`` (two images in the same directory share a stem, e.g.
    ``sprite.png`` + ``sprite.jpg``). Content is the caption text only,
    UTF-8, no trailing newline, no newline translation. An existing file is
    overwritten.

    Args:
        record: An ``ok`` or ``flagged`` record.
        output_dir: Output directory (SPEC §8.1).
        collision_set: Colliding relative paths (``Path`` or POSIX strings);
            ``None`` or empty = no collisions.

    Returns:
        The path written (under ``output_dir``).

    Raises:
        ValueError: ``status`` is not ``ok``/``flagged`` (``failed`` records
            never get a ``.txt``); ``image_rel_path`` is absolute, empty or
            contains ``..``; ``caption_type`` is not a plain name.
        OSError: The file cannot be written.
    """
    if record.status not in _DONE_STATUSES:
        raise ValueError(
            f"txt files are written only for status ok/flagged, got {record.status!r} "
            f"for {_posix(record.image_rel_path)}"
        )
    rel = _safe_rel_path(record.image_rel_path)
    caption_type = record.caption_type
    if (
        not isinstance(caption_type, str)
        or caption_type in ("", ".", "..")
        or any(c in caption_type for c in "/\\:")
    ):
        raise ValueError(f"caption_type is not a valid file-name part: {caption_type!r}")
    collisions = collision_set or ()
    collides = rel.as_posix() in collisions or Path(rel.as_posix()) in collisions
    base = rel.name if collides else rel.stem
    txt_root = Path(output_dir) / _TXT_DIR
    out_path = txt_root.joinpath(*rel.parent.parts, f"{base}.{caption_type}.txt")
    try:
        out_path.resolve().relative_to(txt_root.resolve())
    except ValueError:
        raise ValueError(f"txt path {out_path} escapes {txt_root}") from None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(record.caption_text)
    logger.debug("Wrote %s", out_path.as_posix())
    return out_path


def read_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of JSONL files, in file and line order.

    Blank lines are ignored. A malformed **final** line of a file (invalid
    UTF-8 or JSON, or not a JSON object; typically a crash mid-write) is
    skipped with a WARNING. Records are not otherwise validated.

    Args:
        paths: JSONL files, read in the given order.

    Yields:
        One ``dict`` per record line.

    Raises:
        ValueError: A malformed line that is not the file's final line.
        OSError: A file cannot be read.
    """
    for path in paths:
        for _, rec in _iter_jsonl_file(Path(path)):
            yield rec


# ---------------------------------------------------------------------------
# QA page
# ---------------------------------------------------------------------------


def generate_qa_page(
    output_dir: Path,
    input_dir: Path,
    *,
    n: int = 200,
    seed: int = 0,
    statuses: Collection[str] = ("ok", "flagged"),
    sources: Collection[str] | None = None,
    name: str | None = None,
) -> Path | None:
    """Write a self-contained HTML spot-check page ``<output_dir>/qa/<name>.html``.

    Uses the latest record per (image, caption type) from all
    ``captions/*.jsonl`` files. Samples up to ``n`` images that have at least
    one latest record with a status in ``statuses`` (and a source in
    ``sources``, if given), stratified by source (round-robin over the sources
    in a seeded random order, so small sources are represented). Each image
    is embedded as a base64 JPEG thumbnail next to all its caption types;
    status and flags are highlighted; prompt, seed and token count are shown
    on hover. No external CSS, JS or URLs. An image that cannot be read is
    shown as a placeholder with the error. Same files + same ``seed`` = same
    sample.

    Args:
        output_dir: Output directory (SPEC §8.1).
        input_dir: Dataset root the ``image_rel_path`` s are relative to.
        n: Number of images; ``0`` disables the page (returns ``None``).
        seed: Sampling seed.
        statuses: Statuses that make an image eligible.
        sources: Only these sources (``None`` = all).
        name: File name without ``.html`` (the runner passes its
            ``run_id``); default ``qa-<UTC timestamp>``.

    Returns:
        The written path, or ``None`` when ``n == 0``.

    Raises:
        ValueError: ``n`` is not an int ``>= 0``; ``statuses`` is empty or has
            unknown values; ``name`` is not a plain file name; a malformed
            JSONL line (see ``ResumeIndex.from_output_dir``).
        FileNotFoundError: ``input_dir`` is not a directory.
        OSError: The page cannot be written.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be an int >= 0, got {n!r}")
    if n == 0:
        logger.info("QA page disabled (n=0)")
        return None
    status_set = frozenset(statuses)
    if not status_set or not status_set <= CAPTION_STATUSES:
        raise ValueError(
            f"statuses must be a non-empty subset of {sorted(CAPTION_STATUSES)}, "
            f"got {sorted(status_set)!r}"
        )
    if name is None:
        name = "qa-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not isinstance(name, str) or name in ("", ".", "..") or any(c in name for c in "/\\:"):
        raise ValueError(f"name must be a plain file name, got {name!r}")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir is not a directory: {input_dir}")
    output_dir = Path(output_dir)
    source_filter = None if sources is None else frozenset(sources)

    files = _caption_files(output_dir)
    index = ResumeIndex._scan(files)
    by_source: dict[str, list[str]] = defaultdict(list)
    status_totals: dict[str, int] = defaultdict(int)
    for rel, per in index._entries.items():
        eligible_source: str | None = None
        for _, _, status, gid in per.values():
            source = index._groups[gid][2]
            if source_filter is not None and source not in source_filter:
                continue
            status_totals[status] += 1
            if status in status_set:
                eligible_source = source
        if eligible_source is not None:
            by_source[eligible_source].append(rel)
    matching = sum(len(v) for v in by_source.values())
    sample = _stratified_sample(by_source, n, seed)

    # Second pass: load the full winning records of the sampled images only.
    wanted = set(sample)
    winners = {(rel, ct): e[1] for rel in wanted for ct, e in index._entries[rel].items()}
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for file_idx, path in enumerate(files):
        for lineno, rec in _iter_jsonl_file(path):
            rel, ct = rec.get("image_rel_path"), rec.get("caption_type")
            if rel in wanted and winners.get((rel, ct)) == (file_idx << 32) | lineno:
                records[rel][ct] = rec

    caption_types = sorted({ct for per in records.values() for ct in per})
    rows = sorted(
        records.items(),
        key=lambda kv: (_str_field(next(iter(kv[1].values())), "source"), kv[0]),
    )
    input_root = input_dir.resolve()
    body_rows = [_qa_row(rel, per, caption_types, input_dir, input_root) for rel, per in rows]
    page = _qa_document(
        name=name,
        output_dir=output_dir,
        statuses=sorted(status_set),
        sources=None if source_filter is None else sorted(source_filter),
        seed=seed,
        sampled=len(rows),
        matching=matching,
        status_totals=status_totals,
        caption_types=caption_types,
        body_rows=body_rows,
    )
    out_path = output_dir / _QA_DIR / f"{name}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    logger.info(
        "QA page written: %s (%d of %d matching images)", out_path.as_posix(), len(rows), matching,
    )
    return out_path


def _stratified_sample(by_source: Mapping[str, list[str]], n: int, seed: int) -> list[str]:
    """Round-robin over sources (seeded order), each source's images seeded-shuffled."""
    rng = random.Random(seed)
    pools: list[list[str]] = []
    for source in sorted(by_source):
        items = sorted(by_source[source])
        rng.shuffle(items)
        pools.append(items)
    rng.shuffle(pools)
    sample: list[str] = []
    depth = 0
    while len(sample) < n and any(depth < len(p) for p in pools):
        for pool in pools:
            if depth < len(pool) and len(sample) < n:
                sample.append(pool[depth])
        depth += 1
    return sample


_QA_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:16px;color:#1d1d1f;background:#fafafa}
h1{font-size:20px;margin:0 0 4px}
.meta{color:#555;font-size:13px;margin:0 0 12px}
table{border-collapse:collapse;width:100%;background:#fff}
th,td{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:left;font-size:13px}
th{background:#f0f0f0;position:sticky;top:0}
td.img{width:260px}
td.img img{width:256px;height:auto;image-rendering:pixelated;display:block}
td.img .path{font-family:ui-monospace,Consolas,monospace;font-size:11px;word-break:break-all;color:#444}
.noimg{width:256px;height:64px;background:#eee;color:#900;font-size:11px;padding:4px;box-sizing:border-box}
.text{white-space:pre-wrap;margin-top:4px}
.badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;margin-right:4px;color:#fff}
.st-ok{background:#2e7d32}.st-flagged{background:#ef6c00}.st-failed{background:#c62828}
.flag{background:#6a1b9a}
td.cap-flagged{background:#fff8e1}td.cap-failed{background:#ffebee}
td.missing{color:#999}
"""


def _qa_document(
    *,
    name: str,
    output_dir: Path,
    statuses: list[str],
    sources: list[str] | None,
    seed: int,
    sampled: int,
    matching: int,
    status_totals: Mapping[str, int],
    caption_types: list[str],
    body_rows: list[str],
) -> str:
    e = _esc
    totals = ", ".join(f"{s} {status_totals.get(s, 0)}" for s in ("ok", "flagged", "failed"))
    source_text = ", ".join(sources) if sources is not None else "all"
    meta = (
        f"Generated {e(_utc_now())} · output_dir {e(output_dir.as_posix())} · "
        f"statuses {e(', '.join(statuses))} · sources {e(source_text)} · "
        f"seed {e(seed)} · {sampled} of {matching} matching images · latest records: {e(totals)}"
    )
    head_cells = "".join(f"<th>{e(ct)}</th>" for ct in caption_types)
    if body_rows:
        table = (
            f"<table><thead><tr><th>Image</th>{head_cells}</tr></thead>"
            f"<tbody>\n{''.join(body_rows)}</tbody></table>"
        )
    else:
        table = "<p>No records match.</p>"
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>QA {e(name)}</title>\n<style>{_QA_CSS}</style>\n</head>\n<body>\n"
        f"<h1>QA {e(name)}</h1>\n<p class=\"meta\">{meta}</p>\n{table}\n</body>\n</html>\n"
    )


def _qa_row(
    rel: str,
    per_type: Mapping[str, Mapping[str, Any]],
    caption_types: list[str],
    input_dir: Path,
    input_root: Path,
) -> str:
    e = _esc
    first = next(iter(per_type.values()))
    uri, error = _thumbnail_data_uri(input_dir, input_root, rel)
    if uri is not None:
        img = f'<img src="{uri}" alt="{e(rel)}">'
    else:
        img = f'<div class="noimg">image not available: {e(error)}</div>'
    size = f"{e(first.get('image_width'))}×{e(first.get('image_height'))}"
    source = _str_field(first, "source") or "<root>"
    cells = [
        f'<td class="img">{img}<div class="path">{e(rel)}</div>'
        f"<div>{e(source)} · {e(_str_field(first, 'category'))} · {size}</div></td>"
    ]
    for ct in caption_types:
        rec = per_type.get(ct)
        if rec is None:
            cells.append('<td class="missing">—</td>')
            continue
        status = _str_field(rec, "status")
        flags = rec.get("flags") if isinstance(rec.get("flags"), list) else []
        hover = (
            f"prompt: {rec.get('prompt')}\nseed: {rec.get('seed')}\n"
            f"tokens: {rec.get('num_output_tokens')}\nattempt: {rec.get('attempt')}\n"
            f"finish_reason: {rec.get('finish_reason')}"
        )
        badges = f'<span class="badge st-{e(status)}">{e(status)}</span>' + "".join(
            f'<span class="badge flag">{e(f)}</span>' for f in flags
        )
        cells.append(
            f'<td class="cap cap-{e(status)}" title="{e(hover)}">{badges}'
            f'<div class="text">{e(rec.get("caption_text", ""))}</div></td>'
        )
    return "<tr>" + "".join(cells) + "</tr>\n"


def _thumbnail_data_uri(
    input_dir: Path, input_root: Path, rel: str
) -> tuple[str | None, str | None]:
    """``(data URI, None)`` for the image's JPEG thumbnail, or ``(None, error)``."""
    try:
        rel_path = _safe_rel_path(rel)
        path = input_dir.joinpath(*rel_path.parts)
        path.resolve().relative_to(input_root)
        with PIL.Image.open(path) as src:
            thumb = src.convert("RGB")
        thumb.thumbnail((_QA_THUMB_SIZE, _QA_THUMB_SIZE), PIL.Image.Resampling.NEAREST)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
    except Exception as exc:  # noqa: BLE001 - any unreadable image becomes a placeholder
        logger.warning("QA page: cannot embed %s: %s: %s", rel, type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), None


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: str) -> float | None:
    """Epoch seconds of an ISO 8601 timestamp (``Z`` or offset; naive = UTC)."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _coerce_shard(shard: Any) -> tuple[int, int]:
    """``"i/N"`` or ``(i, N)`` -> ``(i, N)`` with ``0 <= i < N``."""
    if isinstance(shard, str):
        parsed = _parse_shard(shard)
        if parsed is not None:
            return parsed
    elif (
        isinstance(shard, (tuple, list))
        and len(shard) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in shard)
        and 0 <= shard[0] < shard[1]
    ):
        return int(shard[0]), int(shard[1])
    raise ValueError(f"shard must be 'i/N' or (i, N) with 0 <= i < N, got {shard!r}")


def _shard_filename(index: int, count: int) -> str:
    """``shard-<iii>-of-<NNN>.jsonl`` (zero-padded to at least 3 digits)."""
    return f"shard-{index:03d}-of-{count:03d}.jsonl"


def _caption_files(output_dir: Path) -> list[Path]:
    """All ``captions/*.jsonl`` files (case-insensitive suffix), sorted by name."""
    captions = output_dir / _CAPTIONS_DIR
    if not captions.is_dir():
        return []
    return sorted(
        (p for p in captions.iterdir() if p.suffix.lower() == ".jsonl" and p.is_file()),
        key=lambda p: p.name,
    )


def _posix(path: PurePath | str) -> str:
    return path.as_posix() if isinstance(path, PurePath) else str(path)


def _str_field(rec: Mapping[str, Any], key: str) -> str:
    value = rec.get(key)
    return value if isinstance(value, str) else ""


def _safe_rel_path(rel_path: PurePath | str) -> PurePosixPath:
    """Validate a relative image path; reject absolute, empty, ``..`` and drive paths."""
    text = _posix(rel_path)
    rel = PurePosixPath(text)
    if (
        not text
        or rel.is_absolute()
        or not rel.parts
        or rel.name in ("", ".", "..")
        or ".." in rel.parts
        or ":" in rel.parts[0]
    ):
        raise ValueError(f"not a safe relative image path: {text!r}")
    return rel


def _json_default(value: Any) -> Any:
    if isinstance(value, PurePath):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dumps(record: CaptionRecord) -> str:
    """One JSON line: keys in dataclass order, POSIX paths, ``ensure_ascii=False``."""
    data = {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def _parse_line(raw: bytes) -> dict[str, Any] | None:
    """The JSON object on one line, or ``None`` if the line is malformed."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _iter_jsonl_file(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(line number, record)`` per non-blank line; malformed final line skipped."""
    with open(path, "rb") as fh:
        pending: tuple[int, bytes] | None = None
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            if pending is not None:
                rec = _parse_line(pending[1])
                if rec is None:
                    raise ValueError(
                        f"{path.as_posix()}:{pending[0]}: malformed JSONL line "
                        f"(not the last line): {_snippet(pending[1])}"
                    )
                yield pending[0], rec
            pending = (lineno, raw)
        if pending is not None:
            rec = _parse_line(pending[1])
            if rec is None:
                logger.warning(
                    "%s:%d: skipping malformed final line (interrupted write?): %s",
                    path.as_posix(), pending[0], _snippet(pending[1]),
                )
            else:
                yield pending[0], rec


def _snippet(raw: bytes, limit: int = 120) -> str:
    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _repair_tail(path: Path) -> None:
    """Remove a malformed final line and terminate an unterminated valid one."""
    if not path.is_file():
        return
    with open(path, "rb+") as fh:
        size = fh.seek(0, os.SEEK_END)
        if size == 0:
            return
        # End of the last non-blank line (skip trailing whitespace bytes).
        end = size
        while end > 0:
            fh.seek(end - 1)
            if fh.read(1) not in (b" ", b"\t", b"\r", b"\n"):
                break
            end -= 1
        if end == 0:
            return
        start = _line_start(fh, end)
        fh.seek(start)
        last = fh.read(end - start)
        if _parse_line(last) is None:
            fh.truncate(start)
            logger.warning(
                "%s: removed malformed final line (%d bytes, interrupted write?) "
                "before appending: %s",
                path.as_posix(), size - start, _snippet(last),
            )
            return
        fh.seek(size - 1)
        if fh.read(1) != b"\n":
            fh.seek(0, os.SEEK_END)
            fh.write(b"\n")
            logger.warning(
                "%s: terminated an unterminated final line before appending", path.as_posix()
            )


def _line_start(fh: Any, end: int, block: int = 65536) -> int:
    """Offset just after the last ``\\n`` before ``end`` (0 if none)."""
    pos = end
    while pos > 0:
        begin = max(0, pos - block)
        fh.seek(begin)
        chunk = fh.read(pos - begin)
        idx = chunk.rfind(b"\n")
        if idx != -1:
            return begin + idx + 1
        pos = begin
    return 0
