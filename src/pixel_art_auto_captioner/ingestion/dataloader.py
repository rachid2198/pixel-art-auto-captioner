"""Image discovery, sharding, resume planning and threaded decoding (SPEC v1.1 §4).

``ImageDataLoader`` turns one dataset root (``input_dir``) into an ordered
stream of decoded images for the runner:

1. :meth:`ImageDataLoader.discover` walks ``input_dir`` **once** (one
   ``os.walk``, no symlink following), keeps files whose lowercased suffix is
   in the extension set (so ``.JPG`` matches on Linux and the ``.msgpack``
   leftovers are rejected without being opened), applies the
   ``include_sources`` / ``exclude_sources`` filters while walking, and caches
   the sorted relative paths on the instance.
2. :meth:`ImageDataLoader.plan` keeps the paths of this shard (hash of the
   relative path, SPEC D2), asks the runner's resume predicate which caption
   types are still pending, and applies ``max_images`` to the *pending* images
   (SPEC D4). The result is a :class:`Plan`.
3. :meth:`ImageDataLoader.iter_images` decodes the planned images in a thread
   pool with bounded look-ahead and yields one ``ImageRecord`` or
   ``LoadFailure`` per plan item, in plan order (SPEC D5, D10). Images are not
   resized.

The loader knows nothing about output files; resume information comes in
through the ``is_complete`` predicate.

Depends on ``pixel_art_auto_captioner.common`` only (never ``captioning`` or
``batch``) and never imports ``torch`` or ``vllm`` (SPEC §2.2).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixel_art_auto_captioner.common.config import _parse_shard
from pixel_art_auto_captioner.common.image_utils import load_image
from pixel_art_auto_captioner.common.types import ImageRecord, LoadFailure

logger = logging.getLogger(__name__)

#: Default image extensions (SPEC §4.1). Matched case-insensitively.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")

#: Discovery logs an INFO progress line every this many files examined (SPEC §4.2).
_PROGRESS_EVERY = 50_000

#: Max number of colliding paths listed in the collision WARNING.
_MAX_COLLISIONS_LOGGED = 20

#: Name prefix of the decode threads (lets tests check that all were joined).
_THREAD_NAME_PREFIX = "pac-decode"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """The work for one shard after resume filtering (SPEC §4.3).

    Internal to ``ingestion`` and ``batch`` (not part of ``common.types``).
    Built by :meth:`ImageDataLoader.plan` and consumed by
    :meth:`ImageDataLoader.iter_images`, the runner (SPEC §7) and
    ``--dry_run`` (SPEC D7).

    Invariant: ``images_in_shard == images_already_complete + images_deferred
    + images_planned``.

    Attributes:
        shard: ``(index, count)`` this plan was built for.
        caption_types: The requested caption types, in the order given to
            :meth:`ImageDataLoader.plan`.
        images_discovered: Images found by discovery (whole dataset after
            include/exclude).
        images_in_shard: Discovered images that belong to this shard.
        images_already_complete: Images in this shard with no pending caption
            type.
        images_deferred: Pending images cut by ``max_images``.
        items: Planned images in sorted ``rel_path`` order, each with its
            pending caption types (a subset of ``caption_types``, same order).
    """

    shard: tuple[int, int]
    caption_types: list[str]
    images_discovered: int
    images_in_shard: int
    images_already_complete: int
    images_deferred: int
    items: list[tuple[Path, list[str]]] = field(default_factory=list)

    @property
    def images_planned(self) -> int:
        """Number of images to decode and caption (``len(items)``)."""
        return len(self.items)

    @property
    def captions_planned(self) -> int:
        """Number of (image, caption type) pairs to generate."""
        return sum(len(types) for _, types in self.items)

    def captions_by_source(self) -> dict[str, dict[str, int]]:
        """Count planned captions per source and caption type (for ``--dry_run``, D7).

        Returns:
            ``{source: {caption_type: n}}``, sources sorted by name, caption
            types in ``caption_types`` order. Types with no planned caption
            for a source are omitted. Files directly in ``input_dir`` count
            under source ``""``.
        """
        counts: dict[str, dict[str, int]] = defaultdict(dict)
        for rel_path, types in self.items:
            per_type = counts[_source_of(rel_path)]
            for caption_type in types:
                per_type[caption_type] = per_type.get(caption_type, 0) + 1
        order = {t: i for i, t in enumerate(self.caption_types)}
        return {
            source: dict(
                sorted(counts[source].items(), key=lambda kv: order.get(kv[0], len(order)))
            )
            for source in sorted(counts)
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ImageDataLoader:
    """Discovers, shards, plans and decodes the images of one dataset root (SPEC §4).

    Attributes:
        input_dir: Resolved absolute dataset root.
        extensions: Lowercased extension set (each with a leading dot).
        recursive: Whether subdirectories are walked.
        include_sources: Top-level folders to keep, or ``None`` for all.
        exclude_sources: Top-level folders to skip.
        shard: ``(index, count)``.
        max_images: Cap on pending images after resume, or ``None``.
        num_workers: Decode threads; ``0`` decodes inline.
        prefetch: Max number of plan items submitted for decoding and not yet
            handed to the consumer.
    """

    def __init__(self, config: dict) -> None:
        """Validate and store the loader configuration (SPEC §4.1).

        Unknown keys are ignored, so a caller may pass e.g. the merged
        ``input`` and ``runtime`` config sections.

        Args:
            config: Mapping with the SPEC §4.1 keys: ``input_dir`` (required,
                ``str`` or path), ``extensions``, ``recursive``,
                ``include_sources``, ``exclude_sources``, ``shard``
                (``(index, count)``; the config-file form ``"i/N"`` is also
                accepted), ``max_images``, ``num_workers``, ``prefetch``.

        Raises:
            ValueError: If any value is missing or invalid; all problems are
                listed in one message. In particular, ``input_dir`` must exist
                and be a directory.
        """
        if not isinstance(config, dict):
            raise ValueError(f"loader config must be a dict, got {type(config).__name__}")
        errors: list[str] = []

        raw_dir = config.get("input_dir")
        input_dir: Path | None = None
        if not isinstance(raw_dir, (str, os.PathLike)) or not str(raw_dir):
            errors.append(f"input_dir: is required (a directory path), got {raw_dir!r}")
        else:
            candidate = Path(raw_dir)
            if not candidate.exists():
                errors.append(f"input_dir: does not exist: {raw_dir}")
            elif not candidate.is_dir():
                errors.append(f"input_dir: is not a directory: {raw_dir}")
            else:
                input_dir = candidate.resolve()

        extensions = _parse_extensions(config.get("extensions", list(DEFAULT_EXTENSIONS)), errors)

        recursive = config.get("recursive", True)
        if not isinstance(recursive, bool):
            errors.append(f"recursive: must be true or false, got {recursive!r}")

        include_sources = config.get("include_sources")
        if include_sources is not None and not _is_str_list(include_sources):
            errors.append(
                f"include_sources: must be null or a list of folder names, got {include_sources!r}"
            )
        exclude_sources = config.get("exclude_sources")
        if exclude_sources is None:
            exclude_sources = []
        elif not _is_str_list(exclude_sources):
            errors.append(
                f"exclude_sources: must be a list of folder names, got {exclude_sources!r}"
            )

        shard = _coerce_shard(config.get("shard", (0, 1)))
        if shard is None:
            errors.append(
                "shard: must be (index, count) or 'i/N' with 0 <= index < count, "
                f"got {config.get('shard')!r}"
            )

        max_images = config.get("max_images")
        if max_images is not None and not (_is_int(max_images) and max_images > 0):
            errors.append(f"max_images: must be null or a positive integer, got {max_images!r}")

        num_workers = config.get("num_workers", 4)
        if not (_is_int(num_workers) and num_workers >= 0):
            errors.append(f"num_workers: must be an integer >= 0, got {num_workers!r}")

        prefetch = config.get("prefetch", 64)
        if not (_is_int(prefetch) and prefetch >= 1):
            errors.append(f"prefetch: must be an integer >= 1, got {prefetch!r}")

        if errors:
            raise ValueError("Invalid data loader config:\n  - " + "\n  - ".join(errors))

        assert input_dir is not None and shard is not None  # guaranteed by the checks above
        self.input_dir: Path = input_dir
        self.extensions: frozenset[str] = extensions
        self.recursive: bool = recursive
        self.include_sources: list[str] | None = (
            list(include_sources) if include_sources is not None else None
        )
        self.exclude_sources: list[str] = list(exclude_sources)
        self.shard: tuple[int, int] = shard
        self.max_images: int | None = max_images
        self.num_workers: int = num_workers
        self.prefetch: int = prefetch

        self._discovered: list[Path] | None = None
        self._collisions: frozenset[Path] = frozenset()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> list[Path]:
        """Find all image files under ``input_dir`` (SPEC §4.2).

        One ``os.walk`` (no symlink following). Each file name is matched
        once against the lowercased extension set; other files (e.g.
        ``.msgpack``) are rejected by that suffix check and never opened.
        ``include_sources`` / ``exclude_sources`` prune top-level folders
        before they are entered; when ``include_sources`` is set, files
        directly in ``input_dir`` (source ``""``) are excluded too. With
        ``recursive=False`` only files directly in ``input_dir`` are found.

        The result is cached: later calls return a copy without walking
        again. Logs INFO every 50 000 files examined and a final count, and a
        WARNING listing images in the same folder that share a stem
        (case-insensitively), whose ``.txt`` names collide (SPEC §8.2).

        Returns:
            Paths relative to ``input_dir``, sorted by ``as_posix()``.
        """
        if self._discovered is None:
            self._discovered = self._walk()
        return list(self._discovered)

    def _walk(self) -> list[Path]:
        """Do the single walk behind :meth:`discover`."""
        root = self.input_dir
        include = set(self.include_sources) if self.include_sources is not None else None
        exclude = set(self.exclude_sources)
        found: list[Path] = []
        examined = 0
        top_level_dirs: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            rel_dir = Path(dirpath).relative_to(root)
            if not rel_dir.parts:  # input_dir itself
                top_level_dirs = list(dirnames)
                if not self.recursive:
                    dirnames[:] = []
                else:
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if (include is None or d in include) and d not in exclude
                    ]
                if include is not None:
                    # Files directly in input_dir have source "" (SPEC §3).
                    examined += len(filenames)
                    filenames = []
            for name in filenames:
                examined += 1
                if examined % _PROGRESS_EVERY == 0:
                    logger.info(
                        "Discovery: %d files examined, %d images found", examined, len(found)
                    )
                if os.path.splitext(name)[1].lower() in self.extensions:
                    found.append(rel_dir / name)

        for kind, names in (("include_sources", include or set()), ("exclude_sources", exclude)):
            missing = sorted(names - set(top_level_dirs))
            if missing:
                logger.warning("%s: no such top-level folder in %s: %s", kind, root, missing)

        found.sort(key=Path.as_posix)
        self._collisions = _find_stem_collisions(found)
        if self._collisions:
            listed = sorted(p.as_posix() for p in self._collisions)
            more = len(listed) - _MAX_COLLISIONS_LOGGED
            logger.warning(
                "%d images share a stem with another image in the same folder; their .txt "
                "files will be named <stem>.<ext>.<caption_type>.txt: %s%s",
                len(listed),
                listed[:_MAX_COLLISIONS_LOGGED],
                f" (+{more} more)" if more > 0 else "",
            )
        logger.info(
            "Discovery done: %d images found in %s (%d files examined)", len(found), root, examined
        )
        return found

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, is_complete: Callable[[Path, str], bool], caption_types: list[str]) -> Plan:
        """Select this shard's pending work (SPEC §4.3).

        1. Keeps the discovered paths with
           ``int(sha1(rel_path.as_posix()).hexdigest()[:8], 16) % count == index``
           (D2: membership depends only on the path, so it is stable when
           files are added or removed).
        2. For each image, the pending caption types are the requested types
           for which ``is_complete(rel_path, type)`` is ``False``.
        3. Images with no pending type are already complete.
        4. ``max_images`` caps the pending images, in sorted order (D4: the
           cap applies after resume, so repeated limited runs advance).

        Args:
            is_complete: Resume predicate ``(rel_path, caption_type) -> bool``
                (the runner passes ``ResumeIndex.is_complete``, or a predicate
                that is always ``False`` when resume is off).
            caption_types: Caption types to produce; non-empty, no duplicates.

        Returns:
            The :class:`Plan` for this shard.

        Raises:
            ValueError: If ``caption_types`` is empty, contains a non-string
                or contains duplicates.
        """
        if not _is_str_list(caption_types) or not caption_types:
            raise ValueError(
                f"caption_types must be a non-empty list of strings, got {caption_types!r}"
            )
        if len(set(caption_types)) != len(caption_types):
            raise ValueError(f"caption_types contains duplicates: {caption_types!r}")
        types = list(caption_types)
        index, count = self.shard

        discovered = self.discover()
        in_shard = [p for p in discovered if _shard_index(p, count) == index]

        pending: list[tuple[Path, list[str]]] = []
        for rel_path in in_shard:
            todo = [t for t in types if not is_complete(rel_path, t)]
            if todo:
                pending.append((rel_path, todo))

        items = pending if self.max_images is None else pending[: self.max_images]
        plan = Plan(
            shard=(index, count),
            caption_types=types,
            images_discovered=len(discovered),
            images_in_shard=len(in_shard),
            images_already_complete=len(in_shard) - len(pending),
            images_deferred=len(pending) - len(items),
            items=items,
        )
        logger.info(
            "Plan for shard %d/%d: %d discovered, %d in shard, %d already complete, "
            "%d deferred (max_images=%s), %d images / %d captions planned",
            index,
            count,
            plan.images_discovered,
            plan.images_in_shard,
            plan.images_already_complete,
            plan.images_deferred,
            self.max_images,
            plan.images_planned,
            plan.captions_planned,
        )
        return plan

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def iter_images(self, plan: Plan) -> Iterator[ImageRecord | LoadFailure]:
        """Decode the planned images, yielding them in plan order (SPEC §4.4).

        Exactly one item is yielded per ``plan.items`` entry, in the same
        order, so the consumer can pair them with their pending caption
        types. Decoding uses
        :func:`~pixel_art_auto_captioner.common.image_utils.load_image`
        (RGB, original size, handle closed, **not** resized) on
        ``num_workers`` threads, with at most ``prefetch`` items submitted and
        not yet yielded; ``num_workers=0`` decodes inline. Any ``Exception``
        while decoding an image yields a ``LoadFailure`` (logged at WARNING);
        nothing is dropped. ``BaseException`` s such as ``KeyboardInterrupt``
        propagate.

        When the generator is closed before the end (``close()``, or garbage
        collection after the consumer stops early; wrap it in
        ``contextlib.closing`` to make this immediate), queued decodes are
        cancelled and the worker threads are joined. They are also joined
        when the generator ends or a ``BaseException`` propagates from a
        decode.

        Args:
            plan: A plan from :meth:`plan` of this loader.

        Yields:
            ``ImageRecord`` for each decoded image, ``LoadFailure`` for each
            image that could not be decoded.
        """
        paths = [rel_path for rel_path, _ in plan.items]
        if not paths:
            return
        if self.num_workers == 0:
            for rel_path in paths:
                yield self._decode(rel_path)
            return

        executor = ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix=_THREAD_NAME_PREFIX
        )
        queue: deque[Future[ImageRecord | LoadFailure]] = deque()
        next_index = 0
        try:
            while queue or next_index < len(paths):
                while len(queue) < self.prefetch and next_index < len(paths):
                    queue.append(executor.submit(self._decode, paths[next_index]))
                    next_index += 1
                yield queue.popleft().result()
        finally:
            for future in queue:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _decode(self, rel_path: Path) -> ImageRecord | LoadFailure:
        """Decode one image; turn any ``Exception`` into a ``LoadFailure``."""
        path = self.input_dir / rel_path
        try:
            record = load_image(path, self.input_dir)
            if record.rel_path != rel_path:
                # A symlinked file resolves to another key; captioning it under
                # the target's key would break resume for the planned key.
                raise ValueError(f"resolves to {record.rel_path.as_posix()} (symlink?)")
            return record
        except Exception as exc:  # noqa: BLE001 - SPEC §4.4: nothing is silently dropped
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Could not decode %s: %s", rel_path.as_posix(), error)
            return LoadFailure(
                path=path, rel_path=rel_path, source=_source_of(rel_path), error=error
            )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _shard_index(rel_path: Path, count: int) -> int:
    """Shard of ``rel_path`` among ``count`` shards (SPEC §4.3, D2).

    ``int(sha1(rel_path.as_posix() as UTF-8).hexdigest()[:8], 16) % count``;
    POSIX form, so the result is the same on Windows and Linux.
    """
    digest = hashlib.sha1(rel_path.as_posix().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % count


def _source_of(rel_path: Path) -> str:
    """First component of ``rel_path``, or ``""`` for a file directly in the root (SPEC §3)."""
    return rel_path.parts[0] if len(rel_path.parts) > 1 else ""


def _find_stem_collisions(paths: Sequence[Path]) -> frozenset[Path]:
    """Paths sharing (folder, case-folded stem) with another path (SPEC §8.2)."""
    groups: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for p in paths:
        groups[(p.parent, p.stem.casefold())].append(p)
    return frozenset(p for group in groups.values() if len(group) > 1 for p in group)


def _parse_extensions(value: Any, errors: list[str]) -> frozenset[str]:
    """Lowercase the extensions and add a missing leading dot."""
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(e, str) and e.strip(".") for e in value)
    ):
        errors.append(
            f"extensions: must be a non-empty list of extensions like '.jpg', got {value!r}"
        )
        return frozenset()
    return frozenset(e.lower() if e.startswith(".") else "." + e.lower() for e in value)


def _coerce_shard(value: Any) -> tuple[int, int] | None:
    """``(index, count)`` from a 2-sequence of ints or an ``"i/N"`` string; ``None`` if invalid."""
    if isinstance(value, str):
        return _parse_shard(value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(_is_int(v) for v in value):
        index, count = value
        if 0 <= index < count:
            return int(index), int(count)
    return None


def _is_int(value: Any) -> bool:
    """``True`` for ``int`` values that are not ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_list(value: Any) -> bool:
    """``True`` for a list or tuple of strings."""
    return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
