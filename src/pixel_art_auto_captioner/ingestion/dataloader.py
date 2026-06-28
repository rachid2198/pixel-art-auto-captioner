"""Image discovery and lazy-loading dataloader.

Provides ``ImageDataLoader``, a configurable iterator that discovers
images from one or more directories, deduplicates and sorts them,
optionally skips already-captioned images, and lazily yields
``ImageRecord`` objects one at a time.

Depends on: ``pixel_art_auto_captioner.common`` (``types``, ``image_utils``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from pixel_art_auto_captioner.common.image_utils import load_image
from pixel_art_auto_captioner.common.types import ImageRecord

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".webp"]


class ImageDataLoader:
    """Iterable loader that discovers images on disk and yields ``ImageRecord`` objects.

    Images are discovered from one or more ``source_dirs``, optionally
    recursing into subdirectories.  Each image is loaded on demand and
    yielded one at a time — the dataloader does **not** pre-load
    everything into memory.

    When ``skip_existing`` is enabled and ``output_dir`` is set,
    images whose corresponding ``.txt`` sidecar already exists in the
    output tree are silently excluded from iteration.
    """

    def __init__(self, config: dict) -> None:
        # -- validate required key ----------------------------------------
        if "source_dirs" not in config or not config["source_dirs"]:
            raise ValueError("config must contain a non-empty 'source_dirs' list")

        self.source_dirs: list[Path] = [
            Path(d).resolve() for d in config["source_dirs"]
        ]
        self.extensions: list[str] = config.get(
            "extensions", DEFAULT_EXTENSIONS
        )
        self.recursive: bool = config.get("recursive", True)
        self.max_images: Optional[int] = config.get("max_images", None)
        self.skip_existing: bool = config.get("skip_existing", True)
        self.output_dir: Optional[Path] = (
            Path(config["output_dir"]).resolve()
            if config.get("output_dir")
            else None
        )
        self.image_size: Optional[tuple[int, int]] = config.get(
            "image_size", None
        )

        # Cached list of paths *after* skip_existing filtering.
        self._filtered: Optional[list[Path]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[Path]:
        """Discover image paths from all source directories.

        Walks each ``source_dir`` (recursively or flat, per
        ``self.recursive``), collects files matching ``self.extensions``,
        deduplicates by resolved path, sorts alphabetically, and
        optionally truncates to ``self.max_images``.

        Returns:
            Sorted list of unique, resolved ``Path`` objects.
        """
        paths: set[Path] = set()

        for source_dir in self.source_dirs:
            if not source_dir.is_dir():
                logger.warning(
                    "Source directory not found, skipping: %s", source_dir
                )
                continue

            for ext in self.extensions:
                pattern = f"*{ext}"
                found = (
                    source_dir.rglob(pattern)
                    if self.recursive
                    else source_dir.glob(pattern)
                )
                for p in found:
                    if p.is_file():
                        paths.add(p.resolve())

        sorted_paths = sorted(paths)

        if self.max_images is not None and len(sorted_paths) > self.max_images:
            logger.info(
                "Limiting to %d images (discovered %d)",
                self.max_images,
                len(sorted_paths),
            )
            sorted_paths = sorted_paths[: self.max_images]

        logger.info("Discovered %d images", len(sorted_paths))
        return sorted_paths

    def load(self, path: Path) -> ImageRecord:
        """Load a single image from disk as an ``ImageRecord``.

        Delegates to :func:`~pixel_art_auto_captioner.common.image_utils.load_image`,
        automatically determining the correct ``input_root`` from
        ``self.source_dirs``.

        Args:
            path: Absolute or relative path to an image file.

        Returns:
            An ``ImageRecord`` with the loaded PIL image and metadata.

        Raises:
            ValueError: If *path* is not under any configured source
                directory, or if any of the errors documented by
                :func:`load_image` occur.
        """
        source = self._find_source_dir(path)
        if source is None:
            raise ValueError(
                f"Image path {path} is not under any configured source directory"
            )
        return load_image(path, input_root=source, target_size=self.image_size)

    def __iter__(self) -> Iterator[ImageRecord]:
        """Iterate over images, yielding ``ImageRecord`` objects one at a time.

        Images whose output sidecar already exists are skipped when
        ``skip_existing`` is enabled.

        Yields:
            One ``ImageRecord`` per image that passes all filters.
        """
        paths = self._get_filtered_paths()
        total = len(paths)

        for idx, p in enumerate(paths, start=1):
            logger.info("[%d/%d] Processing: %s", idx, total, p.name)
            try:
                yield self.load(p)
            except Exception:
                logger.exception("Failed to load image, skipping: %s", p)

    def __len__(self) -> int:
        """Return the number of images that will be processed (after all filters)."""
        return len(self._get_filtered_paths())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_source_dir(self, path: Path) -> Optional[Path]:
        """Return the source directory that *path* lives under, or ``None``."""
        resolved = path.resolve()
        for sd in self.source_dirs:
            try:
                resolved.relative_to(sd)
                return sd
            except ValueError:
                continue
        return None

    def _get_filtered_paths(self) -> list[Path]:
        """Return discovered paths after ``skip_existing`` filtering.

        Result is cached so ``__len__`` and ``__iter__`` agree without
        re-scanning the filesystem.
        """
        if self._filtered is None:
            discovered = self.discover()

            if not self.skip_existing or self.output_dir is None:
                self._filtered = discovered
            else:
                filtered: list[Path] = []
                for p in discovered:
                    source = self._find_source_dir(p)
                    if source is None:
                        # Shouldn't happen — every discovered path
                        # came from a source_dir.  Include it to be
                        # safe rather than silently dropping it.
                        logger.warning(
                            "Image not under any source dir, including: %s", p
                        )
                        filtered.append(p)
                        continue

                    rel = p.resolve().relative_to(source)
                    sidecar = self.output_dir / rel.with_suffix(".txt")
                    if sidecar.exists():
                        logger.info("Skipping (exists): %s", p.name)
                        continue
                    filtered.append(p)
                self._filtered = filtered

        return self._filtered