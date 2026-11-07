"""Image loading and validation utilities (SPEC v1.1 §4.5).

Two public functions:

- ``load_image``: decodes an image from disk into an ``ImageRecord`` (RGB,
  original size recorded, file handle closed before returning).
- ``validate_image``: checks whether a file is a readable image. Kept as a
  utility; the v1.1 pipeline does not call it (decode errors surface as
  ``LoadFailure`` from the data loader instead).

The two functions are deliberately **separate code paths** with separate
``Image.open()`` calls: ``PIL.Image.verify()`` leaves the image unusable for
pixel access, so a verified image must never be reused for decoding
(AGENTS.md §6.1).

This module imports only the standard library, Pillow and ``common.types``;
it never imports ``torch`` or ``vllm`` (SPEC §2.2).
"""

import logging
from pathlib import Path

from PIL import Image

from pixel_art_auto_captioner.common.types import ImageRecord

logger = logging.getLogger(__name__)


def load_image(
    path: Path, input_root: Path, target_size: tuple[int, int] | None = None
) -> ImageRecord:
    """Decode an image file into an ``ImageRecord``.

    The file is opened in a ``with`` block and ``.convert("RGB")`` is called
    inside it, which forces a full decode, so the returned image owns its
    pixels and the file handle is closed when this function returns (also on
    error). ``width``/``height`` are the image's **original** size, even when
    ``target_size`` is given (SPEC §3, §4.5; fixes v1.0 bugs B5 and B10).

    Args:
        path: Path to the image file. Must lie under ``input_root`` (after
            resolving both).
        input_root: Dataset root (``input_dir``). ``rel_path`` and ``source``
            of the returned record are computed relative to it.
        target_size: Optional ``(width, height)``. If given, the returned
            ``image`` is resized to exactly this size with Lanczos
            resampling; ``width``/``height`` still report the original size.
            The v1.1 pipeline does not resize (the backend's processor does).

    Returns:
        An ``ImageRecord`` with ``path`` (resolved, absolute), ``rel_path``
        (relative to ``input_root``), ``source`` (first component of
        ``rel_path``, or ``""`` for a file directly in ``input_root``),
        ``stem``, the RGB ``image``, and the original ``width``/``height``.

    Raises:
        ValueError: If ``path`` is not under ``input_root``. Checked before
            the file is opened.
        FileNotFoundError: If ``path`` does not exist.
        PIL.UnidentifiedImageError: If the file is not a recognisable image
            (e.g. empty or not an image format).
        OSError: If the file cannot be read or decoded (e.g. a truncated
            JPEG, or ``path`` is a directory). Other exceptions raised by
            Pillow during decoding propagate unchanged; callers that must
            not stop (the data loader) catch ``Exception``.
    """
    abs_path = path.resolve()
    root = input_root.resolve()
    try:
        rel_path = abs_path.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Image path {path} is not under input_root {input_root}"
        ) from None

    with Image.open(abs_path) as img:
        width, height = img.size
        # convert() decodes the full image while the file is still open
        # (it returns a loaded copy even when the mode is already RGB).
        image = img.convert("RGB")

    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    source = rel_path.parts[0] if len(rel_path.parts) > 1 else ""

    return ImageRecord(
        path=abs_path,
        rel_path=rel_path,
        source=source,
        stem=abs_path.stem,
        image=image,
        width=width,
        height=height,
    )


def validate_image(path: Path) -> bool:
    """Check whether a file can be opened and verified as an image.

    Opens the file with Pillow and calls ``verify()``. Any ``Exception``
    (missing file, directory, unknown format, corrupt data, decompression-bomb
    error, ...) yields ``False``; it never raises an ``Exception``.

    .. warning::
        ``PIL.Image.verify()`` leaves the image unusable for pixel access.
        Do not refactor this function to also return the image; use
        :func:`load_image` (a separate ``Image.open()``) to decode
        (AGENTS.md §6.1).

    Args:
        path: Path to the file to check.

    Returns:
        ``True`` if Pillow can identify and verify the file, ``False``
        otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as exc:  # noqa: BLE001 - SPEC §4.5: catch all Exception subclasses
        logger.debug("Invalid image %s: %s: %s", path, type(exc).__name__, exc)
        return False
