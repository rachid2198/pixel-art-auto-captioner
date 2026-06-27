"""Image loading and validation utilities.

Provides two public functions:

- ``load_image``: opens an image from disk, converts to RGB, optionally
  resizes, and returns an ``ImageRecord``.
- ``validate_image``: checks whether a file path points to a valid,
  readable image.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from pixel_art_auto_captioner.common.types import ImageRecord

import logging

logger = logging.getLogger(__name__)


def load_image(
    path: Path, input_root: Path, target_size: tuple[int, int] | None = None
) -> ImageRecord:
    """Open an image file, convert to RGB, optionally resize.

    .. note::
        This function uses ``Image.open()`` to obtain a usable image
        handle.  It is intentionally a **separate code path** from
        :func:`validate_image`, which uses ``Image.open()`` + ``verify()``.
        ``PIL.Image.verify()`` closes the underlying file handle after
        reading the header, so the result of ``verify()`` **cannot** be
        reused for pixel access.  Never merge these two functions.

    Args:
        path: Absolute or relative path to an image file.
        input_root: Root directory of the image source.  The ``rel_path``
            of the returned ``ImageRecord`` is computed relative to this
            root.  Used by export utilities to reconstruct the original
            directory structure in the output tree.
        target_size: Optional ``(width, height)`` tuple. If provided, the
            image is resized to this exact size using ``Image.LANCZOS``.

    Returns:
        An ``ImageRecord`` with the loaded image and metadata.

    Raises:
        FileNotFoundError: If *path* does not exist.
        PIL.UnidentifiedImageError: If *path* is not a valid or
            recognisable image file.
        ValueError: If *path* is not a child of *input_root*.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        logger.error("Cannot identify image file: %s", path)
        raise

    img = img.convert("RGB")

    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)

    # Compute relative path for directory-preserving output
    try:
        rel = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        raise ValueError(
            f"Image path {path} is not under input_root {input_root}"
        )

    return ImageRecord(
        path=path.resolve(),
        stem=path.stem,
        rel_path=rel,
        image=img,
        width=img.width,
        height=img.height,
    )


def validate_image(path: Path) -> bool:
    """Check whether a file can be opened as a valid image.

    Opens the file with PIL and calls ``verify()``.  The file is closed
    after verification.  Only non-corrupt, loadable images return
    ``True``.

    .. warning::
        ``PIL.Image.verify()`` closes the underlying file handle after
        reading the image header.  The verified handle is **not usable**
        for subsequent pixel access or processing.  If you need the
        image data, call :func:`load_image` instead.  Do **not** try
        to refactor this function to also return the verified image;
        keep the two code paths separate.

    Args:
        path: Path to the image file to validate.

    Returns:
        ``True`` if the file is a valid image, ``False`` otherwise.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False