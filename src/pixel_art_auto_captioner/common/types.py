"""Shared data types for the pixel-art auto-captioner pipeline.

Defines the two canonical records that flow through the system:
- ``ImageRecord``: an image loaded from disk, ready for captioning.
- ``CaptionRecord``: a generated caption linked to its source image.
"""

from dataclasses import dataclass
from pathlib import Path

import PIL.Image


@dataclass
class ImageRecord:
    """A single image loaded from disk and ready for captioning.

    Attributes:
        path: Absolute path to the source image file on disk.
        stem: Relative-path identifier from the ``input_root`` directory,
            with path separators replaced by underscores
            (e.g. ``"folder1_sprite"``).  Uniquely identifies the image
            across nested source directories.
        image: The decoded image as a PIL Image in RGB mode.
        width: Original image width in pixels.
        height: Original image height in pixels.
    """

    path: Path
    stem: str
    image: PIL.Image.Image
    width: int
    height: int


@dataclass
class CaptionRecord:
    """A generated caption linked to its source image.

    Attributes:
        image_path: Absolute path to the source image file.
        image_stem: Relative-path identifier (matches
            ``ImageRecord.stem``).  E.g. ``"folder1_sprite"`` instead
            of just ``"sprite"``.
        caption_text: The generated caption string.
        model_name: Identifier for the VLM that produced the caption
            (e.g. ``"joycaption-beta-one"``).
        prompt_template: The user prompt text used for generation.
        generation_params: Dictionary of generation parameters
            (temperature, max_new_tokens, top_p, etc.).
        timestamp_utc: ISO 8601 UTC timestamp of caption generation.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        extra: Optional additional metadata dictionary.
    """

    image_path: Path
    image_stem: str
    caption_text: str
    model_name: str
    prompt_template: str
    generation_params: dict
    timestamp_utc: str
    image_width: int
    image_height: int
    extra: dict
