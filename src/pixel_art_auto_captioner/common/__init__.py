"""Common types, image utilities, and export helpers.

This module has no internal dependencies — it is the leaf module
in the dependency graph. Submodules will be added per SPEC §15.
"""

from pixel_art_auto_captioner.common.image_utils import load_image, validate_image
from pixel_art_auto_captioner.common.types import ImageRecord, CaptionRecord

__all__: list[str] = [
    "ImageRecord",
    "CaptionRecord",
    "load_image",
    "validate_image",
]
