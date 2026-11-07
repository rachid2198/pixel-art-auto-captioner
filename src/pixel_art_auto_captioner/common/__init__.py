"""Common types, image utilities, and export helpers.

This module has no internal dependencies — it is the leaf module
in the dependency graph. Submodules will be added per SPEC §15.
"""

from pixel_art_auto_captioner.common.export_utils import (
    build_record,
    save_jsonl_entry,
    save_txt_sidecar,
)
from pixel_art_auto_captioner.common.image_utils import load_image, validate_image
from pixel_art_auto_captioner.common.types import (
    CAPTION_FLAGS,
    CAPTION_STATUSES,
    FINISH_REASONS,
    CaptionRecord,
    CaptionRequest,
    CaptionResult,
    ImageRecord,
    LoadFailure,
)

__all__: list[str] = [
    "ImageRecord",
    "LoadFailure",
    "CaptionRequest",
    "CaptionResult",
    "CaptionRecord",
    "CAPTION_STATUSES",
    "CAPTION_FLAGS",
    "FINISH_REASONS",
    "build_record",
    "load_image",
    "save_jsonl_entry",
    "save_txt_sidecar",
    "validate_image",
]
