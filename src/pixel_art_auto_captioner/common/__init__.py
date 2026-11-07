"""Common types, configuration, image utilities, and export helpers.

This module has no internal dependencies — it is the leaf module
in the dependency graph. Submodules will be added per SPEC §15.
"""

from pixel_art_auto_captioner.common.config import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
)
from pixel_art_auto_captioner.common.export_utils import (
    ResumeIndex,
    ShardWriter,
    build_record,
    generate_qa_page,
    read_records,
    save_txt,
)
from pixel_art_auto_captioner.common.image_utils import load_image, validate_image
from pixel_art_auto_captioner.common.prompts import (
    prompt_config_hash,
    render_prompt,
    resolve_source,
)
from pixel_art_auto_captioner.common.quality import check_caption
from pixel_art_auto_captioner.common.seeding import derive_seed
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
    "DEFAULT_CONFIG",
    "load_config",
    "validate_config",
    "render_prompt",
    "resolve_source",
    "prompt_config_hash",
    "derive_seed",
    "check_caption",
    "ShardWriter",
    "ResumeIndex",
    "build_record",
    "save_txt",
    "read_records",
    "generate_qa_page",
    "load_image",
    "validate_image",
]
