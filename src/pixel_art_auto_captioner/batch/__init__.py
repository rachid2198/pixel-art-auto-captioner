"""Orchestration runner — wires dataloader → model → exporter.

Depends on: pixel_art_auto_captioner.common, .ingestion, .captioning
"""

from pixel_art_auto_captioner.batch.runner import CaptionRunner

__all__: list[str] = [
    "CaptionRunner",
]
