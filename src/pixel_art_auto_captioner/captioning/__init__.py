"""Captioning backends: the :class:`CaptionModel` interface and :func:`get_model`.

Backend modules (``joycaption_hf``, ``joycaption_vllm``) are not imported here;
:func:`get_model` imports the selected one lazily, so importing this package
never pulls in ``torch``, ``transformers`` or ``vllm`` (SPEC §2.2, D6).

Depends on: ``pixel_art_auto_captioner.common``.
"""

from pixel_art_auto_captioner.captioning.base import CaptionModel, get_model

__all__: list[str] = ["CaptionModel", "get_model"]
