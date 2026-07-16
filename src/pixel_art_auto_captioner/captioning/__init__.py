"""Model abstraction and JoyCaption concrete implementation.

Depends on: pixel_art_auto_captioner.common
"""

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.captioning.joycaption import JoyCaptionModel

__all__: list[str] = [
    "CaptionModel",
    "JoyCaptionModel",
]
