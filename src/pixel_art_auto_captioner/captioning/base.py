"""Abstract base class for captioning vision-language models.

Defines the ``CaptionModel`` ABC that all concrete VLM backends
(e.g. JoyCaption) must implement.  This enforces a narrow, typed
interface so that the batch runner and dataloader are decoupled
from any specific model implementation.
"""

from abc import ABC, abstractmethod

import PIL.Image


class CaptionModel(ABC):
    """Abstract interface for any captioning VLM.

    Concrete subclasses must implement :meth:`load`, :meth:`caption`,
    and :meth:`unload`.  The ``model_name`` attribute is set by the
    subclass constructor and used for metadata in ``CaptionRecord``
    generation.
    """

    model_name: str

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load model weights, move to device, set eval mode.

        Args:
            config: Dictionary with model-loading parameters
                (model_path, torch_dtype, device_map, quantization,
                image_size, etc.).

        Raises:
            RuntimeError: If model loading fails (OOM, missing files).
        """
        ...

    @abstractmethod
    def caption(
        self, image: PIL.Image.Image, prompt: str, **gen_kwargs
    ) -> tuple[str, dict]:
        """Generate a caption for a single image.

        Args:
            image: A PIL Image in RGB mode.
            prompt: The user-prompt text to guide caption generation.
            **gen_kwargs: Optional overrides for generation parameters
                (temperature, top_p, max_new_tokens, etc.).

        Returns:
            A tuple of ``(caption_text, generation_metadata_dict)``.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory and release model resources.

        Called by the runner in a ``finally`` block to guarantee
        cleanup even when errors occur during batch processing.
        """
        ...