"""JoyCaption Beta One concrete implementation.

Provides ``JoyCaptionModel``, a :class:`CaptionModel` subclass that
wraps the LLaVA-based JoyCaption Beta One VLM with NF4/int8/full-precision
loading, vision-tower head fix, and a clean ``caption(image, prompt)``
interface.

Reference: ``main.py`` (preserved prototype).
"""

from __future__ import annotations

import logging
from typing import Any

import PIL.Image
import torch

from pixel_art_auto_captioner.captioning.base import CaptionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default generation parameters (matching reference main.py)
# ---------------------------------------------------------------------------

DEFAULT_GEN_PARAMS: dict[str, Any] = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": None,
    "use_cache": True,
    "suppress_tokens": None,
}

# Default model-loading configuration
DEFAULT_LOAD_CONFIG: dict[str, Any] = {
    "torch_dtype": "bfloat16",
    "device_map": 0,
    "quantization": "nf4",
    "image_size": (384, 384),
}


class JoyCaptionModel(CaptionModel):
    """JoyCaption Beta One captioning model (LLaVA backbone).

    Implements the :class:`CaptionModel` abstract interface for the
    ``fancyfeast/llama-joycaption-beta-one-hf-llava`` model, with
    support for NF4 quantization (default), int8, and full-precision
    loading.

    Example usage::

        model = JoyCaptionModel()
        model.load({"model_path": "./Models"})
        caption, meta = model.caption(image, "Describe this image.")
        model.unload()
    """

    def __init__(self, model_name: str = "joycaption-beta-one") -> None:
        """Initialise the JoyCaption model handle.

        The model is **not** loaded until :meth:`load` is called.

        Args:
            model_name: Identifier stored in ``CaptionRecord`` metadata.
        """
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # CaptionModel interface
    # ------------------------------------------------------------------

    def load(self, config: dict) -> None:
        """Load model weights, move to device, set eval mode.

        Accepts a config dict with the keys documented in SPEC §3.2:

        ================  ==========  ====================================
        Key               Default     Description
        ================  ==========  ====================================
        ``model_path``    *required*  Local path or HF repo ID
        ``torch_dtype``   ``bfloat16``  ``bfloat16``, ``float16``, ``float32``
        ``device_map``    ``0``       ``"auto"``, ``"cuda:0"``, or integer
        ``quantization``  ``nf4``     ``"nf4"``, ``"int8"``, or ``None``
        ``image_size``    ``(384,384)``  Processor resize target
        ================  ==========  ====================================

        Args:
            config: Model-loading configuration dictionary.

        Raises:
            RuntimeError: If model loading fails (OOM, missing files).
            ValueError: If ``model_path`` is missing or empty.
        """
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            LlavaForConditionalGeneration,
        )

        # -- validate required key ----------------------------------------
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("config must contain a non-empty 'model_path'")

        torch_dtype_str = config.get("torch_dtype", DEFAULT_LOAD_CONFIG["torch_dtype"])
        device_map = config.get("device_map", DEFAULT_LOAD_CONFIG["device_map"])
        quantization = config.get("quantization", DEFAULT_LOAD_CONFIG["quantization"])
        image_size = config.get("image_size", DEFAULT_LOAD_CONFIG["image_size"])

        # -- resolve torch dtype ------------------------------------------
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if torch_dtype_str not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype '{torch_dtype_str}'. "
                f"Choose from: {list(dtype_map.keys())}"
            )
        torch_dtype = dtype_map[torch_dtype_str]
        compute_dtype = torch_dtype  # used for bnb compute dtype as well

        # -- quantization config ------------------------------------------
        quantization_config = None
        if quantization == "nf4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif quantization == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization is not None:
            raise ValueError(
                f"Unsupported quantization '{quantization}'. "
                f"Choose from: 'nf4', 'int8', or None."
            )

        logger.info(
            "Loading JoyCaption model from %s (dtype=%s, quantization=%s, device=%s)...",
            model_path,
            torch_dtype_str,
            quantization,
            device_map,
        )

        try:
            # -- processor -------------------------------------------------
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._processor.image_processor.size = {
                "height": image_size[0],
                "width": image_size[1],
            }
            self._processor.image_processor.do_resize = True

            # -- model -----------------------------------------------------
            self._model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                device_map=device_map,
            )
            self._model.eval()

            # -- vision-tower head fix (compatibility shim) ----------------
            _apply_vision_tower_fix(self._model)

            self._loaded = True
            logger.info("JoyCaption model loaded successfully.")

        except Exception:
            logger.exception("Failed to load JoyCaption model from %s", model_path)
            self._model = None
            self._processor = None
            self._loaded = False
            raise

    def caption(
        self, image: PIL.Image.Image, prompt: str, **gen_kwargs: Any
    ) -> tuple[str, dict]:
        """Generate a caption for a single image.

        Args:
            image: A PIL Image in RGB mode.
            prompt: The user-prompt text to guide caption generation.
            **gen_kwargs: Optional overrides for generation parameters
                (``temperature``, ``top_p``, ``max_new_tokens``, etc.).
                If not provided, :data:`DEFAULT_GEN_PARAMS` are used.

        Returns:
            A tuple of ``(caption_text, generation_metadata_dict)``.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if not self._loaded or self._model is None or self._processor is None:
            raise RuntimeError(
                "Model is not loaded. Call model.load(config) before caption()."
            )

        # Merge defaults with user overrides
        merged_params = {**DEFAULT_GEN_PARAMS, **gen_kwargs}

        # Build chat message
        message = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]

        convo_string = self._processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Process text + image
        inputs = self._processor(
            text=[convo_string], images=[image], return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Cast pixel_values to model dtype
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)

        # Generate under inference_mode
        with torch.inference_mode():
            generate_ids = self._model.generate(**inputs, **merged_params)[0]

        # Trim prompt tokens
        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        # Decode
        caption_text = self._processor.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        caption_text = caption_text.strip()

        # Build metadata dict with the actual params used
        metadata = {"gen_params": merged_params}

        return caption_text, metadata

    def unload(self) -> None:
        """Free GPU memory and release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("JoyCaption model unloaded; CUDA cache cleared.")
        else:
            logger.info("JoyCaption model unloaded.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_vision_tower_fix(model: Any) -> None:
    """Apply the LLaVA vision-tower ``out_proj`` linear-layer replacement.

    Some versions of the JoyCaption model ship with a mismatched
    ``out_proj`` layer in the vision tower's attention head.  This shim
    replaces it with a fresh ``nn.Linear`` of the correct shape, placed
    on the model's device with the model's dtype.

    The fix is guarded by a try/except because future model versions
    may not require it.
    """
    try:
        attention = model.model.vision_tower.head.attention
        attention.out_proj = torch.nn.Linear(
            attention.embed_dim,
            attention.embed_dim,
            device=model.device,
            dtype=model.dtype,
        )
        logger.debug("Vision-tower out_proj fix applied.")
    except Exception:
        logger.debug(
            "Vision-tower out_proj fix skipped (model may not need it).",
            exc_info=True,
        )