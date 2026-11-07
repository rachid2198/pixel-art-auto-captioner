"""JoyCaption Beta One on Hugging Face ``transformers`` (local/dev backend).

Provides :class:`JoyCaptionHFModel` (SPEC §6.2), the ``"hf"`` backend used for
local runs on a consumer GPU (RTX 3060, NF4). It replaces the v1.0
``captioning/joycaption.py``; :data:`JoyCaptionModel` stays as an alias for one
release.

Behaviour (SPEC §6.2):

- NF4 / int8 quantization keeps the vision tower and the multi-modal projector
  unquantized (``llm_int8_skip_modules``), so the image encoder runs in the
  configured ``dtype`` (decision Q3). There is no vision-tower layer
  replacement hack.
- The processor keeps its default 384×384 resize.
- Requests are processed one at a time (D9), each with its own seed. The chat
  handling follows ``main.py`` exactly (AGENTS.md §6.6):
  ``apply_chat_template(messages, tokenize=False, add_generation_prompt=True)``
  and then ``processor(text=[convo], images=[image], return_tensors="pt")``.
- A failure in one request becomes ``CaptionResult(finish_reason="error")``;
  ``torch.cuda.OutOfMemoryError`` is systemic and re-raised (SPEC §6.1).

``torch`` and ``transformers`` are imported lazily inside :meth:`load`, so this
module itself imports on a machine without them (the CPU test suite injects
fakes through the constructor). ``bitsandbytes`` is never imported here;
``transformers`` loads it when a quantization config is used.

Depends on ``pixel_art_auto_captioner.captioning.base`` and
``pixel_art_auto_captioner.common`` only.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import logging
import os
from typing import Any

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GEN_PARAMS",
    "DEFAULT_LOAD_CONFIG",
    "QUANT_SKIP_MODULES",
    "SYSTEM_PROMPT",
    "JoyCaptionHFModel",
    "JoyCaptionModel",
]

#: System prompt of every conversation (SPEC §6.2, unchanged from ``main.py``).
SYSTEM_PROMPT = "You are a helpful image captioner."

#: ``load()`` defaults (SPEC §6.2 table). ``quantization`` / ``device_map`` are
#: read from the ``hf`` sub-section of the ``model`` config when present.
DEFAULT_LOAD_CONFIG: dict[str, Any] = {
    "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "revision": None,
    "dtype": "bfloat16",
    "quantization": "nf4",
    "device_map": 0,
}

#: Generation defaults used when a ``CaptionRequest.gen_params`` key is missing
#: or ``None`` (the values ``main.py`` uses; ``top_k=None`` disables top-k).
DEFAULT_GEN_PARAMS: dict[str, Any] = {
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": None,
}

#: Modules kept out of bitsandbytes quantization. ``vision_tower`` and
#: ``multi_modal_projector`` per SPEC §6.2; ``lm_head`` because an explicit
#: ``llm_int8_skip_modules`` list replaces transformers' default skip list
#: (which is ``lm_head``), so it must be listed to stay unquantized as in v1.0.
QUANT_SKIP_MODULES: tuple[str, ...] = ("vision_tower", "multi_modal_projector", "lm_head")

_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")
_QUANTIZATIONS: tuple[str | None, ...] = ("nf4", "int8", None)


def _dist_version(name: str) -> str | None:
    """Return the installed version of distribution ``name``, or ``None``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - broken metadata should not break describe()
        logger.debug("could not read the version of %s", name, exc_info=True)
        return None


def _error_text(exc: BaseException) -> str:
    """Format an exception as ``"<ExceptionType>: <message>"``."""
    return f"{type(exc).__name__}: {exc}"


def _eos_ids(*values: Any) -> frozenset[int]:
    """Collect EOS token ids from ints, lists of ints or ``None`` values."""
    ids: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            ids.update(int(v) for v in value if v is not None)
        else:
            ids.add(int(value))
    return frozenset(ids)


class JoyCaptionHFModel(CaptionModel):
    """JoyCaption Beta One (LLaVA) on Hugging Face ``transformers`` (SPEC §6.2).

    Construct with no arguments (as :func:`~pixel_art_auto_captioner.captioning.get_model`
    does), then call :meth:`load` with the ``model`` config section. The
    keyword parameters exist only to inject fake libraries in CPU tests.

    Example::

        model = JoyCaptionHFModel()
        try:
            model.load({"model_path": "./Models", "hf": {"quantization": "nf4"}})
            result = model.caption(request)
        finally:
            model.unload()

    Args:
        torch_module: Object used instead of ``import torch`` (tests only).
        transformers_module: Object used instead of ``import transformers``
            (tests only).

    Attributes:
        model_name: ``"joycaption-beta-one"``.
        backend: ``"hf"``.
    """

    model_name = "joycaption-beta-one"
    backend = "hf"

    def __init__(self, *, torch_module: Any | None = None, transformers_module: Any | None = None) -> None:
        self._torch_override = torch_module
        self._transformers_override = transformers_module
        self._torch: Any = None
        self._transformers: Any = None
        self._model: Any = None
        self._processor: Any = None
        self._torch_dtype: Any = None
        self._eos_token_ids: frozenset[int] = frozenset()
        self._settings: dict[str, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """``True`` between a successful :meth:`load` and :meth:`unload`."""
        return self._loaded

    # ------------------------------------------------------------------
    # CaptionModel interface
    # ------------------------------------------------------------------

    def load(self, config: dict) -> None:
        """Load the processor and the LLaVA model.

        Reads ``model_path``, ``revision`` and ``dtype`` from ``config`` and
        ``quantization`` / ``device_map`` from ``config["hf"]`` (falling back to
        the top level, then to :data:`DEFAULT_LOAD_CONFIG`). Other keys (e.g.
        ``backend``, ``vllm``) are ignored. If the model is already loaded it
        is unloaded first.

        Args:
            config: The ``model`` section of the merged config (SPEC §9.2).

        Raises:
            ValueError: Invalid ``model_path``, ``dtype``, ``quantization`` or
                ``hf`` section (checked before any library is imported).
            ImportError: ``torch`` or ``transformers`` is not installed.
            Exception: Whatever ``from_pretrained`` raises (missing weights,
                OOM, ...). The partial state is released before re-raising.
        """
        settings = self._resolve_settings(config)
        if self._loaded:
            logger.warning("load() called while a model is loaded; unloading it first")
            self.unload()

        self._settings = settings
        model_path = settings["model_path"]
        logger.info(
            "Loading JoyCaption (hf) from %s (revision=%s, dtype=%s, quantization=%s, device_map=%r)",
            model_path,
            settings["revision"],
            settings["dtype"],
            settings["quantization"],
            settings["device_map"],
        )
        try:
            torch = self._torch_override if self._torch_override is not None else importlib.import_module("torch")
            self._torch = torch
            transformers = (
                self._transformers_override
                if self._transformers_override is not None
                else importlib.import_module("transformers")
            )
            self._transformers = transformers

            torch_dtype = getattr(torch, settings["dtype"])
            quantization_config = self._quantization_config(transformers, settings["quantization"], torch_dtype)

            revision_kwargs: dict[str, Any] = {}
            if settings["revision"] is not None:
                revision_kwargs["revision"] = settings["revision"]

            # Default 384x384 resize: the image processor is not touched (SPEC §6.2.3).
            self._processor = transformers.AutoProcessor.from_pretrained(model_path, **revision_kwargs)

            model_kwargs: dict[str, Any] = {"dtype": torch_dtype, "device_map": settings["device_map"]}
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            model_kwargs.update(revision_kwargs)
            self._model = transformers.LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            self._model.eval()

            self._torch_dtype = torch_dtype
            self._eos_token_ids = _eos_ids(
                getattr(getattr(self._model, "generation_config", None), "eos_token_id", None),
                getattr(getattr(self._processor, "tokenizer", None), "eos_token_id", None),
            )
            if not self._eos_token_ids:
                logger.warning("no EOS token id found; finish_reason will be 'length' whenever max_new_tokens is hit")
        except BaseException:
            logger.exception("Failed to load JoyCaption (hf) from %s", model_path)
            self._release()
            raise

        self._loaded = True
        logger.info("JoyCaption (hf) loaded on %s", getattr(self._model, "device", "?"))

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        """Caption ``requests`` one at a time, in order (D9).

        For each request: seed ``torch`` (CPU and CUDA) with ``request.seed``,
        build the ``[system, user]`` messages, render the chat template
        (``tokenize=False``), run the processor on one conversation and one
        image, move the tensors to the model's device, cast ``pixel_values``
        to the model dtype and sample under ``torch.inference_mode()``. Only
        the newly generated tokens are decoded.

        Args:
            requests: Generation jobs.

        Returns:
            One :class:`CaptionResult` per request, in order. ``finish_reason``
            is ``"length"`` when ``max_new_tokens`` tokens were generated and
            the last one is not EOS, ``"stop"`` otherwise, and ``"error"``
            (with ``error="<Type>: <message>"``) when the request failed.

        Raises:
            RuntimeError: The model is not loaded.
            torch.cuda.OutOfMemoryError: CUDA ran out of memory (systemic).
        """
        if not self._loaded:
            raise RuntimeError("JoyCaptionHFModel is not loaded; call load(config) first")
        results: list[CaptionResult] = []
        for request in requests:
            results.append(self._caption_one(request))
        return results

    def unload(self) -> None:
        """Release the model and processor and empty the CUDA cache.

        Safe at any time: before :meth:`load`, after a failed or partial
        :meth:`load`, and more than once. Never raises because of the model's
        state; a failing ``torch.cuda.empty_cache()`` is logged and ignored.
        """
        was_loaded = self._loaded
        self._release()
        if was_loaded:
            logger.info("JoyCaption (hf) unloaded")

    def describe(self) -> dict[str, Any]:
        """Describe the backend for the run manifest (SPEC §6.1, D11).

        Returns:
            ``model_name``, ``backend``, ``model_path``, ``revision`` (as
            configured, ``None`` if unset), ``dtype``, ``quantization``,
            ``device_map``, ``quant_skip_modules`` (``None`` without
            quantization) and ``torch_version``, ``transformers_version``,
            ``bitsandbytes_version`` (``None`` when unknown). The settings are
            those of the last :meth:`load` call (``None`` before the first)
            and are kept after :meth:`unload`.
        """
        settings = self._settings
        quantization = settings.get("quantization")
        torch = self._torch if self._torch is not None else self._torch_override
        transformers = self._transformers if self._transformers is not None else self._transformers_override
        return {
            **super().describe(),
            "model_path": settings.get("model_path"),
            "revision": settings.get("revision"),
            "dtype": settings.get("dtype"),
            "quantization": quantization,
            "device_map": settings.get("device_map"),
            "quant_skip_modules": list(QUANT_SKIP_MODULES) if quantization is not None and settings else None,
            "torch_version": self._library_version(torch, "torch"),
            "transformers_version": self._library_version(transformers, "transformers"),
            "bitsandbytes_version": _dist_version("bitsandbytes"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_settings(config: dict) -> dict[str, Any]:
        """Merge ``config`` (+ its ``hf`` sub-section) with the defaults and validate."""
        if not isinstance(config, dict):
            raise ValueError(f"model config must be a dict, got {type(config).__name__}")
        hf = config.get("hf")
        if hf is None:
            hf = {}
        if not isinstance(hf, dict):
            raise ValueError(f"model.hf must be a dict or null, got {type(hf).__name__}")

        settings: dict[str, Any] = {}
        for key in ("model_path", "revision", "dtype"):
            settings[key] = config.get(key, DEFAULT_LOAD_CONFIG[key])
        for key in ("quantization", "device_map"):
            if key in hf:
                settings[key] = hf[key]
            else:
                settings[key] = config.get(key, DEFAULT_LOAD_CONFIG[key])

        problems: list[str] = []
        model_path = settings["model_path"]
        if isinstance(model_path, os.PathLike):
            model_path = os.fspath(model_path)
            settings["model_path"] = model_path
        if not isinstance(model_path, str) or not model_path.strip():
            problems.append(f"model_path must be a non-empty string, got {model_path!r}")
        revision = settings["revision"]
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            problems.append(f"revision must be a non-empty string or null, got {revision!r}")
        if settings["dtype"] not in _DTYPES:
            problems.append(f"dtype must be one of {list(_DTYPES)}, got {settings['dtype']!r}")
        if settings["quantization"] not in _QUANTIZATIONS:
            problems.append(f"hf.quantization must be one of ['nf4', 'int8', None], got {settings['quantization']!r}")
        if problems:
            raise ValueError("invalid hf model config: " + "; ".join(problems))
        return settings

    @staticmethod
    def _quantization_config(transformers: Any, quantization: str | None, torch_dtype: Any) -> Any:
        """Build the ``BitsAndBytesConfig`` for ``quantization`` (``None`` → no quantization)."""
        if quantization is None:
            return None
        skip = list(QUANT_SKIP_MODULES)
        if quantization == "nf4":
            return transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
                llm_int8_skip_modules=skip,
            )
        return transformers.BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)

    @staticmethod
    def _library_version(module: Any, dist_name: str) -> str | None:
        """Version of an imported (or injected) library, else its installed distribution."""
        version = getattr(module, "__version__", None) if module is not None else None
        if version is not None:
            return str(version)
        return _dist_version(dist_name)

    @staticmethod
    def _gen_kwargs(gen_params: dict | None) -> tuple[int, dict[str, Any]]:
        """Return ``(max_new_tokens, generate kwargs)`` for one request.

        Missing or ``None`` values use :data:`DEFAULT_GEN_PARAMS`. Unknown keys
        are ignored. ``temperature == 0`` means greedy decoding
        (``do_sample=False``), which ``transformers`` requires.

        Raises:
            ValueError: A parameter has an invalid type or value.
        """
        params = dict(gen_params or {})
        values = {
            key: (params.get(key) if params.get(key) is not None else default)
            for key, default in DEFAULT_GEN_PARAMS.items()
        }
        max_new_tokens = values["max_new_tokens"]
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
        temperature = values["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError(f"temperature must be a number >= 0, got {temperature!r}")
        top_p = values["top_p"]
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")
        top_k = values["top_k"]
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError(f"top_k must be a non-negative int or null, got {top_k!r}")

        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "use_cache": True, "suppress_tokens": None}
        if temperature == 0:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        return max_new_tokens, kwargs

    def _caption_one(self, request: CaptionRequest) -> CaptionResult:
        """Run one request; per-request failures become an error result."""
        torch = self._torch
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        try:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
            max_new_tokens, gen_kwargs = self._gen_kwargs(request.gen_params)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt},
            ]
            convo = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not isinstance(convo, str):
                raise TypeError(f"apply_chat_template returned {type(convo).__name__}, expected str")

            inputs = self._processor(text=[convo], images=[request.image], return_tensors="pt")
            device = self._model.device
            inputs = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

            with torch.inference_mode():
                output = self._model.generate(**inputs, **gen_kwargs)

            prompt_len = int(inputs["input_ids"].shape[1])
            new_tokens = output[0][prompt_len:]
            token_ids = [int(t) for t in new_tokens.tolist()]
            text = self._processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
        except Exception as exc:
            if oom_type is not None and isinstance(exc, oom_type):
                logger.error(
                    "CUDA out of memory on %s (%s); re-raising", request.image_rel_path, request.caption_type
                )
                raise
            logger.warning(
                "caption failed for %s (%s, attempt %s): %s",
                request.image_rel_path,
                request.caption_type,
                request.attempt,
                _error_text(exc),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return CaptionResult(text="", finish_reason="error", num_output_tokens=0, error=_error_text(exc))

        num_output_tokens = len(token_ids)
        hit_limit = num_output_tokens >= max_new_tokens
        ended_with_eos = bool(token_ids) and token_ids[-1] in self._eos_token_ids
        finish_reason = "length" if hit_limit and not ended_with_eos else "stop"
        logger.debug(
            "captioned %s (%s): %d tokens, finish_reason=%s",
            request.image_rel_path,
            request.caption_type,
            num_output_tokens,
            finish_reason,
        )
        return CaptionResult(text=text, finish_reason=finish_reason, num_output_tokens=num_output_tokens)

    def _release(self) -> None:
        """Drop the model/processor references, collect garbage, empty the CUDA cache."""
        self._model = None
        self._processor = None
        self._torch_dtype = None
        self._eos_token_ids = frozenset()
        self._loaded = False
        gc.collect()
        torch = self._torch
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("torch.cuda.empty_cache() failed during unload", exc_info=True)


#: Backwards-compatible alias of :class:`JoyCaptionHFModel` (kept for one release, SPEC §6.2).
JoyCaptionModel = JoyCaptionHFModel
