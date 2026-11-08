"""JoyCaption Beta One on vLLM (cloud backend).

Provides :class:`JoyCaptionVLLMModel` (SPEC §6.3), the ``"vllm"`` backend used
for batched bf16 runs on cloud GPUs (decisions Q2, Q3).

Behaviour (SPEC §6.3):

- :meth:`JoyCaptionVLLMModel.load` builds one ``vllm.LLM`` engine from the
  ``model`` config section and its ``vllm`` sub-section, with
  ``limit_mm_per_prompt={"image": 1}``.
- :meth:`JoyCaptionVLLMModel.caption_batch` builds one chat conversation per
  request (the same ``[system, user]`` structure as the HF backend, with the
  PIL image as a content part of the user message) and one
  ``SamplingParams`` per request (``seed=request.seed``), then makes a single
  ``llm.chat(conversations, sampling_params=[...], use_tqdm=False)`` call.
- Outputs are mapped back to the requests by position: ``text.strip()``,
  ``finish_reason`` (``"stop"`` / ``"length"``; anything else is an error
  result) and ``len(token_ids)``.
- A request whose parameters are invalid becomes an error result and is not
  sent to the engine. An exception from ``llm.chat`` concerns the whole batch
  and is re-raised (the runner then retries the requests one at a time,
  SPEC §7.2).

``vllm`` is imported **only** inside :meth:`JoyCaptionVLLMModel.load`
(SPEC §2.2), so this module imports on a machine without it. ``torch`` is
never imported here: :meth:`JoyCaptionVLLMModel.unload` empties the CUDA cache
only through the ``torch`` module that ``vllm`` itself already loaded.

Pinned version: ``vllm==0.33.0``, API checked against its documentation
(see the step 9 notes in ``TODO.md``): ``LLM(model, *, revision,
tokenizer_revision, dtype, gpu_memory_utilization, enforce_eager, **kwargs)``
with ``max_model_len``, ``max_num_seqs``, ``enable_prefix_caching`` and
``limit_mm_per_prompt`` forwarded to ``EngineArgs``; ``LLM.chat`` takes a list
of conversations plus a list of ``SamplingParams`` paired one by one and
returns the ``RequestOutput`` list in input order; a PIL image is the content
part ``{"type": "image_pil", "image_pil": image}``. The version-sensitive
details are kept in :data:`TOP_K_DISABLED`, :data:`LIMIT_MM_PER_PROMPT` and
:func:`build_conversation`.

Depends on ``pixel_art_auto_captioner.captioning.base`` and
``pixel_art_auto_captioner.common`` only.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import logging
import os
import sys
from typing import Any

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GEN_PARAMS",
    "DEFAULT_LOAD_CONFIG",
    "LIMIT_MM_PER_PROMPT",
    "SYSTEM_PROMPT",
    "TOP_K_DISABLED",
    "JoyCaptionVLLMModel",
    "build_conversation",
]

#: System prompt of every conversation (same text as the HF backend, SPEC §6.2).
SYSTEM_PROMPT = "You are a helpful image captioner."

#: ``load()`` defaults (SPEC §6.3 table). ``model_path``, ``revision`` and
#: ``dtype`` come from the top level of the ``model`` section; the other keys
#: from its ``vllm`` sub-section (falling back to the top level).
DEFAULT_LOAD_CONFIG: dict[str, Any] = {
    "model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "revision": None,
    "dtype": "bfloat16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 128,
    "enable_prefix_caching": True,
    "enforce_eager": False,
}

#: Generation defaults used when a ``CaptionRequest.gen_params`` key is missing
#: or ``None`` (the same values as the HF backend and ``main.py``).
DEFAULT_GEN_PARAMS: dict[str, Any] = {
    "max_new_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": None,
}

#: ``SamplingParams.top_k`` value that disables top-k sampling. vLLM 0.33
#: documents "Set to 0 (or -1) to consider all tokens", accepts -1 only
#: "quietly" and prefers 0 (its default), so 0 is used where SPEC §6.3 writes
#: ``top_k or -1``.
TOP_K_DISABLED = 0

#: One image per prompt (SPEC §6.3.1).
LIMIT_MM_PER_PROMPT: dict[str, int] = {"image": 1}

_DTYPES: tuple[str, ...] = ("bfloat16", "float16", "float32")
_TOP_LEVEL_KEYS: tuple[str, ...] = ("model_path", "revision", "dtype")
_VLLM_KEYS: tuple[str, ...] = (
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "enable_prefix_caching",
    "enforce_eager",
)
_FINISH_REASONS: dict[str, str] = {"stop": "stop", "length": "length"}


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


def _is_int(value: Any) -> bool:
    """``True`` for a real ``int`` (``bool`` excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """``True`` for an ``int`` or ``float`` (``bool`` excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_conversation(prompt: str, image: Any) -> list[dict[str, Any]]:
    """Build the chat conversation of one request for ``LLM.chat``.

    The structure mirrors the HF backend (``[system, user]``, system prompt
    :data:`SYSTEM_PROMPT`, user text = the rendered prompt); the image is
    passed as a PIL content part of the user message, before the text.

    Args:
        prompt: The fully rendered user prompt (SPEC §5.1).
        image: The RGB ``PIL.Image.Image`` of the request.

    Returns:
        ``[{"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image_pil", "image_pil": image},
        {"type": "text", "text": prompt}]}]``.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


class JoyCaptionVLLMModel(CaptionModel):
    """JoyCaption Beta One (LLaVA) on vLLM (SPEC §6.3).

    Construct with no arguments (as :func:`~pixel_art_auto_captioner.captioning.get_model`
    does), then call :meth:`load` with the ``model`` config section. The
    keyword parameters exist only to inject fakes in CPU tests.

    Example::

        model = JoyCaptionVLLMModel()
        try:
            model.load({"model_path": "fancyfeast/llama-joycaption-beta-one-hf-llava",
                        "vllm": {"max_num_seqs": 64}})
            results = model.caption_batch(requests)
        finally:
            model.unload()

    Args:
        vllm_module: Object used instead of ``import vllm`` (tests only). It
            must provide ``LLM`` and ``SamplingParams``.
        torch_module: Object used for ``torch.cuda.empty_cache()`` in
            :meth:`unload` (tests only). Without it, the ``torch`` module that
            ``vllm`` loaded is used, if any; ``torch`` is never imported here.

    Attributes:
        model_name: ``"joycaption-beta-one"``.
        backend: ``"vllm"``.
    """

    model_name = "joycaption-beta-one"
    backend = "vllm"

    def __init__(self, *, vllm_module: Any | None = None, torch_module: Any | None = None) -> None:
        self._vllm_override = vllm_module
        self._torch_override = torch_module
        self._vllm: Any = None
        self._llm: Any = None
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
        """Create the ``vllm.LLM`` engine.

        Reads ``model_path``, ``revision`` and ``dtype`` from ``config`` and
        ``max_model_len``, ``gpu_memory_utilization``, ``max_num_seqs``,
        ``enable_prefix_caching``, ``enforce_eager`` from ``config["vllm"]``
        (falling back to the top level, then to :data:`DEFAULT_LOAD_CONFIG`).
        Other keys (e.g. ``backend``, ``hf``) are ignored. If an engine is
        already loaded it is unloaded first.

        Args:
            config: The ``model`` section of the merged config (SPEC §9.2).

        Raises:
            ValueError: Invalid settings (checked before ``vllm`` is imported;
                all problems in one message).
            ImportError: ``vllm`` is not installed.
            Exception: Whatever ``vllm.LLM(...)`` raises (missing weights,
                OOM, ...). The partial state is released before re-raising.
        """
        settings = self._resolve_settings(config)
        if self._loaded:
            logger.warning("load() called while an engine is loaded; unloading it first")
            self.unload()

        self._settings = settings
        model_path = settings["model_path"]
        logger.info(
            "Loading JoyCaption (vllm) from %s (revision=%s, dtype=%s, max_model_len=%d, "
            "gpu_memory_utilization=%s, max_num_seqs=%d, enable_prefix_caching=%s, enforce_eager=%s)",
            model_path,
            settings["revision"],
            settings["dtype"],
            settings["max_model_len"],
            settings["gpu_memory_utilization"],
            settings["max_num_seqs"],
            settings["enable_prefix_caching"],
            settings["enforce_eager"],
        )
        try:
            vllm = self._vllm_override if self._vllm_override is not None else importlib.import_module("vllm")
            self._vllm = vllm
            self._llm = vllm.LLM(**self._engine_kwargs(settings))
        except BaseException:
            logger.exception("Failed to load JoyCaption (vllm) from %s", model_path)
            self._release()
            raise

        self._loaded = True
        logger.info("JoyCaption (vllm) loaded (vllm %s)", self._library_version(self._vllm, "vllm"))

    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        """Caption ``requests`` with one batched ``llm.chat`` call.

        Each request gets its own conversation (:func:`build_conversation`)
        and its own ``SamplingParams(temperature, top_p, top_k, max_tokens,
        seed=request.seed)``. Requests with invalid generation parameters get
        an error result and are not sent. Results are mapped back by position.

        Args:
            requests: Generation jobs.

        Returns:
            One :class:`CaptionResult` per request, in order. ``finish_reason``
            is vLLM's ``"stop"`` or ``"length"``; any other vLLM finish reason
            (e.g. ``"abort"`` or ``None``), a missing output or an invalid
            request gives ``finish_reason="error"`` with ``error`` set.

        Raises:
            RuntimeError: The engine is not loaded, or ``llm.chat`` returned a
                different number of outputs than conversations sent.
            Exception: Whatever ``llm.chat`` raises (a whole-batch failure).
        """
        if not self._loaded:
            raise RuntimeError("JoyCaptionVLLMModel is not loaded; call load(config) first")

        results: list[CaptionResult | None] = [None] * len(requests)
        conversations: list[list[dict[str, Any]]] = []
        sampling_params: list[Any] = []
        sent: list[int] = []
        for index, request in enumerate(requests):
            try:
                params = self._sampling_params(request)
                conversation = build_conversation(request.prompt, request.image)
            except Exception as exc:
                results[index] = self._error_result(request, exc)
                continue
            conversations.append(conversation)
            sampling_params.append(params)
            sent.append(index)

        if conversations:
            try:
                outputs = list(self._llm.chat(conversations, sampling_params=sampling_params, use_tqdm=False))
            except Exception:
                logger.error("llm.chat failed for a batch of %d requests; re-raising", len(conversations))
                raise
            if len(outputs) != len(conversations):
                raise RuntimeError(
                    f"llm.chat returned {len(outputs)} outputs for {len(conversations)} conversations"
                )
            for index, output in zip(sent, outputs):
                results[index] = self._map_output(requests[index], output)

        return [result for result in results if result is not None]

    def unload(self) -> None:
        """Delete the engine, collect garbage and empty the CUDA cache.

        Safe at any time: before :meth:`load`, after a failed or partial
        :meth:`load`, and more than once. Never imports ``vllm`` or ``torch``;
        a failing ``torch.cuda.empty_cache()`` is logged and ignored.
        """
        was_loaded = self._loaded
        self._release()
        if was_loaded:
            logger.info("JoyCaption (vllm) unloaded")

    def describe(self) -> dict[str, Any]:
        """Describe the backend for the run manifest (SPEC §6.1, D11).

        Returns:
            ``model_name``, ``backend``, ``model_path``, ``revision`` (as
            configured, ``None`` if unset), ``dtype``, ``max_model_len``,
            ``gpu_memory_utilization``, ``max_num_seqs``,
            ``enable_prefix_caching``, ``enforce_eager``,
            ``limit_mm_per_prompt`` and ``vllm_version``, ``torch_version``,
            ``transformers_version`` (``None`` when unknown). The settings are
            those of the last :meth:`load` call (``None`` before the first)
            and are kept after :meth:`unload`.
        """
        settings = self._settings
        vllm = self._vllm if self._vllm is not None else self._vllm_override
        return {
            **super().describe(),
            "model_path": settings.get("model_path"),
            "revision": settings.get("revision"),
            "dtype": settings.get("dtype"),
            "max_model_len": settings.get("max_model_len"),
            "gpu_memory_utilization": settings.get("gpu_memory_utilization"),
            "max_num_seqs": settings.get("max_num_seqs"),
            "enable_prefix_caching": settings.get("enable_prefix_caching"),
            "enforce_eager": settings.get("enforce_eager"),
            "limit_mm_per_prompt": dict(LIMIT_MM_PER_PROMPT) if settings else None,
            "vllm_version": self._library_version(vllm, "vllm"),
            "torch_version": self._library_version(self._torch(), "torch"),
            "transformers_version": _dist_version("transformers"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_settings(config: dict) -> dict[str, Any]:
        """Merge ``config`` (+ its ``vllm`` sub-section) with the defaults and validate."""
        if not isinstance(config, dict):
            raise ValueError(f"model config must be a dict, got {type(config).__name__}")
        section = config.get("vllm")
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError(f"model.vllm must be a dict or null, got {type(section).__name__}")

        settings: dict[str, Any] = {}
        for key in _TOP_LEVEL_KEYS:
            settings[key] = config.get(key, DEFAULT_LOAD_CONFIG[key])
        for key in _VLLM_KEYS:
            settings[key] = section[key] if key in section else config.get(key, DEFAULT_LOAD_CONFIG[key])

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
        for key in ("max_model_len", "max_num_seqs"):
            if not _is_int(settings[key]) or settings[key] < 1:
                problems.append(f"vllm.{key} must be a positive int, got {settings[key]!r}")
        utilization = settings["gpu_memory_utilization"]
        if not _is_number(utilization) or not 0 < utilization <= 1:
            problems.append(f"vllm.gpu_memory_utilization must be in (0, 1], got {utilization!r}")
        for key in ("enable_prefix_caching", "enforce_eager"):
            if not isinstance(settings[key], bool):
                problems.append(f"vllm.{key} must be true or false, got {settings[key]!r}")
        if problems:
            raise ValueError("invalid vllm model config: " + "; ".join(problems))
        return settings

    @staticmethod
    def _engine_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
        """Keyword arguments of ``vllm.LLM(...)`` for validated ``settings``."""
        return {
            "model": settings["model_path"],
            "revision": settings["revision"],
            "tokenizer_revision": settings["revision"],
            "dtype": settings["dtype"],
            "max_model_len": settings["max_model_len"],
            "gpu_memory_utilization": settings["gpu_memory_utilization"],
            "max_num_seqs": settings["max_num_seqs"],
            "enable_prefix_caching": settings["enable_prefix_caching"],
            "enforce_eager": settings["enforce_eager"],
            "limit_mm_per_prompt": dict(LIMIT_MM_PER_PROMPT),
        }

    @staticmethod
    def _gen_values(gen_params: dict | None) -> dict[str, Any]:
        """Resolve and validate one request's generation parameters.

        Missing or ``None`` values use :data:`DEFAULT_GEN_PARAMS`; unknown
        keys are ignored. ``top_k`` ``None`` or ``0`` becomes
        :data:`TOP_K_DISABLED`.

        Returns:
            ``{"temperature", "top_p", "top_k", "max_tokens"}``.

        Raises:
            ValueError: A parameter has an invalid type or value.
        """
        if gen_params is not None and not isinstance(gen_params, dict):
            raise ValueError(f"gen_params must be a dict, got {type(gen_params).__name__}")
        params = dict(gen_params or {})
        values = {
            key: (params.get(key) if params.get(key) is not None else default)
            for key, default in DEFAULT_GEN_PARAMS.items()
        }
        max_new_tokens = values["max_new_tokens"]
        if not _is_int(max_new_tokens) or max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
        temperature = values["temperature"]
        if not _is_number(temperature) or temperature < 0:
            raise ValueError(f"temperature must be a number >= 0, got {temperature!r}")
        top_p = values["top_p"]
        if not _is_number(top_p) or not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p!r}")
        top_k = values["top_k"]
        if top_k is not None and (not _is_int(top_k) or top_k < 0):
            raise ValueError(f"top_k must be a non-negative int or null, got {top_k!r}")
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k or TOP_K_DISABLED,
            "max_tokens": max_new_tokens,
        }

    def _sampling_params(self, request: CaptionRequest) -> Any:
        """Build the ``SamplingParams`` of one request (``seed=request.seed``)."""
        if not _is_int(request.seed):
            raise ValueError(f"seed must be an int, got {request.seed!r}")
        values = self._gen_values(request.gen_params)
        return self._vllm.SamplingParams(**values, seed=request.seed)

    @staticmethod
    def _error_result(request: CaptionRequest, exc: BaseException) -> CaptionResult:
        """Log a per-request failure and return its error result."""
        logger.warning(
            "caption failed for %s (%s, attempt %s): %s",
            request.image_rel_path,
            request.caption_type,
            request.attempt,
            _error_text(exc),
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return CaptionResult(text="", finish_reason="error", num_output_tokens=0, error=_error_text(exc))

    @staticmethod
    def _map_output(request: CaptionRequest, output: Any) -> CaptionResult:
        """Map one ``RequestOutput`` to a :class:`CaptionResult`."""
        try:
            completions = output.outputs
            if not completions:
                raise ValueError("vLLM returned no completion for this request")
            completion = completions[0]
            text = completion.text
            if not isinstance(text, str):
                raise TypeError(f"completion text is {type(text).__name__}, expected str")
            num_output_tokens = len(completion.token_ids)
            raw_reason = completion.finish_reason
        except Exception as exc:
            return JoyCaptionVLLMModel._error_result(request, exc)

        text = text.strip()
        finish_reason = _FINISH_REASONS.get(raw_reason) if isinstance(raw_reason, str) else None
        if finish_reason is None:
            error = f"vLLM finish_reason {raw_reason!r} (expected 'stop' or 'length')"
            logger.warning(
                "caption failed for %s (%s, attempt %s): %s",
                request.image_rel_path,
                request.caption_type,
                request.attempt,
                error,
            )
            return CaptionResult(text=text, finish_reason="error", num_output_tokens=num_output_tokens, error=error)

        logger.debug(
            "captioned %s (%s): %d tokens, finish_reason=%s",
            request.image_rel_path,
            request.caption_type,
            num_output_tokens,
            finish_reason,
        )
        return CaptionResult(text=text, finish_reason=finish_reason, num_output_tokens=num_output_tokens)

    @staticmethod
    def _library_version(module: Any, dist_name: str) -> str | None:
        """Version of an imported (or injected) library, else its installed distribution."""
        version = getattr(module, "__version__", None) if module is not None else None
        if version is not None:
            return str(version)
        return _dist_version(dist_name)

    def _torch(self) -> Any:
        """The injected torch, else the ``torch`` module vllm already loaded, else ``None``."""
        if self._torch_override is not None:
            return self._torch_override
        return sys.modules.get("torch")

    def _release(self) -> None:
        """Drop the engine, collect garbage, empty the CUDA cache (no new imports)."""
        self._llm = None
        self._loaded = False
        gc.collect()
        if self._vllm is None:  # no load() attempt got as far as vllm: nothing on the GPU
            return
        torch = self._torch()
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.warning("torch.cuda.empty_cache() failed during unload", exc_info=True)
