"""Backend interface for captioning models and the lazy backend factory.

Defines :class:`CaptionModel`, the abstract interface every captioning backend
implements (SPEC §6.1), and :func:`get_model`, which picks a backend by name
and imports its module only when called (SPEC §2.2, D6).

This module imports nothing heavy: ``torch``/``transformers`` live only in
``captioning/joycaption_hf.py`` and ``vllm`` only inside
``JoyCaptionVLLMModel.load()``. Importing ``pixel_art_auto_captioner.captioning``
or this module therefore works on a machine without any ML library installed.

Depends on ``pixel_art_auto_captioner.common`` only (never ``ingestion`` or
``batch``).
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from pixel_art_auto_captioner.common.types import CaptionRequest, CaptionResult

logger = logging.getLogger(__name__)

__all__ = ["CaptionModel", "get_model"]

_DEFAULT_MODEL_NAME = "joycaption-beta-one"

# model_name -> backend -> (module path, class name). The modules are imported
# only inside get_model(), so a missing torch/vllm never breaks package import.
_BACKENDS: dict[str, dict[str, tuple[str, str]]] = {
    _DEFAULT_MODEL_NAME: {
        "hf": ("pixel_art_auto_captioner.captioning.joycaption_hf", "JoyCaptionHFModel"),
        "vllm": ("pixel_art_auto_captioner.captioning.joycaption_vllm", "JoyCaptionVLLMModel"),
    },
}


class CaptionModel(ABC):
    """Abstract interface of a captioning backend (SPEC §6.1).

    Subclasses set the class attributes ``model_name`` (e.g.
    ``"joycaption-beta-one"``) and ``backend`` (``"vllm"`` or ``"hf"``) and
    implement :meth:`load`, :meth:`caption_batch` and :meth:`unload`.
    :meth:`caption` and :meth:`describe` have default implementations.

    Contract (relied on by the runner, SPEC §7.2):

    - :meth:`caption_batch` returns exactly ``len(requests)`` results, in the
      same order as ``requests``.
    - Per-request problems are returned as
      ``CaptionResult(finish_reason="error", error=...)``, not raised.
    - Raising is reserved for systemic failures (e.g. CUDA OOM for the whole
      batch); the runner then retries the requests one at a time.
    - :meth:`unload` is safe to call at any time: before :meth:`load`, after a
      failed or partial :meth:`load`, and more than once.

    Attributes:
        model_name: Model identifier written to ``CaptionRecord.model_name``.
        backend: Backend identifier written to ``CaptionRecord.backend``.
    """

    model_name: str
    backend: str

    @abstractmethod
    def load(self, config: dict) -> None:
        """Load the model weights and make the backend ready to caption.

        Args:
            config: The ``model`` section of the merged config (SPEC §9.2),
                including its backend sub-section (``hf`` or ``vllm``). The
                runner passes only this section (SPEC §7.1, fixes B1).

        Raises:
            Exception: Any load failure (missing weights, OOM, bad setting).
                The runner logs it, calls :meth:`unload` and re-raises.
        """

    @abstractmethod
    def caption_batch(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        """Caption a batch of requests.

        Args:
            requests: Generation jobs; each carries its image, rendered
                prompt, generation parameters and seed.

        Returns:
            Exactly ``len(requests)`` results, ``results[i]`` belonging to
            ``requests[i]``. A request that fails on its own yields
            ``CaptionResult(finish_reason="error", error=...)``.

        Raises:
            Exception: Only for systemic failures affecting the whole batch
                (e.g. CUDA OOM, engine crash).
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model and free device memory.

        Must not raise because of the model's state: it is called from a
        ``finally`` block, possibly before :meth:`load`, after a failed or
        partial :meth:`load`, or repeatedly.
        """

    def caption(self, request: CaptionRequest) -> CaptionResult:
        """Caption a single request: ``caption_batch([request])[0]``.

        Args:
            request: The generation job.

        Returns:
            The backend's result for ``request``.

        Raises:
            RuntimeError: If :meth:`caption_batch` does not return exactly one
                result (a backend contract violation).
            Exception: Whatever :meth:`caption_batch` raises (systemic
                failures propagate unchanged).
        """
        results = self.caption_batch([request])
        if len(results) != 1:
            raise RuntimeError(
                f"{type(self).__name__}.caption_batch returned {len(results)} results "
                "for 1 request (contract: exactly one result per request, in order)"
            )
        return results[0]

    def describe(self) -> dict[str, Any]:
        """Describe the model for the run manifest (SPEC §6.1, §8.1).

        The base implementation returns ``model_name`` and ``backend``
        (``None`` if a subclass did not set them). Backends extend the dict
        with their settings and library versions (``model_path``,
        ``revision``, ``dtype``, ``torch``/``transformers``/``vllm`` versions),
        typically as ``{**super().describe(), ...}``.

        Returns:
            A new JSON-serialisable dict on every call.
        """
        return {
            "model_name": getattr(self, "model_name", None),
            "backend": getattr(self, "backend", None),
        }


def get_model(backend: str, model_name: str = _DEFAULT_MODEL_NAME) -> CaptionModel:
    """Create an unloaded backend instance, importing its module lazily.

    The backend module (``captioning.joycaption_hf`` for ``"hf"``,
    ``captioning.joycaption_vllm`` for ``"vllm"``) is imported only here, so
    importing the ``captioning`` package never requires ``torch`` or ``vllm``
    (SPEC §2.2, D6). The backend class is constructed with no arguments; the
    caller then passes the ``model`` config to :meth:`CaptionModel.load`.

    Args:
        backend: ``"vllm"`` or ``"hf"`` (``model.backend`` in the config).
        model_name: Model family. Only ``"joycaption-beta-one"`` exists (Q16).

    Returns:
        A new, unloaded :class:`CaptionModel` instance.

    Raises:
        ValueError: Unknown ``model_name`` or ``backend``.
        ImportError: The backend module cannot be imported (e.g. ``torch`` /
            ``transformers`` missing for ``"hf"``) or lacks its class.
        TypeError: The backend class is not a :class:`CaptionModel` subclass.
    """
    backends = _BACKENDS.get(model_name) if isinstance(model_name, str) else None
    if backends is None:
        raise ValueError(f"unknown model_name {model_name!r}; expected one of {sorted(_BACKENDS)}")
    if not isinstance(backend, str) or backend not in backends:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(backends)}")

    module_path, class_name = backends[backend]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"cannot import the {backend!r} backend module {module_path!r}: {exc} "
            "(the 'hf' backend needs torch + transformers, the 'vllm' backend needs vllm)"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"backend module {module_path!r} has no class {class_name!r}")
    if not (isinstance(cls, type) and issubclass(cls, CaptionModel)):
        raise TypeError(f"{module_path}.{class_name} is not a CaptionModel subclass")

    logger.debug("get_model: %s backend %r -> %s.%s", model_name, backend, module_path, class_name)
    return cls()
