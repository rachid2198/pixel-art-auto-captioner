"""Batch orchestration: plan -> batches -> retry -> writer (SPEC v1.1 §7).

:class:`CaptionRunner` is the only orchestrator of the package. One
:meth:`CaptionRunner.run` call:

1. builds the resume index from the existing output (§8.4) and asks the
   dataloader for this shard's plan (§4.3);
2. returns at once, without loading the model, if nothing is pending;
3. loads the model with **only** the ``model`` config section (fixes B1) and
   always unloads it (the runner owns the model lifecycle, §7.1);
4. consumes the decoded images in groups of ``runtime.batch_size`` images,
   builds one :class:`~pixel_art_auto_captioner.common.types.CaptionRequest`
   per pending caption type (prompt §5.1, seed §5.2), calls
   ``model.caption_batch``, quality-checks every result (§5.3), re-submits
   flagged requests with ``attempt + 1`` and a new seed while
   ``attempt <= quality.max_retries``, and writes one record per final result
   through the :class:`~pixel_art_auto_captioner.common.export_utils.ShardWriter`;
5. logs progress (and GPU cost when ``runtime.gpu_hourly_cost`` is set,
   §11.4) every ``runtime.log_every_s`` seconds;
6. closes the writer, writes the QA page (§10.3) and the run manifest
   ``runs/<run_id>.json`` (D8), and returns the summary (§7.3).

``KeyboardInterrupt`` and ``SIGTERM`` stop the run: the in-flight batch is
abandoned, the writer is flushed and closed, the model unloaded, the manifest
written with ``"interrupted": true``, and the ``KeyboardInterrupt`` re-raised
(``SIGTERM`` is turned into a ``KeyboardInterrupt`` by a handler installed for
the duration of the run, on the main thread only).

``failures/shard-<iii>-of-<NNN>.jsonl`` (§8.1) holds two kinds of lines:

- one ``failed`` caption record (same JSON shape as ``captions/``) per pending
  caption type of an image that could not be decoded (``LoadFailure``, §7.2);
- systemic error events, JSON objects with an ``"event"`` key
  (``"batch_exception"``, ``"request_exception"``, ``"cuda_oom"``).

Resume never reads ``failures/``, so a load failure is retried on the next run.

``prompts.system`` is not passed to the backends: ``load()`` receives only the
``model`` section and ``CaptionRequest`` has no system field, so the backends
use their §6.2 system-prompt constant (see TODO open questions).

This module may import ``common``, ``ingestion`` and ``captioning``; it never
imports ``torch``, ``transformers`` or ``vllm`` (SPEC §2.2).
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import platform
import re
import secrets
import signal
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import FrameType
from typing import Any

import PIL

from pixel_art_auto_captioner import __version__ as _PACKAGE_VERSION
from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.config import _parse_shard, validate_config
from pixel_art_auto_captioner.common.export_utils import (
    ResumeIndex,
    ShardWriter,
    _dumps,
    _repair_tail,
    _shard_filename,
    build_record,
    generate_qa_page,
)
from pixel_art_auto_captioner.common.prompts import (
    prompt_config_hash,
    render_prompt,
    resolve_source,
)
from pixel_art_auto_captioner.common.quality import check_caption
from pixel_art_auto_captioner.common.seeding import derive_seed
from pixel_art_auto_captioner.common.types import (
    CaptionRecord,
    CaptionRequest,
    CaptionResult,
    ImageRecord,
    LoadFailure,
)
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader, Plan

logger = logging.getLogger(__name__)

#: Generation parameters copied from a caption type into each request (SPEC §3).
GEN_PARAM_KEYS: tuple[str, ...] = ("max_new_tokens", "temperature", "top_p", "top_k")

#: Keys of the summary returned by :meth:`CaptionRunner.run` (SPEC §7.3), in order.
SUMMARY_KEYS: tuple[str, ...] = (
    "run_id",
    "shard",
    "images_discovered",
    "images_in_shard",
    "images_already_complete",
    "images_deferred",
    "images_planned",
    "images_load_failed",
    "captions_planned",
    "captions_ok",
    "captions_flagged",
    "captions_failed",
    "retries",
    "interrupted",
    "output_dir",
)

#: Flags that make a final result ``failed``; any other flag gives ``flagged`` (§5.3).
_FAILED_FLAGS: frozenset[str] = frozenset({"error", "empty"})

_FAILURES_DIR = "failures"
_RUNS_DIR = "runs"
_HEX40 = re.compile(r"[0-9a-f]{40}")


def _monotonic() -> float:
    """Monotonic clock for progress/throughput (module-level so tests can replace it)."""
    return time.monotonic()


def _utc_now() -> datetime:
    """Current UTC time (module-level so tests can replace it)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class _Counts:
    """Mutable counters of one run."""

    images_done: int = 0
    images_load_failed: int = 0
    captions_ok: int = 0
    captions_flagged: int = 0
    captions_failed: int = 0
    retries: int = 0

    @property
    def captions_done(self) -> int:
        return self.captions_ok + self.captions_flagged + self.captions_failed


class CaptionRunner:
    """Run one captioning pass over a dataloader's plan (SPEC §7).

    Example::

        cfg = load_config(Path("configs/example_config.json"), overrides)
        validate_config(cfg)
        loader = ImageDataLoader({**cfg["input"], **cfg["runtime"]})
        model = get_model(cfg["model"]["backend"])     # not loaded
        summary = CaptionRunner(loader, model, cfg).run()

    Attributes:
        config: Deep copy of the full merged config.
        output_dir: ``output.output_dir`` as an absolute path.
        shard: ``(index, count)`` of the run (= ``runtime.shard`` = the
            dataloader's shard).
    """

    def __init__(self, dataloader: ImageDataLoader, model: CaptionModel, config: dict) -> None:
        """Validate the config and keep the collaborators.

        Args:
            dataloader: A configured loader; its ``shard`` must equal
                ``runtime.shard``.
            model: An **unloaded** ``CaptionModel``; the runner loads and
                unloads it inside :meth:`run` (the CLI never calls ``load()``).
            config: The full merged config (§9). The runner reads ``model``,
                ``generation``, ``prompts``, ``quality``, ``output`` and
                ``runtime``; ``model.load()`` receives only ``config["model"]``.
                The dict is deep-copied, never modified.

        Raises:
            ValueError: The config fails ``validate_config`` (§9.3);
                ``runtime.shard`` differs from the dataloader's shard;
                ``quality.max_retries`` is not an int >= 0; ``runtime.seed`` is
                not an int; ``runtime.gpu_hourly_cost`` is neither null nor a
                number > 0.
        """
        validate_config(config)
        self.config: dict[str, Any] = copy.deepcopy(config)
        cfg = self.config
        shard = _parse_shard(cfg["runtime"]["shard"])
        loader_shard = tuple(getattr(dataloader, "shard", ()))
        if shard != loader_shard:
            raise ValueError(
                f"runtime.shard {cfg['runtime']['shard']!r} does not match the dataloader's "
                f"shard {loader_shard!r}"
            )
        assert shard is not None  # validate_config checked it
        self.shard: tuple[int, int] = shard
        self.output_dir: Path = Path(cfg["output"]["output_dir"]).absolute()
        self._loader = dataloader
        self._model = model

        generation = cfg["generation"]
        self._caption_types: list[str] = list(generation["enabled_caption_types"])
        self._type_specs: dict[str, dict[str, Any]] = generation["caption_types"]
        self._prompt_config: dict[str, Any] = {
            **cfg["prompts"],
            "caption_types": generation["caption_types"],
        }
        self._quality_cfg: dict[str, Any] = cfg.get("quality") or {}
        max_retries = self._quality_cfg.get("max_retries", 1)
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"quality.max_retries: must be an integer >= 0, got {max_retries!r}")
        self._max_retries: int = max_retries

        runtime = cfg["runtime"]
        self._batch_size: int = runtime["batch_size"]
        seed = runtime.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"runtime.seed: must be an integer, got {seed!r}")
        self._seed: int = seed
        self._log_every_s: float = runtime["log_every_s"]
        cost = runtime.get("gpu_hourly_cost")
        if cost is not None and not (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost > 0
        ):
            raise ValueError(f"runtime.gpu_hourly_cost: must be null or a number > 0, got {cost!r}")
        self._gpu_hourly_cost: float | None = None if cost is None else float(cost)

        output = cfg["output"]
        self._write_txt: bool = bool(output.get("write_txt", False))
        self._resume: bool = bool(output.get("resume", True))
        self._fsync_every: int = output["fsync_every"]
        self._qa_sample: Any = output.get("qa_sample", 100)

        self._reset_run_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Caption every pending (image, caption type) of the shard (SPEC §7.2).

        An empty plan returns the summary at once: the model is not loaded,
        no file is written and no manifest is created. Otherwise the model is
        loaded with ``config["model"]`` only and always unloaded, and the run
        manifest ``runs/<run_id>.json`` is written whatever the outcome.

        Returns:
            The summary (§7.3), keys :data:`SUMMARY_KEYS`. Invariants:
            ``images_in_shard == images_already_complete + images_deferred +
            images_planned`` and (the run was not interrupted)
            ``captions_planned == captions_ok + captions_flagged +
            captions_failed`` (an image load failure counts one failed
            caption per pending caption type).

        Raises:
            KeyboardInterrupt: Ctrl-C or SIGTERM. The writer is flushed and
                closed, the model unloaded and the manifest written with
                ``"interrupted": true`` before re-raising.
            Exception: A model load error (logged, model unloaded, re-raised);
                a CUDA out-of-memory error on a single request (logged
                CRITICAL, re-raised after flushing); a write error (logged
                CRITICAL, re-raised); a malformed caption file (``ValueError``
                from the resume index, before anything is loaded). After a load
                attempt the manifest is written with ``"error"`` set.
        """
        self._reset_run_state()
        self._t_start = _monotonic()
        started = _utc_now()
        self._started_utc = _iso(started)
        self._run_id = f"{started:%Y%m%dT%H%M%SZ}-{self.shard[0]}-{secrets.token_hex(2)}"
        logger.info(
            "Run %s: shard %d/%d, caption types %s, output %s",
            self._run_id, self.shard[0], self.shard[1], self._caption_types,
            self.output_dir.as_posix(),
        )

        if self._resume:
            resume_index = ResumeIndex.from_output_dir(self.output_dir)
        else:
            logger.info("Resume disabled: nothing counts as complete")
            resume_index = ResumeIndex()
        plan = self._loader.plan(resume_index.is_complete, self._caption_types)
        del resume_index  # can be large; the plan keeps only what is pending
        self._plan = plan
        logger.info(
            "Run %s plan: %d images / %d captions planned (%d in shard, %d already complete, "
            "%d deferred)",
            self._run_id, plan.images_planned, plan.captions_planned, plan.images_in_shard,
            plan.images_already_complete, plan.images_deferred,
        )
        logger.debug("Planned captions by source: %s", plan.captions_by_source())

        if plan.images_planned == 0:
            logger.info("Nothing to do for shard %d/%d; the model is not loaded", *self.shard)
            summary = self._summary()
            self._log_summary(summary)
            return summary

        previous_handler = self._install_sigterm_handler()
        try:
            return self._run_plan(plan)
        finally:
            self._restore_sigterm_handler(previous_handler)

    # ------------------------------------------------------------------
    # Run phases
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        """Per-run state; :meth:`run` may be called more than once."""
        self._run_id = ""
        self._started_utc = ""
        self._plan: Plan | None = None
        self._counts = _Counts()
        self._interrupted = False
        self._stopping = False
        self._model_info: dict[str, Any] | None = None
        self._writer: ShardWriter | None = None
        self._failures: _FailureLog | None = None
        self._t_start = 0.0
        self._t_processing: float | None = None
        self._t_last_log = 0.0
        self._prompt_cache: dict[tuple[str, str], str] = {}
        self._source_cache: dict[str, tuple[str, str]] = {}
        self._sigterm_installed = False

    def _run_plan(self, plan: Plan) -> dict[str, Any]:
        """Load, process, shut down; return the summary (non-empty plan)."""
        error: BaseException | None = None
        try:
            self._load_model()
            self._open_writers()
            self._process(plan)
        except KeyboardInterrupt as exc:
            self._interrupted = True
            self._stopping = True
            logger.warning(
                "Run %s interrupted (%s); flushing and shutting down",
                self._run_id, str(exc) or "KeyboardInterrupt",
            )
            error = exc
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._shutdown(error)
        summary = self._summary()
        self._log_summary(summary)
        return summary

    def _load_model(self) -> None:
        """``model.load(config["model"])``; log and re-raise a failure (§7.2 step 3)."""
        model_cfg = copy.deepcopy(self.config["model"])
        logger.info(
            "Loading model %s (backend %s, %s)",
            getattr(self._model, "model_name", "?"), getattr(self._model, "backend", "?"),
            model_cfg.get("model_path"),
        )
        t0 = _monotonic()
        try:
            self._model.load(model_cfg)
        except Exception:
            logger.exception("Model load failed")
            raise
        logger.info("Model loaded in %.1f s", _monotonic() - t0)
        self._model_info = self._describe()

    def _open_writers(self) -> None:
        collisions = self._collision_set() if self._write_txt else None
        self._writer = ShardWriter(
            self.output_dir, self.shard, self._write_txt,
            fsync_every=self._fsync_every, collision_set=collisions,
        )
        self._failures = _FailureLog(self.output_dir, self.shard)

    def _process(self, plan: Plan) -> None:
        """Consume the decoded images in groups of ``batch_size`` images (§7.2 step 4)."""
        self._t_processing = _monotonic()
        self._t_last_log = self._t_processing
        group: list[tuple[ImageRecord, list[str]]] = []
        with contextlib.closing(self._loader.iter_images(plan)) as images:
            for (rel_path, pending), item in zip(plan.items, images):
                if item.rel_path != rel_path:
                    raise RuntimeError(
                        f"dataloader yielded {item.rel_path.as_posix()} for planned "
                        f"{rel_path.as_posix()}"
                    )
                if isinstance(item, LoadFailure):
                    self._handle_load_failure(item, pending)
                    self._maybe_log_progress()
                    continue
                group.append((item, pending))
                if len(group) >= self._batch_size:
                    self._caption_group(group)
                    group = []
                    self._maybe_log_progress()
            if group:
                self._caption_group(group)
        self._log_progress(final=True)

    def _shutdown(self, error: BaseException | None) -> None:
        """Close writers, unload, QA page, manifest. Never masks ``error``."""
        self._stopping = True  # a SIGTERM during shutdown is logged, not raised
        close_error: BaseException | None = None
        for closeable in (self._writer, self._failures):
            if closeable is None:
                continue
            try:
                closeable.close()
            except Exception as exc:  # noqa: BLE001 - disk errors: log, keep cleaning up
                logger.critical("Could not flush/close %s: %s", closeable.path.as_posix(), exc)
                close_error = close_error or exc
        try:
            logger.info("Unloading model")
            self._model.unload()
        except Exception:  # noqa: BLE001 - unload must not mask the run's outcome
            logger.exception("Model unload failed")

        failed = error is not None or close_error is not None
        qa_page = None if failed else self._write_qa_page()
        try:
            self._write_manifest(error or close_error, qa_page)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Could not write the run manifest: %s", exc)
            if not failed:
                raise
        if error is None and close_error is not None:
            raise close_error

    # ------------------------------------------------------------------
    # Captioning
    # ------------------------------------------------------------------

    def _caption_group(self, group: list[tuple[ImageRecord, list[str]]]) -> None:
        """Caption one group of images, re-submitting flagged results (§5.3, §7.2)."""
        pending: list[tuple[CaptionRequest, ImageRecord]] = [
            (self._make_request(image, caption_type, 1), image)
            for image, types in group
            for caption_type in types
        ]
        while pending:
            results = self._submit([request for request, _ in pending])
            retry: list[tuple[CaptionRequest, ImageRecord]] = []
            for (request, image), result in zip(pending, results):
                flags = check_caption(result, self._quality_cfg)
                if flags and request.attempt <= self._max_retries:
                    logger.debug(
                        "Retrying %s [%s] after attempt %d (flags %s)",
                        request.image_rel_path.as_posix(), request.caption_type,
                        request.attempt, flags,
                    )
                    self._counts.retries += 1
                    retry.append(
                        (self._make_request(image, request.caption_type, request.attempt + 1),
                         image)
                    )
                    continue
                self._write_result(image, request, result, flags)
            pending = retry
        self._counts.images_done += len(group)

    def _submit(self, requests: list[CaptionRequest]) -> list[CaptionResult]:
        """``caption_batch``; on an exception (or a wrong result count) retry singly."""
        try:
            results = self._model.caption_batch(requests)
            if not isinstance(results, list) or len(results) != len(requests):
                count = len(results) if isinstance(results, list) else type(results).__name__
                raise RuntimeError(
                    f"caption_batch returned {count} results for {len(requests)} requests"
                )
            return results
        except Exception as exc:  # noqa: BLE001 - §7.2: retry the requests one at a time
            logger.warning(
                "caption_batch failed for %d requests (%s: %s); retrying them one at a time",
                len(requests), type(exc).__name__, exc,
            )
            self._failure_event("batch_exception", exc, requests)
        return [self._submit_one(request) for request in requests]

    def _submit_one(self, request: CaptionRequest) -> CaptionResult:
        """One request via ``model.caption``; an exception becomes an error result."""
        try:
            return self._model.caption(request)
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_oom(exc):
                logger.critical(
                    "CUDA out of memory on a single request (%s [%s], attempt %d); aborting run",
                    request.image_rel_path.as_posix(), request.caption_type, request.attempt,
                )
                self._failure_event("cuda_oom", exc, [request])
                raise
            error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Request %s [%s] (attempt %d) failed: %s",
                request.image_rel_path.as_posix(), request.caption_type, request.attempt, error,
            )
            self._failure_event("request_exception", exc, [request])
            return CaptionResult(text="", finish_reason="error", num_output_tokens=0, error=error)

    def _make_request(self, image: ImageRecord, caption_type: str, attempt: int) -> CaptionRequest:
        return CaptionRequest(
            image_rel_path=image.rel_path,
            caption_type=caption_type,
            image=image.image,
            prompt=self._prompt(image.source, caption_type),
            gen_params=self._gen_params(caption_type),
            seed=derive_seed(self._seed, image.rel_path, caption_type, attempt),
            attempt=attempt,
        )

    def _write_result(
        self,
        image: ImageRecord,
        request: CaptionRequest,
        result: CaptionResult,
        flags: list[str],
    ) -> None:
        status = _status(flags)
        record = self._record(
            image, result, caption_type=request.caption_type, prompt=request.prompt,
            generation_params=request.gen_params, seed=request.seed, attempt=request.attempt,
            status=status, flags=flags,
        )
        self._write(record)
        if status == "ok":
            self._counts.captions_ok += 1
        elif status == "flagged":
            self._counts.captions_flagged += 1
        else:
            self._counts.captions_failed += 1

    def _handle_load_failure(self, failure: LoadFailure, pending: list[str]) -> None:
        """One ``failed`` record per pending type to ``failures/`` (§7.2; the loader logged it)."""
        result = CaptionResult(
            text="", finish_reason="error", num_output_tokens=0, error=failure.error
        )
        flags = check_caption(result, self._quality_cfg)
        assert self._failures is not None
        for caption_type in pending:
            record = self._record(
                failure, result, caption_type=caption_type,
                prompt=self._prompt(failure.source, caption_type),
                generation_params=self._gen_params(caption_type),
                seed=derive_seed(self._seed, failure.rel_path, caption_type, 1), attempt=1,
                status="failed", flags=flags,
            )
            try:
                self._failures.write_record(record)
            except Exception as exc:
                logger.critical("Write error in %s: %s", self._failures.path.as_posix(), exc)
                raise
        self._counts.images_load_failed += 1
        self._counts.captions_failed += len(pending)
        self._counts.images_done += 1

    def _record(
        self,
        image: ImageRecord | LoadFailure,
        result: CaptionResult,
        **fields: Any,
    ) -> CaptionRecord:
        category, source_name = self._resolve(image.source)
        info = self._model_info or {}
        model_cfg = self.config["model"]
        model_path = info.get("model_path") or model_cfg.get("model_path") or ""
        revision = info["revision"] if "revision" in info else model_cfg.get("revision")
        return build_record(
            image, result, category=category,
            model_name=str(getattr(self._model, "model_name", None) or info.get("model_name", "")),
            model_path=str(model_path),
            model_revision=None if revision is None else str(revision),
            backend=str(getattr(self._model, "backend", None) or model_cfg.get("backend", "")),
            run_id=self._run_id, shard=self.shard,
            extra={"source_name": source_name}, **fields,
        )

    def _write(self, record: CaptionRecord) -> None:
        assert self._writer is not None
        try:
            self._writer.write(record)
        except Exception as exc:
            logger.critical("Write error in %s: %s", self._writer.path.as_posix(), exc)
            raise

    def _failure_event(self, event: str, exc: BaseException, requests: list[CaptionRequest]) -> None:
        """Best-effort systemic-error line in ``failures/`` (never raises)."""
        if self._failures is None:
            return
        entry = {
            "event": event,
            "run_id": self._run_id,
            "shard": f"{self.shard[0]}/{self.shard[1]}",
            "timestamp_utc": _iso(_utc_now()),
            "error": f"{type(exc).__name__}: {exc}",
            "requests": [
                {"image_rel_path": r.image_rel_path.as_posix(), "caption_type": r.caption_type,
                 "attempt": r.attempt}
                for r in requests
            ],
        }
        try:
            self._failures.write_event(entry)
        except Exception:  # noqa: BLE001 - the event is diagnostics only
            logger.exception("Could not record the %s event", event)

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _resolve(self, source: str) -> tuple[str, str]:
        if source not in self._source_cache:
            self._source_cache[source] = resolve_source(source, self._prompt_config)
        return self._source_cache[source]

    def _prompt(self, source: str, caption_type: str) -> str:
        key = (source, caption_type)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = render_prompt(caption_type, source, self._prompt_config)
        return self._prompt_cache[key]

    def _gen_params(self, caption_type: str) -> dict[str, Any]:
        spec = self._type_specs[caption_type]
        return {key: spec.get(key) for key in GEN_PARAM_KEYS}

    def _collision_set(self) -> frozenset[Path]:
        """Paths sharing (folder, case-folded stem) with another discovered path (§8.2).

        Same rule as the dataloader's discovery WARNING (case-insensitive
        stems, because Windows file systems are case-insensitive).
        """
        groups: dict[tuple[Path, str], list[Path]] = {}
        for rel_path in self._loader.discover():
            groups.setdefault((rel_path.parent, rel_path.stem.casefold()), []).append(rel_path)
        return frozenset(p for paths in groups.values() if len(paths) > 1 for p in paths)

    # ------------------------------------------------------------------
    # Progress & cost
    # ------------------------------------------------------------------

    def _maybe_log_progress(self) -> None:
        if _monotonic() - self._t_last_log >= self._log_every_s:
            self._log_progress()

    def _throughput(self) -> dict[str, Any]:
        """Rates since processing started (after the model load) and cost figures (§11.4)."""
        now = _monotonic()
        plan = self._plan
        counts = self._counts
        processing_s = max(now - self._t_processing, 0.0) if self._t_processing is not None else 0.0
        total_s = max(now - self._t_start, 0.0)
        images_per_hour = counts.images_done / processing_s * 3600 if processing_s > 0 else None
        captions_per_s = counts.captions_done / processing_s if processing_s > 0 else None
        remaining = (plan.images_planned - counts.images_done) if plan else 0
        deferred = plan.images_deferred if plan else 0
        stats: dict[str, Any] = {
            "elapsed_s": round(total_s, 3),
            "processing_s": round(processing_s, 3),
            "images_done": counts.images_done,
            "images_remaining": remaining,
            "images_per_hour": _round(images_per_hour),
            "captions_per_s": _round(captions_per_s),
            "eta_s": _round(remaining / images_per_hour * 3600) if images_per_hour else None,
            "gpu_hourly_cost": self._gpu_hourly_cost,
            "cost_usd": None,
            "usd_per_1k_images": None,
            "projected_usd_remaining": None,
            "projected_usd_remaining_incl_deferred": None,
        }
        cost = self._gpu_hourly_cost
        if cost is not None:
            stats["cost_usd"] = _round(cost * total_s / 3600)
            if images_per_hour:
                stats["usd_per_1k_images"] = _round(cost / images_per_hour * 1000)
                stats["projected_usd_remaining"] = _round(cost * remaining / images_per_hour)
                stats["projected_usd_remaining_incl_deferred"] = _round(
                    cost * (remaining + deferred) / images_per_hour
                )
        return stats

    def _log_progress(self, final: bool = False) -> None:
        self._t_last_log = _monotonic()
        plan = self._plan
        if plan is None:
            return
        counts = self._counts
        stats = self._throughput()
        done = counts.captions_done
        flagged_pct = 100.0 * counts.captions_flagged / done if done else 0.0
        failed_pct = 100.0 * counts.captions_failed / done if done else 0.0
        eta = stats["eta_s"]
        message = (
            "%s: %d/%d images, %s captions/s, %s images/h, flagged %.1f%%, failed %.1f%%, ETA %s"
            % (
                "Done" if final else "Progress", counts.images_done, plan.images_planned,
                _fmt(stats["captions_per_s"], ".2f"), _fmt(stats["images_per_hour"], ".0f"),
                flagged_pct, failed_pct, "-" if eta is None else _fmt_duration(eta),
            )
        )
        if self._gpu_hourly_cost is not None:
            message += (
                "; cost $%s so far, $%s per 1k images, projected $%s for the remaining images "
                "($%s incl. %d deferred)"
                % (
                    _fmt(stats["cost_usd"], ".2f"), _fmt(stats["usd_per_1k_images"], ".3f"),
                    _fmt(stats["projected_usd_remaining"], ".2f"),
                    _fmt(stats["projected_usd_remaining_incl_deferred"], ".2f"),
                    plan.images_deferred,
                )
            )
        logger.info(message)

    # ------------------------------------------------------------------
    # Summary, QA page, manifest
    # ------------------------------------------------------------------

    def _summary(self) -> dict[str, Any]:
        plan = self._plan
        counts = self._counts
        values: dict[str, Any] = {
            "run_id": self._run_id,
            "shard": f"{self.shard[0]}/{self.shard[1]}",
            "images_discovered": plan.images_discovered if plan else 0,
            "images_in_shard": plan.images_in_shard if plan else 0,
            "images_already_complete": plan.images_already_complete if plan else 0,
            "images_deferred": plan.images_deferred if plan else 0,
            "images_planned": plan.images_planned if plan else 0,
            "images_load_failed": counts.images_load_failed,
            "captions_planned": plan.captions_planned if plan else 0,
            "captions_ok": counts.captions_ok,
            "captions_flagged": counts.captions_flagged,
            "captions_failed": counts.captions_failed,
            "retries": counts.retries,
            "interrupted": self._interrupted,
            "output_dir": self.output_dir.as_posix(),
        }
        return {key: values[key] for key in SUMMARY_KEYS}

    def _log_summary(self, summary: dict[str, Any]) -> None:
        level = logging.WARNING if summary["interrupted"] else logging.INFO
        logger.log(
            level,
            "Run %s %s: %d of %d captions ok, %d flagged, %d failed (%d image load failures), "
            "%d retries",
            summary["run_id"], "interrupted" if summary["interrupted"] else "finished",
            summary["captions_ok"], summary["captions_planned"], summary["captions_flagged"],
            summary["captions_failed"], summary["images_load_failed"], summary["retries"],
        )

    def _write_qa_page(self) -> str | None:
        """QA page ``qa/<run_id>.html`` (§10.3); a failure is logged, not raised."""
        try:
            path = generate_qa_page(
                self.output_dir, Path(self._loader.input_dir), n=self._qa_sample,
                seed=self._seed, name=self._run_id,
            )
        except Exception:  # noqa: BLE001 - the captions are written; the page is a convenience
            logger.exception("Could not write the QA page")
            return None
        if path is None:
            return None
        logger.info("QA page: %s", path.as_posix())
        return path.as_posix()

    def _write_manifest(self, error: BaseException | None, qa_page: str | None) -> Path:
        """Write ``runs/<run_id>.json`` atomically (D8, §8.1)."""
        summary = self._summary()
        info = self._model_info if self._model_info is not None else self._describe()
        failures_path = self._failures.path if self._failures and self._failures.opened else None
        manifest: dict[str, Any] = {
            "run_id": self._run_id,
            "started_utc": self._started_utc,
            "finished_utc": _iso(_utc_now()),
            "interrupted": self._interrupted,
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "summary": summary,
            "throughput": self._throughput(),
            "model": info,
            "versions": _versions(),
            "git_sha": _git_sha(),
            "prompt_config_hash": _safe_prompt_hash(self._prompt_config),
            "files": {
                "captions": self._rel_output(self._writer.path if self._writer else None),
                "failures": self._rel_output(failures_path),
                "qa_page": self._rel_output(Path(qa_page) if qa_page else None),
            },
            "config": self.config,
        }
        runs = self.output_dir / _RUNS_DIR
        runs.mkdir(parents=True, exist_ok=True)
        path = runs / f"{self._run_id}.json"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2, default=_json_default)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        logger.info("Run manifest: %s", path.as_posix())
        return path

    def _rel_output(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return Path(path).relative_to(self.output_dir).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def _describe(self) -> dict[str, Any]:
        try:
            return dict(self._model.describe())
        except Exception as exc:  # noqa: BLE001 - manifest data only
            logger.warning("model.describe() failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # SIGTERM
    # ------------------------------------------------------------------

    def _install_sigterm_handler(self) -> Any:
        """Turn SIGTERM into ``KeyboardInterrupt`` for the run; return the old handler.

        Only possible on the main thread; elsewhere nothing is installed.
        """
        sigterm = getattr(signal, "SIGTERM", None)
        if sigterm is None or threading.current_thread() is not threading.main_thread():
            logger.debug("Not on the main thread: SIGTERM handler not installed")
            return None
        try:
            previous = signal.getsignal(sigterm)
            signal.signal(sigterm, self._on_sigterm)
        except (ValueError, OSError, RuntimeError) as exc:
            logger.debug("SIGTERM handler not installed: %s", exc)
            return None
        self._sigterm_installed = True
        return previous

    def _restore_sigterm_handler(self, previous: Any) -> None:
        if not self._sigterm_installed:
            return
        self._sigterm_installed = False
        try:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            logger.warning("Could not restore the previous SIGTERM handler: %s", exc)

    def _on_sigterm(self, signum: int, frame: FrameType | None) -> None:
        if self._stopping:
            logger.warning("SIGTERM received while shutting down; finishing the shutdown")
            return
        self._stopping = True
        logger.warning("SIGTERM received: stopping the run")
        raise KeyboardInterrupt("SIGTERM")


# ---------------------------------------------------------------------------
# failures/ writer
# ---------------------------------------------------------------------------


class _FailureLog:
    """Appends to ``failures/shard-<iii>-of-<NNN>.jsonl``; the file is opened lazily."""

    def __init__(self, output_dir: Path, shard: tuple[int, int]) -> None:
        self.path = Path(output_dir) / _FAILURES_DIR / _shard_filename(*shard)
        self._fh: Any = None
        self.opened = False
        self.lines_written = 0

    def write_record(self, record: CaptionRecord) -> None:
        self._append(_dumps(record))

    def write_event(self, entry: dict[str, Any]) -> None:
        self._append(json.dumps(entry, ensure_ascii=False, default=_json_default))

    def _append(self, line: str) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _repair_tail(self.path)
            self._fh = open(self.path, "ab")  # binary: no newline translation on Windows
            self.opened = True
        self._fh.write((line + "\n").encode("utf-8"))
        self._fh.flush()
        self.lines_written += 1

    def close(self) -> None:
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        try:
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _status(flags: list[str]) -> str:
    """§5.3: ``error``/``empty`` -> failed; any other flag -> flagged; none -> ok."""
    if any(flag in _FAILED_FLAGS for flag in flags):
        return "failed"
    return "flagged" if flags else "ok"


def _is_cuda_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory, detected without importing torch (class name or message)."""
    if any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _json_default(value: Any) -> Any:
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def _safe_prompt_hash(prompt_config: dict[str, Any]) -> str | None:
    try:
        return prompt_config_hash(prompt_config)
    except Exception as exc:  # noqa: BLE001 - manifest data only
        logger.warning("Could not hash the prompt config: %s", exc)
        return None


def _versions() -> dict[str, str | None]:
    """Package, Python, Pillow and platform versions (model libraries: ``model.describe()``)."""
    return {
        "pixel_art_auto_captioner": _PACKAGE_VERSION,
        "python": platform.python_version(),
        "pillow": getattr(PIL, "__version__", None),
        "platform": platform.platform(),
    }


def _git_sha(start_dirs: Iterable[Path] | None = None) -> str | None:
    """Commit of the checkout containing this package (else the cwd), read from ``.git``.

    No ``git`` subprocess: follows ``.git`` (a directory, or a ``gitdir:``
    file), ``HEAD`` (a detached hash or ``ref:``), the loose ref, then
    ``packed-refs``. Returns ``None`` when there is no checkout (e.g. an
    installed package in Docker) or on any read problem.

    Args:
        start_dirs: Directories to search upwards from (default: this
            module's directory, then the current directory).

    Returns:
        The 40-hex commit hash, or ``None``.
    """
    if start_dirs is None:
        start_dirs = (Path(__file__).resolve().parent, Path.cwd())
    for start in start_dirs:
        try:
            sha = _read_git_sha(Path(start))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            logger.debug("Could not read the git commit under %s: %s", start, exc)
            continue
        if sha is not None:
            return sha
    return None


def _read_git_sha(start: Path) -> str | None:
    for directory in (start, *start.parents):
        dot_git = directory / ".git"
        if dot_git.is_file():
            text = dot_git.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return None
            git_dir = Path(text[len("gitdir:"):].strip())
            if not git_dir.is_absolute():
                git_dir = directory / git_dir
        elif dot_git.is_dir():
            git_dir = dot_git
        else:
            continue
        return _sha_from_git_dir(git_dir)
    return None


def _sha_from_git_dir(git_dir: Path) -> str | None:
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if _HEX40.fullmatch(head):
        return head
    if not head.startswith("ref:"):
        return None
    ref = head[len("ref:"):].strip()
    bases = [git_dir]
    commondir = git_dir / "commondir"  # linked worktrees keep refs in the common dir
    if commondir.is_file():
        bases.append((git_dir / commondir.read_text(encoding="utf-8").strip()).resolve())
    for base in bases:
        loose = base / ref
        if loose.is_file():
            value = loose.read_text(encoding="utf-8").strip()
            return value if _HEX40.fullmatch(value) else None
    for base in bases:
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(" ")
                if len(parts) == 2 and parts[1] == ref and _HEX40.fullmatch(parts[0]):
                    return parts[0]
    return None
