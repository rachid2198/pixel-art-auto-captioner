"""Batch orchestration runner — wires dataloader → model → exporter.

``CaptionRunner`` is the top-level orchestrator.  It takes a
pre-configured :class:`~pixel_art_auto_captioner.ingestion.dataloader.ImageDataLoader`
and a :class:`~pixel_art_auto_captioner.captioning.base.CaptionModel`,
loads the model once, iterates over all images, generates captions,
and writes results through the export utilities in ``common.export_utils``.

Depends on: ``pixel_art_auto_captioner.common``, ``.ingestion``, ``.captioning``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pixel_art_auto_captioner.captioning.base import CaptionModel
from pixel_art_auto_captioner.common.export_utils import (
    build_record,
    save_jsonl_entry,
    save_txt_sidecar,
)
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FORMATS: list[str] = ["txt", "jsonl"]
ALLOWED_OUTPUT_FORMATS: set[str] = {"txt", "jsonl"}


class CaptionRunner:
    """Top-level orchestrator that executes a captioning batch run.

    Wires together the three concerns — image loading, model inference,
    and caption export — with structured logging, per-image error
    resilience, and guaranteed GPU cleanup.

    Example usage::

        dataloader = ImageDataLoader(input_config)
        model = JoyCaptionModel()
        runner = CaptionRunner(dataloader, model, run_config)
        summary = runner.run()
        # {"total": 100, "succeeded": 98, "failed": 1, "skipped": 1, ...}
    """

    def __init__(
        self,
        dataloader: ImageDataLoader,
        model: CaptionModel,
        config: dict,
    ) -> None:
        """Initialise the runner.

        Args:
            dataloader: Pre-configured ``ImageDataLoader``.
            model: Any ``CaptionModel`` implementation (loaded on
                :meth:`run`).
            config: Run-time configuration dictionary with keys:

                ===================  ===========  ==========================
                Key                  Default      Description
                ===================  ===========  ==========================
                ``output_dir``       *required*   Directory for output files
                ``output_formats``   ``["txt", "jsonl"]``  One or both
                ``prompt_template``  *required*   Caption prompt text
                ``generation_params`` ``{}``       Gen-parameter overrides
                ``resume``           ``True``     Skip existing outputs
                ===================  ===========  ==========================

            Raises:
                ValueError: If ``output_dir`` or ``prompt_template`` is
                    missing, or ``output_formats`` is invalid.
        """
        # -- validate required keys ----------------------------------------
        output_dir_raw = config.get("output_dir")
        if not output_dir_raw:
            raise ValueError(
                "config must contain a non-empty 'output_dir' string"
            )
        self.output_dir: Path = Path(output_dir_raw).resolve()

        prompt = config.get("prompt_template", "")
        if not prompt:
            raise ValueError(
                "config must contain a non-empty 'prompt_template' string"
            )
        self.prompt_template: str = prompt

        # -- validate output_formats (Critique 4) --------------------------
        output_formats = config.get("output_formats", DEFAULT_OUTPUT_FORMATS)
        if not isinstance(output_formats, list) or len(output_formats) == 0:
            raise ValueError(
                f"'output_formats' must be a non-empty list, "
                f"got {output_formats!r}"
            )
        for fmt in output_formats:
            if fmt not in ALLOWED_OUTPUT_FORMATS:
                raise ValueError(
                    f"Unsupported output format {fmt!r}. "
                    f"Allowed: {sorted(ALLOWED_OUTPUT_FORMATS)}"
                )
        self.output_formats: list[str] = output_formats

        self.generation_params: dict[str, Any] = config.get(
            "generation_params", {}
        )
        self.resume: bool = config.get("resume", True)

        # -- validate resume consistency with dataloader (Critique 3) ------
        if self.resume and not dataloader.skip_existing:
            logger.warning(
                "Runner configured with resume=True, but the dataloader "
                "has skip_existing=False — existing outputs will NOT be "
                "skipped.  To skip existing outputs, set skip_existing=True "
                "in the dataloader config."
            )

        self._dataloader = dataloader
        self._model = model
        self._full_config = config  # passed through to model.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full captioning batch run.

        1. Loads the model via :meth:`CaptionModel.load`.
        2. Iterates over the dataloader, generating captions.
        3. On each success, creates a ``CaptionRecord`` and exports it.
        4. On each failure, logs the error and continues.
        5. Returns a summary dictionary.

        Returns:
            A summary dict:

            .. code-block:: python

                {
                    "total": int,        # images discovered on disk
                    "succeeded": int,    # successfully captioned
                    "failed": int,       # errors during captioning
                    "skipped": int,      # already existed / filtered
                    "output_dir": str,   # where captions were saved
                }

        Raises:
            RuntimeError: If ``model.load()`` fails (fail-fast per
                SPEC §12.1 — model-load errors must propagate so the
                CLI can exit with code 1).

        The model is **always** unloaded in a ``finally`` block, even
        if the run is terminated by an exception — including if
        ``model.load()`` fails after partial GPU allocation.
        """
        # Count total discovered before filtering
        total_discovered = len(self._dataloader.discover())
        total_to_process = len(self._dataloader)
        skipped = total_discovered - total_to_process
        succeeded = 0
        failed = 0

        logger.info("Loading model...")
        try:
            self._model.load(self._full_config)
            logger.info("Model loaded successfully.")  # Critique 6

            # -- iterate over images ---------------------------------------
            records_received = 0
            for image_record in self._dataloader:
                records_received += 1
                try:
                    caption_text, gen_meta = self._model.caption(
                        image_record.image,
                        self.prompt_template,
                        **self.generation_params,
                    )

                    if not caption_text:
                        logger.warning(
                            "Empty caption for %s — counting as failure.",
                            image_record.path.name,
                        )
                        failed += 1
                        continue

                    record = build_record(
                        image=image_record,
                        caption=caption_text,
                        model_name=self._model.model_name,
                        prompt=self.prompt_template,
                        gen_params=gen_meta.get(
                            "gen_params", self.generation_params
                        ),
                    )

                    # -- export ---------------------------------------------
                    if "txt" in self.output_formats:
                        save_txt_sidecar(record, self.output_dir)
                    if "jsonl" in self.output_formats:
                        save_jsonl_entry(record, self.output_dir)

                    succeeded += 1
                    logger.info("Saved: %s", image_record.path.name)

                except Exception:
                    logger.exception(
                        "Failed to caption image: %s", image_record.path
                    )
                    failed += 1
                    continue

            # -- detect iterator load drops (Critique 5) -------------------
            if records_received < total_to_process:
                load_failures = total_to_process - records_received
                logger.warning(
                    "Dataloader dropped %d image(s) during iteration "
                    "(likely load-time failures).  Counting as failed.",
                    load_failures,
                )
                failed += load_failures

        finally:
            # -- always unload, even if load() partially allocated (Critique 2)
            logger.info("Unloading model...")
            self._model.unload()

        summary: dict = {
            "total": total_discovered,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "output_dir": str(self.output_dir),
        }

        logger.info(
            "Batch complete. %d total, %d succeeded, %d failed, %d skipped.",
            summary["total"],
            summary["succeeded"],
            summary["failed"],
            summary["skipped"],
        )
        return summary