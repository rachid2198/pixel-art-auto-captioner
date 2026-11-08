"""Command-line entry point: caption one shard of the dataset (SPEC v1.1 §10.1).

Thin wiring only (AGENTS.md §4)::

    parse args -> load_config(--config, overrides) -> validate_config
    -> configure logging -> ImageDataLoader -> get_model(backend)
    -> CaptionRunner(...).run() -> log the summary -> exit code

``--dry_run`` stops after planning: it discovers, shards and resume-filters
like the runner and writes the plan as JSON to stdout (counts per source and
caption type, SPEC D7). It never builds a model, so no backend module and no
``torch`` / ``vllm`` is imported.

The model is never loaded here: :class:`CaptionRunner` owns the model
lifecycle and passes only the ``model`` config section to ``load()`` (B1).

Exit codes (SPEC §10.1, §12):

- ``0`` completed (also an empty plan, and every ``--dry_run``);
- ``1`` config error (bad flag value, missing/invalid config file, failed
  validation) or model-load / run error (backend import failure, load error,
  CUDA OOM, write error);
- ``2`` interrupted (Ctrl-C or SIGTERM; the runner flushed, unloaded and
  wrote the manifest first);
- ``3`` completed, but zero captions succeeded while more than zero were
  planned (systemic problem).

Logging goes to stderr (and ``runtime.log_file`` if set) with the SPEC §12
format; stdout carries only the ``--dry_run`` JSON plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

# Fallback for running the script from a checkout where the package is not
# installed (README: ``pip install -e .``). Once installed (pyproject.toml,
# SPEC §15) the import succeeds and ``sys.path`` is left alone.
try:  # pragma: no cover - depends on how the package is installed
    import pixel_art_auto_captioner  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixel_art_auto_captioner.batch.runner import CaptionRunner  # noqa: E402
from pixel_art_auto_captioner.captioning.base import get_model  # noqa: E402
from pixel_art_auto_captioner.common.config import load_config, validate_config  # noqa: E402
from pixel_art_auto_captioner.common.export_utils import ResumeIndex  # noqa: E402
from pixel_art_auto_captioner.ingestion.dataloader import ImageDataLoader, Plan  # noqa: E402

logger = logging.getLogger("pixel_art_auto_captioner.scripts.run_caption")

#: Process exit codes (SPEC §10.1).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 2
EXIT_NO_SUCCESS = 3

#: Values accepted by ``--log_level`` (SPEC §10.1) and ``runtime.log_level``.
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")

#: Log line format (SPEC §12).
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

#: Keys of the ``--dry_run`` JSON plan, in output order.
PLAN_KEYS: tuple[str, ...] = (
    "dry_run",
    "shard",
    "input_dir",
    "output_dir",
    "resume",
    "max_images",
    "caption_types",
    "images_discovered",
    "images_in_shard",
    "images_already_complete",
    "images_deferred",
    "images_planned",
    "captions_planned",
    "captions_by_type",
    "captions_by_source",
)

# Handlers installed by configure_logging(); replaced on every call.
_cli_handlers: list[logging.Handler] = []


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class _ArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose usage errors exit with code 1 (config error).

    argparse exits with 2 by default, which SPEC §10.1 reserves for
    "interrupted".
    """

    def error(self, message: str) -> NoReturn:
        """Print the usage and the message to stderr, then exit with ``EXIT_ERROR``."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SPEC §10.1 command line.

    Every option defaults to ``None`` (or ``False`` for ``--dry_run``) so that
    :func:`build_overrides` can tell which flags the user actually gave.

    Returns:
        The configured parser.
    """
    parser = _ArgumentParser(
        prog="run_caption.py",
        description="Caption one shard of a pixel-art image dataset with JoyCaption.",
        epilog="Exit codes: 0 completed; 1 config or model-load error; 2 interrupted; "
        "3 completed but zero captions succeeded while > 0 were planned.",
    )
    parser.add_argument("--config", type=Path, default=None, metavar="CONFIG",
                        help="JSON config file (defaults < file < flags)")
    parser.add_argument("--input_dir", default=None, metavar="DIR",
                        help="dataset root (input.input_dir)")
    parser.add_argument("--output_dir", default=None, metavar="DIR",
                        help="output directory (output.output_dir)")
    parser.add_argument("--backend", choices=["vllm", "hf"], default=None,
                        help="captioning backend (model.backend)")
    parser.add_argument("--model_path", default=None, metavar="PATH",
                        help="model directory or Hugging Face repo id (model.model_path)")
    parser.add_argument("--revision", default=None, metavar="REV",
                        help="Hugging Face commit hash (model.revision)")
    parser.add_argument("--caption_types", nargs="+", default=None, metavar="TYPE",
                        help="caption types to generate (generation.enabled_caption_types)")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="shard index/count, e.g. 0/4 (runtime.shard)")
    parser.add_argument("--max_images", type=int, default=None, metavar="N",
                        help="cap on pending images after resume (runtime.max_images)")
    parser.add_argument("--batch_size", type=int, default=None, metavar="N",
                        help="images per model batch (runtime.batch_size)")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="global seed (runtime.seed)")
    parser.add_argument("--write_txt", action=argparse.BooleanOptionalAction, default=None,
                        help="also write .txt files (output.write_txt)")
    parser.add_argument("--no-resume", dest="resume", action="store_const", const=False,
                        default=None, help="ignore existing captions (output.resume = false)")
    parser.add_argument("--gpu_hourly_cost", type=float, default=None, metavar="USD",
                        help="GPU price per hour, for cost logging (runtime.gpu_hourly_cost)")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the plan as JSON to stdout and exit; never loads the model")
    parser.add_argument("--log_level", choices=list(LOG_LEVELS), default=None,
                        help="logging level (runtime.log_level)")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv[1:]``.

    Returns:
        The parsed namespace.

    Raises:
        SystemExit: ``--help`` (code 0) or a usage error (code 1).
    """
    return build_parser().parse_args(argv)


#: Flag (namespace attribute) -> nested config key path (SPEC §9.1).
_OVERRIDE_PATHS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("input_dir", ("input", "input_dir")),
    ("output_dir", ("output", "output_dir")),
    ("backend", ("model", "backend")),
    ("model_path", ("model", "model_path")),
    ("revision", ("model", "revision")),
    ("caption_types", ("generation", "enabled_caption_types")),
    ("shard", ("runtime", "shard")),
    ("max_images", ("runtime", "max_images")),
    ("batch_size", ("runtime", "batch_size")),
    ("seed", ("runtime", "seed")),
    ("write_txt", ("output", "write_txt")),
    ("resume", ("output", "resume")),
    ("gpu_hourly_cost", ("runtime", "gpu_hourly_cost")),
    ("log_level", ("runtime", "log_level")),
)


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Turn the given flags into the nested ``overrides`` for ``load_config``.

    Flags the user did not give (value ``None``) are **omitted**, because
    ``load_config`` applies a ``None`` override as an explicit ``null``.

    Args:
        args: Namespace from :func:`parse_args`.

    Returns:
        Nested dict, e.g. ``{"runtime": {"shard": "1/4"}}``; empty when no
        config flag was given.
    """
    overrides: dict[str, Any] = {}
    for attr, (section, key) in _OVERRIDE_PATHS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if isinstance(value, list):
            value = list(value)
        overrides.setdefault(section, {})[key] = value
    return overrides


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(level: str, log_file: str | Path | None = None) -> None:
    """Configure the root logger: stderr plus an optional log file (SPEC §12).

    Replaces the handlers installed by a previous call (other handlers on the
    root logger are left alone). The log file is opened in append mode as
    UTF-8; its parent directory is created if needed.

    Args:
        level: One of :data:`LOG_LEVELS` (case-insensitive).
        log_file: Path of the log file, or ``None`` / ``""`` for stderr only.

    Raises:
        ValueError: ``level`` is not one of :data:`LOG_LEVELS`, or ``log_file``
            is neither a string, a path nor ``None``.
        OSError: The log file cannot be opened.
    """
    if not isinstance(level, str) or level.upper() not in LOG_LEVELS:
        raise ValueError(f"runtime.log_level: must be one of {list(LOG_LEVELS)}, got {level!r}")
    if log_file is not None and not isinstance(log_file, (str, Path)):
        raise ValueError(f"runtime.log_file: must be null or a path, got {log_file!r}")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="a", encoding="utf-8"))

    _remove_cli_handlers()
    formatter = logging.Formatter(LOG_FORMAT)
    root = logging.getLogger()
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _cli_handlers.append(handler)
    root.setLevel(level.upper())


def _remove_cli_handlers() -> None:
    """Detach and close the handlers added by :func:`configure_logging`."""
    root = logging.getLogger()
    while _cli_handlers:
        handler = _cli_handlers.pop()
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - closing a handler should not fail
            pass


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def make_plan(cfg: dict[str, Any], loader: ImageDataLoader) -> Plan:
    """Plan the shard exactly as the runner would, without a model (SPEC §7.2 step 1).

    Args:
        cfg: Validated, merged config.
        loader: Dataloader built from ``cfg``.

    Returns:
        The plan for ``cfg["generation"]["enabled_caption_types"]``.

    Raises:
        ValueError: A caption file in ``output_dir/captions`` is malformed
            (other than a crash-truncated final line).
        OSError: The output or input directory cannot be read.
    """
    output_dir = Path(cfg["output"]["output_dir"]).absolute()
    if cfg["output"].get("resume", True):
        index = ResumeIndex.from_output_dir(output_dir)
    else:
        index = ResumeIndex()
    return loader.plan(index.is_complete, list(cfg["generation"]["enabled_caption_types"]))


def plan_report(cfg: dict[str, Any], loader: ImageDataLoader, plan: Plan) -> dict[str, Any]:
    """Build the ``--dry_run`` JSON plan (SPEC D7), keys :data:`PLAN_KEYS`.

    Args:
        cfg: Validated, merged config.
        loader: The dataloader the plan was made with.
        plan: Result of :func:`make_plan`.

    Returns:
        A JSON-serialisable dict: shard, absolute POSIX ``input_dir`` /
        ``output_dir``, ``resume``, ``max_images``, caption types, the summary
        counts (same names as the run summary, SPEC §7.3), planned captions per
        caption type, and per source and caption type
        (``{source: {caption_type: n}}``; files directly in ``input_dir``
        count under ``""``).
    """
    by_type = {caption_type: 0 for caption_type in plan.caption_types}
    for _, types in plan.items:
        for caption_type in types:
            by_type[caption_type] += 1
    report = {
        "dry_run": True,
        "shard": f"{plan.shard[0]}/{plan.shard[1]}",
        "input_dir": Path(loader.input_dir).as_posix(),
        "output_dir": Path(cfg["output"]["output_dir"]).absolute().as_posix(),
        "resume": bool(cfg["output"].get("resume", True)),
        "max_images": cfg["runtime"].get("max_images"),
        "caption_types": list(plan.caption_types),
        "images_discovered": plan.images_discovered,
        "images_in_shard": plan.images_in_shard,
        "images_already_complete": plan.images_already_complete,
        "images_deferred": plan.images_deferred,
        "images_planned": plan.images_planned,
        "captions_planned": plan.captions_planned,
        "captions_by_type": by_type,
        "captions_by_source": plan.captions_by_source(),
    }
    return {key: report[key] for key in PLAN_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def exit_code_for(summary: dict[str, Any]) -> int:
    """Exit code of a run that returned normally (SPEC §10.1).

    Args:
        summary: The runner's summary (SPEC §7.3).

    Returns:
        :data:`EXIT_NO_SUCCESS` if captions were planned but none ended
        ``ok`` or ``flagged``; otherwise :data:`EXIT_OK`.
    """
    planned = summary.get("captions_planned", 0)
    succeeded = summary.get("captions_ok", 0) + summary.get("captions_flagged", 0)
    return EXIT_NO_SUCCESS if planned > 0 and succeeded == 0 else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code (never raises ``SystemExit``).

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv[1:]``.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_ERROR`, :data:`EXIT_INTERRUPTED` or
        :data:`EXIT_NO_SUCCESS` (SPEC §10.1).
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # --help or a usage error
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    try:
        return _main(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted; exiting with code %d", EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED
    finally:
        _remove_cli_handlers()  # flush + close (no open log file is left behind)


def _main(args: argparse.Namespace) -> int:
    """Body of :func:`main` after argument parsing."""
    # Provisional stderr logging so config errors are reported in the §12 format.
    configure_logging(args.log_level or "INFO")

    # 1. Config: defaults < file < flags; validation; final logging setup.
    try:
        cfg = load_config(args.config, build_overrides(args))
        validate_config(cfg)
        runtime = cfg["runtime"]
        configure_logging(runtime.get("log_level", "INFO"), runtime.get("log_file"))
        loader = ImageDataLoader({**cfg["input"], **cfg["runtime"]})
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_ERROR

    # 2. Dry run: plan only, never a model.
    if args.dry_run:
        try:
            plan = make_plan(cfg, loader)
        except (ValueError, OSError) as exc:
            logger.error("Planning failed: %s: %s", type(exc).__name__, exc)
            return EXIT_ERROR
        report = plan_report(cfg, loader, plan)
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        sys.stdout.flush()
        logger.info(
            "Dry run: %d images / %d captions planned for shard %s; the model was not loaded",
            report["images_planned"], report["captions_planned"], report["shard"],
        )
        return EXIT_OK

    # 3. Model (unloaded; the runner loads it, B1) and runner.
    backend = cfg["model"]["backend"]
    try:
        model = get_model(backend)
    except (ImportError, ValueError, TypeError) as exc:
        logger.error("Cannot create the %r backend: %s", backend, exc)
        return EXIT_ERROR
    try:
        runner = CaptionRunner(loader, model, cfg)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_ERROR

    # 4. Run. The runner turns SIGTERM into KeyboardInterrupt and writes the
    #    manifest before re-raising; no handler is installed here.
    try:
        summary = runner.run()
    except KeyboardInterrupt:
        logger.warning("Run interrupted; exiting with code %d", EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED
    except Exception as exc:
        logger.error("Run failed: %s: %s", type(exc).__name__, exc)
        logger.debug("Run failure traceback", exc_info=True)
        return EXIT_ERROR

    logger.info("Summary: %s", json.dumps(summary))
    code = exit_code_for(summary)
    if code == EXIT_NO_SUCCESS:
        logger.error(
            "%d captions planned but none succeeded (systemic problem); exit code %d",
            summary.get("captions_planned", 0), code,
        )
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
