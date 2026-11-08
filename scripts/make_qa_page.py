"""Write a self-contained QA spot-check page from the JSONL output (SPEC v1.1 §10.3).

Thin CLI over :func:`pixel_art_auto_captioner.common.export_utils.generate_qa_page`
(the same function the runner calls at the end of a run): a random sample of
``--n`` images (stratified by source), each embedded as a base64 JPEG
thumbnail next to all its caption types, with status and flags highlighted
and prompt, seed and token count on hover. No external CSS/JS. The page is
written to ``<output_dir>/qa/qa-<UTC timestamp>.html``; ``input_dir`` is only
read (thumbnails).

Exit codes (same convention as ``run_caption.py``):

- ``0`` page written (also ``--n 0``, which writes nothing);
- ``1`` usage error, ``output_dir`` / ``input_dir`` not a directory, bad
  ``--status`` value, malformed JSONL, write error;
- ``2`` interrupted (Ctrl-C).

Logging goes to stderr with the SPEC §12 format.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

# Fallback for running the script from a checkout where the package is not
# installed (README: ``pip install -e .``). Once installed (pyproject.toml,
# SPEC §15) the import succeeds and ``sys.path`` is left alone.
try:  # pragma: no cover - depends on how the package is installed
    import pixel_art_auto_captioner  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixel_art_auto_captioner.common.export_utils import generate_qa_page  # noqa: E402

logger = logging.getLogger("pixel_art_auto_captioner.scripts.make_qa_page")

#: Process exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 2

#: Log line format (SPEC §12).
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

#: Defaults (SPEC §10.3).
DEFAULT_N = 200
DEFAULT_SEED = 0
DEFAULT_STATUS = "ok,flagged"

# Handlers installed by _configure_logging(); removed when main() returns.
_cli_handlers: list[logging.Handler] = []


class _ArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose usage errors exit with code 1 (config error)."""

    def error(self, message: str) -> NoReturn:
        """Print the usage and the message to stderr, then exit with ``EXIT_ERROR``."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SPEC §10.3 command line.

    Returns:
        The configured parser.
    """
    parser = _ArgumentParser(
        prog="make_qa_page.py",
        description="Write a self-contained HTML spot-check page of sampled captions.",
        epilog="Exit codes: 0 done; 1 usage, input or write error; 2 interrupted.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, metavar="DIR",
                        help="output directory of the captioning runs (with captions/)")
    parser.add_argument("--input_dir", type=Path, required=True, metavar="DIR",
                        help="dataset root the image paths are relative to (read-only)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, metavar="N",
                        help=f"number of images to sample (default {DEFAULT_N}; 0 = no page)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="N",
                        help=f"sampling seed (default {DEFAULT_SEED})")
    parser.add_argument("--status", default=DEFAULT_STATUS, metavar="STATUS[,STATUS]",
                        help="comma-separated statuses that make an image eligible: ok, "
                        f"flagged, failed (default {DEFAULT_STATUS})")
    parser.add_argument("--source", nargs="+", default=None, metavar="SOURCE",
                        help="only these sources (top-level folders; default: all)")
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


def parse_statuses(value: str) -> list[str]:
    """Split a ``--status`` value on commas, dropping blanks (``"ok, flagged"`` -> 2 items).

    Args:
        value: The flag value.

    Returns:
        The status names, in order, without duplicates. Validation is left
        to ``generate_qa_page`` (``ValueError`` for an unknown or empty set).
    """
    out: list[str] = []
    for part in value.split(","):
        name = part.strip()
        if name and name not in out:
            out.append(name)
    return out


def _configure_logging() -> None:
    """INFO logging to stderr with the SPEC §12 format (handlers removed by ``main``)."""
    _remove_cli_handlers()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _cli_handlers.append(handler)


def _remove_cli_handlers() -> None:
    root = logging.getLogger()
    while _cli_handlers:
        handler = _cli_handlers.pop()
        root.removeHandler(handler)
        handler.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code (never raises ``SystemExit``).

    Args:
        argv: Arguments without the program name; ``None`` uses ``sys.argv[1:]``.

    Returns:
        :data:`EXIT_OK`, :data:`EXIT_ERROR` or :data:`EXIT_INTERRUPTED`.
    """
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # --help or a usage error
        return exc.code if isinstance(exc.code, int) else EXIT_ERROR
    _configure_logging()
    try:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"output_dir is not a directory: {args.output_dir}")
        generate_qa_page(
            args.output_dir,
            args.input_dir,
            n=args.n,
            seed=args.seed,
            statuses=parse_statuses(args.status),
            sources=args.source,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted; exiting with code %d", EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED
    except (ValueError, OSError) as exc:  # FileNotFoundError is an OSError
        logger.error("QA page failed: %s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    finally:
        _remove_cli_handlers()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
