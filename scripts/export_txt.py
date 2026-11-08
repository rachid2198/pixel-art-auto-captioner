"""Rebuild ``txt/`` caption files from the JSONL output (SPEC v1.1 §10.2, §8.2).

Thin wiring only (AGENTS.md §4) over ``common.export_utils``:
``read_records`` reads every ``<output_dir>/captions/*.jsonl`` file, the
**latest** record per ``(image_rel_path, caption_type)`` wins (by the parsed
``timestamp_utc``; on a tie the record read last, files in name order, the
same rule as resume), and ``save_txt`` writes
``txt/<rel_dir>/<stem>.<caption_type>.txt`` with the §8.2 collision rule.

- Latest status ``ok`` -> exported; ``flagged`` -> exported only with
  ``--include_flagged``; ``failed`` -> never.
- ``--caption_types`` limits the export to those types (default: all types
  found in the JSONL).
- Collisions (§8.2): images in the same folder whose stems are equal
  (case-insensitively, like discovery) use ``<stem>.<ext>.<caption_type>.txt``.
  The set is computed from every ``image_rel_path`` in the JSONL files (any
  status, any caption type), since this script does not walk the dataset.
- Existing ``.txt`` files are overwritten, never read and never deleted
  (state comes only from the JSONL, AGENTS.md §6.8). ``.txt`` files under
  ``txt/`` that this export did not produce are counted in a WARNING.

Exit codes (same convention as ``run_caption.py``):

- ``0`` done (also when no record qualifies);
- ``1`` usage error, ``output_dir`` missing or without ``captions/*.jsonl``,
  malformed JSONL (other than a crash-truncated final line), write error;
- ``2`` interrupted (Ctrl-C).

Logging goes to stderr with the SPEC §12 format.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

# Fallback for running the script from a checkout where the package is not
# installed (README: ``pip install -e .``). Once installed (pyproject.toml,
# SPEC §15) the import succeeds and ``sys.path`` is left alone.
try:  # pragma: no cover - depends on how the package is installed
    import pixel_art_auto_captioner  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixel_art_auto_captioner.common.export_utils import (  # noqa: E402
    _caption_files,
    _parse_timestamp,
    read_records,
    save_txt,
)
from pixel_art_auto_captioner.common.types import CaptionRecord  # noqa: E402

logger = logging.getLogger("pixel_art_auto_captioner.scripts.export_txt")

#: Process exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 2

#: Log line format (SPEC §12).
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

#: Keys every record needs (non-empty strings), as for resume (SPEC §8.4).
_KEY_FIELDS: tuple[str, ...] = ("image_rel_path", "caption_type", "status", "timestamp_utc")

_RECORD_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(CaptionRecord))

# Handlers installed by _configure_logging(); removed when main() returns.
_cli_handlers: list[logging.Handler] = []


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class _ArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` whose usage errors exit with code 1 (config error)."""

    def error(self, message: str) -> NoReturn:
        """Print the usage and the message to stderr, then exit with ``EXIT_ERROR``."""
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the SPEC §10.2 command line.

    Returns:
        The configured parser.
    """
    parser = _ArgumentParser(
        prog="export_txt.py",
        description="Rebuild txt/ caption files from the JSONL output (latest record per "
        "image and caption type).",
        epilog="Exit codes: 0 done; 1 usage, input or write error; 2 interrupted.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, metavar="DIR",
                        help="output directory of the captioning runs (with captions/)")
    parser.add_argument("--caption_types", nargs="+", default=None, metavar="TYPE",
                        help="only these caption types (default: all found)")
    parser.add_argument("--include_flagged", action="store_true",
                        help="also export captions whose latest status is 'flagged'")
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def collision_set(rel_paths: Iterable[str]) -> frozenset[str]:
    """POSIX paths that share (folder, case-folded stem) with another path (SPEC §8.2).

    Same rule as the dataloader's discovery WARNING and the runner.

    Args:
        rel_paths: Relative image paths (POSIX strings).

    Returns:
        The colliding paths.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for rel in set(rel_paths):
        p = PurePosixPath(rel)
        groups.setdefault((p.parent.as_posix(), p.stem.casefold()), []).append(rel)
    return frozenset(r for group in groups.values() if len(group) > 1 for r in group)


def _check_record(rec: dict[str, Any], position: int) -> tuple[str, str, str, float]:
    values = [rec.get(k) for k in _KEY_FIELDS]
    if not all(isinstance(v, str) and v for v in values):
        raise ValueError(
            f"record #{position}: needs non-empty string fields {list(_KEY_FIELDS)}, "
            f"got {dict(zip(_KEY_FIELDS, values))!r}"
        )
    rel, caption_type, status, timestamp = values
    epoch = _parse_timestamp(timestamp)
    if epoch is None:
        raise ValueError(f"record #{position} ({rel}): unparseable timestamp_utc {timestamp!r}")
    return rel, caption_type, status, epoch


def export_txt(
    output_dir: Path,
    *,
    caption_types: Collection[str] | None = None,
    include_flagged: bool = False,
) -> dict[str, int]:
    """Write ``txt/`` files for the latest qualifying record of each key (SPEC §10.2).

    Two passes over ``captions/*.jsonl``: the first finds the latest record
    per ``(image_rel_path, caption_type)`` and all image paths (for the
    collision set); the second writes the winners that qualify.

    Args:
        output_dir: Output directory with ``captions/*.jsonl``.
        caption_types: Only these caption types (``None`` = all).
        include_flagged: Also export latest records with status ``flagged``.

    Returns:
        Counts: ``records_read``, ``keys`` (distinct image/caption-type
        pairs), ``written``, ``written_ok``, ``written_flagged``,
        ``skipped_failed``, ``skipped_flagged``, ``skipped_type``,
        ``collisions``, ``stale_txt``.

    Raises:
        FileNotFoundError: ``output_dir`` is not a directory or has no
            ``captions/*.jsonl`` file.
        ValueError: A malformed JSONL line (not a file's final line), a
            record without the key fields / with an unparseable timestamp or
            an unsafe ``image_rel_path``, or the files changed between the
            two passes (a run still writing).
        OSError: A file cannot be read or written.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output_dir is not a directory: {output_dir}")
    files = _caption_files(output_dir)
    if not files:
        raise FileNotFoundError(f"no captions/*.jsonl files in {output_dir}")
    type_filter = None if caption_types is None else frozenset(caption_types)
    export_statuses = {"ok", "flagged"} if include_flagged else {"ok"}

    # Pass 1: latest record per key (position = order read), and all image paths.
    latest: dict[tuple[str, str], tuple[float, int, str]] = {}
    rel_paths: set[str] = set()
    position = 0
    for position, rec in enumerate(read_records(files), 1):
        rel, caption_type, status, epoch = _check_record(rec, position)
        rel_paths.add(rel)
        prev = latest.get((rel, caption_type))
        if prev is None or epoch >= prev[0]:
            latest[(rel, caption_type)] = (epoch, position, status)
    records_read = position

    counts = {
        "records_read": records_read, "keys": len(latest), "written": 0, "written_ok": 0,
        "written_flagged": 0, "skipped_failed": 0, "skipped_flagged": 0, "skipped_type": 0,
    }
    winners: dict[int, tuple[str, str]] = {}
    for (rel, caption_type), (_, pos, status) in latest.items():
        if type_filter is not None and caption_type not in type_filter:
            counts["skipped_type"] += 1
        elif status in export_statuses:
            winners[pos] = (rel, caption_type)
        elif status == "flagged":
            counts["skipped_flagged"] += 1
        else:
            counts["skipped_failed"] += 1
    if type_filter is not None:
        unknown = sorted(type_filter - {ct for _, ct in latest})
        if unknown:
            logger.warning("No records for caption types %s", unknown)
    collisions = collision_set(rel_paths)
    counts["collisions"] = len(collisions)
    if collisions:
        logger.warning(
            "%d images share a stem with another image in the same folder; their .txt files "
            "are named <stem>.<ext>.<caption_type>.txt", len(collisions),
        )

    # Pass 2: write the winners.
    written: set[Path] = set()
    for pos, rec in enumerate(read_records(files), 1):
        key = winners.get(pos)
        if key is None:
            continue
        if (rec.get("image_rel_path"), rec.get("caption_type")) != key:
            raise ValueError(
                "the caption files changed during the export (is a run still writing to "
                f"{output_dir}?); re-run when it has finished"
            )
        record = CaptionRecord(**{name: rec.get(name) for name in _RECORD_FIELDS})
        if not isinstance(record.caption_text, str):
            raise ValueError(
                f"record #{pos} ({record.image_rel_path}): caption_text is not a string"
            )
        written.add(save_txt(record, output_dir, collisions).resolve())
        counts["written"] += 1
        counts["written_" + record.status] += 1

    counts["stale_txt"] = _count_stale_txt(output_dir / "txt", written)
    logger.info(
        "Exported %d .txt files (%d ok, %d flagged) from %d records in %d files; "
        "not exported: %d failed, %d flagged (no --include_flagged), %d other caption types",
        counts["written"], counts["written_ok"], counts["written_flagged"], records_read,
        len(files), counts["skipped_failed"], counts["skipped_flagged"], counts["skipped_type"],
    )
    return counts


def _count_stale_txt(txt_root: Path, written: set[Path]) -> int:
    """Count (never read or delete) ``.txt`` files under ``txt_root`` not in ``written``."""
    if not txt_root.is_dir():
        return 0
    stale: list[str] = []
    for dirpath, _, filenames in os.walk(txt_root):
        for name in filenames:
            path = Path(dirpath, name)
            if name.lower().endswith(".txt") and path.resolve() not in written:
                stale.append(path.relative_to(txt_root).as_posix())
    if stale:
        stale.sort()
        logger.warning(
            "%d .txt files under %s were not produced by this export (left untouched; e.g. "
            "older captions, other caption types or collision renames): %s%s",
            len(stale), txt_root.as_posix(), stale[:10], " ..." if len(stale) > 10 else "",
        )
    return len(stale)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        export_txt(
            args.output_dir,
            caption_types=args.caption_types,
            include_flagged=args.include_flagged,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted; exiting with code %d", EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED
    except (ValueError, OSError) as exc:  # FileNotFoundError is an OSError
        logger.error("Export failed: %s: %s", type(exc).__name__, exc)
        return EXIT_ERROR
    finally:
        _remove_cli_handlers()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
