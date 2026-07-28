"""Command line entry point: ``evalpower results.csv --threshold 0.85``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_DISAGREEMENT_TOLERANCE,
)
from .report import analyze, render_markdown
from .verdicts import DEFAULT_THRESHOLD, Verdict

__all__ = ["build_parser", "load_results", "main"]


def load_results(path: Path) -> pd.DataFrame:
    """Load an eval result set from CSV or JSON.

    CSV is read directly. JSON accepts either a list of record objects or an
    object with a ``results`` (or ``rows``) key holding that list. Either way
    each record needs ``item_id``, ``dimension`` and ``passed``.

    Args:
        path: Path to a ``.csv``, ``.json`` or ``.jsonl`` file.

    Returns:
        The results as a DataFrame.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: on an unsupported suffix or an unrecognised JSON shape.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no such file: {0}".format(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("results", "rows", "records"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(
                    "JSON object must hold the records under 'results', 'rows' or 'records'"
                )
        if not isinstance(payload, list):
            raise ValueError("JSON results must be a list of records")
        return pd.DataFrame.from_records(payload)
    raise ValueError("unsupported file type {0!r}; use .csv, .json or .jsonl".format(suffix))


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``evalpower`` command."""
    parser = argparse.ArgumentParser(
        prog="evalpower",
        description=(
            "Check whether an eval has enough data to support the verdict it "
            "just printed. Reads one row per (item, dimension) and writes a "
            "markdown report."
        ),
    )
    parser.add_argument("results", type=Path, help="path to a .csv, .json or .jsonl result set")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="pass rate the eval is held to (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="two sided confidence level for both intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--fdr-q",
        type=float,
        default=None,
        help="Benjamini-Hochberg level (default: the one sided error rate implied by --confidence)",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples per dimension (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="base seed for the bootstrap, so reports are reproducible (default: %(default)s)",
    )
    parser.add_argument(
        "--disagreement-tolerance",
        type=float,
        default=DEFAULT_DISAGREEMENT_TOLERANCE,
        help="bound gap above which Wilson and bootstrap are flagged (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the report here instead of stdout",
    )
    parser.add_argument(
        "--fail-on-indeterminate",
        action="store_true",
        help="exit non-zero when any dimension is indeterminate, for use in CI",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line interface.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on a bad input, and 2 when
        ``--fail-on-indeterminate`` is set and a dimension is indeterminate.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        frame = load_results(args.results)
        analysis = analyze(
            frame,
            threshold=args.threshold,
            confidence=args.confidence,
            fdr_q=args.fdr_q,
            resamples=args.bootstrap_resamples,
            seed=args.seed,
            tolerance=args.disagreement_tolerance,
            source=str(args.results),
        )
    except (FileNotFoundError, ValueError) as error:
        print("evalpower: {0}".format(error), file=sys.stderr)
        return 1

    report = render_markdown(analysis)
    if args.output is None:
        sys.stdout.write(report)
    else:
        args.output.write_text(report, encoding="utf-8")
        print("wrote {0}".format(args.output), file=sys.stderr)

    if args.fail_on_indeterminate and any(
        item.verdict is Verdict.INDETERMINATE for item in analysis.dimensions
    ):
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
