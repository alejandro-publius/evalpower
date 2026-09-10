# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/format/).

## [Unreleased]

### Added

- A note on the pooled aggregate when the rows it pools are not independent.
  Both interval estimators treat their `n` rows as `n` independent draws,
  which fails when the same item is scored on several dimensions -- the shape
  `validate_results` documents and permits. The pooled interval is then
  narrower than the evidence supports, and can return a confident verdict
  where the item level evidence supports none: 30 of 40 items passing every
  one of 10 dimensions pools to a decisive FAIL at the default 0.85
  threshold, while the same rate over the real 40 independent units is
  INDETERMINATE. Exposed as `metrics.pooled_independence_note`, the
  `Analysis.pooled_independence` property, a `pooled_independence` key in
  `--json`, and a line in the markdown Aggregate section. The committed
  examples score each item once, so their reports are unchanged.

## [0.1.0] - 2026-09-07

### Added

- Wilson score and nonparametric bootstrap percentile intervals per
  dimension, reported side by side, with a note when they disagree
  materially (for example, the bootstrap collapsing to a point at a
  saturated pass rate while the Wilson bound does not).
- Three way PASS / FAIL / INDETERMINATE verdicts, using the Wilson interval
  as the exact inversion of a one sided score test against a threshold.
- Benjamini-Hochberg false discovery rate adjustment across dimensions
  tested at once, with a report of which verdicts change under adjustment.
- Closed form required sample size for every INDETERMINATE dimension.
- Pooled aggregate estimate across all dimensions, with a note when it
  disagrees with the per dimension verdicts.
- `evalpower` CLI: reads `.csv`, `.json` or `.jsonl` results and writes a
  markdown report, with `--threshold`, `--confidence`, `--fdr-q`,
  `--bootstrap-resamples`, `--seed`, `--disagreement-tolerance`,
  `--output` and `--fail-on-indeterminate` (exit code 2, for CI).
- `--json` output mode, so a script or a CI step can parse the analysis
  instead of scraping the markdown report; composes with
  `--fail-on-indeterminate`.
- `examples/generate.py`, which reproduces the two committed example CSVs
  from a fixed seed and can sweep many seeds (`--sweep`) to report how
  often the multiple-comparisons failure mode the README describes occurs
  by chance, or find the first seed that displays it (`--choose-seed`).
- `tests/test_oracle.py`: cross-checks of the Wilson interval, the score
  test p values and the closed form required sample size against
  statsmodels and scipy, two libraries the package itself does not depend
  on, covering zero observed failures, zero observed passes, a single item
  eval, and a rate sitting exactly on the threshold.
- Edge case tests for a single-item eval, a saturated (zero-failure) pass
  rate, and a pass rate landing exactly on the threshold, run through the
  full `analyze()` and CLI pipeline, not just the underlying formulas.
- CI: run the test suite across the Python versions `pyproject.toml`
  declares (3.9, 3.12, 3.13), build and check the sdist and wheel, and lint
  with `ruff check`.
- Ruff lint configuration, `.pre-commit-config.yaml`, `CITATION.cff` and
  this changelog.

### Fixed

- README: the seed-sweep output block was missing the `runs:` line that
  `python examples/generate.py --sweep 3000` actually prints first; the
  block now matches the command's real output exactly.
- README: documented the new `--json` flag and added `pip install -e .` to
  the top quickstart, which previously ran the CLI without showing how to
  install it first.
