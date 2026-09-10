"""Assemble the analysis and render it as markdown or JSON.

The report answers one question: does this eval have enough data to support
the verdict it just printed?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_DISAGREEMENT_TOLERANCE,
    DimensionMetrics,
    Interval,
    mean_interval_width,
    pooled_independence_note,
    pooled_metrics,
    summarize_dimensions,
    validate_results,
)
from .planning import SampleSizeRequirement, requirement_for
from .verdicts import (
    DEFAULT_THRESHOLD,
    Verdict,
    VerdictChange,
    benjamini_hochberg,
    count_changed_verdicts,
    default_fdr_q,
    score_test_pvalues,
    verdict_counts,
    verdict_from_interval,
    verdict_from_pvalues,
)

__all__ = ["Analysis", "DimensionAnalysis", "analyze", "analysis_to_dict", "render_json", "render_markdown"]


@dataclass(frozen=True)
class DimensionAnalysis:
    """Everything the report knows about one dimension."""

    metrics: DimensionMetrics
    verdict: Verdict
    adjusted_verdict: Verdict
    p_pass: float
    p_fail: float
    adjusted_p_pass: float
    adjusted_p_fail: float
    requirement: Optional[SampleSizeRequirement]

    @property
    def dimension(self) -> str:
        """Return the dimension name."""
        return self.metrics.dimension


@dataclass(frozen=True)
class Analysis:
    """The complete analysis of one eval run."""

    source: str
    threshold: float
    confidence: float
    fdr_q: float
    n_rows: int
    n_items: int
    dimensions: List[DimensionAnalysis]
    aggregate: DimensionMetrics
    aggregate_verdict: Verdict
    changes: List[VerdictChange]

    @property
    def pooled_independence(self) -> Optional[str]:
        """Return a note when pooling treats correlated rows as independent."""
        return pooled_independence_note(self.n_rows, self.n_items)

    @property
    def k(self) -> int:
        """Return the number of dimensions tested at once."""
        return len(self.dimensions)


def analyze(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
    fdr_q: Optional[float] = None,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    tolerance: float = DEFAULT_DISAGREEMENT_TOLERANCE,
    source: str = "results",
) -> Analysis:
    """Run the full analysis over a results frame.

    Args:
        frame: Results, one row per (item, dimension) pair.
        threshold: The bar the eval is being held to.
        confidence: Two sided confidence level for both intervals.
        fdr_q: False discovery rate for the Benjamini-Hochberg step. Defaults
            to the one sided error rate implied by ``confidence``.
        resamples: Bootstrap resamples per dimension.
        seed: Base seed for the bootstrap.
        tolerance: Bound gap above which the two intervals are flagged.
        source: Label recorded in the report header.

    Returns:
        A fully populated :class:`Analysis`.
    """
    validated = validate_results(frame)
    q = default_fdr_q(confidence) if fdr_q is None else float(fdr_q)

    metrics = summarize_dimensions(validated, confidence, resamples, seed, tolerance)
    verdicts = [verdict_from_interval(item.wilson, threshold) for item in metrics]

    pvalues = [score_test_pvalues(item.passes, item.n, threshold) for item in metrics]
    pass_rejected, pass_adjusted = benjamini_hochberg([pair[0] for pair in pvalues], q)
    fail_rejected, fail_adjusted = benjamini_hochberg([pair[1] for pair in pvalues], q)

    adjusted_verdicts = [
        verdict_from_pvalues(pass_rejected[index], fail_rejected[index], metrics[index].rate, threshold)
        for index in range(len(metrics))
    ]

    dimensions = []
    for index, item in enumerate(metrics):
        requirement = None
        if verdicts[index] is Verdict.INDETERMINATE:
            requirement = requirement_for(item.dimension, item.passes, item.n, threshold, confidence)
        dimensions.append(
            DimensionAnalysis(
                metrics=item,
                verdict=verdicts[index],
                adjusted_verdict=adjusted_verdicts[index],
                p_pass=pvalues[index][0],
                p_fail=pvalues[index][1],
                adjusted_p_pass=pass_adjusted[index],
                adjusted_p_fail=fail_adjusted[index],
                requirement=requirement,
            )
        )

    aggregate = pooled_metrics(validated, "ALL", confidence, resamples, seed, tolerance)
    return Analysis(
        source=source,
        threshold=threshold,
        confidence=confidence,
        fdr_q=q,
        n_rows=len(validated),
        n_items=int(validated["item_id"].nunique()),
        dimensions=dimensions,
        aggregate=aggregate,
        aggregate_verdict=verdict_from_interval(aggregate.wilson, threshold),
        changes=count_changed_verdicts(
            [item.dimension for item in metrics], verdicts, adjusted_verdicts
        ),
    )


def _interval_to_dict(interval: Interval) -> Dict[str, float]:
    """Return an :class:`~evalpower.metrics.Interval` as a JSON-safe dict."""
    return {"lower": interval.lower, "upper": interval.upper}


def _metrics_to_dict(metrics: DimensionMetrics) -> Dict[str, Any]:
    """Return a :class:`~evalpower.metrics.DimensionMetrics` as a JSON-safe dict."""
    return {
        "dimension": metrics.dimension,
        "n": metrics.n,
        "passes": metrics.passes,
        "rate": metrics.rate,
        "wilson": _interval_to_dict(metrics.wilson),
        "bootstrap": _interval_to_dict(metrics.bootstrap),
        "disagreement": metrics.disagreement,
    }


def _requirement_to_dict(requirement: Optional[SampleSizeRequirement]) -> Optional[Dict[str, Any]]:
    """Return a :class:`~evalpower.planning.SampleSizeRequirement` as a dict, or None."""
    if requirement is None:
        return None
    resolves_to = requirement.would_resolve_to
    return {
        "current_n": requirement.current_n,
        "observed_rate": requirement.observed_rate,
        "required_n": requirement.required_n,
        "additional_items": requirement.additional_items,
        "multiple_of_current": requirement.multiple_of_current,
        "would_resolve_to": str(resolves_to) if resolves_to is not None else None,
    }


def _dimension_to_dict(item: DimensionAnalysis) -> Dict[str, Any]:
    """Return a :class:`DimensionAnalysis` as a JSON-safe dict."""
    payload = _metrics_to_dict(item.metrics)
    payload.update(
        {
            "verdict": str(item.verdict),
            "adjusted_verdict": str(item.adjusted_verdict),
            "p_pass": item.p_pass,
            "p_fail": item.p_fail,
            "adjusted_p_pass": item.adjusted_p_pass,
            "adjusted_p_fail": item.adjusted_p_fail,
            "requirement": _requirement_to_dict(item.requirement),
        }
    )
    return payload


def analysis_to_dict(analysis: Analysis) -> Dict[str, Any]:
    """Return a complete :class:`Analysis` as a plain, JSON-safe dict.

    Every field the markdown report renders is present here too, in the same
    units (rates as fractions, not percentages; p values unrounded), so a
    caller can gate a build on it without re-parsing prose.

    Args:
        analysis: The analysis to convert.

    Returns:
        A nested ``dict`` of built-in types, ready for :func:`json.dumps`.
    """
    return {
        "source": analysis.source,
        "threshold": analysis.threshold,
        "confidence": analysis.confidence,
        "fdr_q": analysis.fdr_q,
        "n_rows": analysis.n_rows,
        "n_items": analysis.n_items,
        "k": analysis.k,
        "dimensions": [_dimension_to_dict(item) for item in analysis.dimensions],
        "aggregate": _metrics_to_dict(analysis.aggregate),
        "aggregate_verdict": str(analysis.aggregate_verdict),
        "pooled_independence": analysis.pooled_independence,
        "changes": [
            {
                "dimension": change.dimension,
                "unadjusted": str(change.unadjusted),
                "adjusted": str(change.adjusted),
            }
            for change in analysis.changes
        ],
    }


def render_json(analysis: Analysis, indent: Optional[int] = 2) -> str:
    """Render an :class:`Analysis` as a JSON document.

    Carries the same information as :func:`render_markdown` in a shape meant
    for machines rather than terminals: a CI step can parse it and gate a
    build on ``aggregate_verdict`` or on any per dimension ``verdict``
    without scraping markdown.

    Args:
        analysis: The analysis to render.
        indent: Passed straight to :func:`json.dumps`; ``None`` renders
            compact single-line JSON instead of pretty-printed output.

    Returns:
        A JSON document as a single string, ending in a newline.
    """
    return json.dumps(analysis_to_dict(analysis), indent=indent) + "\n"


def _format_p(value: float) -> str:
    """Format a p value compactly without losing small magnitudes."""
    if value < 1e-4:
        return "{0:.1e}".format(value)
    return "{0:.4f}".format(value)


def _plural(count: int, singular: str, plural: str) -> str:
    """Return ``singular`` or ``plural`` to match ``count``."""
    return singular if count == 1 else plural


def _headline(analysis: Analysis) -> List[str]:
    """Render the headline verdict counts."""
    counts = verdict_counts([item.verdict for item in analysis.dimensions])
    k = analysis.k
    lines = [
        "## Headline",
        "",
        "`{0}` scored {1} {2} across {3} {4} at threshold {5:.2f} "
        "with {6:.0%} intervals.".format(
            analysis.source,
            analysis.n_items,
            _plural(analysis.n_items, "item", "items"),
            k,
            _plural(k, "dimension", "dimensions"),
            analysis.threshold,
            analysis.confidence,
        ),
        "",
        "| Verdict | Dimensions |",
        "| --- | ---: |",
        "| PASS | {0} |".format(counts[Verdict.PASS]),
        "| INDETERMINATE | {0} |".format(counts[Verdict.INDETERMINATE]),
        "| FAIL | {0} |".format(counts[Verdict.FAIL]),
        "",
    ]
    pooled_width = analysis.aggregate.wilson.width
    if k == 1:
        lines.append(
            "One dimension is the degenerate case: the per dimension estimate "
            "and the pooled estimate are the same {0:.0%} interval, of width "
            "{1:.4f}.".format(analysis.confidence, pooled_width)
        )
    else:
        mean_width = mean_interval_width([item.metrics.wilson for item in analysis.dimensions])
        lines.append(
            "Mean {0:.0%} interval width per dimension is {1:.4f}, against "
            "{2:.4f} for the same rows pooled into one estimate. Decomposition "
            "added no evidence, so it bought {3} narrower questions at the cost "
            "of making every answer {4:.1f} times less precise.".format(
                analysis.confidence, mean_width, pooled_width, k, mean_width / pooled_width
            )
        )
    lines.append("")

    indeterminate = counts[Verdict.INDETERMINATE]
    if indeterminate:
        share = indeterminate / k
        lines.append(
            "{0} of {1} {2} ({3:.0%}) carry too little data to support any "
            "verdict at this threshold.".format(
                indeterminate, k, _plural(k, "dimension", "dimensions"), share
            )
        )
        lines.append("")
    return lines


def _per_dimension_table(analysis: Analysis) -> List[str]:
    """Render the per dimension estimates table."""
    lines = [
        "## Per dimension",
        "",
        "| Dimension | n | passes | rate | Wilson {0:.0%} | Bootstrap {0:.0%} | Verdict |".format(
            analysis.confidence
        ),
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in analysis.dimensions:
        metrics = item.metrics
        lines.append(
            "| {0} | {1} | {2} | {3:.4f} | {4} | {5} | {6} |".format(
                metrics.dimension,
                metrics.n,
                metrics.passes,
                metrics.rate,
                metrics.wilson.format(),
                metrics.bootstrap.format(),
                item.verdict,
            )
        )
    lines.append("")

    flagged = [item for item in analysis.dimensions if item.metrics.disagreement]
    if flagged:
        lines.append("Interval disagreements worth reading before quoting a bound:")
        lines.append("")
        for item in flagged:
            lines.append("- `{0}`: {1}".format(item.dimension, item.metrics.disagreement))
        lines.append("")
    return lines


def _multiplicity(analysis: Analysis) -> List[str]:
    """Render the multiple comparisons section."""
    k = analysis.k
    lines = [
        "## Multiple comparisons",
        "",
        "{0} {1} tested at once against the same threshold. Benjamini-Hochberg "
        "at q = {2:.4g} controls the false discovery rate across the family.".format(
            k, _plural(k, "dimension is", "dimensions are"), analysis.fdr_q
        ),
        "",
        "| Dimension | Verdict | BH adjusted | p(pass) | BH p(pass) | p(fail) | BH p(fail) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in analysis.dimensions:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} |".format(
                item.dimension,
                item.verdict,
                item.adjusted_verdict,
                _format_p(item.p_pass),
                _format_p(item.adjusted_p_pass),
                _format_p(item.p_fail),
                _format_p(item.adjusted_p_fail),
            )
        )
    lines.append("")

    if not analysis.changes:
        lines.append("No dimension changes verdict under adjustment.")
    else:
        count = len(analysis.changes)
        lines.append(
            "{0} {1} verdict under adjustment:".format(
                count, _plural(count, "dimension changes", "dimensions change")
            )
        )
        lines.append("")
        for change in analysis.changes:
            lines.append(
                "- `{0}`: {1} becomes {2}".format(change.dimension, change.unadjusted, change.adjusted)
            )
    lines.append("")
    if k > 1:
        lines.append(
            "With {0} independent comparisons at a one sided error rate of "
            "{1:.4g}, the chance of at least one dimension failing by chance "
            "alone when every dimension truly meets the threshold is "
            "{2:.1%}.".format(k, analysis.fdr_q, 1.0 - (1.0 - analysis.fdr_q) ** k)
        )
        lines.append("")
    return lines


def _planning(analysis: Analysis) -> List[str]:
    """Render the required sample size section."""
    pending = [item for item in analysis.dimensions if item.requirement is not None]
    lines = ["## Required sample size", ""]
    if not pending:
        lines.append("No dimension is indeterminate, so no dimension needs more data.")
        lines.append("")
        return lines

    lines.append(
        "For each indeterminate dimension, the smallest n whose interval would "
        "clear the threshold if the observed rate held. This is the number to "
        "budget before running the eval, not after."
    )
    lines.append("")
    lines.append("| Dimension | current n | rate | required n | additional items | would resolve to |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    unresolvable = []
    for item in pending:
        requirement = item.requirement
        if requirement.required_n is None:
            unresolvable.append(item.dimension)
            lines.append(
                "| {0} | {1} | {2:.4f} | never | n/a | n/a |".format(
                    requirement.dimension, requirement.current_n, requirement.observed_rate
                )
            )
            continue
        lines.append(
            "| {0} | {1} | {2:.4f} | {3} | {4} | {5} |".format(
                requirement.dimension,
                requirement.current_n,
                requirement.observed_rate,
                requirement.required_n,
                requirement.additional_items,
                requirement.would_resolve_to,
            )
        )
    lines.append("")

    resolvable = [item.requirement for item in pending if item.requirement.required_n is not None]
    if resolvable:
        total_required = sum(item.required_n for item in resolvable)
        total_current = sum(item.current_n for item in resolvable)
        lines.append(
            "Resolving every indeterminate dimension takes {0} items in total, "
            "against the {1} currently spent on them: a {2:.1f}x increase.".format(
                total_required, total_current, total_required / total_current
            )
        )
        lines.append("")
    if unresolvable:
        lines.append(
            "Sitting exactly on the threshold never resolves at any sample "
            "size: {0}.".format(", ".join("`{0}`".format(name) for name in unresolvable))
        )
        lines.append("")
    return lines


def _aggregate(analysis: Analysis) -> List[str]:
    """Render the aggregate section and any disagreement with the parts."""
    aggregate = analysis.aggregate
    lines = [
        "## Aggregate",
        "",
        "| Scope | n | passes | rate | Wilson {0:.0%} | Bootstrap {0:.0%} | Verdict |".format(
            analysis.confidence
        ),
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
        "| all rows pooled | {0} | {1} | {2:.4f} | {3} | {4} | {5} |".format(
            aggregate.n,
            aggregate.passes,
            aggregate.rate,
            aggregate.wilson.format(),
            aggregate.bootstrap.format(),
            analysis.aggregate_verdict,
        ),
        "",
    ]
    if analysis.pooled_independence:
        lines.append(
            "Treat the pooled interval above as a lower bound on the "
            "uncertainty: {0}.".format(analysis.pooled_independence)
        )
        lines.append("")
    if aggregate.disagreement:
        lines.append("Interval disagreement on the pooled estimate: {0}.".format(aggregate.disagreement))
        lines.append("")

    disagreeing = [item for item in analysis.dimensions if item.verdict is not analysis.aggregate_verdict]
    if not disagreeing:
        lines.append("Every dimension agrees with the aggregate verdict.")
        lines.append("")
        return lines

    counts = verdict_counts([item.verdict for item in disagreeing])
    parts = ["{0} {1}".format(counts[verdict], verdict) for verdict in Verdict if counts[verdict]]
    lines.append(
        "The aggregate verdict is {0}, but {1} of {2} dimensions disagree with "
        "it ({3}). The pooled estimate borrows precision from dimensions that "
        "were never in doubt, so it can look decisive while most of the "
        "sub-components it summarizes are not.".format(
            analysis.aggregate_verdict, len(disagreeing), analysis.k, ", ".join(parts)
        )
    )
    lines.append("")
    if analysis.aggregate_verdict is Verdict.PASS and any(
        item.verdict is Verdict.FAIL for item in disagreeing
    ):
        failing = [item.dimension for item in disagreeing if item.verdict is Verdict.FAIL]
        lines.append(
            "A pooled PASS alongside a per dimension FAIL is the case worth "
            "escalating: {0}. Check the adjusted verdict before treating it as "
            "a real regression.".format(", ".join("`{0}`".format(name) for name in failing))
        )
        lines.append("")
    return lines


def render_markdown(analysis: Analysis) -> str:
    """Render a complete markdown report for an analysis.

    Args:
        analysis: The analysis to render.

    Returns:
        A markdown document as a single string.
    """
    lines = ["# evalpower report", ""]
    lines.extend(_headline(analysis))
    lines.extend(_per_dimension_table(analysis))
    lines.extend(_multiplicity(analysis))
    lines.extend(_planning(analysis))
    lines.extend(_aggregate(analysis))
    lines.append(
        "Wilson intervals and score test p values are two views of the same "
        "test, so an interval verdict and an unadjusted p value never "
        "disagree. Bootstrap intervals are reported alongside because they "
        "make no normal approximation, and because where they diverge from "
        "Wilson is itself informative."
    )
    return "\n".join(lines).rstrip() + "\n"
