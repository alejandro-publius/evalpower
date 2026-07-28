"""Assemble the analysis and render it as markdown.

The report answers one question: does this eval have enough data to support
the verdict it just printed?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .metrics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_DISAGREEMENT_TOLERANCE,
    DimensionMetrics,
    mean_interval_width,
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

__all__ = ["Analysis", "DimensionAnalysis", "analyze", "render_markdown"]


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
