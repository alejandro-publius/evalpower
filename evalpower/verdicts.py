"""Three way verdicts and the multiple comparisons correction.

A verdict is not "pass or fail". An interval that straddles the threshold
supports neither, and calling that a pass or a fail is the error this package
exists to catch. There are three outcomes, and the third one is common.

The tests here are score tests, which is deliberate: the Wilson interval in
:mod:`evalpower.metrics` is the inversion of the score test, so the interval
verdict and the p value verdict agree exactly at matching levels. A two sided
Wilson interval at confidence ``1 - alpha`` has its lower bound at or above
the threshold if and only if the one sided p value for PASS is at most
``alpha / 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Sequence, Tuple

from .metrics import DEFAULT_CONFIDENCE, Interval, normal_cdf, pass_rate

__all__ = [
    "DEFAULT_THRESHOLD",
    "Verdict",
    "benjamini_hochberg",
    "VerdictChange",
    "count_changed_verdicts",
    "default_fdr_q",
    "score_test_pvalues",
    "score_test_z",
    "verdict_from_interval",
    "verdict_from_pvalues",
    "verdict_counts",
]

DEFAULT_THRESHOLD = 0.85


class Verdict(str, Enum):
    """The three outcomes an eval can support."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"

    def __str__(self) -> str:
        return self.value


def default_fdr_q(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Return the false discovery rate level that matches an interval level.

    A two sided interval at ``confidence`` spends ``(1 - confidence) / 2`` on
    each tail, so that is the one sided error rate the correction should
    control. At the default 0.95 confidence this is 0.025.
    """
    return (1.0 - float(confidence)) / 2.0


def verdict_from_interval(interval: Interval, threshold: float = DEFAULT_THRESHOLD) -> Verdict:
    """Return the three way verdict implied by an interval and a threshold.

    PASS when the lower bound is at or above the threshold, FAIL when the
    upper bound is strictly below it, INDETERMINATE when the interval spans
    the threshold.

    Args:
        interval: An interval estimate for the pass rate.
        threshold: The bar the eval is being held to.

    Returns:
        The verdict the evidence supports.

    Raises:
        ValueError: if ``threshold`` lies outside [0, 1]. 0 and 1 themselves
            are allowed (and their behaviour is pinned in
            ``tests/test_oracle.py``: 0 always passes, since every lower
            bound already clears it, and 1 is indeterminate rather than a
            perpetual fail, since a saturated interval's upper bound can sit
            exactly at 1 too). A threshold outside [0, 1] is not a valid bar
            at all -- a stray percentage (85 instead of 0.85) or a sign
            error -- and left unchecked it is silently rewarded with a
            confident PASS or FAIL for every possible interval, which is the
            failure mode this function exists to prevent, not commit.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1], got {0!r}".format(threshold))
    if interval.lower >= threshold:
        return Verdict.PASS
    if interval.upper < threshold:
        return Verdict.FAIL
    return Verdict.INDETERMINATE


def score_test_z(successes: int, n: int, threshold: float = DEFAULT_THRESHOLD) -> float:
    """Return the score test statistic against a fixed threshold.

        z = (p - threshold) / sqrt(threshold * (1 - threshold) / n)

    The standard error uses the null rate, not the observed rate, which is
    what makes the test the exact inversion of the Wilson interval.

    Raises:
        ValueError: if ``threshold`` is not strictly between 0 and 1.
    """
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1, got {0!r}".format(threshold))
    p = pass_rate(successes, n)
    standard_error = math.sqrt(threshold * (1.0 - threshold) / n)
    return (p - threshold) / standard_error


def score_test_pvalues(
    successes: int, n: int, threshold: float = DEFAULT_THRESHOLD
) -> Tuple[float, float]:
    """Return the one sided p values for PASS and for FAIL.

    Both test the null hypothesis that the true rate equals ``threshold``.

    Args:
        successes: Number of passing items.
        n: Number of items evaluated.
        threshold: The bar the eval is being held to.

    Returns:
        A ``(p_pass, p_fail)`` pair. ``p_pass`` is the upper tail probability,
        small when the evidence says the rate is above the threshold.
        ``p_fail`` is the lower tail probability, small when the evidence says
        the rate is below it. The two sum to 1.
    """
    z = score_test_z(successes, n, threshold)
    p_fail = normal_cdf(z)
    p_pass = normal_cdf(-z)
    return p_pass, p_fail


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> Tuple[List[bool], List[float]]:
    """Apply the Benjamini-Hochberg step up procedure.

    Sort the ``k`` p values ascending. Find the largest rank ``i`` for which
    ``p_(i) <= (i / k) * q`` and reject every hypothesis up to that rank.
    The adjusted p values are the running minimum, taken from the largest
    rank downward, of ``(k / j) * p_(j)``.

    Args:
        pvalues: One p value per hypothesis, in any order.
        q: The false discovery rate to control.

    Returns:
        A ``(rejected, adjusted)`` pair, both in the input order. A hypothesis
        is rejected exactly when its adjusted p value is at most ``q``.

    Raises:
        ValueError: on an empty input, a p value outside [0, 1], or a ``q``
            outside (0, 1].
    """
    values = [float(value) for value in pvalues]
    if not values:
        raise ValueError("at least one p value is required")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p values must lie in [0, 1]")
    q = float(q)
    if not 0.0 < q <= 1.0:
        raise ValueError("q must lie in (0, 1], got {0!r}".format(q))

    k = len(values)
    order = sorted(range(k), key=lambda index: values[index])

    adjusted_sorted = [0.0] * k
    running_minimum = 1.0
    for rank in range(k, 0, -1):
        scaled = (k / rank) * values[order[rank - 1]]
        running_minimum = min(running_minimum, scaled, 1.0)
        adjusted_sorted[rank - 1] = running_minimum

    adjusted = [0.0] * k
    rejected = [False] * k
    for rank, index in enumerate(order, start=1):
        adjusted[index] = adjusted_sorted[rank - 1]
        rejected[index] = adjusted_sorted[rank - 1] <= q
    return rejected, adjusted


def verdict_from_pvalues(
    p_pass_rejected: bool, p_fail_rejected: bool, observed_rate: float, threshold: float
) -> Verdict:
    """Return the verdict implied by two one sided test decisions.

    Args:
        p_pass_rejected: Whether the PASS side test was rejected.
        p_fail_rejected: Whether the FAIL side test was rejected.
        observed_rate: The observed pass rate, used only to break the
            contradictory case where both sides reject.
        threshold: The bar the eval is being held to.

    Returns:
        PASS, FAIL, or INDETERMINATE.
    """
    if p_pass_rejected and p_fail_rejected:
        # Both tails cannot carry evidence at once. Defer to the side the
        # point estimate actually sits on rather than inventing a verdict.
        return Verdict.PASS if observed_rate >= threshold else Verdict.FAIL
    if p_pass_rejected:
        return Verdict.PASS
    if p_fail_rejected:
        return Verdict.FAIL
    return Verdict.INDETERMINATE


@dataclass(frozen=True)
class VerdictChange:
    """A dimension whose verdict moved once multiplicity was accounted for."""

    dimension: str
    unadjusted: Verdict
    adjusted: Verdict


def count_changed_verdicts(
    dimensions: Sequence[str],
    unadjusted: Sequence[Verdict],
    adjusted: Sequence[Verdict],
) -> List[VerdictChange]:
    """Return the dimensions whose verdict changes under FDR adjustment.

    Args:
        dimensions: Dimension names.
        unadjusted: Verdicts before correction, aligned with ``dimensions``.
        adjusted: Verdicts after correction, aligned with ``dimensions``.

    Returns:
        One :class:`VerdictChange` per dimension that moved.

    Raises:
        ValueError: if the three sequences are not the same length.
    """
    if not len(dimensions) == len(unadjusted) == len(adjusted):
        raise ValueError("dimensions, unadjusted and adjusted must be the same length")
    return [
        VerdictChange(dimension=name, unadjusted=before, adjusted=after)
        for name, before, after in zip(dimensions, unadjusted, adjusted)
        if before != after
    ]


def verdict_counts(verdicts: Sequence[Verdict]) -> Dict[Verdict, int]:
    """Return a count of each verdict, always including all three keys."""
    counts = {Verdict.PASS: 0, Verdict.FAIL: 0, Verdict.INDETERMINATE: 0}
    for verdict in verdicts:
        counts[verdict] += 1
    return counts
