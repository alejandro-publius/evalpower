"""Point and interval estimates for eval pass rates.

Every number an eval harness reports as a "score" is a proportion estimated
from a finite sample. This module supplies the two interval estimates that
`evalpower` reports side by side: the Wilson score interval (closed form,
well behaved at extreme rates) and the nonparametric bootstrap percentile
interval (assumption light, degenerate at extreme rates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional
from zlib import crc32

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_DISAGREEMENT_TOLERANCE",
    "DimensionMetrics",
    "Interval",
    "REQUIRED_COLUMNS",
    "bootstrap_interval",
    "interval_disagreement",
    "mean_interval_width",
    "normal_cdf",
    "normal_ppf",
    "pass_rate",
    "pooled_metrics",
    "summarize_dimensions",
    "validate_results",
    "wilson_interval",
    "z_for_confidence",
]

DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_DISAGREEMENT_TOLERANCE = 0.02

REQUIRED_COLUMNS = ("item_id", "dimension", "passed")

# Coefficients for Acklam's rational approximation to the inverse of the
# standard normal CDF. The approximation is accurate to about 1.15e-9; a
# single Halley refinement step in normal_ppf pushes it to machine precision.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425


def normal_cdf(z: float) -> float:
    """Return the standard normal CDF at ``z``.

    Computed in closed form from the error function:
    ``Phi(z) = 0.5 * erfc(-z / sqrt(2))``.
    """
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    """Return the standard normal quantile (inverse CDF) at probability ``p``.

    Uses Acklam's rational approximation followed by one Halley refinement
    against :func:`normal_cdf`, which is accurate to machine precision.

    Raises:
        ValueError: if ``p`` is not strictly between 0 and 1.
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1, got {0!r}".format(p))

    if p < _ACKLAM_P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (
            ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= 1.0 - _ACKLAM_P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            ((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r
            + _ACKLAM_A[5]
        ) * q / (
            ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r
            + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(
            ((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q
            + _ACKLAM_C[5]
        ) / ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)

    # Halley refinement. e is the residual of the approximation, u the Newton
    # step; the denominator applies the second order correction.
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def z_for_confidence(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Return the two sided critical value ``z`` for a confidence level.

    For ``confidence = 0.95`` this is the 0.975 quantile of the standard
    normal, about 1.959963984540054.

    Raises:
        ValueError: if ``confidence`` is not strictly between 0 and 1.
    """
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1, got {0!r}".format(confidence))
    return normal_ppf(1.0 - (1.0 - confidence) / 2.0)


@dataclass(frozen=True)
class Interval:
    """A closed interval estimate for a proportion."""

    lower: float
    upper: float

    @property
    def width(self) -> float:
        """Return the width of the interval."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Return True when ``value`` lies inside the closed interval."""
        return self.lower <= value <= self.upper

    def format(self, digits: int = 4) -> str:
        """Return the interval as a human readable string."""
        return "[{0:.{2}f}, {1:.{2}f}]".format(self.lower, self.upper, digits)


def pass_rate(successes: int, n: int) -> float:
    """Return the observed pass rate ``successes / n``.

    Raises:
        ValueError: if ``n`` is not positive or ``successes`` is out of range.
    """
    successes, n = int(successes), int(n)
    if n <= 0:
        raise ValueError("n must be positive, got {0!r}".format(n))
    if not 0 <= successes <= n:
        raise ValueError("successes must lie in [0, n], got {0!r}".format(successes))
    return successes / n


def wilson_interval(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
    """Return the Wilson score interval for a binomial proportion.

    With ``p = successes / n`` and ``z`` the two sided critical value:

        center    = (p + z^2 / (2n)) / (1 + z^2 / n)
        halfwidth = z / (1 + z^2 / n)
                    * sqrt(p(1 - p) / n + z^2 / (4 n^2))

    The Wilson interval is the inversion of the score test, which is what
    makes it interchangeable with the p values in :mod:`evalpower.verdicts`
    and with the closed form in :mod:`evalpower.planning`. It stays inside
    [0, 1] and does not collapse to a point when ``p`` is 0 or 1.

    Args:
        successes: Number of passing items.
        n: Number of items evaluated.
        confidence: Two sided confidence level, default 0.95.

    Returns:
        The Wilson score interval, clipped to [0, 1].
    """
    p = pass_rate(successes, n)
    z = z_for_confidence(confidence)
    z_sq = z * z
    denominator = 1.0 + z_sq / n
    center = (p + z_sq / (2.0 * n)) / denominator
    halfwidth = (z / denominator) * math.sqrt(p * (1.0 - p) / n + z_sq / (4.0 * n * n))
    return Interval(max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def bootstrap_interval(
    successes: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Interval:
    """Return a nonparametric bootstrap percentile interval for a pass rate.

    Resampling ``n`` items with replacement from a vector of ``n`` zeros and
    ones and taking the mean is, exactly, drawing ``Binomial(n, successes / n)
    / n``. We sample that distribution directly, which is the same estimator
    as an explicit item level resample and is far cheaper for large evals.

    The percentile interval is assumption light but inherits a known failure
    mode: when every item passes or every item fails, every resample is
    identical and the interval collapses to a single point. That collapse is
    a property of the method, not evidence of certainty, which is why
    :func:`interval_disagreement` exists.

    Args:
        successes: Number of passing items.
        n: Number of items evaluated.
        confidence: Two sided confidence level, default 0.95.
        resamples: Number of bootstrap resamples.
        seed: Seed for the resampling generator, so results are reproducible.

    Returns:
        The percentile interval at the requested confidence level.
    """
    p = pass_rate(successes, n)
    if resamples <= 0:
        raise ValueError("resamples must be positive, got {0!r}".format(resamples))
    alpha = 1.0 - float(confidence)
    rng = np.random.default_rng(int(seed))
    draws = rng.binomial(n, p, size=int(resamples)) / n
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(float(lower), float(upper))


def interval_disagreement(
    wilson: Interval,
    bootstrap: Interval,
    tolerance: float = DEFAULT_DISAGREEMENT_TOLERANCE,
) -> Optional[str]:
    """Return a note when two intervals for the same quantity differ materially.

    Args:
        wilson: The Wilson score interval.
        bootstrap: The bootstrap percentile interval.
        tolerance: Largest bound difference treated as immaterial.

    Returns:
        A short explanation when either bound differs by more than
        ``tolerance``, otherwise None.
    """
    lower_gap = abs(wilson.lower - bootstrap.lower)
    upper_gap = abs(wilson.upper - bootstrap.upper)
    if max(lower_gap, upper_gap) <= tolerance:
        return None
    if bootstrap.width == 0.0:
        return (
            "bootstrap collapsed to a point at a saturated pass rate; "
            "read the Wilson bound {0} instead".format(wilson.format())
        )
    return "bounds differ by {0:.3f} (lower) and {1:.3f} (upper)".format(lower_gap, upper_gap)


@dataclass(frozen=True)
class DimensionMetrics:
    """Estimates for one dimension, or for a pooled set of dimensions."""

    dimension: str
    n: int
    passes: int
    rate: float
    wilson: Interval
    bootstrap: Interval
    disagreement: Optional[str] = None


def validate_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a results frame.

    Requires the columns ``item_id``, ``dimension`` and ``passed``, with
    ``passed`` in {0, 1}. A single dimension is valid and is the degenerate
    case the package is built to compare against.

    Args:
        frame: Raw results, one row per (item, dimension) pair.

    Returns:
        A copy with ``dimension`` as string and ``passed`` as int.

    Raises:
        ValueError: on missing columns, an empty frame, values outside
            {0, 1}, or a repeated (item_id, dimension) pair.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("results are missing required column(s): {0}".format(", ".join(missing)))
    if len(frame) == 0:
        raise ValueError("results are empty")

    normalized = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    normalized["dimension"] = normalized["dimension"].astype(str)

    passed = pd.to_numeric(normalized["passed"], errors="coerce")
    if passed.isna().any():
        raise ValueError("column 'passed' contains values that are not numeric")
    if not passed.isin([0, 1]).all():
        offenders = sorted(set(passed[~passed.isin([0, 1])].tolist()))
        raise ValueError("column 'passed' must be 0 or 1, found {0}".format(offenders))
    normalized["passed"] = passed.astype(int)

    duplicated = normalized.duplicated(subset=["item_id", "dimension"])
    if duplicated.any():
        example = normalized.loc[duplicated, ["item_id", "dimension"]].iloc[0]
        raise ValueError(
            "repeated (item_id, dimension) pair, for example ({0!r}, {1!r}); "
            "each item should be scored once per dimension".format(example["item_id"], example["dimension"])
        )
    return normalized


def _metrics_for(
    label: str,
    successes: int,
    n: int,
    confidence: float,
    resamples: int,
    seed: int,
    tolerance: float,
) -> DimensionMetrics:
    """Build a :class:`DimensionMetrics` for one labelled group of items."""
    wilson = wilson_interval(successes, n, confidence)
    # Offset the seed by a checksum of the label so each dimension gets its
    # own resampling stream regardless of the order dimensions appear in.
    bootstrap = bootstrap_interval(
        successes, n, confidence, resamples, seed=(int(seed) + crc32(label.encode("utf-8"))) % (2**32)
    )
    return DimensionMetrics(
        dimension=label,
        n=n,
        passes=successes,
        rate=pass_rate(successes, n),
        wilson=wilson,
        bootstrap=bootstrap,
        disagreement=interval_disagreement(wilson, bootstrap, tolerance),
    )


def summarize_dimensions(
    frame: pd.DataFrame,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    tolerance: float = DEFAULT_DISAGREEMENT_TOLERANCE,
) -> List[DimensionMetrics]:
    """Return per dimension estimates, ordered by dimension name.

    Args:
        frame: A validated results frame.
        confidence: Two sided confidence level for both intervals.
        resamples: Bootstrap resamples per dimension.
        seed: Base seed for the bootstrap.
        tolerance: Bound gap above which the two intervals are flagged.

    Returns:
        One :class:`DimensionMetrics` per dimension.
    """
    grouped = frame.groupby("dimension", sort=True)["passed"].agg(["sum", "count"])
    return [
        _metrics_for(str(label), int(row["sum"]), int(row["count"]), confidence, resamples, seed, tolerance)
        for label, row in grouped.iterrows()
    ]


def pooled_metrics(
    frame: pd.DataFrame,
    label: str = "ALL",
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    tolerance: float = DEFAULT_DISAGREEMENT_TOLERANCE,
) -> DimensionMetrics:
    """Return estimates for every row pooled into a single proportion.

    Args:
        frame: A validated results frame.
        label: Name to record on the returned metrics.
        confidence: Two sided confidence level for both intervals.
        resamples: Bootstrap resamples.
        seed: Base seed for the bootstrap.
        tolerance: Bound gap above which the two intervals are flagged.

    Returns:
        Pooled :class:`DimensionMetrics` across all dimensions.
    """
    return _metrics_for(
        label, int(frame["passed"].sum()), int(len(frame)), confidence, resamples, seed, tolerance
    )


def mean_interval_width(intervals: Iterable[Interval]) -> float:
    """Return the mean width of a collection of intervals.

    This is the quantity that grows as a risk is decomposed into more
    sub-components on a fixed evidence budget: the same evidence spread over
    more estimates makes every one of them wider.

    Args:
        intervals: The intervals to average over.

    Returns:
        The mean width.

    Raises:
        ValueError: if no intervals are supplied.
    """
    widths = [interval.width for interval in intervals]
    if not widths:
        raise ValueError("no intervals supplied")
    return float(np.mean(widths))
