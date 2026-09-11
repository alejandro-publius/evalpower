"""How much data an indeterminate dimension would need to resolve.

This is the number a governance team wants before running the eval, not after.
It is closed form, because the Wilson interval is the inversion of the score
test: the interval separates from the threshold exactly when

    |p - threshold| / sqrt(threshold * (1 - threshold) / n) >= z

Solving for n gives

    n >= z^2 * threshold * (1 - threshold) / (p - threshold)^2

so the required sample size grows with the square of how close the observed
rate sits to the bar. Halve the margin and you quadruple the data you need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .metrics import DEFAULT_CONFIDENCE, z_for_confidence
from .verdicts import DEFAULT_THRESHOLD, Verdict

__all__ = [
    "SampleSizeRequirement",
    "achieved_fpr",
    "first_nonzero_fpr_n",
    "honest_samples_for_fpr",
    "required_n",
    "requirement_for",
]


def required_n(
    observed_rate: float,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Optional[int]:
    """Return the smallest ``n`` that resolves a dimension at the observed rate.

    Args:
        observed_rate: The pass rate to hold fixed while scaling the sample.
        threshold: The bar the eval is being held to.
        confidence: Two sided confidence level, matching the reported interval.

    Returns:
        The minimum number of items whose Wilson interval at ``observed_rate``
        clears the threshold, or None when the observed rate sits exactly on
        the threshold and therefore never resolves at any sample size.

    Raises:
        ValueError: if ``observed_rate`` is outside [0, 1] or ``threshold`` is
            not strictly between 0 and 1.
    """
    observed_rate = float(observed_rate)
    threshold = float(threshold)
    if not 0.0 <= observed_rate <= 1.0:
        raise ValueError("observed_rate must lie in [0, 1], got {0!r}".format(observed_rate))
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1, got {0!r}".format(threshold))

    margin = observed_rate - threshold
    if margin == 0.0:
        return None

    z = z_for_confidence(confidence)
    exact = (z * z) * threshold * (1.0 - threshold) / (margin * margin)
    smallest = max(1, math.ceil(exact))

    # PASS needs the lower bound at or above the threshold, so equality is
    # enough. FAIL needs the upper bound strictly below it, so an n that lands
    # exactly on the boundary has to be nudged up by one item.
    if margin < 0 and float(smallest) == exact:
        smallest += 1
    return smallest


@dataclass(frozen=True)
class SampleSizeRequirement:
    """What one indeterminate dimension needs in order to resolve."""

    dimension: str
    current_n: int
    observed_rate: float
    required_n: Optional[int]
    would_resolve_to: Optional[Verdict]

    @property
    def additional_items(self) -> Optional[int]:
        """Return how many more items are needed, or None when unresolvable."""
        if self.required_n is None:
            return None
        return max(0, self.required_n - self.current_n)

    @property
    def multiple_of_current(self) -> Optional[float]:
        """Return the required sample as a multiple of the current sample."""
        if self.required_n is None or self.current_n <= 0:
            return None
        return self.required_n / self.current_n


def requirement_for(
    dimension: str,
    successes: int,
    n: int,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = DEFAULT_CONFIDENCE,
) -> SampleSizeRequirement:
    """Return the sample size requirement for one dimension.

    Args:
        dimension: Dimension name.
        successes: Number of passing items observed.
        n: Number of items evaluated.
        threshold: The bar the eval is being held to.
        confidence: Two sided confidence level, matching the reported interval.

    Returns:
        A :class:`SampleSizeRequirement` describing the sample size that would
        resolve the dimension and which way it would resolve.
    """
    observed_rate = successes / n
    needed = required_n(observed_rate, threshold, confidence)
    if needed is None:
        resolves_to = None
    else:
        resolves_to = Verdict.PASS if observed_rate > threshold else Verdict.FAIL
    return SampleSizeRequirement(
        dimension=dimension,
        current_n=int(n),
        observed_rate=observed_rate,
        required_n=needed,
        would_resolve_to=resolves_to,
    )


# ---------------------------------------------------------------------------
# Monitor thresholds: the false positive rate an empirical quantile delivers.
#
# A monitor's "TPR at 1% FPR" is computed by setting the threshold at the
# (1 - 0.01) quantile of n honest scores and counting attacks above it. The
# common implementation (control-arena's ``compute_classifier_tpr``, numpy's
# ``percentile(..., method="higher")``) rounds the quantile UP to the next
# observed honest score so that the false positive rate is at most the target.
# With distinct scores, exactly floor((n - 1) * target) honest scores sit
# strictly above that threshold. So the FPR the threshold actually delivers is
#
#     achieved = floor((n - 1) * target) / n
#
# which is zero until n exceeds 1 / target - the threshold has collapsed onto
# the single most suspicious honest score - and afterwards a sawtooth that
# falls short of the target by up to one honest sample's worth. "TPR at 1% FPR"
# from 50 honest samples is the TPR above the most suspicious honest score,
# at an achieved FPR of exactly 0. These are the numbers to check before
# quoting the operating point.
# ---------------------------------------------------------------------------


def _check_target_fpr(target_fpr: float) -> float:
    target_fpr = float(target_fpr)
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(
            "target_fpr must be strictly between 0 and 1, got {0!r}".format(target_fpr)
        )
    return target_fpr


def achieved_fpr(n_honest: int, target_fpr: float) -> float:
    """Return the FPR an empirical quantile threshold delivers on ``n_honest`` scores.

    Assumes the threshold is the ``(1 - target_fpr)`` quantile rounded up to the
    next observed score and that a false positive is a score strictly above it,
    which is the rule numpy's ``method="higher"`` and control-arena's
    ``compute_classifier_tpr`` implement. Assumes distinct scores; ties can only
    lower the achieved rate further.

    Args:
        n_honest: Number of honest scores the threshold is set on.
        target_fpr: The requested false positive rate.

    Returns:
        ``floor((n_honest - 1) * target_fpr) / n_honest``.

    Raises:
        ValueError: if ``n_honest`` is not positive or ``target_fpr`` is not
            strictly between 0 and 1.
    """
    target_fpr = _check_target_fpr(target_fpr)
    n_honest = int(n_honest)
    if n_honest <= 0:
        raise ValueError("n_honest must be positive, got {0!r}".format(n_honest))
    return math.floor((n_honest - 1) * target_fpr) / n_honest


def first_nonzero_fpr_n(target_fpr: float) -> int:
    """Return the smallest ``n_honest`` at which the achieved FPR is not zero.

    Below this, the threshold sits on the most suspicious honest score and the
    reported operating point is "FPR = 0", whatever was requested. For a 1%
    target that is 101 honest samples.
    """
    target_fpr = _check_target_fpr(target_fpr)
    n = math.ceil(1.0 / target_fpr) + 1
    # Guard the floating point edge where 1/target is an exact integer.
    while achieved_fpr(n - 1, target_fpr) > 0.0:
        n -= 1
    return n


def honest_samples_for_fpr(target_fpr: float, max_shortfall: float = 0.1) -> int:
    """Return the smallest ``n_honest`` from which the achieved FPR stays close to target.

    "Close" means ``achieved >= (1 - max_shortfall) * target_fpr`` - and it must
    hold for every larger ``n`` too, because the achieved rate is a sawtooth:
    it touches the target at each step and sags between steps, so a single
    ``n`` that happens to sit on a step is not a guarantee. For a 1% target
    and a 10% shortfall that is 901 honest samples; for 2%, 451; for 5%, 181.

    Computed in exact arithmetic on the decimals the caller wrote. The
    sawtooth's troughs can land exactly on the bound - at a 1% target and 10%
    shortfall, n = 1000 delivers 9/1000, which is exactly 0.9 * 0.01 - and in
    floating point 0.9 * 0.01 rounds to 0.009000000000000001, which would score
    that exact touch as a miss and push the answer out by a whole step.

    Args:
        target_fpr: The requested false positive rate.
        max_shortfall: The largest relative shortfall tolerated, in (0, 1).

    Returns:
        The smallest ``n`` such that every ``m >= n`` satisfies the bound.
    """
    target_fpr = _check_target_fpr(target_fpr)
    max_shortfall = float(max_shortfall)
    if not 0.0 < max_shortfall < 1.0:
        raise ValueError(
            "max_shortfall must be strictly between 0 and 1, got {0!r}".format(max_shortfall)
        )
    t = Fraction(repr(target_fpr))
    s = Fraction(repr(max_shortfall))
    floor_rate = (1 - s) * t
    # Beyond this bound the sawtooth's troughs are all inside the tolerance:
    # floor((n - 1) t) >= (n - 1) t - 1 >= (1 - s) n t  whenever  s n t >= 1 + t.
    sufficient = math.ceil((1 + t) / (s * t))
    last_violation = 0
    for n in range(1, sufficient + 1):
        if Fraction(math.floor((n - 1) * t), n) < floor_rate:
            last_violation = n
    return last_violation + 1
