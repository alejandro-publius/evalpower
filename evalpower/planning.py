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
from typing import Optional

from .metrics import DEFAULT_CONFIDENCE, z_for_confidence
from .verdicts import DEFAULT_THRESHOLD, Verdict

__all__ = ["SampleSizeRequirement", "required_n", "requirement_for"]


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
