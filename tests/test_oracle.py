"""Cross-checks against statsmodels and scipy, two implementations evalpower
does not import anywhere in its own source.

test_metrics.py, test_verdicts.py and test_planning.py pin the arithmetic
against hand computed literals, worked out from the same formulas the code
implements. That is a strong regression guard, but it cannot catch a mistake
shared between the code and the hand computation, such as a wrong constant
copied into both. This file recomputes the same quantities with an
independent library instead, so a bug here would have to be a bug in
statsmodels or scipy too.

Every case below is chosen to hit an edge the brief calls out explicitly:
zero observed failures (a saturated pass rate), zero observed passes, a
single item, and a rate sitting exactly on the threshold.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from evalpower.metrics import wilson_interval
from evalpower.planning import required_n
from evalpower.verdicts import Verdict, score_test_pvalues, verdict_from_interval

# (successes, n) pairs covering an ordinary rate, a saturated rate (zero
# observed failures), a totally failing rate (zero observed passes), a
# single item both ways, and a rate exactly on the 0.85 threshold used
# elsewhere in this suite.
WILSON_CASES = [
    (8, 10),
    (529, 600),
    (37, 50),
    (0, 20),  # zero observed passes
    (20, 20),  # zero observed failures / a saturated 100% pass rate
    (50, 50),
    (1, 1),  # one item, passing
    (0, 1),  # one item, failing
    (17, 20),  # rate exactly 0.85
]


@pytest.mark.parametrize("successes,n", WILSON_CASES)
def test_wilson_interval_matches_statsmodels(successes: int, n: int) -> None:
    ours = wilson_interval(successes, n, confidence=0.95)
    lower, upper = proportion_confint(successes, n, alpha=0.05, method="wilson")
    assert ours.lower == pytest.approx(float(lower), abs=1e-9)
    assert ours.upper == pytest.approx(float(upper), abs=1e-9)


@pytest.mark.parametrize("confidence,alpha", [(0.95, 0.05), (0.90, 0.10), (0.99, 0.01), (0.80, 0.20)])
def test_wilson_interval_matches_statsmodels_at_other_confidence_levels(
    confidence: float, alpha: float
) -> None:
    ours = wilson_interval(43, 50, confidence=confidence)
    lower, upper = proportion_confint(43, 50, alpha=alpha, method="wilson")
    assert ours.lower == pytest.approx(float(lower), abs=1e-9)
    assert ours.upper == pytest.approx(float(upper), abs=1e-9)


# (successes, n, threshold): an ordinary case on each side of the bar, the
# committed example's pooled rate, a saturated rate, a single item, and a
# rate exactly on the threshold, where the score test statistic is exactly
# zero and both one sided p values are exactly 0.5.
SCORE_TEST_CASES = [
    (40, 50, 0.85),
    (48, 50, 0.85),
    (529, 600, 0.85),
    (20, 20, 0.85),  # zero observed failures
    (0, 20, 0.5),  # zero observed passes
    (1, 1, 0.5),  # one item
    (17, 20, 0.85),  # exactly at the bar
]


@pytest.mark.parametrize("successes,n,threshold", SCORE_TEST_CASES)
def test_score_test_pvalues_match_statsmodels(successes: int, n: int, threshold: float) -> None:
    p_pass, p_fail = score_test_pvalues(successes, n, threshold)
    _, sm_p_pass = proportions_ztest(successes, n, value=threshold, prop_var=threshold, alternative="larger")
    _, sm_p_fail = proportions_ztest(successes, n, value=threshold, prop_var=threshold, alternative="smaller")
    assert p_pass == pytest.approx(float(sm_p_pass), abs=1e-9)
    assert p_fail == pytest.approx(float(sm_p_fail), abs=1e-9)


def test_score_test_at_the_threshold_is_a_coin_flip_both_ways() -> None:
    # 17 of 20 is exactly 0.85: the score statistic is zero, so both one
    # sided p values are exactly 0.5 under both evalpower and statsmodels.
    p_pass, p_fail = score_test_pvalues(17, 20, 0.85)
    assert p_pass == pytest.approx(0.5, abs=1e-12)
    assert p_fail == pytest.approx(0.5, abs=1e-12)
    _, sm_p_pass = proportions_ztest(17, 20, value=0.85, prop_var=0.85, alternative="larger")
    _, sm_p_fail = proportions_ztest(17, 20, value=0.85, prop_var=0.85, alternative="smaller")
    assert float(sm_p_pass) == pytest.approx(0.5, abs=1e-12)
    assert float(sm_p_fail) == pytest.approx(0.5, abs=1e-12)


# (observed_rate, threshold, confidence) covering an ordinary margin, a
# saturated 100% observed rate (zero observed failures), and a totally
# failing 0% observed rate.
REQUIRED_N_CASES = [
    (0.90, 0.85, 0.95),
    (0.80, 0.85, 0.95),
    (0.875, 0.85, 0.95),
    (1.0, 0.85, 0.95),  # zero observed failures
    (0.0, 0.85, 0.95),  # zero observed passes
    (0.99, 0.85, 0.90),
]


@pytest.mark.parametrize("rate,threshold,confidence", REQUIRED_N_CASES)
def test_required_n_matches_an_independently_computed_closed_form(
    rate: float, threshold: float, confidence: float
) -> None:
    # scipy.stats.norm.ppf is a different implementation of the normal
    # quantile than evalpower.metrics.normal_ppf's Acklam approximation, so
    # this checks the closed form itself, not evalpower's inverse-CDF code.
    margin = rate - threshold
    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    exact = (z * z) * threshold * (1.0 - threshold) / (margin * margin)
    expected = max(1, math.ceil(exact))
    # planning.required_n nudges the FAIL side up by one when an integer n
    # lands exactly on the boundary; mirror that here so the two agree.
    if margin < 0 and float(expected) == exact:
        expected += 1
    assert required_n(rate, threshold, confidence) == expected


def test_required_n_boundary_matches_statsmodels_wilson_interval() -> None:
    # The real world consequence, checked with statsmodels' own Wilson
    # interval rather than evalpower's: 45 of 50 (90%) is indeterminate
    # against a 0.85 bar, but scaled up to the required n it clears.
    needed = required_n(0.90, 0.85, 0.95)
    assert needed == 196
    scaled_successes = round(0.90 * needed)  # 176.4 -> 176, rate slightly under 0.90
    # Use the first achievable count at or above the observed rate, as
    # test_planning.py's equivalent case does.
    if scaled_successes / needed < 0.90:
        scaled_successes += 1
    lower, _ = proportion_confint(scaled_successes, needed, alpha=0.05, method="wilson")
    assert lower >= 0.85


def test_required_n_on_the_fail_side_matches_statsmodels_wilson_interval() -> None:
    # 40 of 50 (80%) is five points below the bar. The required n resolves
    # it to FAIL: the upper bound must sit strictly below the threshold.
    needed = required_n(0.80, 0.85, 0.95)
    scaled_successes = int(0.80 * needed)  # exact: 0.80 * 196 = 156.8 -> 156
    _, upper = proportion_confint(scaled_successes, needed, alpha=0.05, method="wilson")
    assert upper < 0.85


def test_verdict_from_interval_pinned_behaviour_at_degenerate_thresholds() -> None:
    # verdict_from_interval takes threshold as a plain float and does not
    # itself validate it (score_test_z and required_n do; this function does
    # not need to, since a bound comparison is well defined for any float).
    # Called directly with a threshold of 0 or 1, its behaviour is pinned
    # here rather than left to chance: a threshold of 0 always passes,
    # because every Wilson lower bound is already at or above 0.
    interval = wilson_interval(1, 20, confidence=0.95)
    assert verdict_from_interval(interval, 0.0) is Verdict.PASS
    # A threshold of 1 never passes for a finite sample: the Wilson lower
    # bound cannot reach exactly 1.0 unless every item passed with n large
    # enough that rounding makes it so, and even then the upper bound
    # rarely sits below 1 either, so this is indeterminate in practice.
    saturated = wilson_interval(50, 50, confidence=0.95)
    assert verdict_from_interval(saturated, 1.0) is Verdict.INDETERMINATE
