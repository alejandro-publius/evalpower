"""Tests for evalpower.planning.

The required sample size is closed form, so every expectation below is the
result of evaluating that formula by hand and writing the number down.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evalpower.metrics import wilson_interval
from evalpower.planning import (
    achieved_fpr,
    first_nonzero_fpr_n,
    honest_samples_for_fpr,
    required_n,
    requirement_for,
)
from evalpower.verdicts import Verdict, verdict_from_interval

# z^2 for a 95 percent two sided interval:
# 1.959963984540054 ** 2 = 3.8414588206941254
Z_SQUARED_95 = 3.8414588206941254


def test_required_n_five_points_above_threshold() -> None:
    # n = z^2 * threshold * (1 - threshold) / (p - threshold)^2
    #   = 3.8414588206941254 * 0.85 * 0.15 / (0.90 - 0.85)^2
    #   = 3.8414588206941254 * 0.1275 / 0.0025
    #   = 0.489785999638501 / 0.0025
    #   = 195.9143998554004
    # The smallest integer at or above that is 196.
    assert required_n(0.90, 0.85, 0.95) == 196


def test_required_n_is_symmetric_in_the_margin() -> None:
    # The formula depends on the margin only through its square, so five
    # points below the threshold needs the same 196 items as five points
    # above. Only the direction of the eventual verdict differs.
    assert required_n(0.80, 0.85, 0.95) == 196


def test_required_n_two_points_above_threshold() -> None:
    #   = 3.8414588206941254 * 0.1275 / (0.87 - 0.85)^2
    #   = 0.489785999638501 / 0.0004
    #   = 1224.4649990962524
    # The smallest integer at or above that is 1225.
    assert required_n(0.87, 0.85, 0.95) == 1225


def test_required_n_grows_with_the_square_of_the_margin() -> None:
    # Halving the margin from 0.05 to 0.025 quadruples the requirement:
    #   0.489785999638501 / 0.000625 = 783.6575994216015
    # which rounds up to 784, exactly four times the 196 needed at 0.05.
    assert required_n(0.875, 0.85, 0.95) == 784
    assert required_n(0.875, 0.85, 0.95) == 4 * required_n(0.90, 0.85, 0.95)


def test_required_n_at_ninety_percent_confidence() -> None:
    # A weaker confidence level needs less data. With z = 1.6448536269514722,
    # z^2 = 2.705543454095413, so
    #   2.705543454095413 * 0.1275 / 0.0025 = 137.98271615886583
    # which rounds up to 138.
    assert required_n(0.90, 0.85, 0.90) == 138


def test_required_n_is_none_exactly_on_the_threshold() -> None:
    # An observed rate sitting exactly on the bar never separates from it,
    # at any sample size, because the margin in the denominator is zero.
    assert required_n(0.85, 0.85, 0.95) is None


def test_required_n_at_a_saturated_rate() -> None:
    # A perfect observed rate has margin 1 - 0.85 = 0.15, so
    #   0.489785999638501 / 0.0225 = 21.76826665060004
    # which rounds up to 22.
    assert required_n(1.0, 0.85, 0.95) == 22
    # And a rate of zero has margin -0.85, so
    #   0.489785999638501 / 0.7225 = 0.6779044977695518
    # which rounds up to 1, floored at one item.
    assert required_n(0.0, 0.85, 0.95) == 1


def test_required_n_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        required_n(1.5, 0.85, 0.95)
    with pytest.raises(ValueError):
        required_n(-0.1, 0.85, 0.95)
    with pytest.raises(ValueError):
        required_n(0.9, 0.0, 0.95)
    with pytest.raises(ValueError):
        required_n(0.9, 1.0, 0.95)


def _score_statistic(rate: float, n: int, threshold: float) -> float:
    """Score statistic written out directly, for cross checking planning.

    Deliberately independent of the package so this file does not verify
    evalpower against itself.
    """
    return (rate - threshold) / math.sqrt(threshold * (1.0 - threshold) / n)


def test_required_n_is_the_smallest_n_that_separates() -> None:
    # A consistency check rather than a restatement: at the required n the
    # score statistic must reach the critical value, and one item short of it
    # must not. test_verdicts.py separately proves that reaching the critical
    # value is exactly the same event as the Wilson interval clearing the
    # threshold, so the chain from planning to reported verdict is closed.
    threshold, critical = 0.85, 1.959963984540054
    for rate in (0.90, 0.88, 0.80, 0.75, 0.99):
        needed = required_n(rate, threshold, 0.95)
        assert needed is not None
        assert abs(_score_statistic(rate, needed, threshold)) >= critical
        assert abs(_score_statistic(rate, needed - 1, threshold)) < critical


def test_required_n_lands_a_real_count_on_the_right_side() -> None:
    # The same claim at the level a governance team actually experiences it:
    # run 196 items, observe the same 90 percent rate, and the interval that
    # was indeterminate at 50 items now reads PASS.
    assert verdict_from_interval(wilson_interval(45, 50, 0.95), 0.85) is Verdict.INDETERMINATE
    assert required_n(0.90, 0.85, 0.95) == 196
    # 0.90 * 196 = 176.4, so 177 of 196 is the first achievable count at or
    # above the observed rate.
    assert verdict_from_interval(wilson_interval(177, 196, 0.95), 0.85) is Verdict.PASS


def test_requirement_for_reports_the_resolution_direction() -> None:
    # 45 of 50 is a rate of 0.90, which needs 196 items, so 146 more.
    requirement = requirement_for("refusal_consistency", 45, 50, 0.85, 0.95)
    assert requirement.dimension == "refusal_consistency"
    assert requirement.current_n == 50
    assert requirement.observed_rate == 0.90
    assert requirement.required_n == 196
    assert requirement.additional_items == 146
    assert requirement.would_resolve_to is Verdict.PASS
    # 196 / 50 = 3.92
    assert requirement.multiple_of_current == pytest.approx(3.92, abs=1e-12)


def test_requirement_for_below_the_threshold_resolves_to_fail() -> None:
    # 40 of 50 is a rate of 0.80, five points below the bar, so the same 196
    # items, but the verdict it would resolve to is FAIL.
    requirement = requirement_for("tool_misuse", 40, 50, 0.85, 0.95)
    assert requirement.required_n == 196
    assert requirement.would_resolve_to is Verdict.FAIL


def test_requirement_for_on_the_threshold_never_resolves() -> None:
    # 17 of 20 is exactly 0.85.
    requirement = requirement_for("boundary", 17, 20, 0.85, 0.95)
    assert requirement.observed_rate == 0.85
    assert requirement.required_n is None
    assert requirement.would_resolve_to is None
    assert requirement.additional_items is None
    assert requirement.multiple_of_current is None


def test_splitting_evidence_multiplies_the_data_needed() -> None:
    # The planning consequence of decomposition. One dimension observed at
    # 0.90 needs 196 items. Twelve sub-dimensions each observed at 0.90 need
    # 196 items each, so 2352 in total, from a budget that was never
    # increased. Decomposition does not divide the requirement, it multiplies
    # it.
    single = required_n(0.90, 0.85, 0.95)
    assert single == 196
    assert 12 * single == 2352


# ---------------------------------------------------------------------------
# Monitor threshold FPR: achieved_fpr, first_nonzero_fpr_n, honest_samples_for_fpr
# ---------------------------------------------------------------------------


def _rule(honest: np.ndarray, target: float) -> float:
    # The rule the closed form describes: quantile rounded up to the next
    # observed score, false positive = strictly above it. This is what
    # control-arena's compute_classifier_tpr does.
    threshold = np.percentile(honest, (1.0 - target) * 100.0, method="higher")
    return float((honest > threshold).mean())


def test_achieved_fpr_equals_the_rule_exactly_on_distinct_scores() -> None:
    rng = np.random.default_rng(0)
    for target in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
        for n in range(1, 400):
            honest = rng.normal(size=n)
            assert achieved_fpr(n, target) == _rule(honest, target), (n, target)


def test_achieved_fpr_matches_control_arena_cross_check() -> None:
    # Mean actual_fpr from control-arena's compute_classifier_tpr over 200
    # random honest samples of each size, recorded 2026-09-11 at c8dbe40d.
    # Distinct continuous scores, so the mean is the closed form exactly.
    recorded = {
        (20, 0.005): 0.0000, (20, 0.01): 0.0000, (20, 0.02): 0.0000, (20, 0.05): 0.0000,
        (50, 0.005): 0.0000, (50, 0.01): 0.0000, (50, 0.02): 0.0000, (50, 0.05): 0.0400,
        (100, 0.005): 0.0000, (100, 0.01): 0.0000, (100, 0.02): 0.0100, (100, 0.05): 0.0400,
        (200, 0.005): 0.0000, (200, 0.01): 0.0050, (200, 0.02): 0.0150, (200, 0.05): 0.0450,
        (500, 0.005): 0.0040, (500, 0.01): 0.0080, (500, 0.02): 0.0180, (500, 0.05): 0.0480,
        (1000, 0.005): 0.0040, (1000, 0.01): 0.0090, (1000, 0.02): 0.0190, (1000, 0.05): 0.0490,
        (5000, 0.005): 0.0048, (5000, 0.01): 0.0098, (5000, 0.02): 0.0198, (5000, 0.05): 0.0498,
    }
    for (n, target), value in recorded.items():
        assert achieved_fpr(n, target) == pytest.approx(value, abs=5e-5), (n, target)


def test_achieved_fpr_is_zero_below_one_over_target_and_never_exceeds_target() -> None:
    for target in (0.01, 0.02, 0.05):
        for n in range(1, 2000):
            got = achieved_fpr(n, target)
            assert got <= target
            if n * target < 1.0:
                assert got == 0.0


def test_first_nonzero_fpr_n_is_one_more_than_one_over_target() -> None:
    assert first_nonzero_fpr_n(0.01) == 101
    assert first_nonzero_fpr_n(0.02) == 51
    assert first_nonzero_fpr_n(0.05) == 21
    for target in (0.005, 0.01, 0.02, 0.05, 0.1):
        n = first_nonzero_fpr_n(target)
        assert achieved_fpr(n, target) > 0.0
        assert achieved_fpr(n - 1, target) == 0.0


def test_honest_samples_for_fpr_is_a_guarantee_not_a_lucky_step() -> None:
    # Troughs of floor((n - 1) t) / n sit at n = (k + 1) / t, where the rate is
    # k / (k + 1) of the target. Within 10% needs k / (k + 1) >= 0.9, so k >= 9,
    # and the last trough that misses is k = 8, at n = 9 / t. The answer is one
    # past it.
    #   1%:   last miss 900  (8/900 = 0.00889 < 0.009)   -> 901
    #   2%:   last miss 450  (8/450 = 0.01778 < 0.018)   -> 451
    #   5%:   last miss 180  (8/180 = 0.04444 < 0.045)   -> 181
    #   0.5%: last miss 1800 (8/1800 = 0.00444 < 0.0045) -> 1801
    # n = 1000 at 1% delivers 9/1000, EXACTLY the 0.009 bound, and counts as a
    # pass. A floating point comparison scores it as a miss (0.9 * 0.01 is
    # 0.009000000000000001) and returns 1001; this pins the exact answer.
    assert honest_samples_for_fpr(0.01, 0.1) == 901
    assert honest_samples_for_fpr(0.02, 0.1) == 451
    assert honest_samples_for_fpr(0.05, 0.1) == 181
    assert honest_samples_for_fpr(0.005, 0.1) == 1801
    # And 101 (achieved 1/101 = 0.0099) sits on a step, not a floor.
    assert achieved_fpr(101, 0.01) >= 0.009
    assert achieved_fpr(900, 0.01) < 0.009
    for target, shortfall in ((0.01, 0.1), (0.02, 0.1), (0.05, 0.1), (0.01, 0.25)):
        n = honest_samples_for_fpr(target, shortfall)
        floor_rate = (1.0 - shortfall) * target
        assert achieved_fpr(n - 1, target) < floor_rate - 1e-12 or n == 1
        assert all(achieved_fpr(m, target) >= floor_rate - 1e-12 for m in range(n, n + 5000))


def test_fpr_helpers_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        achieved_fpr(0, 0.01)
    with pytest.raises(ValueError):
        achieved_fpr(100, 0.0)
    with pytest.raises(ValueError):
        achieved_fpr(100, 1.0)
    with pytest.raises(ValueError):
        honest_samples_for_fpr(0.01, 0.0)
    with pytest.raises(ValueError):
        first_nonzero_fpr_n(1.5)
