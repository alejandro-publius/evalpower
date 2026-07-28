"""Tests for evalpower.verdicts.

Benjamini-Hochberg is a sort and a running minimum, so every case below is
worked through by hand in the comment above it. The score test values are
worked the same way from the closed form.
"""

from __future__ import annotations

import pytest

from evalpower.metrics import Interval, wilson_interval
from evalpower.verdicts import (
    Verdict,
    benjamini_hochberg,
    count_changed_verdicts,
    default_fdr_q,
    score_test_pvalues,
    score_test_z,
    verdict_counts,
    verdict_from_interval,
    verdict_from_pvalues,
)


def test_verdict_pass_when_lower_bound_clears_threshold() -> None:
    assert verdict_from_interval(Interval(0.86, 0.94), 0.85) is Verdict.PASS


def test_verdict_fail_when_upper_bound_is_below_threshold() -> None:
    assert verdict_from_interval(Interval(0.70, 0.84), 0.85) is Verdict.FAIL


def test_verdict_indeterminate_when_interval_spans_threshold() -> None:
    assert verdict_from_interval(Interval(0.80, 0.90), 0.85) is Verdict.INDETERMINATE


def test_verdict_boundaries_are_explicit() -> None:
    # PASS is "at or above", so a lower bound exactly on the threshold passes.
    assert verdict_from_interval(Interval(0.85, 0.99), 0.85) is Verdict.PASS
    # FAIL is "strictly below", so an upper bound exactly on the threshold
    # does not fail. It is indeterminate.
    assert verdict_from_interval(Interval(0.60, 0.85), 0.85) is Verdict.INDETERMINATE


def test_verdict_is_never_pass_or_fail_by_default() -> None:
    # A wide interval on a small sample supports nothing. Ten items at 90
    # percent gives a Wilson interval of about [0.5958, 0.9834], which spans
    # a threshold of 0.85 from both sides.
    assert verdict_from_interval(wilson_interval(9, 10, 0.95), 0.85) is Verdict.INDETERMINATE


def test_default_fdr_q_matches_the_interval_level() -> None:
    # A two sided 95 percent interval spends (1 - 0.95) / 2 = 0.025 per tail.
    assert default_fdr_q(0.95) == pytest.approx(0.025, abs=1e-15)
    # A two sided 99 percent interval spends (1 - 0.99) / 2 = 0.005 per tail.
    assert default_fdr_q(0.99) == pytest.approx(0.005, abs=1e-15)


def test_score_test_z() -> None:
    # p = 40 / 50 = 0.8, threshold = 0.85
    # standard error = sqrt(0.85 * 0.15 / 50) = sqrt(0.00255)
    #                = 0.05049752469181039
    # z = (0.8 - 0.85) / 0.05049752469181039 = -0.990147542976673
    assert score_test_z(40, 50, 0.85) == pytest.approx(-0.990147542976673, abs=1e-12)
    # p = 48 / 50 = 0.96
    # z = (0.96 - 0.85) / 0.05049752469181039 = 2.178324594548683
    assert score_test_z(48, 50, 0.85) == pytest.approx(2.178324594548683, abs=1e-12)


def test_score_test_z_rejects_degenerate_thresholds() -> None:
    for bad in (0.0, 1.0, -0.2, 1.4):
        with pytest.raises(ValueError):
            score_test_z(40, 50, bad)


def test_score_test_pvalues() -> None:
    # z = -0.990147542976673 for 40 of 50 against 0.85.
    # p_fail = Phi(-0.990147542976673) = 0.16105100406444817
    # p_pass = 1 - p_fail             = 0.8389489959355518
    p_pass, p_fail = score_test_pvalues(40, 50, 0.85)
    assert p_fail == pytest.approx(0.16105100406444817, abs=1e-12)
    assert p_pass == pytest.approx(0.8389489959355518, abs=1e-12)
    # The two are complements of one another by construction.
    assert p_pass + p_fail == pytest.approx(1.0, abs=1e-12)


def test_score_test_pvalues_on_the_pass_side() -> None:
    # z = 2.178324594548683 for 48 of 50 against 0.85.
    # p_pass = Phi(-2.178324594548683) = 0.014690939685036814
    # p_fail = 1 - 0.014690939685036814 = 0.9853090603149632
    p_pass, p_fail = score_test_pvalues(48, 50, 0.85)
    assert p_pass == pytest.approx(0.014690939685036814, abs=1e-12)
    assert p_fail == pytest.approx(0.9853090603149632, abs=1e-12)


def test_wilson_interval_and_score_test_agree_exactly() -> None:
    # The Wilson interval is the inversion of the score test, so a 95 percent
    # interval whose lower bound clears the threshold is exactly the case
    # where the one sided p value for PASS is at most 0.025, and likewise on
    # the FAIL side. This has to hold at every count, not just on average.
    threshold, alpha_one_sided = 0.85, 0.025
    for n in (25, 50, 137, 400):
        for successes in range(0, n + 1):
            verdict = verdict_from_interval(wilson_interval(successes, n, 0.95), threshold)
            p_pass, p_fail = score_test_pvalues(successes, n, threshold)
            assert (verdict is Verdict.PASS) == (p_pass <= alpha_one_sided)
            assert (verdict is Verdict.FAIL) == (p_fail < alpha_one_sided)


def test_benjamini_hochberg_step_up_rescues_middle_ranks() -> None:
    # k = 5, q = 0.05. Critical values (i / k) * q are
    #   0.01, 0.02, 0.03, 0.04, 0.05
    # Sorted p values: 0.001, 0.008, 0.039, 0.041, 0.042
    #   rank 1: 0.001 <= 0.01  yes
    #   rank 2: 0.008 <= 0.02  yes
    #   rank 3: 0.039 <= 0.03  no
    #   rank 4: 0.041 <= 0.04  no
    #   rank 5: 0.042 <= 0.05  yes
    # The largest passing rank is 5, and the procedure steps up, so all five
    # are rejected including ranks 3 and 4.
    # Adjusted values are the running minimum of (k / j) * p_(j) from the top:
    #   j = 5: (5 / 5) * 0.042 = 0.042
    #   j = 4: (5 / 4) * 0.041 = 0.05125 -> min(0.05125, 0.042) = 0.042
    #   j = 3: (5 / 3) * 0.039 = 0.065   -> min(0.065,   0.042) = 0.042
    #   j = 2: (5 / 2) * 0.008 = 0.02    -> min(0.02,    0.042) = 0.02
    #   j = 1: (5 / 1) * 0.001 = 0.005   -> min(0.005,   0.02)  = 0.005
    rejected, adjusted = benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042], q=0.05)
    assert rejected == [True, True, True, True, True]
    assert adjusted == pytest.approx([0.005, 0.02, 0.042, 0.042, 0.042], abs=1e-12)


def test_benjamini_hochberg_rejects_nothing_when_every_p_exceeds_its_critical_value() -> None:
    # k = 3, q = 0.05. Critical values are
    #   (1 / 3) * 0.05 = 0.016666666666666666
    #   (2 / 3) * 0.05 = 0.03333333333333333
    #   (3 / 3) * 0.05 = 0.05
    # Sorted p values 0.04, 0.05, 0.06 exceed every one of them
    # (0.04 > 0.01667, 0.05 > 0.03333, 0.06 > 0.05), so nothing is rejected.
    # Adjusted from the top:
    #   j = 3: (3 / 3) * 0.06 = 0.06
    #   j = 2: (3 / 2) * 0.05 = 0.075 -> min(0.075, 0.06) = 0.06
    #   j = 1: (3 / 1) * 0.04 = 0.12  -> min(0.12,  0.06) = 0.06
    rejected, adjusted = benjamini_hochberg([0.04, 0.05, 0.06], q=0.05)
    assert rejected == [False, False, False]
    assert adjusted == pytest.approx([0.06, 0.06, 0.06], abs=1e-12)


def test_benjamini_hochberg_on_the_exact_boundary() -> None:
    # k = 5, q = 0.05, p values sitting exactly on their critical values.
    #   rank i critical value (i / 5) * 0.05 equals p_(i) for every i, and the
    #   comparison is "at or below", so all five are rejected.
    # Adjusted from the top, every term is exactly 0.05:
    #   j = 5: (5 / 5) * 0.05 = 0.05
    #   j = 4: (5 / 4) * 0.04 = 0.05
    #   j = 3: (5 / 3) * 0.03 = 0.05
    #   j = 2: (5 / 2) * 0.02 = 0.05
    #   j = 1: (5 / 1) * 0.01 = 0.05
    rejected, adjusted = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], q=0.05)
    assert rejected == [True, True, True, True, True]
    assert adjusted == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05], abs=1e-12)


def test_benjamini_hochberg_preserves_input_order() -> None:
    # The same five values as the step up case, shuffled. Results must come
    # back aligned with the input, not with the sort.
    #   input:    0.041, 0.001, 0.042, 0.008, 0.039
    #   adjusted: 0.042, 0.005, 0.042, 0.02,  0.042
    rejected, adjusted = benjamini_hochberg([0.041, 0.001, 0.042, 0.008, 0.039], q=0.05)
    assert rejected == [True, True, True, True, True]
    assert adjusted == pytest.approx([0.042, 0.005, 0.042, 0.02, 0.042], abs=1e-12)


def test_benjamini_hochberg_is_stricter_than_no_correction() -> None:
    # A single p value of 0.02 clears an uncorrected 0.025 bar. Placed in a
    # family of twelve where nothing else is small, it does not survive:
    # its rank is 1, so its critical value is (1 / 12) * 0.025 = 0.00208333,
    # and its adjusted value is (12 / 1) * 0.02 = 0.24, capped by later ranks
    # but still far above 0.025. This is the spurious FAIL rescue.
    pvalues = [0.02] + [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    assert len(pvalues) == 12
    rejected, adjusted = benjamini_hochberg(pvalues, q=0.025)
    assert rejected[0] is False
    assert adjusted[0] == pytest.approx(0.24, abs=1e-12)
    assert not any(rejected)


def test_benjamini_hochberg_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg([], q=0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5], q=0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], q=0.0)
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], q=1.5)


def test_verdict_from_pvalues() -> None:
    assert verdict_from_pvalues(True, False, 0.96, 0.85) is Verdict.PASS
    assert verdict_from_pvalues(False, True, 0.60, 0.85) is Verdict.FAIL
    assert verdict_from_pvalues(False, False, 0.88, 0.85) is Verdict.INDETERMINATE
    # Both tails cannot carry evidence at once. If a caller ever forces that
    # case, the point estimate decides rather than the procedure inventing one.
    assert verdict_from_pvalues(True, True, 0.96, 0.85) is Verdict.PASS
    assert verdict_from_pvalues(True, True, 0.60, 0.85) is Verdict.FAIL


def test_count_changed_verdicts() -> None:
    changes = count_changed_verdicts(
        ["a", "b", "c"],
        [Verdict.FAIL, Verdict.PASS, Verdict.INDETERMINATE],
        [Verdict.INDETERMINATE, Verdict.PASS, Verdict.INDETERMINATE],
    )
    assert len(changes) == 1
    assert changes[0].dimension == "a"
    assert changes[0].unadjusted is Verdict.FAIL
    assert changes[0].adjusted is Verdict.INDETERMINATE


def test_count_changed_verdicts_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError):
        count_changed_verdicts(["a"], [Verdict.PASS, Verdict.FAIL], [Verdict.PASS])


def test_verdict_counts_always_reports_all_three() -> None:
    counts = verdict_counts([Verdict.PASS, Verdict.PASS, Verdict.INDETERMINATE])
    assert counts[Verdict.PASS] == 2
    assert counts[Verdict.INDETERMINATE] == 1
    assert counts[Verdict.FAIL] == 0
