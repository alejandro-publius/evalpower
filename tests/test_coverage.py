"""Exact calibration checks for the intervals and verdicts evalpower ships.

The other test files pin evalpower's arithmetic against hand-computed values
and against statsmodels. This file asks the question a user actually cares
about: when the tool prints a "95% interval", does it contain the truth 95%
of the time, and when it decides PASS/FAIL, how often is that decision wrong?

Both are computed exactly, not by simulation. For a sample of size n drawn at
a true rate p, the coverage of the interval is a finite sum over the binomial
distribution of the counts whose interval contains p; the verdict's error rate
at the threshold is the same sum over the counts whose interval lands entirely
on one side of the bar. So these tests are deterministic — no seed, no
flake — and they exercise `wilson_interval` / `verdict_from_interval` exactly
as shipped, catching any regression that would quietly de-calibrate them.
"""

from __future__ import annotations

import math

import pytest

from evalpower.metrics import wilson_interval
from evalpower.verdicts import Verdict, verdict_from_interval


def _binom_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)


def _interval_coverage(n: int, p: float, confidence: float = 0.95) -> float:
    total = 0.0
    for k in range(n + 1):
        iv = wilson_interval(k, n, confidence)
        if iv.lower <= p <= iv.upper:
            total += _binom_pmf(n, k, p)
    return total


def _false_decision_rate(n: int, threshold: float, confidence: float = 0.95) -> float:
    total = 0.0
    for k in range(n + 1):
        if verdict_from_interval(wilson_interval(k, n, confidence), threshold) in (
            Verdict.PASS,
            Verdict.FAIL,
        ):
            total += _binom_pmf(n, k, threshold)
    return total


_GRID = [0.55 + 0.43 * i / 43 for i in range(44)]  # p in [0.55, 0.98]


class TestIntervalCoverage:
    """The Wilson interval's actual coverage is close to its nominal 95%."""

    @pytest.mark.parametrize("n", [20, 50, 100, 200, 600])
    def test_mean_coverage_is_near_nominal(self, n: int) -> None:
        # Averaged over the operating range, Wilson coverage sits within one
        # point of 0.95 at every sample size the examples use.
        covs = [_interval_coverage(n, p) for p in _GRID]
        mean = sum(covs) / len(covs)
        assert 0.945 <= mean <= 0.960, (n, mean)

    @pytest.mark.parametrize("n", [20, 50, 100, 200, 600])
    def test_worst_case_coverage_never_collapses(self, n: int) -> None:
        # Wilson coverage oscillates and dips below nominal in places (this is
        # the known conservatism/liberality of a discrete interval), but it
        # never falls far: the worst case over the grid stays above 0.90.
        worst = min(_interval_coverage(n, p) for p in _GRID)
        assert worst >= 0.90, (n, worst)

    def test_coverage_stabilises_as_n_grows(self) -> None:
        # More data pulls the worst-case coverage back toward nominal: the
        # dips shrink monotonically-ish with n. Pin the endpoints.
        worst_small = min(_interval_coverage(50, p) for p in _GRID)
        worst_large = min(_interval_coverage(600, p) for p in _GRID)
        assert worst_large > worst_small
        assert worst_large >= 0.94

    def test_a_saturated_rate_is_still_covered(self) -> None:
        # p = 1.0 is the case the README flags (the bootstrap collapses to a
        # point there while Wilson does not). Wilson must still cover it:
        # every sample reads k = n, and its interval's upper bound is 1.0.
        for n in (20, 50, 600):
            assert _interval_coverage(n, 1.0) == pytest.approx(1.0, abs=1e-12)


class TestVerdictErrorRateAtTheThreshold:
    """The honest part: a three-way verdict on discrete counts does NOT hold
    its false-decision rate at the nominal 5%. These pin that it oscillates
    above nominal but stays bounded, which is why the tool prefers
    INDETERMINATE and reports the sample size needed to actually resolve."""

    @pytest.mark.parametrize("n", [20, 50, 100, 200, 600])
    def test_false_decision_rate_is_bounded_but_can_exceed_nominal(self, n: int) -> None:
        fd = _false_decision_rate(n, 0.85)
        # Never wildly wrong ...
        assert fd <= 0.085, (n, fd)
        # ... but genuinely not held at 0.05: at n = 100 and n = 200 it is
        # above nominal, which a naive reading of "95% interval" would miss.
        if n in (100, 200):
            assert fd > 0.05, (n, fd)

    def test_the_worst_case_over_realistic_sizes_is_documented(self) -> None:
        # The number the README quotes: over the sample sizes an eval actually
        # runs at, the verdict's error rate at the bar reaches ~8%.
        worst = max(_false_decision_rate(n, 0.85) for n in range(20, 401))
        assert 0.07 <= worst <= 0.085

    def test_a_rate_off_the_threshold_is_decided_correctly_far_more_often(self) -> None:
        # Sanity: the error rate is a threshold artefact. A true rate a few
        # points above the bar is (correctly) called PASS far more than 5% of
        # the time, and essentially never FAIL.
        n, t = 200, 0.85
        p_true = 0.92
        fail = sum(
            _binom_pmf(n, k, p_true)
            for k in range(n + 1)
            if verdict_from_interval(wilson_interval(k, n), t) is Verdict.FAIL
        )
        assert fail < 0.01


class TestCoverageExampleAgrees:
    """examples/coverage.py prints these same quantities; keep its helpers in
    step with the test's, so the documented table cannot drift from the code."""

    def test_example_helpers_match(self) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).parent.parent / "examples" / "coverage.py"
        spec = importlib.util.spec_from_file_location("coverage_example", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        # Same quantities, but the example's `sum(generator)` and the test's
        # accumulating loop add the binomial terms in a different order, so
        # they can differ in the last floating-point bit; agreement to ~1e-12
        # is exact agreement on the mathematics.
        for n, p in ((50, 0.85), (200, 0.90), (600, 0.95)):
            assert module.interval_coverage(n, p) == pytest.approx(_interval_coverage(n, p), abs=1e-12)
        for n in (50, 100, 600):
            assert module.false_decision_rate(n, 0.85) == pytest.approx(
                _false_decision_rate(n, 0.85), abs=1e-12
            )
