"""Are evalpower's intervals and verdicts calibrated?

    python examples/coverage.py

A tool that prints "95% interval" and decides PASS / FAIL / INDETERMINATE
owes you a demonstration that those numbers mean what they say. This computes
the calibration exactly — no Monte Carlo, no seed. For a given sample size n
and true pass rate p, the coverage of the interval is a finite sum over the
binomial distribution:

    coverage(n, p) = sum_k  C(n, k) p^k (1-p)^(n-k) * [interval(k, n) contains p]

and the verdict's error rate at the threshold is the same sum restricted to
the counts whose interval lands entirely on one side of the bar. Every number
below is that sum evaluated against evalpower's own `wilson_interval` and
`verdict_from_interval`, so it validates the code that ships, not a
restatement of the formula.

Two honest results. The Wilson interval's coverage is close to nominal and
never far below it. The three-way verdict's false-decision rate at the
threshold is NOT bounded by 5%: on discrete counts the actual error rate
oscillates above and below nominal, reaching ~8% at some sample sizes. That
is the standard behaviour of an interval test on a proportion, and this tool's
whole point is to state such things rather than let a "95%" label imply a
guarantee it does not carry.
"""

from __future__ import annotations

import math

from evalpower.metrics import wilson_interval
from evalpower.verdicts import Verdict, verdict_from_interval


def _binom_pmf(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1.0 - p) ** (n - k)


def interval_coverage(n: int, p: float, confidence: float = 0.95) -> float:
    """Exact probability that the Wilson interval on a sample of size n
    contains the true rate p."""
    return sum(
        _binom_pmf(n, k, p)
        for k in range(n + 1)
        if wilson_interval(k, n, confidence).lower <= p <= wilson_interval(k, n, confidence).upper
    )


def false_decision_rate(n: int, threshold: float, confidence: float = 0.95) -> float:
    """Exact probability that the three-way verdict decides PASS or FAIL when
    the true rate sits exactly on the threshold — i.e. makes a call it should
    not be able to make."""
    return sum(
        _binom_pmf(n, k, threshold)
        for k in range(n + 1)
        if verdict_from_interval(wilson_interval(k, n, confidence), threshold)
        in (Verdict.PASS, Verdict.FAIL)
    )


SIZES = [20, 50, 100, 200, 600]
RATES = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]


def main() -> None:
    print(__doc__)
    print("WILSON 95% INTERVAL COVERAGE — exact, should sit near 0.95\n")
    print(f"  {'n':>5} " + " ".join(f"p={p:.2f}" for p in RATES))
    for n in SIZES:
        print(f"  {n:>5} " + " ".join(f"{interval_coverage(n, p):>5.3f}" for p in RATES))

    grid = [0.55 + 0.43 * i / 43 for i in range(44)]
    print("\n  mean / worst coverage over p in [0.55, 0.98]:")
    for n in SIZES:
        covs = [interval_coverage(n, p) for p in grid]
        print(f"    n={n:>4}: mean {sum(covs) / len(covs):.3f}   worst {min(covs):.3f}")

    print("\nTHREE-WAY VERDICT FALSE-DECISION RATE at the threshold — exact")
    print("(the true rate equals the bar, so any PASS/FAIL is a wrong call)\n")
    print(f"  {'n':>5} {'t=0.85':>8}   nominal 0.05")
    for n in SIZES:
        fd = false_decision_rate(n, 0.85)
        flag = "  (> nominal)" if fd > 0.05 else ""
        print(f"  {n:>5} {fd:>8.4f}{flag}")

    print(
        "\nThe interval is well calibrated. The verdict's error rate at the bar\n"
        "oscillates above nominal on discrete counts — up to ~8% at some sizes.\n"
        "That is why the tool reports INDETERMINATE generously and prices the\n"
        "sample size it would take to resolve, rather than treating a lucky\n"
        "PASS or FAIL near the bar as decisive."
    )


if __name__ == "__main__":
    main()
