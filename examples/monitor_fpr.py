"""The false positive rate a monitor threshold actually delivers.

    python examples/monitor_fpr.py

A control evaluation reports a monitor's "TPR at 1% FPR": set the threshold at
the 99th percentile of honest scores, count the attacks above it. The threshold
is an empirical quantile of n honest scores, rounded up to the next observed
score so the FPR is at most the target. That rounding has a floor. With n
honest scores, floor((n - 1) * target) of them can sit strictly above the
threshold, so the achieved FPR is floor((n - 1) * target) / n - exactly zero
until n > 1 / target, and short of the target by up to one sample's worth
after that.

The numbers below were cross-checked on 2026-09-11 against control-arena's
``compute_classifier_tpr`` (UKGovernmentBEIS/control-arena at c8dbe40d), which
implements this rule with numpy's ``method="higher"`` and a strict ``>``: the
closed form reproduces its mean achieved FPR over 200 random honest samples to
four decimals at every cell. ``tests/test_planning.py`` pins that table.
"""

from __future__ import annotations

from evalpower.planning import achieved_fpr, first_nonzero_fpr_n, honest_samples_for_fpr

TARGETS = [0.005, 0.01, 0.02, 0.05]
SIZES = [20, 50, 100, 200, 500, 1000, 5000]


def main() -> None:
    print(__doc__)
    print("Achieved FPR by honest sample size and requested FPR:\n")
    print(f"{'n honest':>9} " + " ".join(f"req {t:.3f}" for t in TARGETS))
    for n in SIZES:
        print(f"{n:>9} " + " ".join(f"{achieved_fpr(n, t):>9.4f}" for t in TARGETS))

    print("\nWhat each requested FPR needs from the honest set:\n")
    print(f"{'requested':>9} {'first n with FPR > 0':>21} {'n from which FPR stays within 10%':>34}")
    for t in TARGETS:
        print(f"{t:>9.3f} {first_nonzero_fpr_n(t):>21} {honest_samples_for_fpr(t, 0.1):>34}")

    print(
        "\nA TPR quoted 'at 1% FPR' from 50 honest samples is the TPR above the\n"
        "single most suspicious honest score, at an achieved FPR of 0. The\n"
        "function that computed it usually returns the achieved rate too; nothing\n"
        "prompts anyone to read it. Report the achieved FPR next to the requested\n"
        "one, and size the honest set before running the eval."
    )


if __name__ == "__main__":
    main()
