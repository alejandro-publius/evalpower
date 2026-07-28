"""Rebuild the two example result sets from a fixed seed.

The two files contain the same 600 items with the same pass or fail bits.
The only difference between them is the ``dimension`` column: one labels
every item with the single general risk, the other labels each item with the
sub-component it actually probes. No evidence is added or removed between
them. That is the point. Decomposition relabels evidence, it does not create
any, and everything the decomposed report says that the single question
report does not is a consequence of splitting the same 600 observations
twelve ways and testing twelve hypotheses against the same bar.

Usage:
    python examples/generate.py              rebuild both CSVs
    python examples/generate.py --sweep 5000 report how often the failure
                                             mode this example displays
                                             occurs across random seeds
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evalpower.metrics import wilson_interval
from evalpower.verdicts import Verdict, verdict_from_interval

HERE = Path(__file__).resolve().parent

RISK = "unsafe_tool_use"
THRESHOLD = 0.85
CONFIDENCE = 0.95
ITEMS_PER_DIMENSION = 50

# Twelve sub-components of one risk, with the true pass rate the simulated
# system has on each. Every rate is at or above the 0.85 threshold, so every
# sub-component genuinely complies. Four are handled well. Eight sit just
# above the bar at 0.855, which is what a system that was tuned to the bar
# looks like. Nothing here is a real defect, so any FAIL the report produces
# is a false positive by construction.
TRUE_RATES: Dict[str, float] = {
    "direct_request_refusal": 0.99,
    "paraphrased_request": 0.99,
    "translated_prompt": 0.99,
    "hypothetical_framing": 0.99,
    "roleplay_persona": 0.855,
    "multi_turn_escalation": 0.855,
    "code_execution_channel": 0.855,
    "tool_chaining": 0.855,
    "citation_laundering": 0.855,
    "encoded_payload": 0.855,
    "authority_impersonation": 0.855,
    "partial_completion": 0.855,
}

# Seed for the committed data. See choose_seed below for how it was picked
# and README.md for the honest frequency of the effect it displays.
SEED = 12


def simulate(seed: int) -> pd.DataFrame:
    """Simulate one eval run of 600 items across the twelve sub-components.

    Args:
        seed: Seed for the generator, so a run is reproducible.

    Returns:
        A frame with columns ``item_id``, ``dimension`` and ``passed``, with
        ``ITEMS_PER_DIMENSION`` items drawn for each sub-component.
    """
    rng = np.random.default_rng(seed)
    item_ids: List[str] = []
    dimensions: List[str] = []
    outcomes: List[int] = []

    index = 0
    for dimension, rate in TRUE_RATES.items():
        draws = (rng.random(ITEMS_PER_DIMENSION) < rate).astype(int)
        for outcome in draws:
            item_ids.append("item_{0:04d}".format(index))
            dimensions.append(dimension)
            outcomes.append(int(outcome))
            index += 1

    return pd.DataFrame({"item_id": item_ids, "dimension": dimensions, "passed": outcomes})


def as_single_question(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Relabel a decomposed run as one general question.

    Same items, same outcomes, one dimension. This is the eval a team runs
    when it asks the risk as a single question rather than decomposing it.
    """
    single = decomposed.copy()
    single["dimension"] = RISK
    return single


def _fast_verdicts(frame: pd.DataFrame) -> List[Verdict]:
    """Return the unadjusted Wilson verdict for each dimension in a frame.

    The sweep runs thousands of simulated evals and only needs the verdict,
    so this skips the bootstrap that the full report pays for.
    """
    grouped = frame.groupby("dimension", sort=True)["passed"].agg(["sum", "count"])
    return [
        verdict_from_interval(wilson_interval(int(row["sum"]), int(row["count"]), CONFIDENCE), THRESHOLD)
        for _, row in grouped.iterrows()
    ]


def displays_the_finding(frame: pd.DataFrame) -> Tuple[bool, int, int]:
    """Report whether one simulated run displays the failure mode.

    A run displays it when the single question is decisive, most
    sub-components are indeterminate, and at least one compliant
    sub-component fails outright.

    Args:
        frame: One simulated decomposed run.

    Returns:
        A ``(displays, indeterminate_count, fail_count)`` triple.
    """
    pooled = wilson_interval(int(frame["passed"].sum()), len(frame), CONFIDENCE)
    single_is_decisive = verdict_from_interval(pooled, THRESHOLD) is Verdict.PASS

    verdicts = _fast_verdicts(frame)
    indeterminate = sum(1 for verdict in verdicts if verdict is Verdict.INDETERMINATE)
    failures = sum(1 for verdict in verdicts if verdict is Verdict.FAIL)
    displays = single_is_decisive and indeterminate >= 7 and failures >= 1
    return displays, indeterminate, failures


def choose_seed(limit: int = 5000) -> int:
    """Return the first seed whose run displays the failure mode.

    The committed seed was picked this way, and the choice is disclosed
    rather than hidden: a spurious FAIL is a low probability event, so a
    randomly chosen seed usually will not show one. :func:`sweep` reports how
    often it happens, which is the number that actually matters.
    """
    for seed in range(limit):
        displays, _, _ = displays_the_finding(simulate(seed))
        if displays:
            return seed
    raise RuntimeError("no seed below {0} displays the finding".format(limit))


def sweep(runs: int = 5000) -> Dict[str, float]:
    """Measure how the decomposed eval behaves across many simulated runs.

    Args:
        runs: Number of independent simulated evals.

    Returns:
        A dictionary of frequencies and averages over the runs.
    """
    spurious_fail_runs = 0
    decisive_single = 0
    indeterminate_total = 0
    fail_total = 0
    for seed in range(runs):
        frame = simulate(seed)
        pooled = wilson_interval(int(frame["passed"].sum()), len(frame), CONFIDENCE)
        if verdict_from_interval(pooled, THRESHOLD) is Verdict.PASS:
            decisive_single += 1
        verdicts = _fast_verdicts(frame)
        indeterminate = sum(1 for verdict in verdicts if verdict is Verdict.INDETERMINATE)
        failures = sum(1 for verdict in verdicts if verdict is Verdict.FAIL)
        indeterminate_total += indeterminate
        fail_total += failures
        if failures >= 1:
            spurious_fail_runs += 1
    return {
        "runs": float(runs),
        "single_question_pass_rate": decisive_single / runs,
        "runs_with_at_least_one_spurious_fail": spurious_fail_runs / runs,
        "mean_indeterminate_dimensions": indeterminate_total / runs,
        "mean_spurious_fails": fail_total / runs,
    }


def write_examples(seed: int = SEED) -> Tuple[Path, Path]:
    """Write both example CSVs and return their paths."""
    decomposed = simulate(seed)
    single = as_single_question(decomposed)

    decomposed_path = HERE / "decomposed.csv"
    single_path = HERE / "single_question.csv"
    decomposed.to_csv(decomposed_path, index=False)
    single.to_csv(single_path, index=False)
    return single_path, decomposed_path


def main() -> None:
    """Rebuild the example data, or run the seed sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED, help="seed for the committed data")
    parser.add_argument("--sweep", type=int, default=0, help="run this many simulated evals and report")
    parser.add_argument("--choose-seed", action="store_true", help="print the first seed that displays the finding")
    args = parser.parse_args()

    if args.choose_seed:
        print("first displaying seed: {0}".format(choose_seed()))
        return

    if args.sweep:
        results = sweep(args.sweep)
        for key, value in results.items():
            print("{0}: {1:.4f}".format(key, value))
        return

    single_path, decomposed_path = write_examples(args.seed)
    frame = pd.read_csv(decomposed_path)
    _, indeterminate, failures = displays_the_finding(frame)
    print("wrote {0}".format(single_path))
    print("wrote {0}".format(decomposed_path))
    print(
        "seed {0}: {1} indeterminate and {2} spurious fail(s) across {3} dimensions".format(
            args.seed, indeterminate, failures, len(TRUE_RATES)
        )
    )


if __name__ == "__main__":
    main()
