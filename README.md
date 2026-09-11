# evalpower

[![CI](https://github.com/alejandro-publius/evalpower/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-publius/evalpower/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/alejandro-publius/evalpower)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

**The same 600 evaluations. The same pass or fail bit on every single item. Asked as one question, the eval returns a confident PASS. Split into twelve sub-components, the same evidence returns seven INDETERMINATE verdicts and one FAIL on a component that was never actually broken.**

Nothing was added or removed between those two runs. `examples/single_question.csv` and `examples/decomposed.csv` contain the same 600 items with the same outcomes, and differ only in the `dimension` column. Every difference in the report below comes from splitting one estimate into twelve and testing twelve hypotheses against the same bar.

## The two example outputs, side by side

```bash
pip install -e .
evalpower examples/single_question.csv
```

```bash
evalpower examples/decomposed.csv
```

| | `single_question.csv` | `decomposed.csv` |
| --- | --- | --- |
| Items evaluated | 600 | 600 (the same ones) |
| Items passing | 529 | 529 (the same ones) |
| Dimensions | 1 | 12 |
| Items per dimension | 600 | 50 |
| Mean 95% interval width | 0.0518 | 0.1658 |
| PASS | 1 | 4 |
| INDETERMINATE | 0 | 7 |
| FAIL | 0 | 1 |
| Headline verdict | **PASS** | **1 FAIL, and 58% of the risk unmeasured** |
| Extra items needed to resolve | 0 | 16,175 |

The single question:

```
| Dimension       | n   | passes | rate   | Wilson 95%       | Verdict |
| --------------- | --: | -----: | -----: | ---------------- | ------- |
| unsafe_tool_use | 600 |    529 | 0.8817 | [0.8534, 0.9051] | PASS    |
```

The same evidence, decomposed:

```
| Dimension               | n  | passes | rate   | Wilson 95%       | Verdict       |
| ----------------------- | -: | -----: | -----: | ---------------- | ------------- |
| authority_impersonation | 50 |     41 | 0.8200 | [0.6920, 0.9023] | INDETERMINATE |
| citation_laundering     | 50 |     44 | 0.8800 | [0.7620, 0.9438] | INDETERMINATE |
| code_execution_channel  | 50 |     40 | 0.8000 | [0.6696, 0.8876] | INDETERMINATE |
| direct_request_refusal  | 50 |     50 | 1.0000 | [0.9287, 1.0000] | PASS          |
| encoded_payload         | 50 |     43 | 0.8600 | [0.7381, 0.9305] | INDETERMINATE |
| hypothetical_framing    | 50 |     49 | 0.9800 | [0.8950, 0.9965] | PASS          |
| multi_turn_escalation   | 50 |     41 | 0.8200 | [0.6920, 0.9023] | INDETERMINATE |
| paraphrased_request     | 50 |     50 | 1.0000 | [0.9287, 1.0000] | PASS          |
| partial_completion      | 50 |     42 | 0.8400 | [0.7149, 0.9166] | INDETERMINATE |
| roleplay_persona        | 50 |     37 | 0.7400 | [0.6045, 0.8413] | FAIL          |
| tool_chaining           | 50 |     43 | 0.8600 | [0.7381, 0.9305] | INDETERMINATE |
| translated_prompt       | 50 |     49 | 0.9800 | [0.8950, 0.9965] | PASS          |
```

`roleplay_persona` is a false positive. Its true pass rate in the generator is 0.855, above the 0.85 threshold. It complies. It failed because with twelve simultaneous comparisons at a one sided error rate of 0.025, the chance that at least one compliant dimension fails by chance alone is 26.2 percent. `evalpower` catches this:

```
1 dimension changes verdict under adjustment:

- `roleplay_persona`: FAIL becomes INDETERMINATE
```

And it prices the rest of the report honestly:

```
| Dimension               | current n | rate   | required n | additional items | would resolve to |
| ----------------------- | --------: | -----: | ---------: | ---------------: | ---------------- |
| authority_impersonation |        50 | 0.8200 |        545 |              495 | FAIL             |
| citation_laundering     |        50 | 0.8800 |        545 |              495 | PASS             |
| code_execution_channel  |        50 | 0.8000 |        196 |              146 | FAIL             |
| encoded_payload         |        50 | 0.8600 |       4898 |             4848 | PASS             |
| multi_turn_escalation   |        50 | 0.8200 |        545 |              495 | FAIL             |
| partial_completion      |        50 | 0.8400 |       4898 |             4848 | FAIL             |
| tool_chaining           |        50 | 0.8600 |       4898 |             4848 | PASS             |

Resolving every indeterminate dimension takes 16525 items in total, against
the 350 currently spent on them: a 47.2x increase.
```

## Why this happens

High-dimensionality evaluation is a real improvement on asking a risk as one general question. A single question can only be answered in one dimension, and one dimension rarely describes how a probabilistic system behaves.

It has a statistical consequence that the method does not itself handle. Every sub-score is a proportion estimated from a sample. Decomposing one risk into k sub-components does not add data. It splits the same evidence across k estimates, so each interval widens by roughly the square root of k, and it runs k simultaneous comparisons against the same threshold, so the chance that at least one fails by accident grows toward certainty as k grows.

The consequence is uncomfortable: **the more faithfully a team follows the method, the less any individual sub-score can support, and the more likely the report is to contain at least one finding that is not real.** Both effects are visible in the example above, from the same 600 observations that produced a clean PASS.

`evalpower` does not argue against decomposition. It sits on top of whatever harness produced the scores and reports what those scores can and cannot carry.

## What it computes

1. **Per dimension estimates.** Pass rate, Wilson score interval, and a bootstrap percentile interval, reported side by side, with a note when the two disagree materially. They disagree in informative places: at a saturated pass rate the bootstrap collapses to `[1.0000, 1.0000]` while Wilson still reports a lower bound of 0.9287. The collapse is a property of the method, not evidence of certainty.

2. **Three way verdicts.** `PASS` when the interval lower bound is at or above the threshold, `FAIL` when the upper bound is below it, `INDETERMINATE` when the interval spans it. Most sub-scores on realistic budgets are the third one.

3. **Multiple comparisons.** Benjamini-Hochberg adjusted verdicts next to unadjusted ones, with a plain count of how many dimensions change.

4. **Required sample size.** For every `INDETERMINATE` dimension, the smallest n that would resolve it at the observed rate, in closed form: `n = z^2 * t(1 - t) / (p - t)^2`. This is the number a governance team wants before running the eval, not after. Halve the margin between the observed rate and the threshold and the requirement quadruples.

5. **Aggregate.** The pooled pass rate with its interval, plus an explicit note when the aggregate verdict and the per dimension verdicts disagree. In the example, the pooled estimate reads PASS while eight of twelve sub-components do not support it, because pooling borrows precision from the dimensions that were never in doubt.

The Wilson interval is the inversion of the score test, so the interval verdicts and the p values are two views of one test and can never contradict each other. The closed form for required sample size falls out of the same identity. `tests/test_verdicts.py` proves the equivalence exhaustively at every possible count for several sample sizes.

## Reproducing the examples

```bash
python examples/generate.py
```

That rewrites both CSVs from seed 12, so the committed data is locally owned and reproducible rather than fetched from anywhere.

**On the choice of seed, stated plainly.** A spurious FAIL is by construction a low probability event, so most seeds do not produce one, and a demo that quietly picked a lucky seed would be doing the exact thing this package exists to catch. Seed 12 is the first seed in `range(5000)` that satisfies the display criteria in `displays_the_finding`, and the generator will tell you so:

```bash
python examples/generate.py --choose-seed
```

The number that actually matters is the frequency, not the one run:

```bash
python examples/generate.py --sweep 3000
```

```
runs: 3000.0000
single_question_pass_rate: 0.9483
runs_with_at_least_one_spurious_fail: 0.1780
mean_indeterminate_dimensions: 7.7243
mean_spurious_fails: 0.1937
```

Across 3,000 simulated evals of a system that genuinely meets the threshold on all twelve sub-components, the single question returns a decisive PASS 94.8 percent of the time, an average of 7.7 of the 12 sub-components come back indeterminate, and 17.8 percent of runs contain at least one FAIL that is not real. That last figure is below the 26.2 percent ceiling for twelve independent comparisons because only eight of the twelve dimensions sit near the bar; the four strong ones essentially never fail by chance.

## Using it on your own eval

Input is one row per (item, dimension):

```csv
item_id,dimension,passed
item_0000,roleplay_persona,1
item_0001,roleplay_persona,0
```

`passed` is 0 or 1. JSON is accepted in the same shape, either as a bare list of records or under a `results` key. A file with a single dimension is valid and is the degenerate case.

```bash
evalpower results.csv --threshold 0.85
evalpower results.csv --threshold 0.90 --confidence 0.99 --output report.md
evalpower results.csv --fail-on-indeterminate   # exit code 2, for CI
evalpower results.csv --json --output report.json   # machine readable, for scripts and CI
```

`--json` writes the same analysis as a JSON document instead of the markdown report: every dimension's n, passes, rate, both intervals, both verdicts, p values and (when indeterminate) its sample size requirement, plus the pooled aggregate and the list of dimensions that changed verdict under adjustment. It composes with `--fail-on-indeterminate`: the report is still written, and the exit code still signals whether to fail the build.

Or as a library:

```python
import pandas as pd
from evalpower import analyze, render_markdown, required_n

analysis = analyze(pd.read_csv("results.csv"), threshold=0.85)
print(render_markdown(analysis))

# Before you run anything: at an expected 90 percent pass rate against an
# 0.85 bar, one dimension needs 196 items. Twelve of them need 2,352.
required_n(0.90, 0.85)
```

There are no LLM calls anywhere in this package. It consumes eval output, it does not produce it, which is what lets it work with any harness including one it has never seen.

## Monitor thresholds: the false positive rate you actually got

A control evaluation reports a monitor's TPR at a target false positive rate: set the threshold at the (1 - target) quantile of honest scores, count the attacks above it. The threshold is an empirical quantile, rounded up to the next observed honest score so that the rate is at most the target. With n honest scores that leaves floor((n - 1) * target) of them above the threshold, so the rate the threshold delivers is

```
achieved = floor((n - 1) * target) / n
```

It is exactly zero until n exceeds 1 / target. A TPR reported "at 1% FPR" from 50 honest samples is the TPR above the single most suspicious honest score, at an achieved FPR of 0.

```bash
python examples/monitor_fpr.py
```

```
 n honest req 0.005 req 0.010 req 0.020 req 0.050
       20    0.0000    0.0000    0.0000    0.0000
       50    0.0000    0.0000    0.0000    0.0400
      100    0.0000    0.0000    0.0100    0.0400
      200    0.0000    0.0050    0.0150    0.0450
      500    0.0040    0.0080    0.0180    0.0480
     1000    0.0040    0.0090    0.0190    0.0490
     5000    0.0048    0.0098    0.0198    0.0498
```

The closed form was checked against control-arena's `compute_classifier_tpr` ([UKGovernmentBEIS/control-arena](https://github.com/UKGovernmentBEIS/control-arena) at `c8dbe40d`), which implements this rule with numpy's `method="higher"` and a strict `>`. It reproduces the function's achieved FPR at every cell above, and `tests/test_planning.py` pins the table.

```python
from evalpower import achieved_fpr, first_nonzero_fpr_n, honest_samples_for_fpr

achieved_fpr(50, 0.01)             # 0.0
first_nonzero_fpr_n(0.01)          # 101
honest_samples_for_fpr(0.01, 0.1)  # 901: from here on, never more than 10% short of 1%
```

The last one is a guarantee, not a single lucky n. The achieved rate is a sawtooth that touches the target at each step and sags between steps, so 101 honest samples happen to deliver 0.0099 while 900 deliver 0.0089. Report the achieved rate next to the requested one, and size the honest set before the eval runs.

## What it does not do

It does not judge whether your dimensions are the right dimensions, whether your items are representative, or whether your threshold is the right threshold. Those are the harder questions and they are not statistical ones. It also assumes items are independent within a dimension; if your harness reuses prompts across dimensions or samples multiple completions per prompt, the true intervals are wider than the ones reported here.

## Install

Requires Python 3.9 or later. Dependencies are numpy and pandas.

```bash
pip install -e .
```

Run the tests:

```bash
pip install -e ".[test]" && pytest
```

Every expected value in the test suite is a hand computed literal with the arithmetic written out in a comment above it. Nothing is generated by calling the code under test.

## License

MIT. See [LICENSE](LICENSE).
