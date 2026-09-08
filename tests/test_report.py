"""End to end tests for the analysis, the markdown report and the CLI.

test_metrics, test_verdicts and test_planning pin the arithmetic against
hand computed literals. This file checks that the pieces compose, and that
the committed example data still shows the finding the README describes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from evalpower.cli import load_results, main
from evalpower.report import analysis_to_dict, analyze, render_json, render_markdown
from evalpower.verdicts import Verdict

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["item_id", "dimension", "passed"])


def test_analyze_on_a_single_dimension() -> None:
    # The degenerate case the README calls out: one dimension is valid input.
    # 45 of 50 at 0.90 gives a Wilson interval that spans a 0.85 threshold,
    # so a single question on this little data is already indeterminate.
    rows = [("item_{0}".format(index), "only", 1 if index < 45 else 0) for index in range(50)]
    analysis = analyze(_frame(rows), threshold=0.85, resamples=500, seed=0)
    assert analysis.k == 1
    assert analysis.dimensions[0].verdict is Verdict.INDETERMINATE
    assert analysis.aggregate_verdict is Verdict.INDETERMINATE
    # Its requirement is the 196 items that test_planning derives by hand.
    assert analysis.dimensions[0].requirement.required_n == 196


def test_analyze_pools_the_aggregate_across_dimensions() -> None:
    rows = [("item_{0}".format(index), "a", 1) for index in range(40)]
    rows += [("item_{0}".format(40 + index), "b", 1 if index < 20 else 0) for index in range(40)]
    analysis = analyze(_frame(rows), threshold=0.85, resamples=500, seed=0)
    # 40 of 40 on a, 20 of 40 on b, so 60 of 80 pooled.
    assert (analysis.aggregate.passes, analysis.aggregate.n) == (60, 80)
    assert analysis.n_rows == 80
    assert analysis.n_items == 80
    # a is saturated and clears the bar, b is at 0.50 and is nowhere near it.
    verdicts = {item.dimension: item.verdict for item in analysis.dimensions}
    assert verdicts["a"] is Verdict.PASS
    assert verdicts["b"] is Verdict.FAIL


def test_analyze_records_only_indeterminate_requirements() -> None:
    rows = [("item_{0}".format(index), "a", 1) for index in range(40)]
    rows += [("item_{0}".format(40 + index), "b", 1 if index < 20 else 0) for index in range(40)]
    analysis = analyze(_frame(rows), threshold=0.85, resamples=500, seed=0)
    requirements = {item.dimension: item.requirement for item in analysis.dimensions}
    assert requirements["a"] is None
    assert requirements["b"] is None


def test_render_markdown_has_every_section() -> None:
    rows = [("item_{0}".format(index), "only", 1 if index < 45 else 0) for index in range(50)]
    report = render_markdown(analyze(_frame(rows), threshold=0.85, resamples=500, seed=0))
    for heading in (
        "# evalpower report",
        "## Headline",
        "## Per dimension",
        "## Multiple comparisons",
        "## Required sample size",
        "## Aggregate",
    ):
        assert heading in report
    assert report.endswith("\n")


def test_load_results_round_trips_csv_and_json(tmp_path: Path) -> None:
    frame = _frame([("i1", "a", 1), ("i2", "a", 0)])

    csv_path = tmp_path / "results.csv"
    frame.to_csv(csv_path, index=False)
    assert load_results(csv_path)["passed"].tolist() == [1, 0]

    json_path = tmp_path / "results.json"
    json_path.write_text(frame.to_json(orient="records"), encoding="utf-8")
    assert load_results(json_path)["passed"].tolist() == [1, 0]

    wrapped_path = tmp_path / "wrapped.json"
    wrapped_path.write_text(
        '{"results": [{"item_id": "i1", "dimension": "a", "passed": 1}]}', encoding="utf-8"
    )
    assert load_results(wrapped_path)["dimension"].tolist() == ["a"]


def test_load_results_rejects_unknown_shapes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "absent.csv")
    bad_suffix = tmp_path / "results.txt"
    bad_suffix.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        load_results(bad_suffix)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"unexpected": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_results(bad_json)


def test_cli_writes_a_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "report.md"
    code = main(
        [
            str(EXAMPLES / "decomposed.csv"),
            "--threshold",
            "0.85",
            "--bootstrap-resamples",
            "500",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "## Multiple comparisons" in output.read_text(encoding="utf-8")


def test_cli_reports_bad_input_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert "evalpower:" in capsys.readouterr().err


def test_cli_can_fail_a_build_on_indeterminate_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    code = main(
        [
            str(EXAMPLES / "decomposed.csv"),
            "--bootstrap-resamples",
            "200",
            "--output",
            str(output),
            "--fail-on-indeterminate",
        ]
    )
    assert code == 2


def test_cli_fails_a_build_when_only_the_aggregate_is_indeterminate(tmp_path: Path) -> None:
    # 25 items at 10/25 (rate 0.40) is a decisive FAIL: its Wilson upper
    # bound sits well below the 0.85 threshold. 135 items at 125/135 (rate
    # 0.926) is a decisive PASS: its Wilson lower bound sits well above it.
    # Every per dimension verdict is therefore decisive. But pooling both
    # into one estimate (160 items, 135 passes, rate 0.8438) lands close
    # enough to the threshold that the pooled Wilson interval straddles it:
    # the aggregate verdict is INDETERMINATE even though no dimension is.
    # A CI gate that only checks per dimension verdicts would exit 0 here,
    # while the report it just wrote says INDETERMINATE in the aggregate
    # row -- the tool would disagree with itself.
    rows = [("A-{0}".format(i), "A", 1 if i < 10 else 0) for i in range(25)]
    rows += [("B-{0}".format(i), "B", 1 if i < 125 else 0) for i in range(135)]
    input_path = tmp_path / "aggregate_indeterminate.csv"
    _frame(rows).to_csv(input_path, index=False)

    analysis = analyze(pd.read_csv(input_path), threshold=0.85, resamples=200, seed=0)
    assert all(item.verdict is not Verdict.INDETERMINATE for item in analysis.dimensions)
    assert analysis.aggregate_verdict is Verdict.INDETERMINATE

    output = tmp_path / "report.md"
    code = main(
        [
            str(input_path),
            "--bootstrap-resamples",
            "200",
            "--output",
            str(output),
            "--fail-on-indeterminate",
        ]
    )
    assert "INDETERMINATE" in output.read_text(encoding="utf-8")
    assert code == 2

    json_output = tmp_path / "report.json"
    json_code = main(
        [
            str(input_path),
            "--bootstrap-resamples",
            "200",
            "--json",
            "--output",
            str(json_output),
            "--fail-on-indeterminate",
        ]
    )
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["aggregate_verdict"] == "INDETERMINATE"
    assert json_code == 2


def test_committed_examples_hold_identical_evidence() -> None:
    # The two files must differ only in the dimension column. If they ever
    # stop being the same 600 outcomes, the comparison in the README stops
    # meaning anything.
    single = pd.read_csv(EXAMPLES / "single_question.csv")
    decomposed = pd.read_csv(EXAMPLES / "decomposed.csv")
    assert single["item_id"].tolist() == decomposed["item_id"].tolist()
    assert single["passed"].tolist() == decomposed["passed"].tolist()
    assert single["dimension"].nunique() == 1
    assert decomposed["dimension"].nunique() == 12
    assert len(single) == 600


def test_committed_examples_still_show_the_finding() -> None:
    # Counts read straight off the committed CSVs with pandas, independent of
    # any evalpower code.
    decomposed = pd.read_csv(EXAMPLES / "decomposed.csv")
    counts = decomposed.groupby("dimension")["passed"].agg(["sum", "count"])
    assert int(decomposed["passed"].sum()) == 529
    assert int(counts.loc["roleplay_persona", "sum"]) == 37

    single_analysis = analyze(
        pd.read_csv(EXAMPLES / "single_question.csv"), threshold=0.85, resamples=500, seed=0
    )
    assert single_analysis.dimensions[0].verdict is Verdict.PASS

    analysis = analyze(decomposed, threshold=0.85, resamples=500, seed=0)
    verdicts = [item.verdict for item in analysis.dimensions]
    assert sum(1 for verdict in verdicts if verdict is Verdict.PASS) == 4
    assert sum(1 for verdict in verdicts if verdict is Verdict.INDETERMINATE) == 7
    assert sum(1 for verdict in verdicts if verdict is Verdict.FAIL) == 1

    # The one FAIL is roleplay_persona at 37 of 50, a rate of 0.74. Its
    # Wilson upper bound is 0.8413, below the 0.85 bar, so the unadjusted
    # verdict is FAIL. Its true rate in examples/generate.py is 0.855, so
    # that FAIL is a false positive.
    failing = [item for item in analysis.dimensions if item.verdict is Verdict.FAIL]
    assert [item.dimension for item in failing] == ["roleplay_persona"]
    assert failing[0].metrics.wilson.upper == pytest.approx(0.8413, abs=5e-5)

    # Benjamini-Hochberg across the twelve rescues it, and nothing else moves.
    assert len(analysis.changes) == 1
    assert analysis.changes[0].dimension == "roleplay_persona"
    assert analysis.changes[0].unadjusted is Verdict.FAIL
    assert analysis.changes[0].adjusted is Verdict.INDETERMINATE

    # The pooled estimate still reads PASS while eight of twelve parts do not.
    assert analysis.aggregate_verdict is Verdict.PASS
    assert analysis.aggregate.passes == 529


def test_analyze_handles_a_single_item_dimension() -> None:
    # n = 1 is the smallest possible input. A lone pass at threshold 0.85
    # gives a Wilson interval of [0.2065, 1.0] (statsmodels-verified in
    # test_oracle.py), which spans the threshold, so this is indeterminate
    # rather than a crash or a spurious PASS from a single lucky draw.
    analysis = analyze(_frame([("item_0", "only", 1)]), threshold=0.85, resamples=200, seed=0)
    assert analysis.dimensions[0].metrics.n == 1
    assert analysis.dimensions[0].verdict is Verdict.INDETERMINATE
    assert analysis.dimensions[0].requirement is not None
    # Rendering must not fail on the smallest possible input, in either
    # format.
    assert "## Headline" in render_markdown(analysis)
    json.loads(render_json(analysis))


def test_analyze_handles_zero_observed_failures() -> None:
    # A saturated 100% pass rate: the bootstrap interval collapses to a
    # point (test_metrics.py), which must not break the full pipeline or
    # its JSON rendering.
    rows = [("item_{0}".format(index), "only", 1) for index in range(30)]
    analysis = analyze(_frame(rows), threshold=0.85, resamples=200, seed=0)
    dimension = analysis.dimensions[0]
    assert (dimension.metrics.passes, dimension.metrics.n) == (30, 30)
    assert dimension.metrics.bootstrap.width == 0.0
    assert dimension.metrics.disagreement is not None
    assert dimension.verdict is Verdict.PASS
    assert dimension.requirement is None
    payload = analysis_to_dict(analysis)
    assert payload["dimensions"][0]["requirement"] is None
    assert payload["dimensions"][0]["bootstrap"] == {"lower": 1.0, "upper": 1.0}


def test_analyze_handles_a_rate_exactly_on_the_threshold() -> None:
    # 17 of 20 is exactly 0.85. The dimension is indeterminate and never
    # resolves at any sample size (test_planning.py), so the report must
    # show "never" rather than a number, and the JSON must carry an
    # explicit null rather than omitting the field.
    rows = [("item_{0}".format(index), "only", 1 if index < 17 else 0) for index in range(20)]
    analysis = analyze(_frame(rows), threshold=0.85, resamples=200, seed=0)
    dimension = analysis.dimensions[0]
    assert dimension.verdict is Verdict.INDETERMINATE
    assert dimension.requirement is not None
    assert dimension.requirement.required_n is None
    assert dimension.requirement.additional_items is None
    report = render_markdown(analysis)
    assert "| only | 20 | 0.8500 | never | n/a | n/a |" in report
    payload = analysis_to_dict(analysis)
    assert payload["dimensions"][0]["requirement"]["required_n"] is None
    assert payload["dimensions"][0]["requirement"]["would_resolve_to"] is None


def test_render_json_round_trips_the_committed_decomposed_example() -> None:
    analysis = analyze(
        pd.read_csv(EXAMPLES / "decomposed.csv"), threshold=0.85, resamples=500, seed=0
    )
    payload = json.loads(render_json(analysis))
    assert payload["k"] == 12
    assert payload["aggregate_verdict"] == "PASS"
    assert payload["n_items"] == 600
    verdicts = {item["dimension"]: item["verdict"] for item in payload["dimensions"]}
    assert verdicts["roleplay_persona"] == "FAIL"
    assert verdicts["direct_request_refusal"] == "PASS"
    assert [change["dimension"] for change in payload["changes"]] == ["roleplay_persona"]
    assert payload["changes"][0] == {
        "dimension": "roleplay_persona",
        "unadjusted": "FAIL",
        "adjusted": "INDETERMINATE",
    }
    # Every dimension keeps a requirement key, present or explicitly null,
    # so a consumer never has to special case a missing field.
    for item in payload["dimensions"]:
        assert "requirement" in item
        if item["verdict"] == "INDETERMINATE":
            assert item["requirement"] is not None
        else:
            assert item["requirement"] is None


def test_cli_json_flag_writes_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    code = main(
        [
            str(EXAMPLES / "decomposed.csv"),
            "--bootstrap-resamples",
            "300",
            "--json",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate_verdict"] == "PASS"
    assert payload["k"] == 12


def test_cli_json_and_fail_on_indeterminate_compose(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    code = main(
        [
            str(EXAMPLES / "decomposed.csv"),
            "--bootstrap-resamples",
            "300",
            "--json",
            "--fail-on-indeterminate",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    # The report is still written even though the exit code signals failure,
    # so a CI step can both gate on the exit code and archive the JSON.
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["k"] == 12


@pytest.mark.parametrize("bad_threshold", ["0", "1", "1.5", "-0.1"])
def test_cli_rejects_a_threshold_of_zero_one_or_out_of_range(
    bad_threshold: str, capsys: pytest.CaptureFixture
) -> None:
    # A bar of 0 or 1 is degenerate: score_test_z and required_n both raise
    # rather than silently dividing by zero or reporting a meaningless
    # verdict. The CLI must surface that as a clean exit 1 with a message on
    # stderr, not a traceback.
    code = main([str(EXAMPLES / "single_question.csv"), "--threshold", bad_threshold])
    assert code == 1
    err = capsys.readouterr().err
    assert "evalpower:" in err
    assert "threshold" in err
