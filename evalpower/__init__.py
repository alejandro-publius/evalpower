"""evalpower: does this eval have enough data to support its verdict?

Decomposing one risk into k sub-components does not add data. It splits the
same evidence across k estimates, each with its own sampling error, and it
runs k simultaneous comparisons against the same threshold. This package sits
on top of any eval harness, consumes its per item output, and reports what
that output can and cannot support.
"""

from __future__ import annotations

from .metrics import (
    DimensionMetrics,
    Interval,
    bootstrap_interval,
    pass_rate,
    pooled_metrics,
    summarize_dimensions,
    validate_results,
    wilson_interval,
    z_for_confidence,
)
from .planning import (
    SampleSizeRequirement,
    achieved_fpr,
    first_nonzero_fpr_n,
    honest_samples_for_fpr,
    required_n,
    requirement_for,
)
from .report import Analysis, DimensionAnalysis, analysis_to_dict, analyze, render_json, render_markdown
from .verdicts import (
    Verdict,
    benjamini_hochberg,
    score_test_pvalues,
    verdict_from_interval,
)

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "DimensionAnalysis",
    "DimensionMetrics",
    "Interval",
    "SampleSizeRequirement",
    "Verdict",
    "achieved_fpr",
    "analysis_to_dict",
    "analyze",
    "benjamini_hochberg",
    "bootstrap_interval",
    "first_nonzero_fpr_n",
    "honest_samples_for_fpr",
    "pass_rate",
    "pooled_metrics",
    "render_json",
    "render_markdown",
    "required_n",
    "requirement_for",
    "score_test_pvalues",
    "summarize_dimensions",
    "validate_results",
    "verdict_from_interval",
    "wilson_interval",
    "z_for_confidence",
]
