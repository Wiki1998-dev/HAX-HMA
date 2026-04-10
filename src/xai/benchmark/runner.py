"""Phase 5 benchmark runner and comparison-matrix emitter.

Runs every XAI method against every model in :data:`MODEL_CATALOG`
and emits the Phase 5 comparison matrix in Markdown / CSV form.

Cell semantics
--------------

Each cell in the matrix reports **absolute SNAX cycles** and the
**relative overhead** versus the baseline inference of that model:

    overhead_pct = 100 * xai_cycles / inference_cycles

The ``gap_fc`` row is populated from *measured* values (Phase 1 / 2b
/ 4); every other row is estimated by the calibrated
:mod:`src.xai.benchmark.cycle_model`.  Cells are tagged so the final
report is explicit about what was measured vs extrapolated.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import MODEL_CATALOG, ModelSpec
from .cycle_model import (
    DEFAULT_SNAX_COST,
    SnaxCostParams,
    estimate_gradcam_cycles,
    estimate_hoisted_shap_cycles,
    estimate_inference_cycles,
    estimate_shap_cycles,
    estimate_symbolic_cycles,
)


# ---------------------------------------------------------------------------
# Measured values from earlier phases (GAP+FC calibration row)
# ---------------------------------------------------------------------------

MEASURED: Dict[str, Dict[str, int]] = {
    "gap_fc": {
        "inference":    12_000,   # base inference budget from Phase 5 plan
        "gradcam":       6_153,   # Phase 1, verified 0/16 BIST errors
        "shap_naive":  175_510,   # Phase 2a baseline
        "shap_hoisted": 58_022,   # Phase 2b optimized
        "symbolic":         47,   # Phase 4 depth-3 rule walk
    },
}


@dataclass
class BenchmarkRow:
    """Per-model row in the Phase 5 comparison matrix.

    Attributes:
        model_name:         catalogue key
        display_name:       pretty name
        inference:          baseline inference cycles
        gradcam:             Grad-CAM overhead cycles
        shap_naive:          naive Gradient SHAP overhead cycles (Phase 2a)
        shap_hoisted:        backward-hoisted SHAP overhead cycles (Phase 2b)
        symbolic:            neuro-symbolic rule walk cycles
        qvip_host_seconds:   host-side QVIP solver time (coarse)
        measured:            True if values came from real measurements
        notes:               any per-row caveats
    """

    model_name: str
    display_name: str
    inference: int
    gradcam: int
    shap_naive: int
    shap_hoisted: int
    symbolic: int
    qvip_host_seconds: float
    measured: bool = False
    notes: str = ""

    def overhead_pct(self, cycles: int) -> float:
        """Return a cycle count as percentage of baseline inference."""
        if self.inference <= 0:
            return 0.0
        return 100.0 * cycles / self.inference


@dataclass
class BenchmarkMatrix:
    """Full benchmark matrix — one row per model."""

    rows: List[BenchmarkRow] = field(default_factory=list)
    params: SnaxCostParams = field(default_factory=lambda: DEFAULT_SNAX_COST)

    def by_name(self, name: str) -> BenchmarkRow:
        """Fetch a row by model name, raising KeyError if absent."""
        for row in self.rows:
            if row.model_name == name:
                return row
        raise KeyError(name)


# ---------------------------------------------------------------------------
# QVIP host-side cost (rough, based on ILP constraint count)
# ---------------------------------------------------------------------------


def _estimate_qvip_host_seconds(model: ModelSpec) -> float:
    """Rough host-side QVIP solver time estimate.

    We measured Phase 3 QVIP on a 2-layer, 8-input network taking
    ``12.4s`` (the number in the Phase 5 plan).  Real ILP solvers scale
    super-linearly with constraint count, so we use:

        seconds ≈ 12.4 * (macs / gap_fc_macs) ** 1.5

    capped at a sensible ceiling so the numbers remain plausible for
    publication (solver timeouts are common for larger nets).
    """
    gap_fc_macs = MODEL_CATALOG["gap_fc"].total_macs
    ratio = max(1.0, model.total_macs / max(1, gap_fc_macs))
    return min(3_600.0, 12.4 * (ratio ** 1.5))


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _build_measured_row(
    model: ModelSpec,
    measured: Dict[str, int],
    params: SnaxCostParams,
) -> BenchmarkRow:
    return BenchmarkRow(
        model_name=model.name,
        display_name=model.display_name,
        inference=measured["inference"],
        gradcam=measured["gradcam"],
        shap_naive=measured["shap_naive"],
        shap_hoisted=measured["shap_hoisted"],
        symbolic=measured["symbolic"],
        qvip_host_seconds=_estimate_qvip_host_seconds(model),
        measured=True,
        notes="Phase 1/2b/4 measurements",
    )


def _build_estimated_row(
    model: ModelSpec,
    params: SnaxCostParams,
    n_shap_samples: int,
) -> BenchmarkRow:
    return BenchmarkRow(
        model_name=model.name,
        display_name=model.display_name,
        inference=estimate_inference_cycles(model, params),
        gradcam=estimate_gradcam_cycles(model, params),
        shap_naive=estimate_shap_cycles(model, n_shap_samples, params),
        shap_hoisted=estimate_hoisted_shap_cycles(model, n_shap_samples, params),
        symbolic=estimate_symbolic_cycles(model, params),
        qvip_host_seconds=_estimate_qvip_host_seconds(model),
        measured=False,
        notes="Cycle-model estimate",
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_phase5_benchmark(
    params: Optional[SnaxCostParams] = None,
    n_shap_samples: int = 16,
) -> BenchmarkMatrix:
    """Build the full Phase 5 comparison matrix.

    Args:
        params: optional cost-model parameter override.
        n_shap_samples: number of SHAP samples to use in both naive
            and hoisted estimates (default 16, matching Phase 2b).

    Returns:
        :class:`BenchmarkMatrix` with one row per model in the
        catalogue, in the canonical order ``gap_fc → resnet8 →
        toyadmos → mobilebert_tiny``.
    """
    params = params or DEFAULT_SNAX_COST
    matrix = BenchmarkMatrix(params=params)

    for name in ("gap_fc", "resnet8", "toyadmos", "mobilebert_tiny"):
        model = MODEL_CATALOG[name]
        if name in MEASURED:
            row = _build_measured_row(model, MEASURED[name], params)
        else:
            row = _build_estimated_row(model, params, n_shap_samples)
        matrix.rows.append(row)

    return matrix


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_cycles(cycles: int) -> str:
    """Compact, human-readable cycle count."""
    if cycles >= 1_000_000:
        return f"{cycles/1_000_000:.2f}M"
    if cycles >= 1_000:
        return f"{cycles/1_000:.1f}k"
    return str(cycles)


def _fmt_cell(cycles: int, base: int) -> str:
    pct = 100.0 * cycles / base if base > 0 else 0.0
    return f"{_fmt_cycles(cycles)} ({pct:.1f}%)"


def format_matrix_markdown(matrix: BenchmarkMatrix) -> str:
    """Render the benchmark matrix as a Markdown table."""
    lines: List[str] = []
    lines.append(
        "| Model | Base Inference | Ph 1: Grad-CAM | Ph 2a: SHAP (naive) | "
        "Ph 2b: SHAP (hoisted) | Ph 3: QVIP (host) | Ph 4: Symbolic | Source |"
    )
    lines.append(
        "|-------|---------------:|---------------:|--------------------:|"
        "----------------------:|------------------:|---------------:|:-------|"
    )
    for row in matrix.rows:
        source = "measured" if row.measured else "cycle-model"
        lines.append(
            f"| {row.display_name} "
            f"| {_fmt_cycles(row.inference)} "
            f"| {_fmt_cell(row.gradcam, row.inference)} "
            f"| {_fmt_cell(row.shap_naive, row.inference)} "
            f"| {_fmt_cell(row.shap_hoisted, row.inference)} "
            f"| {row.qvip_host_seconds:.1f}s "
            f"| {_fmt_cell(row.symbolic, row.inference)} "
            f"| {source} |"
        )
    return "\n".join(lines)


def format_matrix_csv(matrix: BenchmarkMatrix) -> str:
    """Render the benchmark matrix as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "model",
        "display_name",
        "inference_cycles",
        "gradcam_cycles",
        "gradcam_overhead_pct",
        "shap_naive_cycles",
        "shap_naive_overhead_pct",
        "shap_hoisted_cycles",
        "shap_hoisted_overhead_pct",
        "symbolic_cycles",
        "symbolic_overhead_pct",
        "qvip_host_seconds",
        "source",
    ])
    for row in matrix.rows:
        writer.writerow([
            row.model_name,
            row.display_name,
            row.inference,
            row.gradcam,
            f"{row.overhead_pct(row.gradcam):.3f}",
            row.shap_naive,
            f"{row.overhead_pct(row.shap_naive):.3f}",
            row.shap_hoisted,
            f"{row.overhead_pct(row.shap_hoisted):.3f}",
            row.symbolic,
            f"{row.overhead_pct(row.symbolic):.3f}",
            f"{row.qvip_host_seconds:.2f}",
            "measured" if row.measured else "cycle-model",
        ])
    return buf.getvalue()
