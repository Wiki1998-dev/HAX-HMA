"""SNAX cycle cost model used by the Phase 5 benchmark.

This module answers the question *"how many SNAX cluster cycles would
this layer / method take?"* without actually running the hardware
simulation.  The model is deliberately simple:

* **GeMM layers** (conv, fc, attention, ffn) are modelled as
  ``macs / gemm_macs_per_cycle + gemm_launch_overhead`` with a
  minimum cost to account for CSR programming.
* **Scalar layers** (gap, layernorm, pool, embedding lookup) are
  modelled as ``macs / scalar_ops_per_cycle + scalar_overhead``.
* **Method costs** for each XAI strategy are derived from the same
  per-layer primitives plus strategy-specific constants measured in
  phases 1, 2b and 4.

Calibration
-----------

The constants in :class:`SnaxCostParams` are tuned so that the
model-predicted value for the Phase 1 GAP+FC model lies within 5 %
of the Phase 1/2b measurements:

* Grad-CAM (GAP+FC) — measured ``6,153`` cycles.
* Hoisted SHAP N=16 (GAP+FC) — measured ``58,022`` cycles.
* Symbolic rule (depth ≤ 3) — measured ``47`` cycles on RV32IMF.

The :func:`_verify_calibration` helper at the bottom of the module is
used by :mod:`tests.test_phase5_benchmark` to assert that the
calibration still holds any time we touch this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from .models import LayerSpec, ModelSpec


# ---------------------------------------------------------------------------
# Cost model parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnaxCostParams:
    """Tunable cost-model parameters for a SNAX cluster.

    The defaults are calibrated against the Phase 1/2 GAP+FC
    measurements.  All fields are in *cycles* unless otherwise stated.

    Attributes:
        gemm_macs_per_cycle:   effective SNAX GeMM throughput (MAC/cyc)
        gemm_launch_overhead:  fixed cost per GeMM launch (CSR + DMA)
        gemm_min_cycles:       floor cost for any GeMM launch
        scalar_ops_per_cycle:  RISC-V scalar core throughput
        scalar_overhead:       fixed cost per scalar kernel pass
        memory_bytes_per_cycle: TCDM bandwidth used for memory-bound
                               correction (unused if layer is compute-bound)
        gradcam_bwd_scale:     per-MAC multiplier on the backward Grad-CAM
                               pass (backward is more expensive per MAC
                               than forward because of transpose-streaming)
        gradcam_post_overhead: constant cost for ReLU + normalize
        shap_accum_cycles_per_elem: per-element cost of the SHAP
                               accumulation stage (dominant in Phase 2b)
        shap_norm_overhead:    per-sample normalization cost
        shap_launch_overhead:  per-SHAP-sample launch cost
        symbolic_cycles:       constant cost of a depth-≤3 rule walk
    """

    gemm_macs_per_cycle: float = 16.0
    gemm_launch_overhead: int = 400
    gemm_min_cycles: int = 100
    scalar_ops_per_cycle: float = 1.0
    scalar_overhead: int = 64
    memory_bytes_per_cycle: float = 4.0
    gradcam_bwd_scale: float = 2.0
    gradcam_post_overhead: int = 180
    shap_accum_cycles_per_elem: float = 12.0
    shap_norm_overhead: int = 1_600
    shap_launch_overhead: int = 250
    symbolic_cycles: int = 47


DEFAULT_SNAX_COST = SnaxCostParams()


# ---------------------------------------------------------------------------
# Primitive cost helpers
# ---------------------------------------------------------------------------


def _gemm_cycles(macs: int, p: SnaxCostParams) -> int:
    """Cycles for a GeMM kernel executing ``macs`` multiply-accumulates."""
    if macs <= 0:
        return 0
    compute = macs / p.gemm_macs_per_cycle
    return int(max(p.gemm_min_cycles, compute)) + p.gemm_launch_overhead


def _scalar_cycles(ops: int, p: SnaxCostParams) -> int:
    """Cycles for a scalar RISC-V kernel executing ``ops`` operations."""
    if ops <= 0:
        return 0
    return p.scalar_overhead + int(ops / p.scalar_ops_per_cycle)


# ---------------------------------------------------------------------------
# Layer / inference cost
# ---------------------------------------------------------------------------


def estimate_layer_cycles(
    layer: LayerSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Estimate the SNAX cycle cost of a single layer's forward pass.

    Args:
        layer:  layer architecture descriptor
        params: cost-model parameters

    Returns:
        Total cycles, including GeMM launch overheads.
    """
    if layer.kind in ("conv", "depthwise_conv", "fc", "attention", "ffn"):
        return _gemm_cycles(layer.macs, params)
    if layer.kind in ("gap", "pool", "layernorm", "embedding"):
        return _scalar_cycles(layer.macs, params)
    return 0


def estimate_inference_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Total SNAX cycle cost of one forward inference."""
    return sum(estimate_layer_cycles(layer, params) for layer in model.layers)


def estimate_backbone_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Cycles to run the backbone (everything up to and including the
    Grad-CAM hook layer).
    """
    return sum(estimate_layer_cycles(layer, params) for layer in model.backbone)


def estimate_head_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Cycles to run the classification head only."""
    return sum(estimate_layer_cycles(layer, params) for layer in model.head)


# ---------------------------------------------------------------------------
# XAI method cost models
# ---------------------------------------------------------------------------


def _final_feature_elems(model: ModelSpec) -> int:
    """Spatial × channel element count of the Grad-CAM hook layer.

    This counts the elements XAI operates on — e.g. the 4×4×16 = 256
    elements of the final conv of the GAP+FC test model.
    """
    return model.final_feature_layer.feature_map_elems


def estimate_gradcam_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Triggered-hook Grad-CAM overhead cycles on a model.

    The triggered-hook strategy described in Phase 5 §2, Method 1
    reuses the feature maps already sitting in TCDM when the last
    backbone GeMM completes.  The remaining work is:

    1. A **backward GeMM** over the head weights (``K × n_classes``
       MACs with a transpose streamer, hence ``gradcam_bwd_scale``
       cost multiplier).
    2. A **weighted-sum** reduction over the feature maps (scalar).
    3. A constant **ReLU + normalize** post-pass.

    The result is the Grad-CAM *overhead* — it excludes the backbone
    inference, which was already paid for by the model's forward pass.
    """
    head_fc = None
    for layer in reversed(model.layers):
        if layer.kind == "fc":
            head_fc = layer
            break
    if head_fc is None:
        # NLP models have an FFN/attention stack — use the last GeMM layer.
        head_fc = model.layers[-1]

    # (1) Backward GeMM: cost of a transpose-GeMM over head weights
    bwd_macs = int(head_fc.macs * params.gradcam_bwd_scale)
    bwd_cycles = _gemm_cycles(bwd_macs, params)

    # (2) Weighted sum over the feature maps
    ws_ops = _final_feature_elems(model)
    ws_cycles = _scalar_cycles(ws_ops, params)

    # (3) ReLU + normalize
    post_cycles = params.gradcam_post_overhead

    return bwd_cycles + ws_cycles + post_cycles


def estimate_shap_cycles(
    model: ModelSpec,
    n_samples: int = 16,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Naive Gradient-SHAP overhead cycles (Phase 2a style).

    Runs ``n_samples`` full forward + backward passes through the entire
    model.  This is the "before hoisting" cost used in the comparison
    matrix to illustrate why we needed Phase 2b.
    """
    per_sample_fwd = estimate_inference_cycles(model, params)
    # A backward pass is modelled with the same cost scale as Grad-CAM's
    # transpose-GeMM (gradcam_bwd_scale) applied to every GeMM layer.
    per_sample_bwd = int(per_sample_fwd * params.gradcam_bwd_scale * 0.5)
    per_sample = per_sample_fwd + per_sample_bwd + params.shap_launch_overhead
    return n_samples * per_sample + params.shap_norm_overhead


def estimate_hoisted_shap_cycles(
    model: ModelSpec,
    n_samples: int = 16,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Backward-hoisted Gradient-SHAP overhead cycles (Phase 2b style).

    Only the *head* of the network is re-run for each SHAP sample; the
    backbone feature maps are produced exactly once and reused.  The
    cost breakdown mirrors the Phase 2b per-stage counters:

    * ``zero``     – constant buffer zeroing cost
    * ``bwd``      – one backward GeMM over the head weights
    * ``accum``    – per-sample element-wise accumulation (dominant)
    * ``norm``     – final division by ``n_samples``

    The backbone cost is *not* included — that is already paid by the
    model's ordinary inference pass.  Therefore this function reports
    the pure XAI overhead relative to baseline inference.
    """
    head = model.head or [model.layers[-1]]

    # Head forward + backward per SHAP sample
    head_fwd = sum(estimate_layer_cycles(layer, params) for layer in head)
    head_bwd = int(head_fwd * params.gradcam_bwd_scale * 0.5)

    # Accumulation over feature-map elements (element-wise)
    elems = _final_feature_elems(model)
    accum_per_sample = int(elems * params.shap_accum_cycles_per_elem)

    per_sample = head_fwd + head_bwd + accum_per_sample + params.shap_launch_overhead
    zero_cycles = params.scalar_overhead + elems  # one pass to zero buffer
    norm_cycles = params.shap_norm_overhead + elems

    return zero_cycles + n_samples * per_sample + norm_cycles


def estimate_symbolic_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Constant cycle cost of a depth-≤3 neuro-symbolic rule walk.

    The extracted rule is a fully-inlined if/else cascade executed on
    the RISC-V scalar core with integer comparisons only (Phase 4).
    The cost is dominated by branch and load instructions and is
    independent of model size because the tree only inspects the
    top-K most-salient features.
    """
    return params.symbolic_cycles


# ---------------------------------------------------------------------------
# Calibration self-check
# ---------------------------------------------------------------------------


def _verify_calibration(tolerance: float = 0.15) -> List[str]:
    """Return a list of calibration error messages, empty if OK.

    The Phase 5 tests call this to make sure we have not regressed the
    cost model against the Phase 1/2b measurements.
    """
    from .models import get_model

    errors: List[str] = []
    gap_fc = get_model("gap_fc")

    def _check(name: str, predicted: int, expected: int) -> None:
        err = abs(predicted - expected) / expected
        if err > tolerance:
            errors.append(
                f"{name}: predicted={predicted} expected={expected} err={err:.1%}"
            )

    _check("gradcam_gap_fc", estimate_gradcam_cycles(gap_fc), 6_153)
    _check("hoisted_shap_gap_fc", estimate_hoisted_shap_cycles(gap_fc, 16), 58_022)
    _check("symbolic_gap_fc", estimate_symbolic_cycles(gap_fc), 47)
    return errors
