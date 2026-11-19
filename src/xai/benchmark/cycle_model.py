"""SNAX cycle cost model used by the Phase 5 benchmark.

This module answers the question *"how many SNAX cluster cycles would
this layer / method take?"* without actually running the hardware
simulation.  The model is deliberately simple:

* **GeMM layers** (conv, fc, attention, ffn) are modelled as
  ``macs / gemm_macs_per_cycle + gemm_launch_overhead`` with a
  minimum cost to account for CSR programming.
* **Scalar layers** (gap, layernorm, pool, embedding lookup) are
  modelled as ``macs / scalar_ops_per_cycle + scalar_overhead``.
* **XAI method costs** are modelled on the RISC-V *scalar* core, because
  that is where the measured Phase 1/2 kernels run (no XAI kernel uses
  the GeMM accelerator yet).  Each method is a sum of per-stage costs
  whose structure mirrors the measured C kernels.

Calibration
-----------

The per-element / per-MAC constants in :class:`SnaxCostParams` are fitted
to the per-stage Verilator counters of the GAP+FC model (4x4x16, C=10):

* Grad-CAM — measured ``4,757`` cycles (2026-11-19, after fixing an
  out-of-bounds read in the GAP step; the originally published 6,153
  included ~1,400 cycles of that bug).
* Naive Gradient SHAP N=16 (Phase 2a) — measured ``175,510`` cycles.
* Hoisted Gradient SHAP N=16 (Phase 2b) — measured ``58,022`` cycles.
* Symbolic rule (depth <= 3) — measured ``47`` cycles.

:func:`_verify_calibration` asserts all four stay within tolerance.

Only the GAP+FC numbers are calibrated.  Inference cycles (GeMM path) and
every number for ResNet-8 / ToyAdmos / MobileBERT-tiny are *estimates*:
they extrapolate the per-element constants to larger shapes and have not
been checked against simulation.
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
                               [inference only, uncalibrated]
        gemm_launch_overhead:  fixed cost per GeMM launch (CSR + DMA)
        gemm_min_cycles:       floor cost for any GeMM launch
        scalar_ops_per_cycle:  RISC-V scalar throughput for inference-side
                               scalar layers [uncalibrated]
        scalar_overhead:       fixed cost per scalar kernel pass
        memory_bytes_per_cycle: TCDM bandwidth (currently unused)
        scalar_mac_cycles:     fp32 MAC (load-load-fmadd) on the scalar
                               core; fits Grad-CAM transpose/gemm/wsum and
                               SHAP backward (~7.3-7.9 measured)
        gradcam_gap_cycles_per_chan: per-channel cost of the alpha step
        gradcam_post_cycles_per_px:  ReLU + normalize per spatial position
        shap_zero_cycles_per_elem:   accumulator clear
        shap_interp_cycles_per_elem: x' + a(x - x') per element (2a only)
        shap_gap_cycles_per_op:      GAP reduction in the head forward (2a)
        shap_accum_cycles_per_elem:  (x - x') * grad per element, per sample
        shap_norm_cycles_per_elem:   final division by N
        symbolic_cycles:       constant cost of a depth-<=3 rule walk
    """

    gemm_macs_per_cycle: float = 16.0
    gemm_launch_overhead: int = 400
    gemm_min_cycles: int = 100
    scalar_ops_per_cycle: float = 1.0
    scalar_overhead: int = 64
    memory_bytes_per_cycle: float = 4.0
    scalar_mac_cycles: float = 7.5
    gradcam_gap_cycles_per_chan: float = 3.0
    gradcam_post_cycles_per_px: float = 20.0
    shap_zero_cycles_per_elem: float = 4.0
    shap_interp_cycles_per_elem: float = 12.0
    shap_gap_cycles_per_op: float = 5.0
    shap_accum_cycles_per_elem: float = 13.0
    shap_norm_cycles_per_elem: float = 7.0
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
    """Spatial x channel element count of the Grad-CAM hook layer.

    This counts the elements XAI operates on — e.g. the 4x4x16 = 256
    elements of the final conv of the GAP+FC test model.
    """
    return model.final_feature_layer.feature_map_elems


_GEMM_KINDS = ("conv", "depthwise_conv", "fc", "attention", "ffn")


def _head_layers(model: ModelSpec) -> List[LayerSpec]:
    return list(model.head) or [model.layers[-1]]


def _head_gemm_macs(model: ModelSpec) -> int:
    """MACs of every GeMM-type layer in the classification head."""
    return sum(l.macs for l in _head_layers(model) if l.kind in _GEMM_KINDS)


def _head_gap_ops(model: ModelSpec) -> int:
    return sum(l.macs for l in _head_layers(model) if l.kind in ("gap", "pool"))


def head_is_linear(model: ModelSpec) -> bool:
    """True if the head has at most one GeMM layer (e.g. GAP + FC).

    Backward hoisting (Phase 2b) is exact only for such heads; with
    several GeMM layers there are non-linearities between them, so the
    gradient must be recomputed per SHAP sample.
    """
    return sum(1 for l in _head_layers(model) if l.kind in _GEMM_KINDS) <= 1


def estimate_gradcam_cycles(
    model: ModelSpec,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Grad-CAM overhead cycles on the RISC-V scalar core.

    Mirrors the measured kernel (``snax_cluster/sw/xai/gradcam``) stage by
    stage; the backbone forward pass is excluded (already paid by
    inference):

    1. transpose the head weights            — scalar_mac_cycles x head MACs
    2. backward GeMM through the head        — scalar_mac_cycles x head MACs
    3. alpha (GAP of the gradient)           — overhead + per-channel cost
    4. weighted sum over the feature maps    — scalar_mac_cycles x E
    5. ReLU + normalize                      — per spatial position
    """
    p = params
    elems = _final_feature_elems(model)
    k = max(1, _head_layers(model)[0].c_in)
    hw = max(1, elems // k)
    head_macs = _head_gemm_macs(model)

    transpose = p.scalar_mac_cycles * head_macs
    bwd = p.scalar_mac_cycles * head_macs
    gap = p.scalar_overhead + p.gradcam_gap_cycles_per_chan * k
    wsum = p.scalar_mac_cycles * elems
    post = p.gradcam_post_cycles_per_px * hw
    return int(transpose + bwd + gap + wsum + post)


def _shap_head_fwd(model: ModelSpec, p: SnaxCostParams) -> float:
    return (p.scalar_mac_cycles * _head_gemm_macs(model)
            + p.shap_gap_cycles_per_op * _head_gap_ops(model))


def _shap_head_bwd(model: ModelSpec, p: SnaxCostParams) -> float:
    """Gradient w.r.t. the hook features: broadcast over E elements, plus a
    backward pass through any extra head GeMM layers (non-linear heads)."""
    bwd = p.scalar_mac_cycles * _final_feature_elems(model)
    if not head_is_linear(model):
        bwd += p.scalar_mac_cycles * _head_gemm_macs(model)
    return bwd


def estimate_shap_cycles(
    model: ModelSpec,
    n_samples: int = 16,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Naive Gradient-SHAP overhead cycles (Phase 2a structure, scalar core).

    Attribution is over the hook-layer features (as in the measured
    kernel), so each sample re-runs the head, not the backbone:
    ``zero + N x (interp + head_fwd + head_bwd + accum) + norm``.
    """
    p = params
    e = _final_feature_elems(model)
    per_sample = (p.shap_interp_cycles_per_elem * e
                  + _shap_head_fwd(model, p)
                  + _shap_head_bwd(model, p)
                  + p.shap_accum_cycles_per_elem * e)
    return int(p.shap_zero_cycles_per_elem * e
               + n_samples * per_sample
               + p.shap_norm_cycles_per_elem * e)


def estimate_hoisted_shap_cycles(
    model: ModelSpec,
    n_samples: int = 16,
    params: SnaxCostParams = DEFAULT_SNAX_COST,
) -> int:
    """Backward-hoisted Gradient-SHAP cycles (Phase 2b structure, scalar core).

    For a linear head the gradient is constant along the integration path,
    so it is computed once and only the accumulation runs per sample:
    ``zero + bwd + N x accum + norm`` (matches the 2b per-stage counters).

    For a non-linear head hoisting is not exact, so this falls back to the
    naive cost (see :func:`head_is_linear`).
    """
    if not head_is_linear(model):
        return estimate_shap_cycles(model, n_samples, params)
    p = params
    e = _final_feature_elems(model)
    return int(p.shap_zero_cycles_per_elem * e
               + _shap_head_bwd(model, p)
               + n_samples * p.shap_accum_cycles_per_elem * e
               + p.shap_norm_cycles_per_elem * e)


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


#: Verilator measurements on the GAP+FC model (scalar core, fp32, N=16).
MEASURED_GAP_FC = {
    "gradcam": 4_757,
    "shap_naive": 175_510,
    "shap_hoisted": 58_022,
    "symbolic": 47,
}


def _verify_calibration(tolerance: float = 0.05) -> List[str]:
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

    _check("gradcam_gap_fc", estimate_gradcam_cycles(gap_fc), MEASURED_GAP_FC["gradcam"])
    _check("naive_shap_gap_fc", estimate_shap_cycles(gap_fc, 16), MEASURED_GAP_FC["shap_naive"])
    _check("hoisted_shap_gap_fc", estimate_hoisted_shap_cycles(gap_fc, 16),
           MEASURED_GAP_FC["shap_hoisted"])
    _check("symbolic_gap_fc", estimate_symbolic_cycles(gap_fc), MEASURED_GAP_FC["symbolic"])
    return errors
