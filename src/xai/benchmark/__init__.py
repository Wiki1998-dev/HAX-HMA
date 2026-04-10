"""Phase 5: full-matrix benchmark & scaling analysis.

Exposes the model catalogue, SNAX cycle cost model, scaling strategies,
and the end-to-end benchmark runner that produces the Phase 5 comparison
matrix across four XAI methods and four model architectures.
"""

from .models import (
    LayerSpec,
    ModelSpec,
    MODEL_CATALOG,
    get_model,
)
from .cycle_model import (
    SnaxCostParams,
    DEFAULT_SNAX_COST,
    estimate_layer_cycles,
    estimate_inference_cycles,
    estimate_backbone_cycles,
    estimate_head_cycles,
    estimate_gradcam_cycles,
    estimate_shap_cycles,
    estimate_hoisted_shap_cycles,
    estimate_symbolic_cycles,
)
from .hoisted_shap import hoisted_gradient_shap, HoistedShapResult
from .ecq_filter import ecqx_weight_mask, ecqx_bitwidth_policy
from .topk_filter import topk_saliency_filter
from .runner import (
    BenchmarkRow,
    BenchmarkMatrix,
    run_phase5_benchmark,
    format_matrix_markdown,
    format_matrix_csv,
)
from .model_runners import (
    ModelRunner,
    build_runner,
    build_gap_fc,
    build_resnet8,
    build_toyadmos,
    build_mobilebert_tiny,
    RUNNER_FACTORIES,
)

__all__ = [
    "LayerSpec",
    "ModelSpec",
    "MODEL_CATALOG",
    "get_model",
    "SnaxCostParams",
    "DEFAULT_SNAX_COST",
    "estimate_layer_cycles",
    "estimate_inference_cycles",
    "estimate_backbone_cycles",
    "estimate_head_cycles",
    "estimate_gradcam_cycles",
    "estimate_shap_cycles",
    "estimate_hoisted_shap_cycles",
    "estimate_symbolic_cycles",
    "hoisted_gradient_shap",
    "HoistedShapResult",
    "ecqx_weight_mask",
    "ecqx_bitwidth_policy",
    "topk_saliency_filter",
    "BenchmarkRow",
    "BenchmarkMatrix",
    "run_phase5_benchmark",
    "format_matrix_markdown",
    "format_matrix_csv",
    "ModelRunner",
    "build_runner",
    "build_gap_fc",
    "build_resnet8",
    "build_toyadmos",
    "build_mobilebert_tiny",
    "RUNNER_FACTORIES",
]
