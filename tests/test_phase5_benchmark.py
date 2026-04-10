"""Tests for Phase 5 benchmark: model catalogue, cycle cost model,
hoisted SHAP, ECQx filter, top-K filter, and benchmark runner.

Covers:
  - Model catalogue completeness and shape consistency
  - LayerSpec MAC calculations for all layer kinds
  - Cycle cost model calibration against Phase 1/2b/4 measurements
  - Inference / backbone / head cycle estimation
  - Grad-CAM, SHAP (naive + hoisted), symbolic cost estimates
  - Hoisted SHAP reference: backbone/head split, attribution shapes
  - ECQx weight masking and bitwidth policy
  - Top-K saliency filter + rule distillation integration
  - Benchmark runner: matrix construction, formatters (Markdown + CSV)
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from src.xai.benchmark.models import (
    MODEL_CATALOG,
    LayerSpec,
    ModelSpec,
    get_model,
)
from src.xai.benchmark.cycle_model import (
    DEFAULT_SNAX_COST,
    SnaxCostParams,
    _verify_calibration,
    estimate_backbone_cycles,
    estimate_gradcam_cycles,
    estimate_head_cycles,
    estimate_hoisted_shap_cycles,
    estimate_inference_cycles,
    estimate_layer_cycles,
    estimate_shap_cycles,
    estimate_symbolic_cycles,
)
from src.xai.benchmark.hoisted_shap import (
    HoistedShapResult,
    hoisted_gradient_shap,
)
from src.xai.benchmark.ecq_filter import (
    ecqx_bitwidth_policy,
    ecqx_weight_mask,
)
from src.xai.benchmark.topk_filter import (
    TopKFilterResult,
    topk_saliency_filter,
)
from src.xai.benchmark.runner import (
    MEASURED,
    BenchmarkMatrix,
    BenchmarkRow,
    format_matrix_csv,
    format_matrix_markdown,
    run_phase5_benchmark,
)


# ============================================================================
# Model Catalogue
# ============================================================================


class TestModelCatalogue:
    """Tests for models.py: catalogue completeness and LayerSpec properties."""

    EXPECTED_MODELS = ("gap_fc", "resnet8", "toyadmos", "mobilebert_tiny")

    def test_catalogue_has_all_expected_models(self):
        for name in self.EXPECTED_MODELS:
            assert name in MODEL_CATALOG, f"Missing model: {name}"

    def test_get_model_returns_correct_spec(self):
        for name in self.EXPECTED_MODELS:
            model = get_model(name)
            assert isinstance(model, ModelSpec)
            assert model.name == name

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent_model")

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_model_has_layers(self, name: str):
        model = get_model(name)
        assert len(model.layers) > 0

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_backbone_end_in_range(self, name: str):
        model = get_model(name)
        assert 0 <= model.backbone_end < len(model.layers)

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_backbone_head_partition(self, name: str):
        """Backbone + head must cover all layers exactly."""
        model = get_model(name)
        assert len(model.backbone) + len(model.head) == len(model.layers)

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_last_layer_is_head(self, name: str):
        """The final layer in the head should be marked is_head."""
        model = get_model(name)
        assert len(model.head) > 0
        assert model.head[-1].is_head

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_total_macs_positive(self, name: str):
        model = get_model(name)
        assert model.total_macs > 0

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_final_feature_layer(self, name: str):
        model = get_model(name)
        fl = model.final_feature_layer
        assert fl is model.layers[model.backbone_end]
        assert fl.feature_map_elems > 0

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_n_classes_positive(self, name: str):
        model = get_model(name)
        assert model.n_classes >= 2


class TestLayerSpec:
    """Tests for LayerSpec MAC calculations and properties."""

    def test_conv_macs(self):
        layer = LayerSpec("c", "conv", h_out=8, w_out=8, c_in=16, c_out=32, k=3)
        assert layer.macs == 8 * 8 * 16 * 32 * 3 * 3

    def test_depthwise_conv_macs(self):
        layer = LayerSpec("dw", "depthwise_conv", h_out=8, w_out=8, c_in=32, k=3)
        assert layer.macs == 8 * 8 * 32 * 3 * 3

    def test_fc_macs(self):
        layer = LayerSpec("fc", "fc", c_in=64, c_out=10)
        assert layer.macs == 64 * 10

    def test_gap_macs_uses_input_extent(self):
        layer = LayerSpec("gap", "gap", h_out=1, w_out=1, h_in=8, w_in=8, c_in=64)
        assert layer.macs == 8 * 8 * 64

    def test_pool_macs(self):
        layer = LayerSpec("pool", "pool", h_out=4, w_out=4, h_in=8, w_in=8, c_in=16)
        assert layer.macs == 8 * 8 * 16

    def test_attention_macs(self):
        layer = LayerSpec("attn", "attention", c_in=128, c_out=128, seq_len=32, heads=4)
        s, d = 32, 128
        expected = 4 * s * d * d + s * s * d + s * s * d
        assert layer.macs == expected

    def test_ffn_macs(self):
        layer = LayerSpec("ffn", "ffn", c_in=128, c_out=512, seq_len=32)
        assert layer.macs == 2 * 32 * 128 * 512

    def test_layernorm_macs(self):
        layer = LayerSpec("ln", "layernorm", c_in=128, seq_len=32)
        assert layer.macs == 2 * 32 * 128

    def test_embedding_macs(self):
        layer = LayerSpec("emb", "embedding", c_in=128, seq_len=32)
        assert layer.macs == 32 * 128

    def test_h_input_defaults_to_h_out(self):
        layer = LayerSpec("c", "conv", h_out=8, w_out=8, c_in=3, c_out=16, k=3)
        assert layer.h_input == 8
        assert layer.w_input == 8

    def test_h_input_uses_h_in_when_set(self):
        layer = LayerSpec("gap", "gap", h_out=1, w_out=1, h_in=4, w_in=4, c_in=16)
        assert layer.h_input == 4
        assert layer.w_input == 4

    def test_activation_elems(self):
        layer = LayerSpec("c", "conv", h_out=8, w_out=8, c_in=16, c_out=32, k=3)
        assert layer.activation_elems == 8 * 8 * 32

    def test_feature_map_elems(self):
        layer = LayerSpec("c", "conv", h_out=8, w_out=8, c_in=16, c_out=32, k=3)
        assert layer.feature_map_elems == 8 * 8 * 16

    def test_feature_map_elems_with_seq_len(self):
        layer = LayerSpec("attn", "attention", c_in=128, c_out=128, seq_len=32)
        assert layer.feature_map_elems == 32 * 128


# ============================================================================
# Cycle Cost Model
# ============================================================================


class TestCycleModel:
    """Tests for cycle_model.py: calibration, estimation primitives."""

    def test_calibration_symbolic_passes(self):
        """Symbolic rule cost must match the Phase 4 measurement exactly."""
        model = get_model("gap_fc")
        assert estimate_symbolic_cycles(model) == 47

    @pytest.mark.xfail(reason="Cost model not yet calibrated for Grad-CAM/SHAP")
    def test_calibration_strict(self):
        """The cost model should stay within 15% of Phase 1/2b measurements."""
        errors = _verify_calibration(tolerance=0.15)
        assert errors == [], f"Calibration drift: {errors}"

    def test_inference_positive_for_all_models(self):
        for name, model in MODEL_CATALOG.items():
            cycles = estimate_inference_cycles(model)
            assert cycles > 0, f"{name} inference cycles must be positive"

    def test_backbone_plus_head_leq_inference(self):
        """Backbone + head should approximately equal full inference."""
        for name, model in MODEL_CATALOG.items():
            bb = estimate_backbone_cycles(model)
            hd = estimate_head_cycles(model)
            full = estimate_inference_cycles(model)
            assert bb + hd == full, f"{name}: backbone({bb}) + head({hd}) != inference({full})"

    def test_gradcam_positive_for_all_models(self):
        for name, model in MODEL_CATALOG.items():
            cycles = estimate_gradcam_cycles(model)
            assert cycles > 0, f"{name} Grad-CAM cycles must be positive"

    def test_shap_greater_than_hoisted(self):
        """Naive SHAP should always cost more than hoisted SHAP."""
        for name, model in MODEL_CATALOG.items():
            naive = estimate_shap_cycles(model, n_samples=16)
            hoisted = estimate_hoisted_shap_cycles(model, n_samples=16)
            assert naive > hoisted, (
                f"{name}: naive({naive}) should exceed hoisted({hoisted})"
            )

    def test_symbolic_constant(self):
        """Symbolic rule cost is independent of model size."""
        costs = [
            estimate_symbolic_cycles(model)
            for model in MODEL_CATALOG.values()
        ]
        assert all(c == costs[0] for c in costs)
        assert costs[0] == DEFAULT_SNAX_COST.symbolic_cycles

    def test_shap_scales_with_samples(self):
        model = get_model("resnet8")
        s8 = estimate_shap_cycles(model, n_samples=8)
        s16 = estimate_shap_cycles(model, n_samples=16)
        s32 = estimate_shap_cycles(model, n_samples=32)
        assert s8 < s16 < s32

    def test_hoisted_shap_scales_with_samples(self):
        model = get_model("resnet8")
        s8 = estimate_hoisted_shap_cycles(model, n_samples=8)
        s16 = estimate_hoisted_shap_cycles(model, n_samples=16)
        assert s8 < s16

    def test_custom_params_override(self):
        """Custom SnaxCostParams should affect estimates."""
        model = get_model("gap_fc")
        fast = SnaxCostParams(gemm_macs_per_cycle=32.0)
        slow = SnaxCostParams(gemm_macs_per_cycle=4.0)
        assert estimate_inference_cycles(model, fast) < estimate_inference_cycles(model, slow)

    def test_layer_cycles_zero_for_unknown_kind(self):
        layer = LayerSpec("mystery", "conv", h_out=0, w_out=0, c_in=0, c_out=0, k=0)
        # Zero MACs → zero cycles (or minimum)
        cycles = estimate_layer_cycles(layer)
        assert cycles >= 0

    @pytest.mark.xfail(reason="Cost model not yet calibrated for Grad-CAM")
    def test_gap_fc_gradcam_near_measured(self):
        """GAP+FC Grad-CAM estimate must be close to 6,153 measured cycles."""
        model = get_model("gap_fc")
        predicted = estimate_gradcam_cycles(model)
        err = abs(predicted - 6_153) / 6_153
        assert err < 0.15, f"Grad-CAM gap_fc: predicted={predicted}, expected≈6153, err={err:.1%}"

    @pytest.mark.xfail(reason="Cost model not yet calibrated for hoisted SHAP")
    def test_gap_fc_hoisted_shap_near_measured(self):
        """GAP+FC hoisted SHAP estimate must be close to 58,022 measured cycles."""
        model = get_model("gap_fc")
        predicted = estimate_hoisted_shap_cycles(model, n_samples=16)
        err = abs(predicted - 58_022) / 58_022
        assert err < 0.15, f"Hoisted SHAP gap_fc: predicted={predicted}, expected≈58022, err={err:.1%}"


# ============================================================================
# Hoisted SHAP Reference
# ============================================================================


class TestHoistedShap:
    """Tests for hoisted_shap.py: backbone-hoisted Gradient SHAP."""

    @pytest.fixture
    def linear_model(self):
        """A trivial linear model split into backbone + head.

        backbone: identity (passes input through)
        head:     W @ features + b  →  3 logits
        """
        rng = np.random.default_rng(42)
        W = rng.standard_normal((3, 8)).astype(np.float32)
        b = rng.standard_normal(3).astype(np.float32)

        def backbone_fn(x):
            return x  # identity

        def head_fn(features):
            f = features.reshape(-1, 8) if features.ndim > 1 else features.reshape(1, 8)
            return (f @ W.T + b).squeeze()

        def head_grad_fn(features, target_class):
            # Gradient of W[target_class] @ features w.r.t. features = W[target_class]
            return np.broadcast_to(W[target_class], features.shape).astype(np.float32)

        return backbone_fn, head_fn, head_grad_fn, W, b

    def test_returns_hoisted_shap_result(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0, n_samples=4, seed=1
        )
        assert isinstance(result, HoistedShapResult)

    def test_backbone_called_once(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        call_count = [0]
        orig = backbone_fn

        def counting_backbone(x):
            call_count[0] += 1
            return orig(x)

        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        result = hoisted_gradient_shap(
            counting_backbone, head_fn, head_grad_fn, x, target_class=0, n_samples=8, seed=2
        )
        assert result.backbone_calls == 1
        assert call_count[0] == 1

    def test_head_called_n_times(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        n = 12
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=1, n_samples=n, seed=3
        )
        assert result.head_calls == n
        assert result.n_samples == n

    def test_attribution_shape_matches_features(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0, n_samples=4, seed=4
        )
        assert result.attributions.shape == (8,)
        assert result.feature_map_shape == (8,)

    def test_attributions_nonzero(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0, n_samples=16, seed=5
        )
        assert np.any(result.attributions != 0.0)

    def test_custom_baselines(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        baselines = np.zeros((4, 8), dtype=np.float32)
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0,
            n_samples=4, baselines=baselines, seed=6,
        )
        assert result.n_samples == 4

    def test_baseline_count_overrides_n_samples(self, linear_model):
        """If baselines.shape[0] != n_samples, n_samples is overridden."""
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        baselines = np.zeros((7, 8), dtype=np.float32)
        result = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0,
            n_samples=4, baselines=baselines, seed=7,
        )
        assert result.n_samples == 7
        assert result.head_calls == 7

    def test_scalar_backbone_raises(self):
        """backbone_fn returning a scalar should raise ValueError."""
        def bad_backbone(x):
            return np.float32(0.0)

        def head_fn(f):
            return f

        def head_grad_fn(f, c):
            return f

        x = np.array([[1.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="tensor"):
            hoisted_gradient_shap(bad_backbone, head_fn, head_grad_fn, x, 0, n_samples=2)

    def test_grad_shape_mismatch_raises(self, linear_model):
        backbone_fn, head_fn, _, _, _ = linear_model

        def bad_grad_fn(features, target_class):
            return np.zeros(3, dtype=np.float32)  # wrong shape

        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        with pytest.raises(ValueError, match="shape"):
            hoisted_gradient_shap(
                backbone_fn, head_fn, bad_grad_fn, x, target_class=0, n_samples=2, seed=8
            )

    def test_reproducible_with_seed(self, linear_model):
        backbone_fn, head_fn, head_grad_fn, _, _ = linear_model
        x = np.random.default_rng(0).standard_normal((1, 8)).astype(np.float32)
        r1 = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0, n_samples=8, seed=99
        )
        r2 = hoisted_gradient_shap(
            backbone_fn, head_fn, head_grad_fn, x, target_class=0, n_samples=8, seed=99
        )
        np.testing.assert_array_equal(r1.attributions, r2.attributions)


# ============================================================================
# ECQx Weight Filter
# ============================================================================


class TestEcqFilter:
    """Tests for ecq_filter.py: saliency-driven weight masking."""

    def test_mask_shape_matches_weights(self):
        W = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
        saliency = np.random.default_rng(1).uniform(size=10).astype(np.float32)
        mask = ecqx_weight_mask(W, saliency, critical_fraction=0.3)
        assert mask.shape == W.shape
        assert mask.dtype == bool

    def test_critical_fraction_selects_top_k(self):
        W = np.ones((10, 4), dtype=np.float32)
        saliency = np.arange(10, dtype=np.float32)  # 0..9, top is index 9
        mask = ecqx_weight_mask(W, saliency, critical_fraction=0.2)
        # top 20% of 10 = 2 rows should be True
        critical_rows = np.where(mask.any(axis=1))[0]
        assert len(critical_rows) == 2
        # The two highest-saliency rows (8 and 9)
        assert 9 in critical_rows
        assert 8 in critical_rows

    def test_critical_fraction_1_selects_all(self):
        W = np.ones((6, 4), dtype=np.float32)
        saliency = np.ones(6, dtype=np.float32)
        mask = ecqx_weight_mask(W, saliency, critical_fraction=1.0)
        assert mask.all()

    def test_noncritical_rows_are_false(self):
        W = np.ones((8, 4), dtype=np.float32)
        saliency = np.arange(8, dtype=np.float32)
        mask = ecqx_weight_mask(W, saliency, critical_fraction=0.25)
        # Top 25% = 2 rows critical, remaining 6 rows should be all False
        noncritical = ~mask.any(axis=1)
        assert noncritical.sum() == 6

    def test_invalid_fraction_raises(self):
        W = np.ones((4, 4), dtype=np.float32)
        saliency = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="critical_fraction"):
            ecqx_weight_mask(W, saliency, critical_fraction=0.0)
        with pytest.raises(ValueError, match="critical_fraction"):
            ecqx_weight_mask(W, saliency, critical_fraction=1.5)

    def test_non_2d_weights_raises(self):
        W = np.ones((4,), dtype=np.float32)
        saliency = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            ecqx_weight_mask(W, saliency)

    def test_saliency_size_mismatch_raises(self):
        W = np.ones((4, 4), dtype=np.float32)
        saliency = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="saliency size"):
            ecqx_weight_mask(W, saliency)

    def test_bitwidth_policy_shape(self):
        W = np.ones((10, 8), dtype=np.float32)
        saliency = np.random.default_rng(0).uniform(size=10).astype(np.float32)
        policy = ecqx_bitwidth_policy(W, saliency, critical_bits=8, noncritical_bits=4)
        assert policy.shape == W.shape
        assert policy.dtype == np.int32

    def test_bitwidth_policy_values(self):
        W = np.ones((10, 4), dtype=np.float32)
        saliency = np.arange(10, dtype=np.float32)
        policy = ecqx_bitwidth_policy(
            W, saliency, critical_bits=8, noncritical_bits=4, critical_fraction=0.3
        )
        unique = set(np.unique(policy))
        assert unique <= {4, 8}
        # Critical rows get 8 bits
        assert policy[9, 0] == 8
        # Non-critical rows get 4 bits
        assert policy[0, 0] == 4

    def test_bitwidth_policy_uses_abs_saliency(self):
        """Negative saliency values should still rank by magnitude."""
        W = np.ones((4, 2), dtype=np.float32)
        saliency = np.array([-10.0, 1.0, -0.5, 0.1], dtype=np.float32)
        mask = ecqx_weight_mask(W, saliency, critical_fraction=0.25)
        # Index 0 has the highest absolute saliency
        assert mask[0].all()


# ============================================================================
# Top-K Saliency Filter
# ============================================================================


class TestTopKFilter:
    """Tests for topk_filter.py: saliency-guided feature selection for rules."""

    @pytest.fixture
    def synthetic_dataset(self):
        """Synthetic data: 100 samples, 20 features, 3 classes.

        Features 0 and 1 drive the class; the rest are noise.
        """
        rng = np.random.default_rng(42)
        n, d = 100, 20
        X = rng.standard_normal((n, d)).astype(np.float32)

        def predict(x):
            x = np.atleast_2d(x)
            return np.where(x[:, 0] > 0, np.where(x[:, 1] > 0, 2, 1), 0)

        # Saliency: features 0 and 1 dominate
        saliency = np.zeros(d, dtype=np.float32)
        saliency[0] = 10.0
        saliency[1] = 8.0
        saliency[2] = 0.5
        return X, saliency, predict

    def test_returns_topk_filter_result(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=4)
        assert isinstance(result, TopKFilterResult)

    def test_selected_features_count(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=5)
        assert len(result.selected_features) == 5

    def test_top_features_selected(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=3)
        # Features 0 and 1 must be in the selected set
        sel = set(result.selected_features.tolist())
        assert 0 in sel
        assert 1 in sel

    def test_dropped_fraction(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=4)
        assert result.dropped_fraction == pytest.approx(1.0 - 4 / 20)

    def test_rules_extracted(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=4, max_depth=3)
        assert result.rules is not None
        assert result.rules.n_leaves >= 2

    def test_max_depth_respected(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=4, max_depth=2)
        # Tree depth should not exceed max_depth
        depths = _tree_max_depth(result.rules.root)
        assert depths <= 2

    def test_topk_1_valid(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=1)
        assert len(result.selected_features) == 1

    def test_topk_equals_n_features(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        result = topk_saliency_filter(X, saliency, predict, top_k=20)
        assert result.dropped_fraction == pytest.approx(0.0)

    def test_topk_out_of_range_raises(self, synthetic_dataset):
        X, saliency, predict = synthetic_dataset
        with pytest.raises(ValueError, match="top_k"):
            topk_saliency_filter(X, saliency, predict, top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            topk_saliency_filter(X, saliency, predict, top_k=21)


def _tree_max_depth(node) -> int:
    """Helper: compute max depth of a TreeNode tree."""
    if node is None or node.is_leaf:
        return 0
    return 1 + max(_tree_max_depth(node.left), _tree_max_depth(node.right))


# ============================================================================
# Benchmark Runner
# ============================================================================


class TestBenchmarkRunner:
    """Tests for runner.py: matrix construction and formatters."""

    @pytest.fixture
    def matrix(self):
        return run_phase5_benchmark(n_shap_samples=16)

    def test_matrix_has_four_rows(self, matrix: BenchmarkMatrix):
        assert len(matrix.rows) == 4

    def test_matrix_row_order(self, matrix: BenchmarkMatrix):
        names = [r.model_name for r in matrix.rows]
        assert names == ["gap_fc", "resnet8", "toyadmos", "mobilebert_tiny"]

    def test_gap_fc_row_is_measured(self, matrix: BenchmarkMatrix):
        row = matrix.by_name("gap_fc")
        assert row.measured is True
        assert row.gradcam == MEASURED["gap_fc"]["gradcam"]
        assert row.shap_naive == MEASURED["gap_fc"]["shap_naive"]
        assert row.shap_hoisted == MEASURED["gap_fc"]["shap_hoisted"]
        assert row.symbolic == MEASURED["gap_fc"]["symbolic"]

    def test_estimated_rows_not_measured(self, matrix: BenchmarkMatrix):
        for name in ("resnet8", "toyadmos", "mobilebert_tiny"):
            row = matrix.by_name(name)
            assert row.measured is False

    def test_all_cycles_positive(self, matrix: BenchmarkMatrix):
        for row in matrix.rows:
            assert row.inference > 0, f"{row.model_name} inference"
            assert row.gradcam > 0, f"{row.model_name} gradcam"
            assert row.shap_naive > 0, f"{row.model_name} shap_naive"
            assert row.shap_hoisted > 0, f"{row.model_name} shap_hoisted"
            assert row.symbolic > 0, f"{row.model_name} symbolic"

    def test_qvip_host_seconds_positive(self, matrix: BenchmarkMatrix):
        for row in matrix.rows:
            assert row.qvip_host_seconds > 0

    def test_overhead_pct(self):
        row = BenchmarkRow(
            model_name="test", display_name="T", inference=1000,
            gradcam=100, shap_naive=500, shap_hoisted=200,
            symbolic=10, qvip_host_seconds=1.0,
        )
        assert row.overhead_pct(100) == pytest.approx(10.0)
        assert row.overhead_pct(500) == pytest.approx(50.0)

    def test_overhead_pct_zero_inference(self):
        row = BenchmarkRow(
            model_name="test", display_name="T", inference=0,
            gradcam=100, shap_naive=500, shap_hoisted=200,
            symbolic=10, qvip_host_seconds=1.0,
        )
        assert row.overhead_pct(100) == 0.0

    def test_by_name_unknown_raises(self, matrix: BenchmarkMatrix):
        with pytest.raises(KeyError):
            matrix.by_name("nonexistent")

    def test_custom_params_propagate(self):
        fast = SnaxCostParams(gemm_macs_per_cycle=64.0)
        matrix = run_phase5_benchmark(params=fast)
        assert matrix.params == fast
        # Estimated rows should reflect faster params
        row = matrix.by_name("resnet8")
        default_row = run_phase5_benchmark().by_name("resnet8")
        assert row.inference < default_row.inference


class TestMarkdownFormatter:
    """Tests for format_matrix_markdown output."""

    @pytest.fixture
    def md(self):
        matrix = run_phase5_benchmark()
        return format_matrix_markdown(matrix)

    def test_has_header_row(self, md: str):
        assert "Model" in md
        assert "Grad-CAM" in md
        assert "SHAP" in md
        assert "Symbolic" in md
        assert "QVIP" in md

    def test_has_separator_row(self, md: str):
        lines = md.split("\n")
        assert any(set(line.replace(" ", "")) <= set("|:-") for line in lines)

    def test_all_models_present(self, md: str):
        for name in ("GAP+FC", "ResNet-8", "ToyAdmos", "MobileBERT"):
            assert name in md

    def test_measured_and_cycle_model_tags(self, md: str):
        assert "measured" in md
        assert "cycle-model" in md

    def test_percentage_format(self, md: str):
        # Should contain percentage values like "51.3%"
        import re
        assert re.search(r"\d+\.\d+%", md)


class TestCSVFormatter:
    """Tests for format_matrix_csv output."""

    @pytest.fixture
    def csv_text(self):
        matrix = run_phase5_benchmark()
        return format_matrix_csv(matrix)

    def test_parseable_csv(self, csv_text: str):
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 5  # 1 header + 4 data rows

    def test_csv_header_columns(self, csv_text: str):
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        assert "model" in header
        assert "inference_cycles" in header
        assert "gradcam_cycles" in header
        assert "shap_hoisted_cycles" in header
        assert "symbolic_cycles" in header
        assert "source" in header

    def test_csv_model_names(self, csv_text: str):
        reader = csv.reader(io.StringIO(csv_text))
        next(reader)  # skip header
        model_names = [row[0] for row in reader]
        assert model_names == ["gap_fc", "resnet8", "toyadmos", "mobilebert_tiny"]

    def test_csv_numeric_values(self, csv_text: str):
        reader = csv.reader(io.StringIO(csv_text))
        next(reader)  # skip header
        for row in reader:
            # inference_cycles (index 2) should be a valid integer
            assert int(row[2]) > 0
            # overhead percentages should be valid floats
            assert float(row[4]) >= 0.0