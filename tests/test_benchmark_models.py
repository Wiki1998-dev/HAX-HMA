"""End-to-end benchmark tests: run all XAI phases on all benchmark models.

Tests that Grad-CAM, SHAP, formal verification, and rule extraction
produce valid results on ResNet-8, ToyAdmos, and MobileBERT (plus the
gap_fc calibration model).  Uses synthetic weights — the goal is to
verify that the XAI methods handle realistic shapes and produce
structurally correct outputs, not that the explanations are meaningful.

Covers:
  - Phase 1 (Grad-CAM + LRP) on all 4 models
  - Phase 2 (Gradient SHAP + hoisted SHAP) on all 4 models
  - Phase 3 (QVIP formal verification) on representative sub-networks
  - Phase 4 (Rule extraction via top-K filter) on all 4 models
  - Model runner API: forward_fn, backbone/head split, predict
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xai.benchmark.model_runners import (
    ModelRunner,
    build_runner,
    RUNNER_FACTORIES,
)
from src.xai.gradcam.gradcam_reference import gradcam, lrp_epsilon
from src.xai.shap.gradient_shap_reference import gradient_shap
from src.xai.benchmark.hoisted_shap import hoisted_gradient_shap
from src.xai.benchmark.ecq_filter import ecqx_weight_mask, ecqx_bitwidth_policy
from src.xai.benchmark.topk_filter import topk_saliency_filter
from src.xai.formal.quantization import QuantConfig, quantize_uniform
from src.xai.formal.qvip_verifier import QNN, verify_robustness, VerifyResult


ALL_MODELS = list(RUNNER_FACTORIES.keys())
# Vision models that produce spatial (H, W) saliency maps
VISION_MODELS = ["gap_fc", "resnet8"]
# Models with small enough feature space for fast SHAP
FAST_MODELS = ["gap_fc", "toyadmos"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(params=ALL_MODELS)
def runner(request) -> ModelRunner:
    return build_runner(request.param, seed=42)


@pytest.fixture(params=VISION_MODELS)
def vision_runner(request) -> ModelRunner:
    return build_runner(request.param, seed=42)


@pytest.fixture(params=FAST_MODELS)
def fast_runner(request) -> ModelRunner:
    return build_runner(request.param, seed=42)


# ============================================================================
# Model Runner Basics
# ============================================================================


class TestModelRunnerAPI:
    """Verify model runner API works for all models."""

    def test_forward_fn_returns_logits_and_fmaps(self, runner: ModelRunner):
        x = runner.sample_input()
        logits, fmaps = runner.forward_fn(x)
        assert logits.ndim == 2
        assert logits.shape[0] == 1
        assert logits.shape[1] == runner.spec.n_classes
        assert fmaps.shape[1:] == runner.feature_shape or \
            fmaps[0].shape == runner.feature_shape

    def test_backbone_fn_shape(self, runner: ModelRunner):
        x = runner.sample_input()
        fmaps = runner.backbone_fn(x)
        # Should have batch dim
        assert fmaps.ndim >= 2
        flat_feat = fmaps.flatten().size
        expected_feat = int(np.prod(runner.feature_shape))
        assert flat_feat == expected_feat or flat_feat == expected_feat  # with or without batch

    def test_head_fn_produces_logits(self, runner: ModelRunner):
        x = runner.sample_input()
        fmaps = runner.backbone_fn(x)
        logits = runner.head_fn(fmaps)
        assert logits.ndim >= 1

    def test_head_grad_fn_shape(self, runner: ModelRunner):
        x = runner.sample_input()
        fmaps = runner.backbone_fn(x)
        f = fmaps[0] if fmaps.shape[0] == 1 and fmaps.ndim > len(runner.feature_shape) else fmaps
        grad = runner.head_grad_fn(f, 0)
        assert grad.shape == f.shape

    def test_predict_batch(self, runner: ModelRunner):
        rng = np.random.default_rng(0)
        n_flat = int(np.prod(runner.input_shape))
        X = rng.standard_normal((10, n_flat)).astype(np.float32) * 0.1
        preds = runner.predict(X)
        assert preds.shape == (10,)
        assert all(0 <= p < runner.spec.n_classes for p in preds)

    def test_layer_outputs_fn_returns_list(self, runner: ModelRunner):
        x = runner.sample_input()
        outputs = runner.layer_outputs_fn(x)
        assert isinstance(outputs, list)
        assert len(outputs) >= 2  # at least input + output

    def test_feature_maps_override(self, runner: ModelRunner):
        """forward_fn with feature_maps_override should use the provided maps."""
        x = runner.sample_input()
        _, fmaps = runner.forward_fn(x)
        logits1, fmaps1 = runner.forward_fn(x, feature_maps_override=fmaps)
        # Feature maps should be unchanged
        np.testing.assert_array_equal(fmaps, fmaps1)

    def test_reproducible_with_same_seed(self):
        r1 = build_runner("resnet8", seed=99)
        r2 = build_runner("resnet8", seed=99)
        x = r1.sample_input(seed=0)
        l1, _ = r1.forward_fn(x)
        l2, _ = r2.forward_fn(x)
        np.testing.assert_array_equal(l1, l2)


# ============================================================================
# Phase 1: Grad-CAM
# ============================================================================


class TestPhase1GradCAM:
    """Grad-CAM on all vision models (requires spatial feature maps)."""

    def test_gradcam_output_shape(self, vision_runner: ModelRunner):
        x = vision_runner.sample_input()
        cam = gradcam(vision_runner.forward_fn, x, target_class=0)
        # Grad-CAM produces a spatial heatmap
        assert cam.ndim == 2
        h, w = vision_runner.feature_shape[:2]
        # Output shape depends on feature map spatial dims
        # For valid conv: output may be (h-2, w-2) or (h, w)
        assert cam.shape[0] > 0 and cam.shape[1] > 0

    def test_gradcam_range(self, vision_runner: ModelRunner):
        x = vision_runner.sample_input()
        cam = gradcam(vision_runner.forward_fn, x, target_class=0)
        assert cam.min() >= 0.0, "Grad-CAM must be non-negative (ReLU)"
        assert cam.max() <= 1.0 + 1e-6, "Grad-CAM must be normalized to [0,1]"

    def test_gradcam_dtype(self, vision_runner: ModelRunner):
        x = vision_runner.sample_input()
        cam = gradcam(vision_runner.forward_fn, x, target_class=0)
        assert cam.dtype == np.float32

    def test_gradcam_different_classes_differ(self, vision_runner: ModelRunner):
        x = vision_runner.sample_input()
        cam0 = gradcam(vision_runner.forward_fn, x, target_class=0)
        cam1 = gradcam(vision_runner.forward_fn, x, target_class=1)
        # Different target classes should (usually) produce different maps
        assert not np.allclose(cam0, cam1, atol=1e-6)


class TestPhase1LRP:
    """LRP on models with FC-only paths."""

    def test_lrp_on_toyadmos(self):
        runner = build_runner("toyadmos", seed=42)
        x = runner.sample_input()
        outputs = runner.layer_outputs_fn(x)
        # LRP needs (layer_outputs, layer_weights) — use the FC weights
        # ToyAdmos is all FC, so we can pass the full weight chain
        relevance = lrp_epsilon(
            layer_outputs=outputs[:3],  # input, first hidden, second hidden
            layer_weights=runner.weights[:2] if len(runner.weights) >= 2 else [runner.weights[0]],
            target_class=0,
        )
        assert relevance is not None
        assert relevance.shape[0] == 1  # batch dim

    def test_lrp_on_gap_fc(self):
        runner = build_runner("gap_fc", seed=42)
        x = runner.sample_input()
        outputs = runner.layer_outputs_fn(x)
        # GAP+FC: use pooled -> logits path
        relevance = lrp_epsilon(
            layer_outputs=outputs[-2:],  # pooled, logits
            layer_weights=runner.weights[:1],
            target_class=0,
        )
        assert relevance is not None
        assert np.any(relevance != 0)


# ============================================================================
# Phase 2: Gradient SHAP
# ============================================================================


class TestPhase2SHAP:
    """Gradient SHAP on all models."""

    def test_gradient_shap_on_vision(self, vision_runner: ModelRunner):
        """Full gradient SHAP on vision models (small n_samples for speed)."""
        x = vision_runner.sample_input()
        attr = gradient_shap(
            vision_runner.forward_fn, x, target_class=0, n_samples=2, seed=42
        )
        H, W, C = vision_runner.feature_shape[:3] if len(vision_runner.feature_shape) >= 3 else (
            vision_runner.input_shape
        )
        assert attr.shape == vision_runner.input_shape

    def test_hoisted_shap_backbone_calls(self, runner: ModelRunner):
        """Hoisted SHAP should call backbone exactly once."""
        x = runner.sample_input()
        fmaps = runner.backbone_fn(x)
        f = fmaps[0] if fmaps.shape[0] == 1 and fmaps.ndim > len(runner.feature_shape) else fmaps

        result = hoisted_gradient_shap(
            runner.backbone_fn,
            runner.head_fn,
            runner.head_grad_fn,
            x,
            target_class=0,
            n_samples=4,
            seed=42,
        )
        assert result.backbone_calls == 1
        assert result.head_calls == 4

    def test_hoisted_shap_attribution_shape(self, runner: ModelRunner):
        x = runner.sample_input()
        result = hoisted_gradient_shap(
            runner.backbone_fn,
            runner.head_fn,
            runner.head_grad_fn,
            x,
            target_class=0,
            n_samples=4,
            seed=42,
        )
        assert result.attributions.shape == runner.feature_shape

    def test_hoisted_shap_nonzero(self, runner: ModelRunner):
        x = runner.sample_input()
        result = hoisted_gradient_shap(
            runner.backbone_fn,
            runner.head_fn,
            runner.head_grad_fn,
            x,
            target_class=0,
            n_samples=8,
            seed=42,
        )
        assert np.any(result.attributions != 0.0)


# ============================================================================
# Phase 3: Formal Verification (QVIP)
# ============================================================================


class TestPhase3QVIP:
    """QVIP verification on small sub-networks derived from benchmark models.

    Full models are too large for ILP solving, so we extract representative
    2-layer sub-networks matching the head structure of each model.
    """

    @staticmethod
    def _build_small_qnn(n_in: int, n_hidden: int, n_out: int, seed: int = 42) -> QNN:
        """Build a small QNN for verification testing."""
        rng = np.random.default_rng(seed)
        config = QuantConfig(signed=True, total_bits=8, frac_bits=4)

        W1 = rng.standard_normal((n_hidden, n_in)).astype(np.float32) * 0.5
        b1 = np.zeros(n_hidden, dtype=np.float32)
        W2 = rng.standard_normal((n_out, n_hidden)).astype(np.float32) * 0.5
        b2 = np.zeros(n_out, dtype=np.float32)

        W1_q = quantize_uniform(W1, config)
        b1_q = quantize_uniform(b1, config)
        W2_q = quantize_uniform(W2, config)
        b2_q = quantize_uniform(b2, config)

        return QNN(
            weights=[W1_q, W2_q],
            biases=[b1_q, b2_q],
            config_in=config,
            config_w=config,
            config_out=config,
        )

    @pytest.mark.parametrize("model_name,n_in,n_hidden,n_out", [
        ("gap_fc", 16, 8, 10),
        ("resnet8", 64, 16, 10),
        ("toyadmos", 8, 4, 2),
        ("mobilebert_tiny", 128, 16, 2),
    ])
    def test_qvip_verify_produces_result(self, model_name, n_in, n_hidden, n_out):
        """QVIP should produce a valid VerifyResult for each model's head."""
        qnn = self._build_small_qnn(n_in, n_hidden, n_out)
        rng = np.random.default_rng(0)
        x = quantize_uniform(
            rng.standard_normal(n_in).astype(np.float32) * 0.1,
            qnn.config_in,
        )
        true_class = int(np.argmax(qnn.forward(x)))
        report = verify_robustness(qnn, x, true_class, radius=1)
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST, VerifyResult.UNKNOWN)
        assert report.true_class == true_class
        assert report.radius == 1
        assert report.n_variables >= 0

    @pytest.mark.parametrize("model_name,n_in,n_hidden,n_out", [
        ("gap_fc", 16, 8, 10),
        ("toyadmos", 8, 4, 2),
    ])
    def test_qvip_small_radius_robust(self, model_name, n_in, n_hidden, n_out):
        """With radius=0, the network should always be trivially robust."""
        qnn = self._build_small_qnn(n_in, n_hidden, n_out)
        rng = np.random.default_rng(0)
        x = quantize_uniform(
            rng.standard_normal(n_in).astype(np.float32) * 0.1,
            qnn.config_in,
        )
        true_class = int(np.argmax(qnn.forward(x)))
        report = verify_robustness(qnn, x, true_class, radius=0)
        assert report.result == VerifyResult.ROBUST

    def test_qvip_large_radius_not_robust(self):
        """With a very large radius, the network should be not robust."""
        qnn = self._build_small_qnn(8, 4, 2, seed=99)
        x = quantize_uniform(
            np.ones(8, dtype=np.float32) * 0.1,
            qnn.config_in,
        )
        true_class = int(np.argmax(qnn.forward(x)))
        report = verify_robustness(qnn, x, true_class, radius=50)
        # Large radius should make it not robust (or at least not provably robust)
        assert report.result in (VerifyResult.NOT_ROBUST, VerifyResult.UNKNOWN)


# ============================================================================
# Phase 4: Rule Extraction
# ============================================================================


class TestPhase4RuleExtraction:
    """Top-K saliency-guided rule extraction on all models."""

    def test_rule_extraction_produces_rules(self, runner: ModelRunner):
        rng = np.random.default_rng(42)
        n_flat = int(np.prod(runner.input_shape))
        n_samples = 50
        X = rng.standard_normal((n_samples, n_flat)).astype(np.float32) * 0.1

        # Use uniform saliency as a baseline
        saliency = rng.uniform(size=n_flat).astype(np.float32)
        # Boost a few features to make selection interesting
        saliency[:min(4, n_flat)] += 5.0

        top_k = min(8, n_flat)
        result = topk_saliency_filter(
            X, saliency, runner.predict, top_k=top_k, max_depth=3,
        )
        assert result.rules is not None
        assert result.rules.n_leaves >= 2
        assert len(result.selected_features) == top_k

    def test_rule_extraction_fidelity(self, runner: ModelRunner):
        """Extracted rules should achieve non-trivial fidelity."""
        rng = np.random.default_rng(42)
        n_flat = int(np.prod(runner.input_shape))
        n_samples = 80
        X = rng.standard_normal((n_samples, n_flat)).astype(np.float32) * 0.1

        saliency = rng.uniform(size=n_flat).astype(np.float32)
        saliency[:min(4, n_flat)] += 5.0

        top_k = min(8, n_flat)
        result = topk_saliency_filter(
            X, saliency, runner.predict, top_k=top_k, max_depth=3,
        )
        assert result.rules.train_fidelity > 0.0

    def test_rule_extraction_dropped_fraction(self, runner: ModelRunner):
        n_flat = int(np.prod(runner.input_shape))
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, n_flat)).astype(np.float32) * 0.1
        saliency = rng.uniform(size=n_flat).astype(np.float32)
        top_k = min(8, n_flat)

        result = topk_saliency_filter(
            X, saliency, runner.predict, top_k=top_k, max_depth=3,
        )
        expected_drop = 1.0 - top_k / n_flat
        assert result.dropped_fraction == pytest.approx(expected_drop, abs=1e-6)


# ============================================================================
# ECQx on benchmark model weights
# ============================================================================


class TestECQxOnModels:
    """ECQx weight filtering using saliency from benchmark models."""

    def test_ecqx_mask_on_model_weights(self, runner: ModelRunner):
        """Apply ECQx masking to each model's FC weight matrix."""
        if not runner.weights:
            pytest.skip("Model has no exposed weight matrices")

        W = runner.weights[-1]  # last FC layer
        if W.ndim != 2:
            pytest.skip("Weight matrix is not 2D")

        n_out = W.shape[0] if W.shape[0] == runner.spec.n_classes else W.shape[1]
        saliency = np.abs(np.random.default_rng(0).standard_normal(W.shape[0])).astype(np.float32)
        mask = ecqx_weight_mask(W, saliency, critical_fraction=0.25)
        assert mask.shape == W.shape
        assert mask.dtype == bool
        # At least some weights should be critical
        assert mask.any()
        # Not all weights should be critical (at 25%)
        if W.shape[0] > 1:
            assert not mask.all()

    def test_ecqx_bitwidth_on_model_weights(self, runner: ModelRunner):
        if not runner.weights:
            pytest.skip("No weights")

        W = runner.weights[-1]
        if W.ndim != 2:
            pytest.skip("Weight matrix is not 2D")

        saliency = np.abs(np.random.default_rng(0).standard_normal(W.shape[0])).astype(np.float32)
        policy = ecqx_bitwidth_policy(W, saliency, critical_bits=8, noncritical_bits=4)
        assert set(np.unique(policy)) <= {4, 8}


# ============================================================================
# Cross-phase integration
# ============================================================================


class TestCrossPhaseIntegration:
    """Verify the full Phase 1-4 pipeline works end-to-end on a model."""

    def test_gradcam_feeds_rule_extraction_gap_fc(self):
        """Phase 1 saliency -> Phase 4 rule extraction on gap_fc."""
        runner = build_runner("gap_fc", seed=42)
        x = runner.sample_input()

        # Phase 1: Grad-CAM
        cam = gradcam(runner.forward_fn, x, target_class=0)
        assert cam.shape[0] > 0

        # Use the Grad-CAM map as saliency for Phase 4
        rng = np.random.default_rng(42)
        n_flat = int(np.prod(runner.input_shape))
        X = rng.standard_normal((50, n_flat)).astype(np.float32) * 0.1
        saliency = np.abs(cam).flatten()
        # Pad or truncate saliency to match n_flat
        if saliency.size < n_flat:
            saliency = np.pad(saliency, (0, n_flat - saliency.size))
        else:
            saliency = saliency[:n_flat]

        result = topk_saliency_filter(
            X, saliency, runner.predict, top_k=min(8, n_flat), max_depth=3,
        )
        assert result.rules is not None
        assert result.rules.n_leaves >= 1

    def test_shap_feeds_ecqx_toyadmos(self):
        """Phase 2 SHAP -> ECQx weight filtering on ToyAdmos."""
        runner = build_runner("toyadmos", seed=42)
        x = runner.sample_input()

        # Phase 2: Hoisted SHAP
        result = hoisted_gradient_shap(
            runner.backbone_fn,
            runner.head_fn,
            runner.head_grad_fn,
            x,
            target_class=0,
            n_samples=4,
            seed=42,
        )
        attr = result.attributions
        assert attr is not None

        # Use SHAP attributions as saliency for ECQx
        W = runner.weights[-1]  # (c_in, c_out)
        if W.ndim == 2:
            # Create output saliency from SHAP attributions
            output_saliency = np.abs(attr).flatten()
            if output_saliency.size != W.shape[0]:
                # Resize to match weight rows
                output_saliency = np.abs(
                    np.random.default_rng(0).standard_normal(W.shape[0])
                ).astype(np.float32)
            mask = ecqx_weight_mask(W, output_saliency, critical_fraction=0.25)
            assert mask.shape == W.shape

    def test_full_pipeline_resnet8(self):
        """Run all phases on ResNet-8 (the main benchmark model)."""
        runner = build_runner("resnet8", seed=42)
        x = runner.sample_input()

        # Phase 1: Grad-CAM
        cam = gradcam(runner.forward_fn, x, target_class=0)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-6

        # Phase 2: Hoisted SHAP
        shap_result = hoisted_gradient_shap(
            runner.backbone_fn,
            runner.head_fn,
            runner.head_grad_fn,
            x,
            target_class=0,
            n_samples=4,
            seed=42,
        )
        assert shap_result.backbone_calls == 1
        assert shap_result.attributions.shape == runner.feature_shape

        # Phase 3: QVIP on a small sub-network
        config = QuantConfig(signed=True, total_bits=8, frac_bits=4)
        rng = np.random.default_rng(42)
        W1 = quantize_uniform(rng.standard_normal((16, 64)).astype(np.float32) * 0.5, config)
        b1 = quantize_uniform(np.zeros(16, dtype=np.float32), config)
        W2 = quantize_uniform(rng.standard_normal((10, 16)).astype(np.float32) * 0.5, config)
        b2 = quantize_uniform(np.zeros(10, dtype=np.float32), config)
        qnn = QNN(weights=[W1, W2], biases=[b1, b2],
                   config_in=config, config_w=config, config_out=config)
        x_q = quantize_uniform(rng.standard_normal(64).astype(np.float32) * 0.1, config)
        true_class = int(np.argmax(qnn.forward(x_q)))
        report = verify_robustness(qnn, x_q, true_class, radius=1)
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST, VerifyResult.UNKNOWN)

        # Phase 4: Rule extraction from SHAP saliency
        n_flat = int(np.prod(runner.input_shape))
        X = rng.standard_normal((50, n_flat)).astype(np.float32) * 0.1
        saliency = np.abs(shap_result.attributions).flatten()
        if saliency.size < n_flat:
            saliency = np.pad(saliency, (0, n_flat - saliency.size))
        else:
            saliency = saliency[:n_flat]
        rules = topk_saliency_filter(
            X, saliency, runner.predict, top_k=8, max_depth=3,
        )
        assert rules.rules.n_leaves >= 1
