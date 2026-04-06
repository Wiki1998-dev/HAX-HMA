"""
Tests for Gradient SHAP reference implementation.
Run: pytest tests/test_shap_reference.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.xai.shap.gradient_shap_reference import (
    gradient_shap,
    gradient_shap_analytical,
    expected_gradients,
)


# ── Tiny toy model for testing (same as gradcam tests) ──────────────────────


def make_toy_model():
    """
    Create a minimal 2-layer model (conv + fc) for testing.
    Weights are fixed for reproducibility.
    Returns (forward_fn, weights_dict).
    """
    rng = np.random.default_rng(42)
    W_conv = rng.standard_normal((3, 3, 1, 4)).astype(np.float32) * 0.1
    W_fc = rng.standard_normal((4, 2)).astype(np.float32) * 0.1
    b_fc = np.zeros(2, dtype=np.float32)

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        """x: (1, 8, 8, 1)"""
        if feature_maps_override is None:
            N, H, W, C = x.shape
            fH, fW = H - 2, W - 2
            fmaps = np.zeros((N, fH, fW, 4), dtype=np.float32)
            for k in range(4):
                for i in range(fH):
                    for j in range(fW):
                        fmaps[0, i, j, k] = np.sum(
                            x[0, i : i + 3, j : j + 3, :] * W_conv[:, :, :, k]
                        )
            fmaps = np.maximum(fmaps, 0)  # ReLU
        else:
            fmaps = feature_maps_override

        pooled = fmaps.mean(axis=(1, 2))  # (N, 4)
        logits = pooled @ W_fc + b_fc  # (N, 2)
        return logits, fmaps

    return forward_fn, {"W_conv": W_conv, "W_fc": W_fc}


def make_linear_model():
    """
    Simple GAP + FC model (no conv, no ReLU) for analytical verification.
    Input: (1, H, W, K) → GAP → (1, K) → FC → (1, C).
    """
    rng = np.random.default_rng(99)
    K, C = 8, 3
    W_fc = rng.standard_normal((K, C)).astype(np.float32) * 0.1

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        fmaps = x if feature_maps_override is None else feature_maps_override
        pooled = fmaps.mean(axis=(1, 2))  # (1, K)
        logits = pooled @ W_fc  # (1, C)
        return logits, fmaps

    return forward_fn, W_fc


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGradientSHAP:
    def setup_method(self):
        self.forward_fn, self.weights = make_toy_model()
        rng = np.random.default_rng(0)
        self.input_tensor = rng.standard_normal((1, 8, 8, 1)).astype(np.float32)

    def test_output_shape(self):
        attr = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=0,
            n_samples=4, seed=42,
        )
        assert attr.shape == (8, 8, 1), f"Expected (8,8,1), got {attr.shape}"

    def test_output_dtype(self):
        attr = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=0,
            n_samples=4, seed=42,
        )
        assert attr.dtype == np.float32

    def test_different_classes_differ(self):
        attr0 = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=0,
            n_samples=8, seed=42,
        )
        attr1 = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=1,
            n_samples=8, seed=42,
        )
        assert not np.allclose(attr0, attr1), \
            "Attributions for different classes should differ"

    def test_zero_input_zero_baseline_gives_zero(self):
        zero_input = np.zeros((1, 8, 8, 1), dtype=np.float32)
        zero_baselines = np.zeros((4, 8, 8, 1), dtype=np.float32)
        attr = gradient_shap(
            self.forward_fn, zero_input, target_class=0,
            baselines=zero_baselines, seed=42,
        )
        # Zero input and zero baselines → zero difference → zero attributions
        assert np.allclose(attr, 0.0, atol=1e-5), \
            "Zero input with zero baselines should give zero attributions"

    def test_custom_baselines(self):
        rng = np.random.default_rng(123)
        baselines = rng.standard_normal((8, 8, 8, 1)).astype(np.float32) * 0.01
        attr = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, seed=42,
        )
        assert attr.shape == (8, 8, 1)
        # Should not be all zeros (input differs from baselines)
        assert not np.allclose(attr, 0.0, atol=1e-6)

    def test_more_samples_reduces_variance(self):
        """More samples should give more consistent results across seeds."""
        attrs_few = []
        attrs_many = []
        for seed in [10, 20, 30]:
            attrs_few.append(gradient_shap(
                self.forward_fn, self.input_tensor, target_class=0,
                n_samples=4, seed=seed,
            ))
            attrs_many.append(gradient_shap(
                self.forward_fn, self.input_tensor, target_class=0,
                n_samples=32, seed=seed,
            ))
        var_few = np.var([a.flatten() for a in attrs_few], axis=0).mean()
        var_many = np.var([a.flatten() for a in attrs_many], axis=0).mean()
        assert var_many < var_few, \
            "More samples should reduce variance across seeds"


class TestAnalyticalSHAP:
    """Test analytical SHAP for linear (GAP+FC) model against numerical."""

    def setup_method(self):
        self.forward_fn, self.W_fc = make_linear_model()

    def test_matches_numerical(self):
        """Analytical and numerical SHAP should agree for linear model."""
        rng = np.random.default_rng(42)
        H, W, K = 4, 4, 8
        input_tensor = rng.standard_normal((1, H, W, K)).astype(np.float32)
        baselines = rng.standard_normal((16, H, W, K)).astype(np.float32) * 0.1

        analytical = gradient_shap_analytical(
            input_tensor[0], self.W_fc, target_class=1, baselines=baselines,
        )

        numerical = gradient_shap(
            self.forward_fn, input_tensor, target_class=1,
            baselines=baselines, seed=42,
        )

        # For linear model, gradient is constant, so analytical should be close
        # to numerical (difference comes from alpha interpolation in numerical)
        np.testing.assert_allclose(
            analytical, numerical, rtol=0.3, atol=1e-3,
        )

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        H, W, K = 4, 4, 8
        inp = rng.standard_normal((H, W, K)).astype(np.float32)
        baselines = rng.standard_normal((8, H, W, K)).astype(np.float32)

        attr = gradient_shap_analytical(inp, self.W_fc, 0, baselines)
        assert attr.shape == (H, W, K)

    def test_completeness_approximate(self):
        """
        SHAP completeness: sum of attributions ≈ f(x) - E[f(x')].
        For a linear model, this should hold closely.
        """
        rng = np.random.default_rng(7)
        H, W, K = 4, 4, 8
        input_tensor = rng.standard_normal((1, H, W, K)).astype(np.float32)
        baselines = rng.standard_normal((32, H, W, K)).astype(np.float32) * 0.1
        target_class = 0

        attr = gradient_shap_analytical(
            input_tensor[0], self.W_fc, target_class, baselines,
        )

        # f(x) - E[f(x')]
        logit_x, _ = self.forward_fn(input_tensor)
        baseline_logits = []
        for i in range(baselines.shape[0]):
            bl = baselines[i][np.newaxis, ...]
            logit_bl, _ = self.forward_fn(bl)
            baseline_logits.append(logit_bl[0, target_class])
        expected_diff = logit_x[0, target_class] - np.mean(baseline_logits)

        attr_sum = attr.sum()
        # Should be approximately equal (linear model)
        assert abs(attr_sum - expected_diff) / (abs(expected_diff) + 1e-8) < 0.3, \
            f"Completeness violated: sum={attr_sum:.4f}, expected={expected_diff:.4f}"


class TestExpectedGradients:
    def setup_method(self):
        self.forward_fn, self.weights = make_toy_model()
        rng = np.random.default_rng(0)
        self.input_tensor = rng.standard_normal((1, 8, 8, 1)).astype(np.float32)

    def test_output_shape(self):
        rng = np.random.default_rng(5)
        baselines = rng.standard_normal((4, 8, 8, 1)).astype(np.float32) * 0.1
        attr = expected_gradients(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, n_steps=5,
        )
        assert attr.shape == (8, 8, 1)

    def test_output_dtype(self):
        rng = np.random.default_rng(5)
        baselines = rng.standard_normal((4, 8, 8, 1)).astype(np.float32) * 0.1
        attr = expected_gradients(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, n_steps=5,
        )
        assert attr.dtype == np.float32

    def test_zero_baseline(self):
        """With zero baselines, expected gradients = integrated gradients."""
        baselines = np.zeros((2, 8, 8, 1), dtype=np.float32)
        attr = expected_gradients(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, n_steps=10,
        )
        # Should not be all zeros (non-zero input)
        assert not np.allclose(attr, 0.0, atol=1e-6)

    def test_consistent_with_gradient_shap(self):
        """Expected gradients and gradient SHAP should be roughly aligned."""
        rng = np.random.default_rng(5)
        baselines = rng.standard_normal((8, 8, 8, 1)).astype(np.float32) * 0.1

        eg = expected_gradients(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, n_steps=10,
        )
        gs = gradient_shap(
            self.forward_fn, self.input_tensor, target_class=0,
            baselines=baselines, seed=42,
        )
        # Both should have same sign pattern for most features
        sign_agreement = np.mean(np.sign(eg) == np.sign(gs))
        assert sign_agreement > 0.5, \
            f"Expected >50% sign agreement, got {sign_agreement:.1%}"
