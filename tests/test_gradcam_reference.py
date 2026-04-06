"""
Tests for Grad-CAM reference implementation.
Run: pytest tests/test_gradcam_reference.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.xai.gradcam.gradcam_reference import (
    gradcam,
    lrp_epsilon,
    upsample_saliency,
    faithfulness_score,
)


# ── Tiny toy model for testing ────────────────────────────────────────────────

def make_toy_model():
    """
    Create a minimal 2-layer model (conv + fc) for testing.
    Weights are fixed for reproducibility.
    Returns (forward_fn, weights_dict).
    """
    rng = np.random.default_rng(42)
    W_conv = rng.standard_normal((3, 3, 1, 4)).astype(np.float32) * 0.1
    W_fc   = rng.standard_normal((4, 2)).astype(np.float32) * 0.1
    b_fc   = np.zeros(2, dtype=np.float32)

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        """x: (1, 8, 8, 1)"""
        # Simple conv (no padding, stride 1)
        if feature_maps_override is None:
            N, H, W, C = x.shape
            fH, fW = H - 2, W - 2
            fmaps = np.zeros((N, fH, fW, 4), dtype=np.float32)
            for k in range(4):
                for i in range(fH):
                    for j in range(fW):
                        fmaps[0, i, j, k] = np.sum(x[0, i:i+3, j:j+3, :] * W_conv[:, :, :, k])
            fmaps = np.maximum(fmaps, 0)  # ReLU
        else:
            fmaps = feature_maps_override

        # Global average pool
        pooled = fmaps.mean(axis=(1, 2))  # (N, 4)
        # FC
        logits = pooled @ W_fc + b_fc     # (N, 2)
        return logits, fmaps

    return forward_fn, {"W_conv": W_conv, "W_fc": W_fc}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGradCAM:
    def setup_method(self):
        self.forward_fn, self.weights = make_toy_model()
        rng = np.random.default_rng(0)
        self.input_tensor = rng.standard_normal((1, 8, 8, 1)).astype(np.float32)

    def test_output_shape(self):
        cam = gradcam(self.forward_fn, self.input_tensor, target_class=0)
        # Conv with 3×3 kernel on 8×8 input → 6×6 feature maps
        assert cam.shape == (6, 6), f"Expected (6,6), got {cam.shape}"

    def test_output_dtype(self):
        cam = gradcam(self.forward_fn, self.input_tensor, target_class=0)
        assert cam.dtype == np.float32

    def test_output_range(self):
        cam = gradcam(self.forward_fn, self.input_tensor, target_class=0)
        assert cam.min() >= 0.0, "Grad-CAM must be non-negative after ReLU"
        assert cam.max() <= 1.0 + 1e-6, "Grad-CAM must be normalized to [0,1]"

    def test_different_classes_differ(self):
        cam0 = gradcam(self.forward_fn, self.input_tensor, target_class=0)
        cam1 = gradcam(self.forward_fn, self.input_tensor, target_class=1)
        # Saliency maps for different classes should generally differ
        assert not np.allclose(cam0, cam1), "Saliency maps for different classes should differ"

    def test_zero_input_gives_zero_saliency(self):
        zero_input = np.zeros((1, 8, 8, 1), dtype=np.float32)
        cam = gradcam(self.forward_fn, zero_input, target_class=0)
        # Zero input → zero activations → zero gradients → zero saliency
        assert np.allclose(cam, 0.0, atol=1e-6), "Zero input should give zero saliency"


class TestLRP:
    def test_output_shape(self):
        layer_outputs = [
            np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),  # input layer (1, 4)
            np.array([[0.5, 1.5]], dtype=np.float32),              # hidden layer (1, 2)
        ]
        layer_weights = [
            np.random.randn(4, 2).astype(np.float32) * 0.1,       # (4, 2)
        ]
        R = lrp_epsilon(layer_outputs, layer_weights, target_class=0)
        assert R.shape == (1, 4)

    def test_relevance_conservation(self):
        """Total relevance at input should approximately equal output relevance."""
        rng = np.random.default_rng(1)
        layer_outputs = [
            np.abs(rng.standard_normal((1, 8)).astype(np.float32)),
            np.abs(rng.standard_normal((1, 4)).astype(np.float32)),
            np.abs(rng.standard_normal((1, 2)).astype(np.float32)),
        ]
        layer_weights = [
            rng.standard_normal((8, 4)).astype(np.float32) * 0.1,
            rng.standard_normal((4, 2)).astype(np.float32) * 0.1,
        ]
        R = lrp_epsilon(layer_outputs, layer_weights, target_class=0)
        input_relevance_sum = R.sum()
        output_relevance = layer_outputs[-1][0, 0]
        # With epsilon stabilization, sums won't be exact but should be close
        assert abs(input_relevance_sum - output_relevance) / (abs(output_relevance) + 1e-8) < 0.5


class TestUpsample:
    def test_output_shape(self):
        cam = np.random.rand(6, 6).astype(np.float32)
        upsampled = upsample_saliency(cam, target_shape=(8, 8))
        assert upsampled.shape == (8, 8)

    def test_dtype(self):
        cam = np.random.rand(6, 6).astype(np.float32)
        upsampled = upsample_saliency(cam, target_shape=(32, 32))
        assert upsampled.dtype == np.float32


class TestFaithfulness:
    def setup_method(self):
        self.forward_fn, _ = make_toy_model()
        self.input_tensor = np.ones((1, 8, 8, 1), dtype=np.float32)
        # Synthetic saliency: highlight corner region
        self.saliency = np.zeros((8, 8), dtype=np.float32)
        self.saliency[:3, :3] = 1.0

    def test_returns_dict(self):
        scores = faithfulness_score(
            self.forward_fn, self.input_tensor, self.saliency, target_class=0
        )
        assert isinstance(scores, dict)
        assert "baseline" in scores

    def test_has_expected_keys(self):
        scores = faithfulness_score(
            self.forward_fn, self.input_tensor, self.saliency, target_class=0,
            top_k_fractions=[0.1, 0.5]
        )
        assert "top_10pct_deleted" in scores
        assert "top_50pct_deleted" in scores
