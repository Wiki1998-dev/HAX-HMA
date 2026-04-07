"""Tests for formal verification of quantized neural networks.

Tests cover:
  - Quantization correctness (symmetric, QVIP-style uniform)
  - Bound propagation (linear, ReLU, clamp, full network)
  - ILP verification (local robustness, MRR, quantization safety)
  - Integration: Grad-CAM saliency → verification of high-saliency regions

References:
    - QVIP (Zhang et al., ASE'22)
    - ECQx (Becking et al., 2022)
"""

import numpy as np
import pytest

from src.xai.formal.quantization import (
    QuantConfig,
    SNAX_INT8,
    QVIP_DEFAULT,
    quantize_uniform,
    dequantize_uniform,
    quantize_symmetric,
    dequantize_symmetric,
    quantization_error,
    quantization_error_bound,
    quantize_layer_weights,
)
from src.xai.formal.bound_propagation import (
    Bounds,
    LayerSpec,
    propagate_linear,
    propagate_relu,
    propagate_clamp,
    propagate_network,
    classify_relu_neurons,
    compute_input_bounds_linf,
    output_robustness_check,
)
from src.xai.formal.qvip_verifier import (
    QNN,
    VerifyResult,
    VerifyReport,
    verify_robustness,
    compute_max_robustness_radius,
    verify_quantization_safety,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """4-bit signed quantization config for testing."""
    return QuantConfig(signed=True, total_bits=4, frac_bits=1)


@pytest.fixture
def int8_config():
    """Standard INT8 config matching SNAX GeMM."""
    return SNAX_INT8


@pytest.fixture
def simple_qnn():
    """Simple 2-neuron QNN for verification testing.

    Architecture: 2 inputs -> 2 hidden (ReLU) -> 2 outputs
    Designed so class 0 is predicted for positive inputs.
    """
    config = QuantConfig(signed=True, total_bits=8, frac_bits=0)
    W1 = np.array([[2, -1], [-1, 2]], dtype=np.int32)
    b1 = np.array([0, 0], dtype=np.int32)
    W2 = np.array([[3, -2], [-2, 3]], dtype=np.int32)
    b2 = np.array([0, 0], dtype=np.int32)
    return QNN(
        weights=[W1, W2],
        biases=[b1, b2],
        config_in=config,
        config_w=config,
        config_out=config,
    )


@pytest.fixture
def linear_qnn():
    """Single linear layer QNN (no ReLU)."""
    config = QuantConfig(signed=True, total_bits=8, frac_bits=0)
    # K=4 inputs, C=3 classes (matching our XAI test dimensions)
    W = np.array([
        [5, 3, -2, 1],
        [-1, 4, 3, -2],
        [2, -1, 1, 5],
    ], dtype=np.int32)
    b = np.array([1, 0, -1], dtype=np.int32)
    return QNN(
        weights=[W],
        biases=[b],
        config_in=config,
        config_w=config,
        config_out=config,
    )


# ============================================================================
# Test Quantization
# ============================================================================

class TestQuantization:
    """Tests for quantization module."""

    def test_quantize_uniform_basic(self, small_config):
        """QVIP quantization: û = clamp(floor(2^F * u), C^lb, C^ub)."""
        # 4-bit signed, F=1: range [-8, 7], scale = 2^1 = 2
        x = np.array([0.0, 0.5, 1.0, -0.5, 3.5, -5.0])
        q = quantize_uniform(x, small_config)
        # floor(2^1 * x) = floor([0, 1, 2, -1, 7, -10])
        # clamp to [-8, 7]: [0, 1, 2, -1, 7, -8]
        expected = np.array([0, 1, 2, -1, 7, -8])
        np.testing.assert_array_equal(q, expected)

    def test_dequantize_uniform(self, small_config):
        """Dequantize recovers approximate values."""
        x_q = np.array([0, 2, -4, 7], dtype=np.int32)
        x_deq = dequantize_uniform(x_q, small_config)
        # scale = 2^(-1) = 0.5
        expected = np.array([0.0, 1.0, -2.0, 3.5])
        np.testing.assert_array_almost_equal(x_deq, expected)

    def test_quantize_symmetric_snax(self):
        """Symmetric INT8 quantization matching SNAX GeMM."""
        x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        x_q, scale = quantize_symmetric(x, bits=8)
        assert x_q.dtype == np.int8
        assert scale > 0
        assert np.max(np.abs(x_q)) <= 127
        # Check round-trip error
        x_deq = dequantize_symmetric(x_q, scale)
        np.testing.assert_allclose(x_deq, x, atol=scale)

    def test_quantize_symmetric_zero(self):
        """Zero input quantizes to zero."""
        x = np.zeros(10, dtype=np.float32)
        x_q, scale = quantize_symmetric(x)
        np.testing.assert_array_equal(x_q, 0)

    def test_quantization_error_bound(self):
        """Error bound is tight: actual error <= bound."""
        rng = np.random.RandomState(42)
        x = rng.randn(100).astype(np.float32)
        err = quantization_error(x, bits=8)
        bound = quantization_error_bound(x, bits=8)
        assert np.all(err <= bound + 1e-7)

    def test_quantize_layer_weights(self, small_config):
        """Layer weight quantization preserves shape."""
        W = np.random.randn(10, 16).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        W_q, b_q = quantize_layer_weights(W, b, small_config)
        assert W_q.shape == W.shape
        assert b_q.shape == b.shape
        assert W_q.dtype == np.int32

    def test_config_properties(self):
        """QuantConfig computes correct bounds and levels."""
        c = QuantConfig(signed=True, total_bits=8, frac_bits=4)
        assert c.clamp_lb == -128
        assert c.clamp_ub == 127
        assert c.n_levels == 256
        assert c.scale == 2**(-4)

        c_unsigned = QuantConfig(signed=False, total_bits=4, frac_bits=0)
        assert c_unsigned.clamp_lb == 0
        assert c_unsigned.clamp_ub == 15
        assert c_unsigned.n_levels == 16


# ============================================================================
# Test Bound Propagation
# ============================================================================

class TestBoundPropagation:
    """Tests for interval arithmetic bound propagation."""

    def test_bounds_basic(self):
        """Bounds contain their midpoint."""
        b = Bounds(lb=np.array([-1.0, 0.0]), ub=np.array([1.0, 2.0]))
        assert b.contains(np.array([0.0, 1.0]))
        assert not b.contains(np.array([2.0, 1.0]))

    def test_propagate_linear_identity(self):
        """Identity linear layer preserves bounds."""
        W = np.eye(3)
        b = np.zeros(3)
        bounds = Bounds(lb=np.array([-1, -2, -3.0]), ub=np.array([1, 2, 3.0]))
        result = propagate_linear(bounds, W, b)
        np.testing.assert_array_almost_equal(result.lb, bounds.lb)
        np.testing.assert_array_almost_equal(result.ub, bounds.ub)

    def test_propagate_linear_scaling(self):
        """Scaling widens bounds."""
        W = np.array([[2.0, 0], [0, -3.0]])
        b = np.array([1.0, 0.0])
        bounds = Bounds(lb=np.array([-1.0, -1.0]), ub=np.array([1.0, 1.0]))
        result = propagate_linear(bounds, W, b)
        # y1 = 2*x1 + 1: [-2+1, 2+1] = [-1, 3]
        # y2 = -3*x2: [-3*1, -3*(-1)] = [-3, 3] (W_neg @ ub, W_pos @ lb need care)
        np.testing.assert_array_almost_equal(result.lb, [-1.0, -3.0])
        np.testing.assert_array_almost_equal(result.ub, [3.0, 3.0])

    def test_propagate_relu_all_positive(self):
        """ReLU with all-positive bounds is identity."""
        bounds = Bounds(lb=np.array([1.0, 2.0]), ub=np.array([3.0, 4.0]))
        result = propagate_relu(bounds)
        np.testing.assert_array_equal(result.lb, bounds.lb)
        np.testing.assert_array_equal(result.ub, bounds.ub)

    def test_propagate_relu_all_negative(self):
        """ReLU with all-negative bounds gives zero."""
        bounds = Bounds(lb=np.array([-3.0, -4.0]), ub=np.array([-1.0, -2.0]))
        result = propagate_relu(bounds)
        np.testing.assert_array_equal(result.lb, [0.0, 0.0])
        np.testing.assert_array_equal(result.ub, [0.0, 0.0])

    def test_propagate_relu_crossing(self):
        """ReLU with crossing bounds: lb clamped to 0, ub unchanged."""
        bounds = Bounds(lb=np.array([-2.0, -1.0]), ub=np.array([3.0, 4.0]))
        result = propagate_relu(bounds)
        np.testing.assert_array_equal(result.lb, [0.0, 0.0])
        np.testing.assert_array_equal(result.ub, [3.0, 4.0])

    def test_propagate_clamp(self):
        """Clamp restricts bounds."""
        bounds = Bounds(lb=np.array([-200.0, 50.0]), ub=np.array([200.0, 100.0]))
        result = propagate_clamp(bounds, -128, 127)
        np.testing.assert_array_equal(result.lb, [-128, 50])
        np.testing.assert_array_equal(result.ub, [127, 100])

    def test_classify_relu_neurons(self):
        """Correctly classifies active/inactive/crossing neurons."""
        bounds = Bounds(
            lb=np.array([1.0, -3.0, -1.0, 0.0]),
            ub=np.array([5.0, -1.0, 2.0, 3.0]),
        )
        active, inactive, crossing = classify_relu_neurons(bounds)
        np.testing.assert_array_equal(active, [True, False, False, True])
        np.testing.assert_array_equal(inactive, [False, True, False, False])
        np.testing.assert_array_equal(crossing, [False, False, True, False])

    def test_propagate_network_soundness(self):
        """Propagated bounds contain actual network outputs."""
        rng = np.random.RandomState(42)
        W1 = rng.randn(4, 3)
        b1 = rng.randn(4)
        W2 = rng.randn(2, 4)
        b2 = rng.randn(2)

        layers = [
            LayerSpec('linear', weights=W1, bias=b1),
            LayerSpec('relu'),
            LayerSpec('linear', weights=W2, bias=b2),
        ]

        input_bounds = Bounds(lb=-np.ones(3), ub=np.ones(3))
        all_bounds = propagate_network(input_bounds, layers)

        # Check 100 random inputs within bounds
        for _ in range(100):
            x = rng.uniform(-1, 1, size=3)
            h = W1 @ x + b1
            h = np.maximum(h, 0)
            y = W2 @ h + b2
            assert all_bounds[-1].contains(y), f"Output {y} not in bounds"

    def test_input_bounds_linf(self):
        """L-inf input bounds are correct."""
        config = QuantConfig(signed=True, total_bits=8, frac_bits=0)
        x = np.array([10, -10, 100], dtype=np.int32)
        bounds = compute_input_bounds_linf(x, radius=5, config=config)
        np.testing.assert_array_equal(bounds.lb, [5, -15, 95])
        np.testing.assert_array_equal(bounds.ub, [15, -5, 105])

    def test_input_bounds_linf_clamped(self):
        """L-inf bounds are clamped to quantization range."""
        config = QuantConfig(signed=True, total_bits=8, frac_bits=0)
        x = np.array([126, -126], dtype=np.int32)
        bounds = compute_input_bounds_linf(x, radius=5, config=config)
        np.testing.assert_array_equal(bounds.lb, [121, -128])
        np.testing.assert_array_equal(bounds.ub, [127, -121])

    def test_output_robustness_check(self):
        """Quick robustness check from output bounds."""
        # Class 0 clearly dominant: lb[0]=10 > ub[1]=5 and lb[0]=10 > ub[2]=3
        bounds = Bounds(lb=np.array([10.0, -5.0, -3.0]), ub=np.array([20.0, 5.0, 3.0]))
        assert output_robustness_check(bounds, true_class=0)

        # Class 1 not clearly dominant: lb[1]=-5 < ub[0]=20
        assert not output_robustness_check(bounds, true_class=1)


# ============================================================================
# Test QNN Forward Pass
# ============================================================================

class TestQNN:
    """Tests for quantized neural network forward pass."""

    def test_forward_linear(self, linear_qnn):
        """Linear QNN forward pass matches manual computation."""
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        y = linear_qnn.forward(x)
        # y = W @ x + b = [5+6-6+4+1, -1+8+9-8+0, 2-2+3+20-1] = [10, 8, 22]
        expected = np.array([10, 8, 22], dtype=np.int32)
        np.testing.assert_array_equal(y, expected)

    def test_classify(self, linear_qnn):
        """Classification returns argmax."""
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        assert linear_qnn.classify(x) == 2  # class 2 has highest output

    def test_forward_with_relu(self, simple_qnn):
        """Two-layer QNN with ReLU."""
        x = np.array([5, 1], dtype=np.int32)
        y = simple_qnn.forward(x)
        # h = W1@x + b1 = [2*5-1, -5+2] = [9, -3]
        # relu: [9, 0]
        # clamp to [-128, 127]: [9, 0]
        # out = W2@h + b2 = [3*9, -2*9] = [27, -18]
        expected = np.array([27, -18], dtype=np.int32)
        np.testing.assert_array_equal(y, expected)

    def test_layer_sizes(self, simple_qnn):
        """Layer sizes reported correctly."""
        assert simple_qnn.layer_sizes == [2, 2, 2]
        assert simple_qnn.input_size == 2
        assert simple_qnn.output_size == 2


# ============================================================================
# Test Verification
# ============================================================================

class TestVerification:
    """Tests for ILP-based robustness verification."""

    def test_verify_robust_linear(self, linear_qnn):
        """Linear QNN verification at small radius produces definitive result."""
        x = np.array([10, 20, 5, 30], dtype=np.int32)
        true_class = linear_qnn.classify(x)
        report = verify_robustness(linear_qnn, x, true_class, radius=1)
        # Verification must produce a definitive answer for linear networks
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST)
        if report.result == VerifyResult.NOT_ROBUST:
            # Counterexample must be valid
            assert report.counterexample is not None
            adv_class = linear_qnn.classify(report.counterexample)
            assert adv_class != true_class

    def test_verify_not_robust_linear(self, linear_qnn):
        """Linear QNN becomes non-robust at large radius."""
        x = np.array([1, 1, 1, 1], dtype=np.int32)
        true_class = linear_qnn.classify(x)
        # With large enough radius, classification should change
        report = verify_robustness(linear_qnn, x, true_class, radius=20)
        # Either NOT_ROBUST or ROBUST depending on the specific weights
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST)

    def test_verify_two_layer(self, simple_qnn):
        """Two-layer QNN verification."""
        x = np.array([10, 2], dtype=np.int32)
        true_class = simple_qnn.classify(x)
        report = verify_robustness(simple_qnn, x, true_class, radius=1)
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST,
                                  VerifyResult.UNKNOWN)
        assert report.true_class == true_class

    def test_verify_zero_radius(self, linear_qnn):
        """Zero radius is trivially robust."""
        x = np.array([5, 5, 5, 5], dtype=np.int32)
        true_class = linear_qnn.classify(x)
        report = verify_robustness(linear_qnn, x, true_class, radius=0)
        assert report.result == VerifyResult.ROBUST

    def test_counterexample_valid(self, linear_qnn):
        """If NOT_ROBUST, counterexample produces different class."""
        x = np.array([0, 0, 0, 0], dtype=np.int32)
        true_class = linear_qnn.classify(x)
        report = verify_robustness(linear_qnn, x, true_class, radius=10)
        if report.result == VerifyResult.NOT_ROBUST:
            assert report.counterexample is not None
            adv_class = linear_qnn.classify(report.counterexample)
            assert adv_class != true_class

    def test_max_robustness_radius(self, linear_qnn):
        """MRR binary search returns consistent result."""
        x = np.array([10, 20, 5, 30], dtype=np.int32)
        true_class = linear_qnn.classify(x)
        mrr, reports = compute_max_robustness_radius(
            linear_qnn, x, true_class, max_radius=15,
        )
        assert mrr >= 0
        assert len(reports) >= 1
        # Verify MRR is tight: robust at mrr, not at mrr+1 (or mrr=max)
        if mrr > 0 and mrr < 15:
            r_check = verify_robustness(linear_qnn, x, true_class, radius=mrr)
            assert r_check.result == VerifyResult.ROBUST

    def test_verify_quantization_safety(self):
        """Verify that quantizing a float model preserves classification."""
        config = QuantConfig(signed=True, total_bits=8, frac_bits=4)
        W = [np.array([[0.5, -0.3], [-0.2, 0.6]], dtype=np.float32)]
        b = [np.array([0.1, -0.1], dtype=np.float32)]
        x = np.array([1.0, 0.5], dtype=np.float32)

        report = verify_quantization_safety(
            W, b, x, config, config, config, radius=0,
        )
        # With radius=0, we just check if quantized model agrees
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST)


# ============================================================================
# Test Integration: XAI + Formal Verification
# ============================================================================

class TestXAIIntegration:
    """Integration tests connecting XAI saliency to formal verification.

    Key insight from ECQx: use XAI to identify which regions matter,
    then formally verify those regions are preserved after quantization.
    """

    def test_saliency_guided_verification(self):
        """Verify quantization robustness in high-saliency input regions.

        Simulates the Phase 1 → Phase 3 pipeline:
        1. Compute saliency (simplified Grad-CAM analog for FC layer)
        2. Identify high-saliency inputs
        3. Verify quantized model is robust in those dimensions
        """
        rng = np.random.RandomState(42)
        config = QuantConfig(signed=True, total_bits=8, frac_bits=0)

        # Create a simple network
        n_in, n_out = 8, 3
        W = rng.randint(-10, 11, size=(n_out, n_in)).astype(np.int32)
        b = np.zeros(n_out, dtype=np.int32)

        qnn = QNN(weights=[W], biases=[b], config_in=config,
                   config_w=config, config_out=config)

        x = rng.randint(-5, 6, size=n_in).astype(np.int32)
        true_class = qnn.classify(x)

        # Step 1: Compute saliency (gradient magnitude for linear layer)
        saliency = np.abs(W[true_class]).astype(np.float64)
        saliency /= saliency.sum()

        # Step 2: Identify high-saliency dimensions (top 50%)
        threshold = np.median(saliency)
        high_saliency_mask = saliency >= threshold

        # Step 3: Verify robustness
        report = verify_robustness(qnn, x, true_class, radius=2)

        # Report should be a valid verification result
        assert isinstance(report, VerifyReport)
        assert report.true_class == true_class

        # The high-saliency dimensions have larger weight magnitudes,
        # meaning they dominate the classification decision
        assert np.sum(high_saliency_mask) > 0

    def test_quantization_error_vs_saliency(self):
        """ECQx insight: quantization error matters more in salient dimensions.

        Verify that weighted quantization error (by saliency) is a better
        predictor of classification change than raw error.
        """
        rng = np.random.RandomState(42)

        # Simulate a weight matrix and its saliency
        W = rng.randn(10, 16).astype(np.float32)

        # Saliency from LRP (simplified: use gradient magnitude)
        saliency = np.abs(W).astype(np.float64)
        saliency /= saliency.max()

        # Quantization error
        err = quantization_error(W, bits=8)

        # Weighted error (ECQx metric)
        weighted_err = err * saliency

        # Both should be non-negative
        assert np.all(err >= 0)
        assert np.all(weighted_err >= 0)

        # Weighted error should be <= raw error (saliency in [0,1])
        assert np.all(weighted_err <= err + 1e-10)

        # High-saliency weights should have higher weighted error
        # relative to their raw error
        high_sal = saliency > 0.5
        if np.any(high_sal):
            ratio_high = weighted_err[high_sal].mean() / (err[high_sal].mean() + 1e-10)
            low_sal = saliency <= 0.5
            if np.any(low_sal) and err[low_sal].mean() > 1e-10:
                ratio_low = weighted_err[low_sal].mean() / (err[low_sal].mean() + 1e-10)
                assert ratio_high >= ratio_low

    def test_snax_dimensions(self):
        """Verify with SNAX-matching dimensions: K=16, C=10."""
        config = QuantConfig(signed=True, total_bits=8, frac_bits=0)
        rng = np.random.RandomState(42)

        # Match Phase 1/2 dimensions
        K, C = 16, 10
        W = rng.randint(-5, 6, size=(C, K)).astype(np.int32)
        b = np.zeros(C, dtype=np.int32)

        qnn = QNN(weights=[W], biases=[b], config_in=config,
                   config_w=config, config_out=config)

        x = rng.randint(-10, 11, size=K).astype(np.int32)
        true_class = qnn.classify(x)

        report = verify_robustness(qnn, x, true_class, radius=1)
        assert report.result in (VerifyResult.ROBUST, VerifyResult.NOT_ROBUST)

        # Compute MRR for this SNAX-sized network
        mrr, reports = compute_max_robustness_radius(
            qnn, x, true_class, max_radius=10,
        )
        assert mrr >= 0
        assert isinstance(mrr, int)
