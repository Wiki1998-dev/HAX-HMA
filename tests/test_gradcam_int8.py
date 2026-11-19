# tests/test_gradcam_int8.py
import numpy as np
import pytest
from src.xai.gradcam.gradcam_reference import gradcam


def _quantize_symmetric(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization. Returns (w_i8, scale)."""
    scale = float(np.max(np.abs(w))) / 127.0
    if scale == 0.0:
        scale = 1.0
    w_i8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_i8, scale


def _dequantize(w_i8: np.ndarray, scale: float) -> np.ndarray:
    return w_i8.astype(np.float32) * scale


def _gradcam_backward_gemm_i8(
    d_logit: np.ndarray,   # (C,) float32
    w_fc_i8: np.ndarray,   # (K, C) int8
    scale: float,
) -> np.ndarray:
    """Python model of the INT8 backward GeMM: grad[k] = d_logit @ w_fc_i8.T * scale."""
    # INT32 accumulation, then scale
    acc = d_logit.astype(np.float64) @ w_fc_i8.T.astype(np.float64)
    return (acc * scale).astype(np.float32)


def test_int8_backward_gemm_close_to_fp32():
    """INT8 backward GeMM gradient should be within 1% of float32 reference."""
    rng = np.random.default_rng(42)
    K, C = 16, 10
    w_fc = rng.random((K, C)).astype(np.float32)
    target_class = 3

    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Float32 reference
    grad_fp32 = d_logit @ w_fc.T  # shape (K,)

    # INT8 path
    w_fc_i8, scale = _quantize_symmetric(w_fc)
    grad_i8 = _gradcam_backward_gemm_i8(d_logit, w_fc_i8, scale)

    max_err = float(np.max(np.abs(grad_fp32 - grad_i8)))
    rel_err = max_err / (float(np.max(np.abs(grad_fp32))) + 1e-9)
    assert rel_err < 0.01, f"Relative error {rel_err:.4f} >= 1%"


def test_int8_quantization_range():
    """Quantized weights must stay within [-128, 127]."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 10)).astype(np.float32)
    w_i8, _ = _quantize_symmetric(w)
    assert w_i8.dtype == np.int8
    assert np.all(w_i8 >= -128) and np.all(w_i8 <= 127)


def test_int8_scale_positive():
    """Scale must always be positive."""
    w = np.ones((4, 4), dtype=np.float32) * 0.5
    _, scale = _quantize_symmetric(w)
    assert scale > 0.0


def test_zero_weight_matrix():
    """All-zero weights should not divide by zero."""
    w = np.zeros((8, 4), dtype=np.float32)
    w_i8, scale = _quantize_symmetric(w)
    assert scale == 1.0
    assert np.all(w_i8 == 0)
