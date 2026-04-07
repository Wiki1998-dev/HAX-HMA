"""INT8 quantization matching SNAX GeMM accelerator scheme.

Implements symmetric per-tensor uniform quantization as used by SNAX GeMM:
  û = clamp(round(x / scale), -128, 127)
  x_approx = û * scale

where scale = max(|x|) / 127 (symmetric range).

This module also implements the QVIP quantization encoding:
  û = clamp(floor(2^F * u), C^lb, C^ub)

for formal verification of quantized neural networks.

References:
    - QVIP (Zhang et al., ASE'22): Section 2.2, quantization of DNNs
    - SNAX GeMM: INT8 symmetric per-tensor, 8x8x8 tiles
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QuantConfig:
    """Quantization configuration tuple (τ, Q, F) from QVIP paper.

    Attributes:
        signed: True for signed (τ=±), False for unsigned (τ=+)
        total_bits: Q, total number of quantization bits
        frac_bits: F, number of fractional bits
    """
    signed: bool
    total_bits: int
    frac_bits: int

    @property
    def clamp_lb(self) -> int:
        """Lower bound C^lb of quantization grid."""
        if self.signed:
            return -(2 ** (self.total_bits - 1))
        return 0

    @property
    def clamp_ub(self) -> int:
        """Upper bound C^ub of quantization grid."""
        if self.signed:
            return 2 ** (self.total_bits - 1) - 1
        return 2 ** self.total_bits - 1

    @property
    def scale(self) -> float:
        """Scale factor: 2^(-F)."""
        return 2.0 ** (-self.frac_bits)

    @property
    def n_levels(self) -> int:
        """Number of distinct quantization levels."""
        return self.clamp_ub - self.clamp_lb + 1


# Standard configurations
SNAX_INT8 = QuantConfig(signed=True, total_bits=8, frac_bits=0)
QVIP_DEFAULT = QuantConfig(signed=True, total_bits=8, frac_bits=4)


def quantize_uniform(x: np.ndarray, config: QuantConfig) -> np.ndarray:
    """Quantize float values to fixed-point integers per QVIP encoding.

    Implements: û = clamp(floor(2^F * u), C^lb, C^ub)

    Args:
        x: float array to quantize
        config: quantization configuration

    Returns:
        Integer array of quantized values in [C^lb, C^ub]
    """
    scaled = np.floor(x * (2.0 ** config.frac_bits))
    return np.clip(scaled, config.clamp_lb, config.clamp_ub).astype(np.int32)


def dequantize_uniform(x_q: np.ndarray, config: QuantConfig) -> np.ndarray:
    """Dequantize fixed-point integers back to float.

    Implements: x_approx = û * 2^(-F)

    Args:
        x_q: integer array of quantized values
        config: quantization configuration

    Returns:
        Float array of dequantized values
    """
    return x_q.astype(np.float64) * config.scale


def quantize_symmetric(x: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization matching SNAX GeMM scheme.

    scale = max(|x|) / (2^(bits-1) - 1)
    x_q = clamp(round(x / scale), -2^(bits-1), 2^(bits-1) - 1)

    Args:
        x: float array to quantize
        bits: bit width (default 8 for INT8)

    Returns:
        Tuple of (quantized int array, scale factor)
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    abs_max = np.max(np.abs(x))
    if abs_max == 0:
        return np.zeros_like(x, dtype=np.int8), 1.0

    scale = abs_max / qmax
    x_q = np.clip(np.round(x / scale), qmin, qmax).astype(np.int8)
    return x_q, float(scale)


def dequantize_symmetric(x_q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize symmetric quantized values.

    Args:
        x_q: quantized integer array
        scale: scale factor from quantize_symmetric

    Returns:
        Float array of dequantized values
    """
    return x_q.astype(np.float64) * scale


def quantization_error(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """Compute per-element quantization error for symmetric scheme.

    Args:
        x: original float array
        bits: bit width

    Returns:
        Per-element absolute quantization error |x - dequant(quant(x))|
    """
    x_q, scale = quantize_symmetric(x, bits)
    x_deq = dequantize_symmetric(x_q, scale)
    return np.abs(x.astype(np.float64) - x_deq)


def quantization_error_bound(x: np.ndarray, bits: int = 8) -> float:
    """Compute worst-case quantization error bound for symmetric scheme.

    For symmetric quantization with scale s:
        max error = s/2 = max(|x|) / (2 * (2^(bits-1) - 1))

    Args:
        x: original float array
        bits: bit width

    Returns:
        Upper bound on per-element quantization error
    """
    qmax = 2 ** (bits - 1) - 1
    abs_max = float(np.max(np.abs(x)))
    if abs_max == 0:
        return 0.0
    scale = abs_max / qmax
    return scale / 2.0


def quantize_layer_weights(
    weights: np.ndarray,
    bias: Optional[np.ndarray],
    config: QuantConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Quantize a layer's weights and bias using QVIP-style encoding.

    Weights use config C_w, bias uses C_b (same config here for simplicity).

    Args:
        weights: float weight matrix (out_features, in_features)
        bias: optional float bias vector (out_features,)
        config: quantization configuration

    Returns:
        Tuple of (quantized weights, quantized bias or None)
    """
    w_q = quantize_uniform(weights, config)
    b_q = quantize_uniform(bias, config) if bias is not None else None
    return w_q, b_q
