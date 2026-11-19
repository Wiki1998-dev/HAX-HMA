"""MLPerf Tiny ToyAdmos autoencoder (ad01_int8.tflite) in GeMMX integer semantics.

Every layer is a per-tensor-quantised FULLY_CONNECTED (weight zero point 0).
On GeMMX a layer is

    acc = x_q @ W^T + b'                      (int8 x int8 -> int32, C = b')
    y_q = clip(floor(acc * M0 / 2^s) + zp_y)  (SIMD requantiser, double_round 0)

with the input zero point and half an output LSB folded into the bias:

    b' = b - zp_x * sum_k W[n, k] + round(2^(s-1) / M0)

so the floor becomes round-half-up. TFLite rounds twice (a rounding doubling
high multiply at bit 31, then a rounding shift); a single multiply-shift
cannot reproduce that, so outputs differ from TFLite by +-1 LSB on ~0.5% of
activations. This module is the bit-exact reference for the simulator.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../../../util/sim/"))
from snax_utils import postprocessing_simd_golden_model  # noqa: E402

DEFAULT_MODEL = os.path.normpath(os.path.join(
    HERE, "../../../../../external/tiny/benchmark/training/anomaly_detection/"
          "trained_models/ad01_int8.tflite"))


@dataclass(frozen=True)
class Layer:
    """One FC layer: W (out, in) int8, folded bias int32, requant params."""

    W: np.ndarray
    bias: np.ndarray       # b' (int32), see module docstring
    M0: int                # Q31 multiplier
    shift: int             # total right shift s
    zp_x: int
    zp_y: int
    s_x: float
    s_w: float
    s_y: float

    @property
    def relu(self) -> bool:
        """Hidden layers: zp_y = -128 with output clamp == fused ReLU."""
        return self.zp_y == -128


def _quantize_multiplier(real: float) -> tuple:
    m, e = np.frexp(real)
    m0 = int(np.round(m * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        e += 1
    return m0, int(e)


def load_layers(model_path: str = DEFAULT_MODEL) -> List[Layer]:
    """Read weights, biases and quantisation from the TFLite file."""
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=model_path)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    layers = []
    for op in it._get_ops_details():
        xi, wi, bi = op["inputs"]
        yi = op["outputs"][0]
        W = it.get_tensor(wi).astype(np.int8)
        b = it.get_tensor(bi).astype(np.int64)
        (s_x, zp_x), (s_w, zp_w), (s_y, zp_y) = (td[xi]["quantization"],
                                                  td[wi]["quantization"],
                                                  td[yi]["quantization"])
        assert zp_w == 0, "per-tensor symmetric weights expected"
        m0, e = _quantize_multiplier(s_x * s_w / s_y)
        s = 31 - e
        bias = b - zp_x * W.astype(np.int64).sum(axis=1) + int(round(2 ** (s - 1) / m0))
        assert np.all(np.abs(bias) < 2 ** 31)
        layers.append(Layer(W, bias.astype(np.int32), m0, s, int(zp_x), int(zp_y),
                            float(s_x), float(s_w), float(s_y)))
    return layers


def requant(acc: np.ndarray, L: Layer) -> np.ndarray:
    """GeMMX SIMD requantisation of int32 accumulators (bias already added)."""
    return postprocessing_simd_golden_model(
        acc.astype(np.int32), 0, L.zp_y, L.shift, 127, -128, 0, L.M0).astype(np.int8)


def forward(xq: np.ndarray, layers: List[Layer]) -> List[np.ndarray]:
    """Batch forward pass. xq: (B, 640) int8. Returns [x, a1, ..., a10]."""
    acts = [xq.astype(np.int8)]
    for L in layers:
        acc = acts[-1].astype(np.int64) @ L.W.T.astype(np.int64) + L.bias
        acts.append(requant(acc, L))
    return acts


def quantize_input(x: np.ndarray, layers: List[Layer]) -> np.ndarray:
    """Float log-mel vectors -> int8 model input."""
    L = layers[0]
    return np.clip(np.round(x / L.s_x + L.zp_x), -128, 127).astype(np.int8)


def anomaly_score(xq: np.ndarray, layers: List[Layer]) -> float:
    """Mean over vectors of the dequantised reconstruction MSE (MLPerf)."""
    y = forward(xq, layers)[-1]
    L0, L9 = layers[0], layers[-1]
    xd = (xq.astype(np.float64) - L0.zp_x) * L0.s_x
    yd = (y.astype(np.float64) - L9.zp_y) * L9.s_y
    return float(np.mean(np.mean((xd - yd) ** 2, axis=1)))
