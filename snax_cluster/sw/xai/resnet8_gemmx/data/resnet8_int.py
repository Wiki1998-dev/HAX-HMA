"""MLPerf Tiny ResNet-8 (pretrainedResnet_quant.tflite) in GeMMX integer
semantics, as a NumPy reference for the GeMMX schedule and for accuracy.

GeMMX mapping per op:

* CONV_2D (per-channel int8 weights): implicit im2col on GeMMX, one launch
  per 8 output channels (the SIMD unit has 8 per-column multiplier / shift
  slots). Input zero point and half an LSB folded into the int32 bias, the
  padding filled with the input zero point; floor requantisation becomes
  round-half-up (TFLite rounds twice: +-1 LSB on a few activations).
* ADD (+ fused ReLU): block-diagonal launch acc = a*x1 + b*x2 + c with
  a / b ~ s1 / s2 (|a|, |b| <= 127), c holding the zero points and the
  rounding offset, then the SIMD requantiser to the output scale.
* AVERAGE_POOL_2D 8x8: one GeMM against a ones block, floor((sum + 32) / 64).
* FULLY_CONNECTED (per-tensor): GeMM, like a 1x1 conv.
* SOFTMAX: not needed for the class decision (argmax of the logits).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../../../util/sim/"))
from snax_utils import postprocessing_simd_golden_model  # noqa: E402

DEFAULT_MODEL = os.path.normpath(os.path.join(
    HERE, "../../../../../external/tiny/benchmark/training/image_classification/"
          "trained_models/pretrainedResnet_quant.tflite"))


def quantize_multiplier(real: float) -> Tuple[int, int]:
    """real ~ M0 / 2^shift with M0 in [2^30, 2^31)."""
    m, e = np.frexp(real)
    m0 = int(np.round(m * (1 << 31)))
    if m0 == (1 << 31):
        m0 //= 2
        e += 1
    return m0, 31 - int(e)


def requant(acc: np.ndarray, m0: np.ndarray, sh: np.ndarray, zp: int,
            lo: int = -128, hi: int = 127) -> np.ndarray:
    """GeMMX SIMD: floor(acc * M0 / 2^shift) + zp, clip; per-channel M0 / shift
    broadcast over the last axis."""
    out = np.empty(acc.shape, dtype=np.int32)
    for c in range(acc.shape[-1]):
        out[..., c] = postprocessing_simd_golden_model(
            acc[..., c].astype(np.int32), 0, zp, int(sh[c]), hi, lo, 0, int(m0[c]))
    return out.astype(np.int8)


@dataclass
class Conv:
    W: np.ndarray        # (cout, kh, kw, cin) int8
    bias: np.ndarray     # folded int32 (cout,)
    m0: np.ndarray       # (cout,)
    sh: np.ndarray       # (cout,)
    zp_x: int
    zp_y: int
    stride: int
    pad: int             # TF 'SAME' padding before (top / left); see same_pad()
    relu: bool


@dataclass
class Add:
    a: int
    b: int
    c: int               # int32 constant (zero points + rounding)
    m0: int
    sh: int
    zp_y: int


@dataclass
class Model:
    convs: Dict[str, Conv]
    adds: List[Add]
    fc: Conv             # as a 1x1 conv on a 1x1 map
    s_in: float
    zp_in: int


def _conv_from(it, td, op, stride: int) -> Conv:
    xi, wi, bi = op["inputs"]
    yi = op["outputs"][0]
    W = it.get_tensor(wi).astype(np.int8)
    b = it.get_tensor(bi).astype(np.int64)
    qx, qw, qy = (td[i]["quantization_parameters"] for i in (xi, wi, yi))
    s_x, zp_x = float(qx["scales"][0]), int(qx["zero_points"][0])
    s_y, zp_y = float(qy["scales"][0]), int(qy["zero_points"][0])
    sw = np.asarray(qw["scales"], dtype=np.float64)
    if sw.size == 1:
        sw = np.repeat(sw, W.shape[0])
    ms = [quantize_multiplier(s_x * s / s_y) for s in sw]
    m0 = np.array([m for m, _ in ms], dtype=np.int64)
    sh = np.array([s for _, s in ms], dtype=np.int64)
    wsum = W.reshape(W.shape[0], -1).astype(np.int64).sum(axis=1)
    half = np.array([int(round(2 ** (s - 1) / m)) for m, s in zip(m0, sh)], dtype=np.int64)
    bias = b - zp_x * wsum + half
    assert np.all(np.abs(bias) < 2 ** 31)
    k = W.shape[1] if W.ndim == 4 else 1
    return Conv(W if W.ndim == 4 else W[:, None, None, :], bias.astype(np.int32), m0, sh,
                zp_x, zp_y, stride, (k - 1) // 2, zp_y == -128)


def _best_pair(ratio: float) -> Tuple[int, int]:
    """Integers a, b in [1, 127] with a / b closest to ratio."""
    f = Fraction(ratio).limit_denominator(127)
    best = (1, 1, 1e9)
    for b in range(1, 128):
        a = int(round(ratio * b))
        if 1 <= a <= 127 and abs(a / b - ratio) < best[2]:
            best = (a, b, abs(a / b - ratio))
    del f
    return best[0], best[1]


def load_model(path: str = DEFAULT_MODEL) -> Model:
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    names = ["conv1", "b1c1", "b1c2", None, "b2c1", "b2c2", "b2sc", None,
             "b3c1", "b3c2", "b3sc", None]
    strides = {"conv1": 1, "b1c1": 1, "b1c2": 1, "b2c1": 2, "b2c2": 1, "b2sc": 2,
               "b3c1": 2, "b3c2": 1, "b3sc": 2}
    convs, adds = {}, []
    for op, nm in zip(ops[:12], names):
        if op["op_name"] == "CONV_2D":
            convs[nm] = _conv_from(it, td, op, strides[nm])
        else:
            x1, x2 = op["inputs"]
            q1, q2, qy = (td[i]["quantization_parameters"] for i in (x1, x2, op["outputs"][0]))
            s1, z1 = float(q1["scales"][0]), int(q1["zero_points"][0])
            s2, z2 = float(q2["scales"][0]), int(q2["zero_points"][0])
            sy, zy = float(qy["scales"][0]), int(qy["zero_points"][0])
            a, b = _best_pair(s1 / s2)
            m0, sh = quantize_multiplier((s2 / b) / sy)
            c = -(a * z1 + b * z2) + int(round(2 ** (sh - 1) / m0))
            adds.append(Add(a, b, c, m0, sh, zy))
    fc_op = [o for o in ops if o["op_name"] == "FULLY_CONNECTED"][0]
    fc = _conv_from(it, td, fc_op, 1)
    qi = td[ops[0]["inputs"][0]]["quantization_parameters"]
    return Model(convs, adds, fc, float(qi["scales"][0]), int(qi["zero_points"][0]))


def same_pad(n: int, k: int, s: int) -> Tuple[int, int, int]:
    """TF 'SAME': output ceil(n / s), total padding split with the extra row
    after (stride-2 3x3 on an even size pads 0 before, 1 after)."""
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return out, total // 2, total - total // 2


def conv2d(x: np.ndarray, L: Conv) -> np.ndarray:
    """x: (H, W, Cin) int8 -> (H', W', Cout) int8, GeMMX semantics; padding
    filled with the input zero point."""
    H, Wd, Ci = x.shape
    s = L.stride
    kh, kw = L.W.shape[1:3]
    Ho, pt, pb = same_pad(H, kh, s)
    Wo, pl, pr = same_pad(Wd, kw, s)
    xp = np.full((H + pt + pb, Wd + pl + pr, Ci), L.zp_x, dtype=np.int64)
    xp[pt:pt + H, pl:pl + Wd] = x
    acc = np.zeros((Ho, Wo, L.W.shape[0]), dtype=np.int64)
    Wt = L.W.astype(np.int64)
    for i in range(kh):
        for j in range(kw):
            patch = xp[i:i + s * Ho:s, j:j + s * Wo:s]           # (Ho, Wo, Ci)
            acc += patch @ Wt[:, i, j, :].T
    acc += L.bias
    return requant(acc, L.m0, L.sh, L.zp_y)


def add(x1: np.ndarray, x2: np.ndarray, A: Add) -> np.ndarray:
    acc = A.a * x1.astype(np.int64) + A.b * x2.astype(np.int64) + A.c
    return requant(acc, np.full(acc.shape[-1], A.m0), np.full(acc.shape[-1], A.sh), A.zp_y)


def avgpool(x: np.ndarray) -> np.ndarray:
    """8x8 global average pool, floor((sum + 32) / 64) (same scale / zp)."""
    s = x.astype(np.int64).sum(axis=(0, 1))
    return np.clip(np.floor_divide(s + 32, 64), -128, 127).astype(np.int8)


def forward(xq: np.ndarray, M: Model) -> Dict[str, np.ndarray]:
    """xq: (32, 32, 3) int8. Returns all intermediate tensors and logits."""
    c = M.convs
    t = {"x": xq}
    t["conv1"] = conv2d(xq, c["conv1"])
    t["b1c1"] = conv2d(t["conv1"], c["b1c1"])
    t["b1c2"] = conv2d(t["b1c1"], c["b1c2"])
    t["add1"] = add(t["conv1"], t["b1c2"], M.adds[0])
    t["b2c1"] = conv2d(t["add1"], c["b2c1"])
    t["b2c2"] = conv2d(t["b2c1"], c["b2c2"])
    t["b2sc"] = conv2d(t["add1"], c["b2sc"])
    t["add2"] = add(t["b2sc"], t["b2c2"], M.adds[1])
    t["b3c1"] = conv2d(t["add2"], c["b3c1"])
    t["b3c2"] = conv2d(t["b3c1"], c["b3c2"])
    t["b3sc"] = conv2d(t["add2"], c["b3sc"])
    t["add3"] = add(t["b3sc"], t["b3c2"], M.adds[2])
    t["pool"] = avgpool(t["add3"])
    t["logits"] = conv2d(t["pool"][None, None, :], M.fc)[0, 0]
    return t


def quantize_image(img: np.ndarray, M: Model) -> np.ndarray:
    return np.clip(np.round(img / M.s_in + M.zp_in), -128, 127).astype(np.int8)
