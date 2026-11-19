"""MLPerf Tiny keyword spotting (DS-CNN, kws_ref_model.tflite) and visual wake
words (MobileNetV1 0.25, vww_96_int8.tflite) in GeMMX integer semantics.

Both are a plain chain: CONV_2D, then (DEPTHWISE_CONV_2D, CONV_2D 1x1) pairs,
AVERAGE_POOL_2D over the whole map, FULLY_CONNECTED, SOFTMAX. All convs
use TF 'SAME' padding and per-channel int8 weights; every activation after
the first conv has zero point -128 (ReLU folded into the clip).

GeMMX semantics (as in resnet8_int):
* convs and depthwise convs: int32 accumulate, input zero point and half an
  LSB folded into the bias, padding filled with the input zero point, SIMD
  requantiser floor(acc * M0 / 2^shift) + zp (TFLite rounds twice: +-1 LSB
  on a few activations);
* average pool: round(sum / count), exact for odd counts (no ties);
* FULLY_CONNECTED: per-tensor requant, like a 1x1 conv;
* SOFTMAX: not needed (argmax of the logits).

KWS runs transposed (time along the width: input (10, 49) instead of
(49, 10), kernels transposed accordingly); the result is identical, but the
25-wide time axis fills GeMMX's 8-pixel rows far better than the 5-wide
frequency axis.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../resnet8_gemmx/data")))
from resnet8_int import quantize_multiplier, requant, same_pad  # noqa: E402

TINY = os.path.normpath(os.path.join(HERE, "../../../../../external/tiny/benchmark/training"))
MODEL_PATHS = {
    "kws": os.path.join(TINY, "keyword_spotting/trained_models/kws_ref_model.tflite"),
    "vww": os.path.join(TINY, "visual_wake_words/trained_models/vww_96_int8.tflite"),
}


@dataclass
class Layer:
    """CONV_2D / DEPTHWISE_CONV_2D / FULLY_CONNECTED in GeMMX semantics."""

    kind: str            # "conv" | "dw" | "fc"
    W: np.ndarray        # conv / fc: (cout, kh, kw, cin); dw: (c, kh, kw)
    bias: np.ndarray     # folded int32 (cout,)
    m0: np.ndarray       # (cout,)
    sh: np.ndarray       # (cout,)
    zp_x: int
    zp_y: int
    stride: int

    @property
    def cout(self) -> int:
        return self.W.shape[0]

    @property
    def k(self) -> Tuple[int, int]:
        return (self.W.shape[1], self.W.shape[2]) if self.kind != "fc" else (1, 1)


@dataclass
class Model:
    task: str
    layers: List[Layer]  # convs and depthwise convs in order
    fc: Layer
    pool_count: int
    in_shape: Tuple[int, int, int]   # (H, W, C) as run on GeMMX (KWS transposed)
    s_in: float
    zp_in: int
    n_classes: int
    transposed: bool


def _layer(it, td, op, kind: str, stride: int) -> Layer:
    xi, wi, bi = op["inputs"]
    yi = op["outputs"][0]
    W = it.get_tensor(wi).astype(np.int8)
    b = it.get_tensor(bi).astype(np.int64)
    qx, qw, qy = (td[i]["quantization_parameters"] for i in (xi, wi, yi))
    s_x, zp_x = float(qx["scales"][0]), int(qx["zero_points"][0])
    s_y, zp_y = float(qy["scales"][0]), int(qy["zero_points"][0])
    if kind == "dw":
        W = np.ascontiguousarray(W[0].transpose(2, 0, 1))         # (1,kh,kw,c) -> (c,kh,kw)
    elif kind == "fc":
        W = W[:, None, None, :]
    sw = np.asarray(qw["scales"], dtype=np.float64)
    if sw.size == 1:
        sw = np.repeat(sw, W.shape[0])
    ms = [quantize_multiplier(s_x * s / s_y) for s in sw]
    m0 = np.array([m for m, _ in ms], dtype=np.int64)
    sh = np.array([s for _, s in ms], dtype=np.int64)
    wsum = W.reshape(W.shape[0], -1).astype(np.int64).sum(axis=1)
    half = np.array([int(round(2 ** (s - 1) / m)) for m, s in zip(m0, sh)], dtype=np.int64)
    bias = b - zp_x * wsum + half
    # Pruned channels (all weights zero, |bias| ~ 2^30, tiny scale) output a
    # constant: TFLite's fixed-point result of bias * real, re-encoded as
    # bias' = 2 * const with a x1/2 multiplier so it fits int32.
    dead = ~W.reshape(W.shape[0], -1).any(axis=1)
    for c in np.where(dead)[0]:
        bias[c] = 2 * tflite_mul(int(b[c]), int(m0[c]), int(sh[c]))
        m0[c], sh[c] = 1 << 30, 31
    assert np.all(np.abs(bias) < 2 ** 31)
    return Layer(kind, W, bias.astype(np.int32), m0, sh, zp_x, zp_y, stride)


def tflite_mul(x: int, m0: int, sh: int) -> int:
    """TFLite MultiplyByQuantizedMultiplier for a right shift: rounding
    doubling high multiply by m0, then rounding divide by 2^(sh - 31)."""
    ab = x * m0
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    v = ab + nudge
    hi = -((-v) >> 31) if v < 0 else v >> 31                     # truncating division
    r = sh - 31
    if r <= 0:
        return hi << -r
    mask = (1 << r) - 1
    rem, thr = hi & mask, (mask >> 1) + (1 if hi < 0 else 0)
    return (hi >> r) + (1 if rem > thr else 0)


def _transpose(L: Layer) -> Layer:
    W = L.W.transpose(0, 2, 1, 3) if L.kind == "conv" else L.W.transpose(0, 2, 1)
    return Layer(L.kind, np.ascontiguousarray(W), L.bias, L.m0, L.sh, L.zp_x, L.zp_y, L.stride)


def load_model(task: str) -> Model:
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=MODEL_PATHS[task])
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    layers, fc, count = [], None, 0
    for op in it._get_ops_details():
        name = op["op_name"]
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            hin = td[op["inputs"][0]]["shape"][1]
            hout = td[op["outputs"][0]]["shape"][1]
            stride = int(round(hin / hout))
            assert -(-hin // stride) == hout
            layers.append(_layer(it, td, op, "conv" if name == "CONV_2D" else "dw", stride))
        elif name == "AVERAGE_POOL_2D":
            sh = td[op["inputs"][0]]["shape"]
            count = int(sh[1] * sh[2])
        elif name == "FULLY_CONNECTED":
            fc = _layer(it, td, op, "fc", 1)
    inp = td[it.get_input_details()[0]["index"]]
    s_in, zp_in = inp["quantization"]
    shape = tuple(int(v) for v in inp["shape"][1:])
    transposed = task == "kws"
    if transposed:
        layers = [_transpose(L) for L in layers]
        shape = (shape[1], shape[0], shape[2])
    return Model(task, layers, fc, count, shape, float(s_in), int(zp_in), fc.cout, transposed)


def conv2d(x: np.ndarray, L: Layer) -> np.ndarray:
    """x: (H, W, Cin) int8 -> (H', W', Cout) int8; conv or depthwise."""
    H, Wd, Ci = x.shape
    s = L.stride
    kh, kw = L.k
    Ho, pt, pb = same_pad(H, kh, s)
    Wo, pl, pr = same_pad(Wd, kw, s)
    xp = np.full((H + pt + pb, Wd + pl + pr, Ci), L.zp_x, dtype=np.int64)
    xp[pt:pt + H, pl:pl + Wd] = x
    acc = np.zeros((Ho, Wo, L.cout), dtype=np.int64)
    Wt = L.W.astype(np.int64)
    for i in range(kh):
        for j in range(kw):
            patch = xp[i:i + s * Ho:s, j:j + s * Wo:s]
            acc += patch * Wt[:, i, j] if L.kind == "dw" else patch @ Wt[:, i, j, :].T
    acc += L.bias
    return requant(acc, L.m0, L.sh, L.zp_y)


def avgpool(x: np.ndarray) -> np.ndarray:
    """Global average pool, round(sum / count) on the raw int8 values (input
    and output share scale and zero point); exact for odd counts."""
    n = x.shape[0] * x.shape[1]
    assert n % 2 == 1
    s = x.astype(np.int64).sum(axis=(0, 1))
    return np.clip(np.floor_divide(s + n // 2, n), -128, 127).astype(np.int8)


def forward(xq: np.ndarray, M: Model) -> Dict[str, np.ndarray]:
    """xq: input as run on GeMMX (KWS transposed). Returns every activation
    ("l0" .. "lN"), the pooled vector and the logits."""
    t = {"x": xq}
    a = xq
    for i, L in enumerate(M.layers):
        a = conv2d(a, L)
        t[f"l{i}"] = a
    t["pool"] = avgpool(a)
    t["logits"] = conv2d(t["pool"][None, None, :], M.fc)[0, 0]
    return t


def quantize_input(x: np.ndarray, M: Model) -> np.ndarray:
    """Float model input (KWS: (49, 10) MFCC; VWW: (96, 96, 3) in [0, 1]) ->
    int8 tensor in GeMMX orientation."""
    q = np.clip(np.round(x / M.s_in + M.zp_in), -128, 127).astype(np.int8)
    if q.ndim == 2:
        q = q[:, :, None]
    return np.ascontiguousarray(q.transpose(1, 0, 2)) if M.transposed else q


def load_inputs(task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Prepared inputs (scripts/kws_vww_data.py) as float model inputs + labels."""
    ext = os.path.normpath(os.path.join(HERE, "../../../../../external"))
    if task == "kws":
        d = np.load(os.path.join(ext, "kws/features_test.npz"))
        return d["x"].astype(np.float32), d["y"]
    d = np.load(os.path.join(ext, "vww/images.npz"))
    return d["x"].astype(np.float32) / 255.0, d["y"]
