"""Input-level Integrated Gradients and Gradient SHAP for MLPerf Tiny
ResNet-8: an integer reference in GeMMX semantics (what the hardware
schedule reproduces) and a float32 reference (PyTorch autograd on the
dequantised model).

Explained score: the logit of the class predicted for the input x.
  IG:   16 steps from a black image, alpha = 1..16 (/16)
  SHAP: 16 baselines drawn from CIFAR-10 training images, alpha in 0..16 (/16)
  attr = (1/N) sum_i (x - b_i) * d logit / d x~_i,   x~_i = b_i + alpha_i/16 (x - b_i)

Integer backward pass (per sample), all gradients int8 with zero point 0 and
a per-tensor scale S (real = q * S), calibrated statically:
  avgpool+FC  G_add3 = W_fc[pred] broadcast over the 8x8 map, S = s_wfc / 64
  ReLU        G * (act != -128)                         (core masks)
  ADD         identity: both branches receive the same tensor
  conv        transposed convolution with backward weights requantised per
              input channel (int8, scale s_b[ci]); int32 accumulate; SIMD
              requantisation to the output scale with exact per-channel
              multipliers (half an LSB folded into the bias)
  fork sum    a*G1 + b*G2 (a/b ~ S1/S2), requantised     (block-diagonal add)
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import resnet8_int as R  # noqa: E402

N_SAMPLES = 16
EXT = os.path.normpath(os.path.join(HERE, "../../../../../external"))

# backward conv order: (forward conv, input tensor of the forward conv, its shape key)
BWD_CONVS = ["b3c2", "b3c1", "b3sc", "b2c2", "b2c1", "b2sc", "b1c2", "b1c1", "conv1"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def cifar_train(n: int = 512, seed: int = 0) -> np.ndarray:
    d = pickle.load(open(os.path.join(EXT, "cifar-10-batches-py/data_batch_1"), "rb"), encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X[np.random.default_rng(seed).choice(len(X), n, replace=False)]


def baselines(method: str, M: R.Model, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """int8 baselines (N, 32, 32, 3) and alphas (N,) in 0..16."""
    if method == "ig":
        b = R.quantize_image(np.zeros((32, 32, 3)), M)
        return np.repeat(b[None], N_SAMPLES, axis=0), np.arange(1, N_SAMPLES + 1)
    bg = cifar_train()
    idx = rng.choice(len(bg), N_SAMPLES, replace=False)
    alpha = rng.integers(0, 17, size=N_SAMPLES)
    return np.stack([R.quantize_image(bg[i], M) for i in idx]), alpha


def interpolate(xq: np.ndarray, bq: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """x~_i = floor((alpha_i x + (16 - alpha_i) b_i) / 16) in the int8 domain."""
    a = alpha[:, None, None, None].astype(np.int64)
    return ((a * xq[None].astype(np.int64) + (16 - a) * bq.astype(np.int64)) // 16).astype(np.int8)


# --------------------------------------------------------------------------
# Integer backward pass
# --------------------------------------------------------------------------
@dataclass
class BwdConv:
    Wb: np.ndarray        # (cout_fwd, kh, kw, cin_fwd) int8, per-cin scale
    s_b: np.ndarray       # (cin_fwd,) backward weight scales
    stride: int


def backward_weights(M: R.Model) -> Dict[str, BwdConv]:
    """Requantise every conv's real weights per input channel (the backward
    conv's output channel), so a per-tensor gradient feeds every branch."""
    from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=R.DEFAULT_MODEL)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    names = ["conv1", "b1c1", "b1c2", None, "b2c1", "b2c2", "b2sc", None, "b3c1", "b3c2", "b3sc", None]
    out = {}
    for op, nm in zip(it._get_ops_details()[:12], names):
        if nm is None:
            continue
        wi = op["inputs"][1]
        sw = np.asarray(td[wi]["quantization_parameters"]["scales"], dtype=np.float64)
        Wr = it.get_tensor(wi).astype(np.float64) * sw[:, None, None, None]
        s_b = np.abs(Wr).max(axis=(0, 1, 2)) / 127.0
        s_b[s_b == 0] = 1.0
        Wb = np.clip(np.round(Wr / s_b[None, None, None, :]), -127, 127).astype(np.int8)
        out[nm] = BwdConv(Wb, s_b, M.convs[nm].stride)
    return out


def conv_bwd_acc(G: np.ndarray, B: BwdConv, in_hw: Tuple[int, int]) -> np.ndarray:
    """Transposed convolution: (Ho, Wo, Co) int8 -> int32 acc (H, W, Ci)."""
    H, W = in_hw
    co, kh, kw, ci = B.Wb.shape
    s = B.stride
    _, pt, _ = R.same_pad(H, kh, s)
    _, pl, _ = R.same_pad(W, kw, s)
    Ho, Wo = G.shape[:2]
    acc = np.zeros((H + kh + s, W + kw + s, ci), dtype=np.int64)       # with margin
    Wt = B.Wb.astype(np.int64)
    Gi = G.astype(np.int64)
    for i in range(kh):
        for j in range(kw):
            contrib = Gi @ Wt[:, i, j, :]                                # (Ho, Wo, ci)
            acc[i:i + s * Ho:s, j:j + s * Wo:s] += contrib
    return acc[pt:pt + H, pl:pl + W]


def fit_ms(ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel (M0, shift) with M0 / 2^shift = ratio."""
    ms = [R.quantize_multiplier(float(r)) for r in np.atleast_1d(ratio)]
    return np.array([m for m, _ in ms]), np.array([s for _, s in ms])


def requant_g(acc: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """GeMMX SIMD with half an LSB folded in: floor((acc + c) M0 / 2^sh)."""
    m0, sh = fit_ms(np.broadcast_to(ratio, (acc.shape[-1],)))
    half = np.array([int(round(2 ** (s - 1) / m)) for m, s in zip(m0, sh)])
    return R.requant(acc + half, m0, sh, 0, -127, 127)


def mask(G: np.ndarray, act: np.ndarray) -> np.ndarray:
    return np.where(act != -128, G, 0).astype(np.int8)


def best_pair(ratio: float) -> Tuple[int, int]:
    return R._best_pair(ratio)


def fork_sum(G1: np.ndarray, S1: float, G2: np.ndarray, S2: float, S_out: float,
             real: Dict[str, np.ndarray] = None, key: str = "") -> np.ndarray:
    """requant(a G1 + b G2) with a / b ~ S1 / S2 (block-diagonal add launch)."""
    a, b = best_pair(S1 / S2)
    acc = a * G1.astype(np.int64) + b * G2.astype(np.int64)
    if real is not None:
        real[key] = acc * (S2 / b)
    return requant_g(acc, np.array([S2 / b / S_out]))


def int_input_grad(t: Dict[str, np.ndarray], pred: int, M: R.Model, BW: Dict[str, BwdConv],
                   S: Dict[str, float], real: Dict[str, np.ndarray] = None) -> np.ndarray:
    """d logit[pred] / d x (int8, scale S['x']) from one sample's forward
    tensors t. ``real`` (optional) receives every stage's pre-requantisation
    value in real units (for calibration)."""
    def bconv(name: str, Gin: np.ndarray, Sin: float, key: str, in_hw) -> np.ndarray:
        acc = conv_bwd_acc(Gin, BW[name], in_hw)
        units = Sin * BW[name].s_b
        if real is not None:
            real[key] = acc * units
        return requant_g(acc, units / S[key])

    gp = np.clip(M.fc.W[pred, 0, 0, :], -127, 127).astype(np.int8)          # symmetric gradients
    g = mask(np.broadcast_to(gp, (8, 8, 64)).copy(), t["add3"])
    # block 3 (strided 1x1 shortcut)
    b3 = mask(bconv("b3c2", g, S["add3"], "b3c1", (8, 8)), t["b3c1"])
    m3 = bconv("b3c1", b3, S["b3c1"], "m3", (16, 16))
    s3 = bconv("b3sc", g, S["add3"], "s3", (16, 16))
    g = mask(fork_sum(s3, S["s3"], m3, S["m3"], S["add2"], real, "add2"), t["add2"])
    # block 2 (strided 1x1 shortcut)
    b2 = mask(bconv("b2c2", g, S["add2"], "b2c1", (16, 16)), t["b2c1"])
    m2 = bconv("b2c1", b2, S["b2c1"], "m2", (32, 32))
    s2 = bconv("b2sc", g, S["add2"], "s2", (32, 32))
    g = mask(fork_sum(s2, S["s2"], m2, S["m2"], S["add1"], real, "add1"), t["add1"])
    # block 1 (identity shortcut)
    b1 = mask(bconv("b1c2", g, S["add1"], "b1c1", (32, 32)), t["b1c1"])
    m1 = bconv("b1c1", b1, S["b1c1"], "m1", (32, 32))
    g = mask(fork_sum(m1, S["m1"], g, S["add1"], S["conv1"], real, "conv1"), t["conv1"])
    return bconv("conv1", g, S["conv1"], "x", (32, 32))


GRAD_KEYS = ["b3c1", "m3", "s3", "add2", "b2c1", "m2", "s2", "add1", "b1c1", "m1", "conv1", "x"]


def fc_scale() -> float:
    from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=R.DEFAULT_MODEL)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][0]
    return float(td[fc["inputs"][1]]["quantization_parameters"]["scales"][0])


def calibrate(M: R.Model, BW: Dict[str, BwdConv], n_img: int = 8, pct: float = 99.5) -> Dict[str, float]:
    """Static per-tensor gradient scales: the pct-th percentile of |real
    value| over interpolated CIFAR-10 training images maps to 127. Stages are
    fixed in order, each using the already calibrated upstream scales."""
    S: Dict[str, float] = {"add3": fc_scale() / 64}
    S.update({k: 1.0 for k in GRAD_KEYS})
    imgs = cifar_train(64, seed=1)
    rng = np.random.default_rng(5)
    samples = []
    for k in range(n_img):
        xq = R.quantize_image(imgs[k], M)
        bq, al = baselines("shap", M, rng)
        xt = interpolate(xq, bq, al)
        pred = int(np.argmax(R.forward(xq, M)["logits"][:10]))
        samples += [(R.forward(xt[i], M), pred) for i in range(0, N_SAMPLES, 4)]
    for key in GRAD_KEYS:
        vals = []
        for t, pred in samples:
            real: Dict[str, np.ndarray] = {}
            int_input_grad(t, pred, M, BW, S, real)
            vals.append(np.abs(real[key]).ravel())
        S[key] = max(float(np.percentile(np.concatenate(vals), pct)), 1e-30) / 127.0
    return S


def int_attr(xq: np.ndarray, bq: np.ndarray, alpha: np.ndarray, M: R.Model,
             BW: Dict[str, BwdConv], S: Dict[str, float]) -> Tuple[np.ndarray, int]:
    """Integer attribution (32, 32, 3), int64, and the explained class."""
    pred = int(np.argmax(R.forward(xq, M)["logits"][:10]))
    xt = interpolate(xq, bq, alpha)
    acc = np.zeros((32, 32, 3), dtype=np.int64)
    for i in range(len(alpha)):
        gx = int_input_grad(R.forward(xt[i], M), pred, M, BW, S)[:, :, :3]
        acc += (xq.astype(np.int64) - bq[i].astype(np.int64)) * gx.astype(np.int64)
    return acc, pred


# --------------------------------------------------------------------------
# Float32 reference (PyTorch autograd on the dequantised model)
# --------------------------------------------------------------------------
def torch_model():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../../../scripts")))
    from resnet8_gradcam_fidelity import float_model  # noqa: F401
    from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=R.DEFAULT_MODEL)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    names = ["conv1", "b1c1", "b1c2", None, "b2c1", "b2c2", "b2sc", None, "b3c1", "b3c2", "b3sc", None]
    strides = {"conv1": 1, "b1c1": 1, "b1c2": 1, "b2c1": 2, "b2c2": 1, "b2sc": 2,
               "b3c1": 2, "b3c2": 1, "b3sc": 2}
    P = {}
    for op, nm in zip(ops[:12], names):
        if nm is None:
            continue
        xi, wi, bi = op["inputs"]
        sx = float(td[xi]["quantization_parameters"]["scales"][0])
        sw = np.asarray(td[wi]["quantization_parameters"]["scales"], dtype=np.float64)
        W = it.get_tensor(wi).astype(np.float64) * sw[:, None, None, None]   # (co, kh, kw, ci)
        b = it.get_tensor(bi).astype(np.float64) * sx * sw
        P[nm] = (torch.tensor(W.transpose(0, 3, 1, 2), dtype=torch.float64),
                 torch.tensor(b, dtype=torch.float64), strides[nm])
    fc = [o for o in ops if o["op_name"] == "FULLY_CONNECTED"][0]
    xi, wi, bi = fc["inputs"]
    sx = float(td[xi]["quantization_parameters"]["scales"][0])
    sw = float(td[wi]["quantization_parameters"]["scales"][0])
    Wfc = torch.tensor(it.get_tensor(wi).astype(np.float64) * sw)
    bfc = torch.tensor(it.get_tensor(bi).astype(np.float64) * sx * sw)

    def conv(nm, x):
        W, b, s = P[nm]
        k = W.shape[2]
        n = x.shape[2]
        out = -(-n // s)
        total = max((out - 1) * s + k - n, 0)
        x = F.pad(x, (total // 2, total - total // 2, total // 2, total - total // 2))
        return F.conv2d(x, W, b, stride=s)

    def forward(x):                         # x: (B, 3, 32, 32) real pixels
        c1 = F.relu(conv("conv1", x))
        a1 = F.relu(c1 + conv("b1c2", F.relu(conv("b1c1", c1))))
        a2 = F.relu(conv("b2sc", a1) + conv("b2c2", F.relu(conv("b2c1", a1))))
        a3 = F.relu(conv("b3sc", a2) + conv("b3c2", F.relu(conv("b3c1", a2))))
        return a3.mean(dim=(2, 3)) @ Wfc.T + bfc

    return forward


def float_attr(fwd, xq: np.ndarray, bq: np.ndarray, alpha: np.ndarray, M: R.Model,
               pred: int) -> np.ndarray:
    """Float32 attribution (32, 32, 3) on the same int8-domain inputs."""
    import torch
    deq = lambda q: (q.astype(np.float64) - M.zp_in) * M.s_in   # noqa: E731
    x, b = deq(xq), deq(bq)
    xt = b + alpha[:, None, None, None] / 16.0 * (x[None] - b)
    xt_t = torch.tensor(xt.transpose(0, 3, 1, 2), requires_grad=True)
    fwd(xt_t)[:, pred].sum().backward()
    g = xt_t.grad.numpy().transpose(0, 2, 3, 1)
    return ((x[None] - b) * g).mean(axis=0)
