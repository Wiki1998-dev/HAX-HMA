#!/usr/bin/env python3
"""Fidelity of the int8 GeMMX Grad-CAM for MLPerf Tiny ResNet-8 against a
float32 Grad-CAM of the dequantised model, on the 200 MLPerf performance
samples of CIFAR-10.

The integer Grad-CAM (resnet8.gradcam_ref, the computation the simulator
reproduces bit-exactly) uses the int8 feature map of the last residual block
and alpha = FC row of the predicted class (the head, average pool + FC, is
linear). The float reference runs the whole network in float32 with the
dequantised weights and computes alpha = d logit / d feature map.

Metrics per image: Pearson r of the 8x8 maps, top-1 cell agreement, top-8
cell overlap. Also reported: accuracy of the GeMMX-semantics model.
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "snax_cluster/sw/xai/resnet8_gemmx/data"))

import resnet8 as RR  # noqa: E402
import resnet8_int as R  # noqa: E402


def conv_float(x: np.ndarray, L: R.Conv, s_x: float, s_w: np.ndarray) -> np.ndarray:
    """Float conv with the dequantised weights and the TFLite bias."""
    H, W, _ = x.shape
    kh, kw = L.W.shape[1:3]
    Ho, pt, pb = R.same_pad(H, kh, L.stride)
    Wo, pl, pr = R.same_pad(W, kw, L.stride)
    xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
    Wr = L.W.astype(np.float64) * s_w[:, None, None, None]
    y = np.zeros((Ho, Wo, L.W.shape[0]))
    for i in range(kh):
        for j in range(kw):
            y += xp[i:i + L.stride * Ho:L.stride, j:j + L.stride * Wo:L.stride] @ Wr[:, i, j, :].T
    return y


def float_model(path: str = R.DEFAULT_MODEL):
    """Dequantised float32 ResNet-8: returns a forward function."""
    from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=path)
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
        qx, qw = td[xi]["quantization_parameters"], td[wi]["quantization_parameters"]
        sw = np.asarray(qw["scales"], dtype=np.float64)
        W = it.get_tensor(wi).astype(np.int8)
        b = it.get_tensor(bi).astype(np.float64) * float(qx["scales"][0]) * sw
        P[nm] = (R.Conv(W, None, None, None, 0, 0, strides[nm], 0, False), float(qx["scales"][0]), sw, b)
    fc = [o for o in ops if o["op_name"] == "FULLY_CONNECTED"][0]
    xi, wi, bi = fc["inputs"]
    sx = float(td[xi]["quantization_parameters"]["scales"][0])
    sw = float(td[wi]["quantization_parameters"]["scales"][0])
    Wfc = it.get_tensor(wi).astype(np.float64) * sw
    bfc = it.get_tensor(bi).astype(np.float64) * sx * sw

    def conv(nm, x, relu):
        L, s_x, s_w, b = P[nm]
        y = conv_float(x, L, s_x, s_w) + b
        return np.maximum(y, 0) if relu else y

    def forward(img: np.ndarray):
        x = img.astype(np.float64)              # input scale 1, zp -128: real = pixel
        c1 = conv("conv1", x, True)
        a1 = np.maximum(c1 + conv("b1c2", conv("b1c1", c1, True), False), 0)
        a2 = np.maximum(conv("b2sc", a1, False) + conv("b2c2", conv("b2c1", a1, True), False), 0)
        a3 = np.maximum(conv("b3sc", a2, False) + conv("b3c2", conv("b3c1", a2, True), False), 0)
        logits = a3.mean(axis=(0, 1)) @ Wfc.T + bfc
        return a3, logits

    return forward, Wfc


def main() -> None:
    M = R.load_model()
    fwd, Wfc = float_model()
    rs, top1, top8, acc_int, acc_flt = [], [], [], 0, 0
    for k in range(200):
        img, label = RR.load_image(k)
        ref = R.forward(R.quantize_image(img, M), M)
        cam_i = RR.gradcam_ref(ref, M).astype(np.float64)
        pred = int(np.argmax(ref["logits"][:10]))
        a3, logits = fwd(img)
        acc_int += pred == label
        acc_flt += int(np.argmax(logits)) == label
        cam_f = np.maximum(a3 @ Wfc[pred], 0).reshape(-1)       # alpha = d logit / d a3 (x 64)
        if cam_i.std() == 0 or cam_f.std() == 0:
            continue
        rs.append(np.corrcoef(cam_i, cam_f)[0, 1])
        top1.append(int(np.argmax(cam_i)) == int(np.argmax(cam_f)))
        top8.append(len(set(np.argsort(-cam_i)[:8]) & set(np.argsort(-cam_f)[:8])) / 8)
    print(f"ResNet-8, 200 MLPerf CIFAR-10 samples: top-1 GeMMX-int {acc_int / 200:.3f}, "
          f"dequantised float {acc_flt / 200:.3f}")
    print(f"Grad-CAM int8 GeMMX vs float32 (predicted class, {len(rs)} maps): "
          f"Pearson r {np.mean(rs):.3f} (min {np.min(rs):.3f}), top-1 cell {np.mean(top1):.3f}, "
          f"top-8 overlap {np.mean(top8):.3f}")


if __name__ == "__main__":
    main()
