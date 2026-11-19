#!/usr/bin/env python3
"""Accuracy and Grad-CAM fidelity of the GeMMX-semantics keyword-spotting
(DS-CNN) and visual-wake-words (MobileNetV1) models.

* Accuracy: GeMMX integer model (mobile_int.forward, which the simulator
  reproduces bit-exactly) vs the TFLite int8 reference, and how often the two
  predict the same class, on the full prepared test sets.
* Grad-CAM fidelity: integer Grad-CAM (mobile.gradcam_ref: int8 last feature
  map, alpha = FC row of the predicted class) vs a float32 Grad-CAM of the
  dequantised network (float activations, each layer clipped to the float
  range of its int8 output, alpha = d logit / d feature map). Pearson r over
  the map, top-1 cell agreement.

    python scripts/mobile_gradcam_fidelity.py [--task kws|vww] [--n N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "snax_cluster/sw/xai/mobile_gemmx/data"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import mobile as MB  # noqa: E402
import mobile_int as MI  # noqa: E402
import tflite_eval as TE  # noqa: E402


def float_model(task: str, M: MI.Model):
    """Dequantised float32 network in GeMMX orientation: returns
    forward(x_float) -> (last feature map, logits) and the float FC weights."""
    from tflite_runtime.interpreter import Interpreter

    it = Interpreter(model_path=MI.MODEL_PATHS[task])
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    convs = [o for o in it._get_ops_details() if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    P = []
    for op, L in zip(convs, M.layers):
        xi, wi, bi = op["inputs"]
        qx = td[xi]["quantization_parameters"]
        qw = td[wi]["quantization_parameters"]
        qy = td[op["outputs"][0]]["quantization_parameters"]
        sx = float(qx["scales"][0])
        sw = np.asarray(qw["scales"], dtype=np.float64)
        sy, zy = float(qy["scales"][0]), int(qy["zero_points"][0])
        b = it.get_tensor(bi).astype(np.float64) * sx * sw
        W = L.W.astype(np.float64) * (sw[:, None, None, None] if L.kind == "conv" else sw[:, None, None])
        P.append((L, W, b, (-128 - zy) * sy, (127 - zy) * sy))
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][0]
    xi, wi, bi = fc["inputs"]
    sx = float(td[xi]["quantization_parameters"]["scales"][0])
    sw = float(td[wi]["quantization_parameters"]["scales"][0])
    Wfc = it.get_tensor(wi).astype(np.float64) * sw
    bfc = it.get_tensor(bi).astype(np.float64) * sx * sw

    def conv(x: np.ndarray, L: MI.Layer, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        H, Wd, _ = x.shape
        kh, kw = L.k
        Ho, pt, pb = MI.same_pad(H, kh, L.stride)
        Wo, pl, pr = MI.same_pad(Wd, kw, L.stride)
        xp = np.pad(x, ((pt, pb), (pl, pr), (0, 0)))
        y = np.zeros((Ho, Wo, W.shape[0]))
        s = L.stride
        for i in range(kh):
            for j in range(kw):
                patch = xp[i:i + s * Ho:s, j:j + s * Wo:s]
                y += patch * W[:, i, j] if L.kind == "dw" else patch @ W[:, i, j, :].T
        return y + b

    def forward(x: np.ndarray):
        a = x.astype(np.float64)
        a = a[:, :, None] if a.ndim == 2 else a
        if M.transposed:
            a = a.transpose(1, 0, 2)
        for L, W, b, lo, hi in P:
            a = np.clip(conv(a, L, W, b), lo, hi)
        return a, a.mean(axis=(0, 1)) @ Wfc.T + bfc

    return forward, Wfc


def run(task: str, n_cam: int) -> None:
    M = MI.load_model(task)
    x, y = MI.load_inputs(task)
    tfl = TE.tflite_logits(task, x).argmax(1)
    fwd, Wfc = float_model(task, M)
    pred_i = np.empty(len(y), dtype=np.int64)
    rs, top1 = [], []
    for k in range(len(y)):
        ref = MI.forward(MI.quantize_input(x[k], M), M)
        pred_i[k] = int(np.argmax(ref["logits"]))
        if k >= n_cam:
            continue
        cam_i = MB.gradcam_ref(ref, M).astype(np.float64)
        fm, _ = fwd(x[k])
        cam_f = np.maximum(fm @ Wfc[pred_i[k]], 0).reshape(-1)
        if cam_i.std() == 0 or cam_f.std() == 0:
            continue
        rs.append(np.corrcoef(cam_i, cam_f)[0, 1])
        top1.append(int(np.argmax(cam_i)) == int(np.argmax(cam_f)))
    print(f"{task}: {len(y)} inputs: top-1 GeMMX-int {np.mean(pred_i == y):.4f}, "
          f"TFLite {np.mean(tfl == y):.4f}, same prediction {np.mean(pred_i == tfl):.4f}")
    print(f"{task}: Grad-CAM int8 GeMMX vs float32 ({len(rs)} maps, {M.layers[-1].cout} ch): "
          f"Pearson r {np.mean(rs):.3f} (min {np.min(rs):.3f}, median {np.median(rs):.3f}), "
          f"top-1 cell {np.mean(top1):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["kws", "vww", "both"], default="both")
    ap.add_argument("--n", type=int, default=500, help="inputs for the Grad-CAM comparison")
    a = ap.parse_args()
    for t in (["kws", "vww"] if a.task == "both" else [a.task]):
        run(t, a.n)
