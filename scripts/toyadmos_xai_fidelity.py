#!/usr/bin/env python3
"""Fidelity of the int8 GeMMX attributions (Integrated Gradients, Gradient
SHAP) for ToyAdmos against float32 attributions on the dequantised model.

The integer pipeline used here (toyadmos.int_xai_reference) is the one the
simulator reproduces bit-exactly, so this extends the single simulated input
to many real windows. Per explained window: the most anomalous 5-frame window
of each of N anomalous and N normal ToyCar test clips.

Metrics per window: Pearson r, Spearman rho, top-k feature overlap (k = 32),
sign agreement on the top-k float features.

Usage: python scripts/toyadmos_xai_fidelity.py [--clips N]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "snax_cluster/sw/xai/toyadmos_gemmx/data"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import ad01_int as A  # noqa: E402
import toyadmos as T  # noqa: E402


def float_model(layers: List[A.Layer]) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """Dequantised weights / biases: (W_real (out, in), b_real, relu)."""
    out = []
    for L in layers:
        b_real = (L.bias.astype(np.float64) - int(round(2 ** (L.shift - 1) / L.M0))
                  + L.zp_x * L.W.astype(np.int64).sum(axis=1)) * L.s_x * L.s_w
        out.append((L.W.astype(np.float64) * L.s_w, b_real, L.relu))
    return out


def score_grad(fm, x: np.ndarray) -> np.ndarray:
    """d/dx mean((x - f(x))^2) for a batch x (B, 640)."""
    acts, pre = [x], []
    for W, b, relu in fm:
        z = acts[-1] @ W.T + b
        pre.append(z)
        acts.append(np.maximum(z, 0) if relu else z)
    e = x - acts[-1]                                   # (B, 640)
    g = -2.0 * e / x.shape[1]                          # d loss / d y
    for (W, b, relu), z in zip(reversed(fm), reversed(pre)):
        if relu:
            g = g * (z > 0)
        g = g @ W
    return g + 2.0 * e / x.shape[1]                    # + direct term


def float_attr(fm, x: np.ndarray, base: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    xt = base + (alpha[:, None] / 16.0) * (x[None] - base)
    return ((x[None] - base) * score_grad(fm, xt)).mean(axis=0)


def metrics(a_int: np.ndarray, a_flt: np.ndarray, k: int = 32) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    ti = set(np.argsort(-np.abs(a_int))[:k])
    tf = np.argsort(-np.abs(a_flt))[:k]
    return {"pearson": pearsonr(a_int, a_flt)[0], "spearman": spearmanr(a_int, a_flt)[0],
            "topk": len(ti & set(tf)) / k,
            "sign": float(np.mean(np.sign(a_int[tf]) == np.sign(a_flt[tf])))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=50)
    args = ap.parse_args()
    layers, feats, bg = T.load_data()
    fm = float_model(layers)
    sh = T.calibrate(layers, bg, np.random.default_rng(7))
    scores = np.load(os.path.join(ROOT, "external/toycar/gemmx_int_scores.npy"))
    rng = np.random.default_rng(3)
    picks = []
    for lab in (1, 0):
        idx = np.where(feats["labels"] == lab)[0]
        picks += [(lab, int(c)) for c in rng.choice(idx, args.clips, replace=False)]
    L0 = layers[0]
    res = {"ig": [], "shap": []}
    for lab, c in picks:
        xq_all = A.quantize_input(feats["vectors"][c], layers)
        y = A.forward(xq_all, layers)[-1]
        err = np.mean(((xq_all.astype(np.float64) - L0.zp_x) * L0.s_x
                       - (y.astype(np.float64) - layers[-1].zp_y) * layers[-1].s_y) ** 2, axis=1)
        xq = xq_all[int(np.argmax(err))]
        for m in ("ig", "shap"):
            bq, al = T.baselines(m, layers, bg, np.random.default_rng(11 + c))
            r = T.int_xai_reference(layers, xq, bq, al, sh)
            xr = (xq.astype(np.float64) - L0.zp_x) * L0.s_x
            br = (bq.astype(np.float64) - L0.zp_x) * L0.s_x
            res[m].append(metrics(r["attr"].astype(np.float64), float_attr(fm, xr, br, al)))
    print(f"ToyAdmos int8 GeMMX attribution vs float32 (dequantised model), "
          f"{len(picks)} windows ({args.clips} anomalous + {args.clips} normal clips)")
    for m in ("ig", "shap"):
        arr = {k: np.array([d[k] for d in res[m]]) for k in res[m][0]}
        print(f"  {m:4s}: " + "  ".join(f"{k} {v.mean():.3f} (min {v.min():.3f})" for k, v in arr.items()))


if __name__ == "__main__":
    main()
