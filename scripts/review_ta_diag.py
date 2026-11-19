#!/usr/bin/env python3
"""Item 12 follow-up: where the ToyAdmos int8-vs-float32 attribution gap comes
from. Float32 IG with the backward pass in float, swapping in pieces of the
int8 forward pass one at a time (ReLU masks, reconstruction y, interpolated
input x~). 100 windows of item 2, IG from the mean normal spectrum, 16 steps.

    python scripts/review_ta_diag.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import review_experiments as RE  # noqa: E402


def grad(fm, xt, masks=None, y=None):
    """d/dx~ mean((x~ - f(x~))^2) in float; masks / y optionally replaced."""
    acts, pre = [xt], []
    for W, b, relu in fm:
        z = acts[-1] @ W.T + b
        pre.append(z)
        acts.append(np.maximum(z, 0) if relu else z)
    yy = acts[-1] if y is None else y
    e = xt - yy
    g = -2.0 * e / xt.shape[1]
    for li, ((W, b, relu), z) in enumerate(zip(reversed(fm), reversed(pre))):
        if relu:
            m = (z > 0) if masks is None else masks[len(fm) - 1 - li]
            g = g * m
        g = g @ W
    return g + 2.0 * e / xt.shape[1]


def main():
    E = RE._ta_setup()
    A, T, layers, bg, fm = E["A"], E["T"], E["layers"], E["bg"], E["fm"]
    L9 = layers[-1]
    res = {k: [] for k in ("float", "int8 masks", "int8 y", "int8 x~", "int8 masks + y + x~", "int8 pipeline")}
    for lab, c, xq in RE._ta_windows(50):
        bq, al = T.baselines("ig", layers, bg, None)
        xr, br = RE._ta_real(xq, layers), RE._ta_real(bq, layers)
        xt_f = br + (al[:, None] / 16.0) * (xr[None] - br)
        xt_q = ((al[:, None] * xq[None].astype(np.int64) + (16 - al[:, None]) * bq.astype(np.int64)) // 16).astype(np.int8)
        acts = A.forward(xt_q, layers)
        masks = [acts[i + 1] != -128 for i in range(len(layers) - 1)] + [None]
        y_q = (acts[-1].astype(np.float64) - L9.zp_y) * L9.s_y
        xt_qr = RE._ta_real(xt_q, layers)
        ref = ((xr[None] - br) * grad(fm, xt_f)).mean(axis=0)

        def at(xt, **kw):
            return ((xr[None] - br) * grad(fm, xt, **kw)).mean(axis=0)
        res["float"].append(RE.pearson(at(xt_f), ref))
        res["int8 masks"].append(RE.pearson(at(xt_f, masks=masks), ref))
        res["int8 y"].append(RE.pearson(at(xt_f, y=y_q), ref))
        res["int8 x~"].append(RE.pearson(at(xt_qr), ref))
        res["int8 masks + y + x~"].append(RE.pearson(at(xt_qr, masks=masks, y=y_q), ref))
        ai = T.int_xai_reference(layers, xq, bq, al, E["sh"])["attr"]
        res["int8 pipeline"].append(RE.pearson(ai, ref))
    print("ToyAdmos IG, 100 windows: float32 backward with pieces of the int8 forward swapped in, r vs float32 IG")
    for k, v in res.items():
        print("  " + RE.summ(k, v))


if __name__ == "__main__":
    main()
