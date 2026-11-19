#!/usr/bin/env python3
"""Fidelity of the int8 GeMMX-semantics input attributions (Integrated
Gradients, Gradient SHAP) for MLPerf Tiny ResNet-8 against float32
attributions (PyTorch autograd on the dequantised model), on MLPerf
performance samples of CIFAR-10.

Metrics per image on the channel-summed 32x32 map: Pearson r, Spearman rho,
top-64 pixel overlap; plus Pearson r of the full 32x32x3 attribution.

Usage: python scripts/resnet8_xai_fidelity.py [--images N] [--pct P]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "snax_cluster/sw/xai/resnet8_gemmx/data"))

import resnet8 as RR  # noqa: E402
import resnet8_int as R  # noqa: E402
import resnet8_xai as X  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--pct", type=float, default=99.5)
    a = ap.parse_args()
    from scipy.stats import pearsonr, spearmanr
    M = R.load_model()
    BW = X.backward_weights(M)
    S = X.calibrate(M, BW, pct=a.pct)
    fwd = X.torch_model()
    res = {"ig": [], "shap": []}
    for k in range(a.images):
        img, _ = RR.load_image(k)
        xq = R.quantize_image(img, M)
        for m in ("ig", "shap"):
            bq, al = X.baselines(m, M, np.random.default_rng(100 + k))
            ai, pred = X.int_attr(xq, bq, al, M, BW, S)
            af = X.float_attr(fwd, xq, bq, al, M, pred)
            mi, mf = ai.sum(axis=2).ravel().astype(np.float64), af.sum(axis=2).ravel()
            top = len(set(np.argsort(-np.abs(mi))[:64]) & set(np.argsort(-np.abs(mf))[:64])) / 64
            res[m].append((pearsonr(mi, mf)[0], spearmanr(mi, mf)[0], top,
                           pearsonr(ai.ravel().astype(np.float64), af.ravel())[0]))
    print(f"ResNet-8 int8 GeMMX-semantics attribution vs float32, {a.images} images "
          f"(calibration percentile {a.pct})")
    for m in ("ig", "shap"):
        r = np.array(res[m])
        print(f"  {m:4s}: map pearson {r[:, 0].mean():.3f} (min {r[:, 0].min():.3f})  "
              f"spearman {r[:, 1].mean():.3f}  top-64 overlap {r[:, 2].mean():.3f}  "
              f"full pearson {r[:, 3].mean():.3f}")


if __name__ == "__main__":
    main()
