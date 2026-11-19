#!/usr/bin/env python3
"""Reviewer experiments (docs/REVIEW_EXPERIMENTS.md), on the bit-exact Python
models of the GeMMX pipelines.

    python scripts/review_experiments.py <item> [--workers 4]

items: 1 (ResNet-8 full test set), r8 (per-image ResNet-8 data for 2/4/5/13),
2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15.

Every random choice is seeded; the seed is printed with the result. The
integer attributions come from the same functions the simulator was checked
against (resnet8_xai.int_attr, toyadmos.int_xai_reference); where an item
needs a variant (another mask, sample count, precision), the variant is a
local copy that is first checked to reproduce the original exactly.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("snax_cluster/sw/xai/resnet8_gemmx/data", "snax_cluster/sw/xai/toyadmos_gemmx/data",
          "snax_cluster/sw/xai/mobile_gemmx/data", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, p))
CACHE = os.path.join(ROOT, "external/review")
os.makedirs(CACHE, exist_ok=True)
N_BOOT = 10_000
BOOT_SEED = 0


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def boot_ci(v: np.ndarray, seed: int = BOOT_SEED) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean over inputs (10,000 resamples)."""
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: int, n: int) -> Tuple[float, float]:
    z = 1.959963984540054
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def summ(name: str, v) -> str:
    v = np.asarray(v, dtype=np.float64)
    lo, hi = boot_ci(v)
    return f"{name}: mean {v.mean():.4f}  min {v.min():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(v)})"


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pool_map(fn: Callable, items: List, workers: int) -> List:
    if workers <= 1:
        return [fn(i) for i in items]
    from multiprocessing import Pool
    with Pool(workers) as p:
        return p.map(fn, items, chunksize=1)


# --------------------------------------------------------------------------
# Item 1: ResNet-8 on the full CIFAR-10 test set
# --------------------------------------------------------------------------
def _cifar_test():
    import pickle
    d = pickle.load(open(os.path.join(ROOT, "external/cifar-10-batches-py/test_batch"), "rb"),
                    encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])


def _r8_int_pred(chunk: Tuple[int, int]) -> np.ndarray:
    import resnet8_int as R
    M = R.load_model()
    X, _ = _cifar_test()
    return np.array([int(np.argmax(R.forward(R.quantize_image(X[i], M), M)["logits"][:10]))
                     for i in range(*chunk)])


def item1(a) -> None:
    import resnet8_int as R
    from tflite_runtime.interpreter import Interpreter
    X, y = _cifar_test()
    it = Interpreter(model_path=R.DEFAULT_MODEL, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][0]
    M = R.load_model()
    tfl_logit, tfl_out = [], []
    for i in range(len(y)):
        it.set_tensor(inp["index"], R.quantize_image(X[i], M)[None])
        it.invoke()
        tfl_logit.append(int(np.argmax(it.get_tensor(fc["outputs"][0])[0])))
        tfl_out.append(int(np.argmax(it.get_tensor(it.get_output_details()[0]["index"])[0])))
    tfl_logit, tfl_out = np.array(tfl_logit), np.array(tfl_out)
    chunks = [(s, min(s + 500, len(y))) for s in range(0, len(y), 500)]
    gem = np.concatenate(pool_map(_r8_int_pred, chunks, a.workers))
    print(f"ResNet-8 pretrainedResnet_quant, CIFAR-10 test set, {len(y)} images (no sampling, no seed)")
    for name, p in (("TFLite int8 (argmax of the FC logits)", tfl_logit),
                    ("TFLite int8 (argmax of the softmax output)", tfl_out),
                    ("GeMMX integer semantics (resnet8_int)", gem)):
        k = int((p == y).sum())
        lo, hi = wilson(k, len(y))
        print(f"  {name}: top-1 {k / len(y):.4f} ({k}/{len(y)}), 95% Wilson CI [{lo:.4f}, {hi:.4f}]")
    d = gem != tfl_logit
    print(f"  predictions differing GeMMX vs TFLite logits: {int(d.sum())} / {len(y)}; "
          f"of these GeMMX right {int(((gem == y) & d).sum())}, TFLite right {int(((tfl_logit == y) & d).sum())}")
    print(f"  predictions differing TFLite logits vs TFLite softmax argmax (int8 ties): {int((tfl_logit != tfl_out).sum())}")
    np.savez(os.path.join(CACHE, "item1.npz"), tfl=tfl_logit, tfl_out=tfl_out, gem=gem, y=y)


# --------------------------------------------------------------------------
# ResNet-8 per-image data (items 2, 4, 5, 13)
# --------------------------------------------------------------------------
R8_MASKED = ["conv1", "b1c1", "add1", "b2c1", "add2", "b3c1", "add3"]
_R8 = {}


def _r8_setup():
    if not _R8:
        import resnet8 as RR
        import resnet8_int as R
        import resnet8_xai as X
        M = R.load_model()
        BW = X.backward_weights(M)
        S = X.calibrate(M, BW, pct=99.5)
        _R8.update(RR=RR, R=R, X=X, M=M, BW=BW, S=S, fwd=X.torch_model())
    return _R8


def _mask_hi(G: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Variant mask: also zero positions where the forward output saturated
    at +127 (the upper clamp has zero derivative too)."""
    return np.where((act != -128) & (act != 127), G, 0).astype(np.int8)


def _r8_image(k: int) -> Dict[str, np.ndarray]:
    import torch
    E = _r8_setup()
    RR, R, X, M, BW, S, fwd = E["RR"], E["R"], E["X"], E["M"], E["BW"], E["S"], E["fwd"]
    img, label = RR.load_image(k)
    xq = R.quantize_image(img, M)
    out = {"label": label}
    deq = lambda q: (q.astype(np.float64) - M.zp_in) * M.s_in   # noqa: E731
    for m in ("ig", "shap"):
        bq, al = X.baselines(m, M, np.random.default_rng(100 + k))   # seed as the fidelity script
        ai, pred = X.int_attr(xq, bq, al, M, BW, S)
        af = X.float_attr(fwd, xq, bq, al, M, pred)
        orig = X.mask
        X.mask = _mask_hi
        try:
            ah, _ = X.int_attr(xq, bq, al, M, BW, S)
        finally:
            X.mask = orig
        out[f"{m}_int"], out[f"{m}_flt"], out[f"{m}_hi"] = ai, af, ah
        out["pred"] = pred
        # saturation of the masked forward tensors over the 16 interpolated samples
        xt = X.interpolate(xq, bq, al)
        sat = []
        for i in range(len(al)):
            t = R.forward(xt[i], M)
            sat.append([(t[n] == 127).mean() for n in R8_MASKED] + [(t[n] == -128).mean() for n in R8_MASKED])
        out[f"{m}_sat"] = np.array(sat).mean(axis=0)
        if m == "ig":                                       # completeness (baseline = black)
            lq_x = R.forward(xq, M)["logits"][:10].astype(np.float64)
            lq_b = R.forward(bq[0], M)["logits"][:10].astype(np.float64)
            s_l, zp_l = _r8_logit_q(M)
            out["f_int"] = ((lq_x[pred] - zp_l) - (lq_b[pred] - zp_l)) * s_l
            with torch.no_grad():
                fx = fwd(torch.tensor(deq(xq)[None].transpose(0, 3, 1, 2)))[0, pred].item()
                fb = fwd(torch.tensor(deq(bq[0])[None].transpose(0, 3, 1, 2)))[0, pred].item()
            out["f_flt"] = fx - fb
            out["sum_int_real"] = float(ai.sum()) * M.s_in * S["x"] / len(al)
            out["sum_flt"] = float(af.sum())
    return out


def _r8_logit_q(M) -> Tuple[float, int]:
    from tflite_runtime.interpreter import Interpreter
    import resnet8_int as R
    it = Interpreter(model_path=R.DEFAULT_MODEL)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    fc = [o for o in it._get_ops_details() if o["op_name"] == "FULLY_CONNECTED"][0]
    q = td[fc["outputs"][0]]["quantization_parameters"]
    return float(q["scales"][0]), int(q["zero_points"][0])


def item_r8(a) -> None:
    res = pool_map(_r8_image, list(range(a.images)), a.workers)
    keys = [k for k in res[0] if k not in ("label",)]
    np.savez_compressed(os.path.join(CACHE, f"r8_{a.images}.npz"),
                        **{k: np.array([r[k] for r in res]) for k in keys},
                        label=np.array([r["label"] for r in res]))
    print(f"cached {a.images} images -> external/review/r8_{a.images}.npz")


def _r8_cache(n: int = 200) -> Dict[str, np.ndarray]:
    d = np.load(os.path.join(CACHE, f"r8_{n}.npz"))
    return {k: d[k] for k in d.files}


def _r8_metrics(ai: np.ndarray, af: np.ndarray) -> Tuple[float, float, float, float]:
    from scipy.stats import spearmanr
    mi, mf = ai.sum(axis=2).ravel().astype(np.float64), af.sum(axis=2).ravel()
    top = len(set(np.argsort(-np.abs(mi))[:64]) & set(np.argsort(-np.abs(mf))[:64])) / 64
    return pearson(mi, mf), float(spearmanr(mi, mf)[0]), top, pearson(ai, af)


# --------------------------------------------------------------------------
# ToyAdmos shared
# --------------------------------------------------------------------------
_TA = {}


def _ta_setup():
    if not _TA:
        import ad01_int as A
        import toyadmos as T
        import toyadmos_xai_fidelity as F
        layers, feats, bg = T.load_data()
        sh = T.calibrate(layers, bg, np.random.default_rng(7))
        _TA.update(A=A, T=T, F=F, layers=layers, feats=feats, bg=bg, sh=sh, fm=F.float_model(layers))
    return _TA


def _ta_windows(n_clips: int = 50) -> List[Tuple[int, int, np.ndarray]]:
    """The fidelity script's picks: rng(3), n anomalous + n normal clips, the
    most anomalous 5-frame window of each."""
    E = _ta_setup()
    A, layers, feats = E["A"], E["layers"], E["feats"]
    rng = np.random.default_rng(3)
    picks = []
    for lab in (1, 0):
        idx = np.where(feats["labels"] == lab)[0]
        picks += [(lab, int(c)) for c in rng.choice(idx, n_clips, replace=False)]
    L0 = layers[0]
    out = []
    for lab, c in picks:
        xq_all = A.quantize_input(feats["vectors"][c], layers)
        y = A.forward(xq_all, layers)[-1]
        err = np.mean(((xq_all.astype(np.float64) - L0.zp_x) * L0.s_x
                       - (y.astype(np.float64) - layers[-1].zp_y) * layers[-1].s_y) ** 2, axis=1)
        out.append((lab, c, xq_all[int(np.argmax(err))]))
    return out


def ta_int_xai(layers, xq, bq, alpha, sh, den: int = 16, mask_hi: bool = False,
               prec: Dict[str, str] = None) -> Dict[str, np.ndarray]:
    """Copy of toyadmos.int_xai_reference with three switches: interpolation
    denominator (sample-count sweep), +127 masking, and per-stage gradient
    precision {"loss"|"bwd1".."bwd10": "int8"|"int16"|"float"}.

    Gradients are held as values v on the int8 grid of their stage (real =
    v * stage scale): int8 = floor + clip to [-128, 127] (the GeMMX SIMD);
    int16 = the same scale with 8 extra fractional bits, floor to 1/256 and
    saturate at +-32767/256; float = no rounding or clipping. With the
    defaults this is exactly int_xai_reference (checked by _ta_check_copy)."""
    import ad01_int as A
    import toyadmos as T
    prec = prec or {}

    def q(v: np.ndarray, mode: str) -> np.ndarray:
        if mode == "int8":
            return np.clip(np.floor(v), -128, 127)
        if mode == "int16":
            return np.clip(np.floor(v * 256.0) / 256.0, -32767 / 256.0, 32767 / 256.0)
        return v

    xt = ((alpha[:, None].astype(np.int64) * xq[None].astype(np.int64)
           + (den - alpha[:, None]).astype(np.int64) * bq.astype(np.int64)) // den).astype(np.int8)
    acts = A.forward(xt, layers)
    y = acts[-1]
    gacc = (T.LOSS_A * y.astype(np.int64) - T.LOSS_B * xt.astype(np.int64)).astype(np.float64)
    m, s = sh["loss"]
    gy = q(gacc * m / 2.0 ** s + sh["loss_zp"], prec.get("loss", "int8"))
    g = gy
    for li in range(len(layers) - 1, -1, -1):
        L = layers[li]
        if li < len(layers) - 1:
            keep = acts[li + 1] != -128
            if mask_hi:
                keep &= acts[li + 1] != 127
            g = np.where(keep, g, 0.0)
        acc = g @ L.W.astype(np.float64)                  # exact: |acc * mult| < 2^53
        m, s = sh[f"bwd{li + 1}"]
        g = q(acc * m / 2.0 ** s, prec.get(f"bwd{li + 1}", "int8"))
    d = ((xq[None].astype(np.int64) - bq.astype(np.int64)) // 2).astype(np.float64)
    attr = sh["m0"] * (d * g).sum(axis=0) - sh["m1"] * (d * gy).sum(axis=0)
    return {"attr": attr, "acts": acts, "xt": xt}


def _ta_check_copy() -> None:
    """The local copy with default switches equals the original pipeline."""
    E = _ta_setup()
    T, layers, sh, bg = E["T"], E["layers"], E["sh"], E["bg"]
    for lab, c, xq in _ta_windows(50)[:6]:
        for m in ("ig", "shap"):
            bq, al = T.baselines(m, layers, bg, np.random.default_rng(11 + c))
            r0 = T.int_xai_reference(layers, xq, bq, al, sh)["attr"]
            r1 = ta_int_xai(layers, xq, bq, al, sh)["attr"]
            assert np.array_equal(r0.astype(np.float64), r1), "ta_int_xai != int_xai_reference"


def _ta_float_attr(fm, xr, br, alpha_frac) -> np.ndarray:
    import toyadmos_xai_fidelity as F
    xt = br + alpha_frac[:, None] * (xr[None] - br)
    return ((xr[None] - br) * F.score_grad(fm, xt)).mean(axis=0)


def _ta_real(q: np.ndarray, layers) -> np.ndarray:
    return (q.astype(np.float64) - layers[0].zp_x) * layers[0].s_x


# --------------------------------------------------------------------------
# Item 2: faithfulness with bootstrap CIs
# --------------------------------------------------------------------------
def item2(a) -> None:
    # (a) ResNet-8 IG / SHAP from the per-image cache (same seeds as the script)
    d = _r8_cache(a.images)
    print(f"ResNet-8 IG / SHAP int8 vs float32, {a.images} MLPerf samples "
          f"(baselines seeded rng(100 + k), calibration pct 99.5, bootstrap seed {BOOT_SEED})")
    for m in ("ig", "shap"):
        r = np.array([_r8_metrics(ai, af) for ai, af in zip(d[f"{m}_int"], d[f"{m}_flt"])])
        for j, nm in enumerate(["map Pearson r", "map Spearman rho", "top-64 overlap", "full 32x32x3 Pearson r"]):
            print("  " + summ(f"{m} {nm}", r[:, j]))
    # (b) ResNet-8 Grad-CAM, 200 images (resnet8_gradcam_fidelity.py logic)
    import resnet8 as RR
    import resnet8_int as R
    import resnet8_gradcam_fidelity as G
    M = R.load_model()
    fwd, Wfc = G.float_model()
    rs, top1, top8 = [], [], []
    for k in range(200):
        img, _ = RR.load_image(k)
        ref = R.forward(R.quantize_image(img, M), M)
        cam_i = RR.gradcam_ref(ref, M).astype(np.float64)
        pred = int(np.argmax(ref["logits"][:10]))
        a3, _ = fwd(img)
        cam_f = np.maximum(a3 @ Wfc[pred], 0).reshape(-1)
        if cam_i.std() == 0 or cam_f.std() == 0:
            continue
        rs.append(pearson(cam_i, cam_f))
        top1.append(float(np.argmax(cam_i) == np.argmax(cam_f)))
        top8.append(len(set(np.argsort(-cam_i)[:8]) & set(np.argsort(-cam_f)[:8])) / 8)
    print(f"ResNet-8 Grad-CAM int8 vs float32, 200 MLPerf samples ({len(rs)} non-constant maps)")
    for nm, v in (("Pearson r", rs), ("top-1 cell", top1), ("top-8 overlap", top8)):
        print("  " + summ(nm, v))
    # (c) ToyAdmos IG / SHAP, 50 + 50 clips (toyadmos_xai_fidelity.py logic)
    E = _ta_setup()
    T, F, layers, sh, bg, fm = E["T"], E["F"], E["layers"], E["sh"], E["bg"], E["fm"]
    res = {"ig": [], "shap": []}
    for lab, c, xq in _ta_windows(50):
        for m in ("ig", "shap"):
            bq, al = T.baselines(m, layers, bg, np.random.default_rng(11 + c))
            r = T.int_xai_reference(layers, xq, bq, al, sh)
            res[m].append(F.metrics(r["attr"].astype(np.float64),
                                    F.float_attr(fm, _ta_real(xq, layers), _ta_real(bq, layers), al)))
    print("ToyAdmos IG / SHAP int8 vs float32, 100 windows (clips rng(3), baselines rng(11 + clip), "
          "calibration rng(7))")
    for m in ("ig", "shap"):
        for k in res[m][0]:
            print("  " + summ(f"{m} {k}", [x[k] for x in res[m]]))


# --------------------------------------------------------------------------
# Item 3: ToyAdmos calibration ablation on held-out training windows
# --------------------------------------------------------------------------
def calibrate_variant(layers, bg, rng, pct: float, exact: bool, n_cal: int = 16):
    """toyadmos.calibrate with the percentile and the multiplier form as
    parameters (exact: mult / 2^shift, mult < 2^15; else 2 / 2^(s+1) =
    1 / 2^s with the smallest s that maps the percentile into int8)."""
    import ad01_int as A
    import toyadmos as T

    def fit(v):
        P = np.percentile(np.abs(v), pct)
        if exact:
            sc = 127.0 / P
            sft = int(np.floor(np.log2(2 ** 14 / sc)))
            return int(round(sc * 2 ** sft)), sft
        s = max(0, int(np.ceil(np.log2(max(P, 1e-30) / 127.0))))
        return 2, s + 1
    sh = {"loss": (2, 1), "loss_zp": 0}
    sh.update({f"bwd{i + 1}": (2, 1) for i in range(len(layers))})
    xs = A.quantize_input(bg[rng.choice(len(bg), 32, replace=False)], layers)
    bs = A.quantize_input(bg[rng.choice(len(bg), 32 * n_cal, replace=False)], layers)
    keys = ["loss"] + [f"bwd{i}" for i in range(len(layers), 0, -1)]
    for k in keys:
        vals = []
        for c in range(len(xs)):
            al = rng.integers(0, 17, size=n_cal)
            r = T.int_xai_reference(layers, xs[c], bs[c * n_cal:(c + 1) * n_cal], al, sh)
            vals.append(r["gacc"] if k == "loss" else r["accs"][int(k[3:]) - 1])
        sh[k] = fit(np.concatenate([v.reshape(-1) for v in vals]))
        if k == "loss":
            const = T.LOSS_B * layers[0].zp_x - T.LOSS_A * layers[-1].zp_y
            sh["loss_zp"] = int(round(const * sh["loss"][0] / 2 ** sh["loss"][1]))
    r = float(np.prod([L.s_w * 2.0 ** sh[f"bwd{i + 1}"][1] / sh[f"bwd{i + 1}"][0]
                       for i, L in enumerate(layers)]))
    k = 11 - max(0, int(np.ceil(np.log2(max(r, 1.0)))))
    sh["m0"], sh["m1"] = int(round(r * 2 ** k)), 2 ** k
    sh["ratio"] = r
    return sh


def item3(a) -> None:
    E = _ta_setup()
    T, F, layers, bg, fm = E["T"], E["F"], E["layers"], E["bg"], E["fm"]
    # the windows the calibration draws (same rng(7) call sequence)
    rng = np.random.default_rng(7)
    used = set(rng.choice(len(bg), 32, replace=False).tolist())
    used |= set(rng.choice(len(bg), 32 * 16, replace=False).tolist())
    free = np.array(sorted(set(range(len(bg))) - used))
    held = np.random.default_rng(21).choice(free, 100, replace=False)
    shap_pool = np.array(sorted(set(free.tolist()) - set(held.tolist())))
    ref = T.calibrate(layers, bg, np.random.default_rng(7))
    cells = {}
    for exact in (True, False):
        for pct in (99.5, 99.9):
            sh = calibrate_variant(layers, bg, np.random.default_rng(7), pct, exact)
            if exact and pct == 99.5:
                assert all(sh[k] == ref[k] for k in ref if k != "ratio"), "variant != repo calibration"
            rs = {"ig": [], "shap": []}
            for j, w in enumerate(held):
                xq = E["A"].quantize_input(bg[w][None], layers)[0]
                for m in ("ig", "shap"):
                    if m == "ig":
                        bq, al = T.baselines("ig", layers, bg, None)
                    else:
                        r2 = np.random.default_rng(1000 + j)
                        idx = r2.choice(shap_pool, 16, replace=False)
                        al = r2.integers(0, 17, size=16)
                        al[0] = 16
                        bq = E["A"].quantize_input(bg[idx], layers)
                    ai = T.int_xai_reference(layers, xq, bq, al, sh)["attr"].astype(np.float64)
                    af = F.float_attr(fm, _ta_real(xq, layers), _ta_real(bq, layers), al)
                    rs[m].append(pearson(ai, af))
            cells[(exact, pct)] = rs
    print(f"ToyAdmos calibration ablation on 100 held-out NORMAL TRAINING windows "
          f"(not used by the calibration; picked with rng(21) from {len(free)} unused windows; "
          f"SHAP baselines rng(1000 + j) from the remaining unused windows; calibration rng(7))")
    print(f"{'multiplier':>14} {'pct':>6} | IG r mean [95% CI] (min) | SHAP r mean [95% CI] (min)")
    for (exact, pct), rs in cells.items():
        row = []
        for m in ("ig", "shap"):
            v = np.array(rs[m])
            lo, hi = boot_ci(v)
            row.append(f"{v.mean():.4f} [{lo:.4f}, {hi:.4f}] ({v.min():.3f})")
        print(f"{'exact' if exact else 'power of two':>14} {pct:>6} | {row[0]} | {row[1]}")
    best = {m: max(cells, key=lambda c: np.mean(cells[c][m])) for m in ("ig", "shap")}
    print("best cell: " + ", ".join(f"{m}: {'exact' if b[0] else 'pow2'} @ {b[1]}" for m, b in best.items()))
    # paired comparison of the repo's choice with each other cell
    base = cells[(True, 99.5)]
    for c, rs in cells.items():
        if c == (True, 99.5):
            continue
        for m in ("ig", "shap"):
            dlt = np.array(rs[m]) - np.array(base[m])
            lo, hi = boot_ci(dlt)
            print(f"  {m}: ({'exact' if c[0] else 'pow2'}, {c[1]}) - (exact, 99.5): "
                  f"mean {dlt.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")


# --------------------------------------------------------------------------
# Item 4: IG completeness
# --------------------------------------------------------------------------
def item4(a) -> None:
    d = _r8_cache(a.images)
    rel_f = np.abs(d["sum_flt"] - d["f_flt"]) / np.abs(d["f_flt"])
    rel_i = np.abs(d["sum_int_real"] - d["f_int"]) / np.abs(d["f_int"])
    rel_if = np.abs(d["sum_int_real"] - d["f_flt"]) / np.abs(d["f_flt"])
    print(f"ResNet-8 IG completeness, {a.images} images, black baseline, 16-step Riemann sum "
          f"(right endpoints alpha = 1/16..1), f = logit of the predicted class")
    for nm, v in (("float32: |sum(attr) - (f(x)-f(b))| / |f(x)-f(b)|", rel_f),
                  ("int8 pipeline vs int8 model's own logits", rel_i),
                  ("int8 pipeline vs float32 f(x)-f(b)", rel_if)):
        print(f"  {nm}: median {np.median(v):.4f}, 90th pct {np.percentile(v, 90):.4f}")
    # ToyAdmos: 100 windows, IG from the mean normal spectrum, f = MSE score
    E = _ta_setup()
    T, F, A, layers, sh, bg, fm = E["T"], E["F"], E["A"], E["layers"], E["sh"], E["bg"], E["fm"]
    L0, L9 = layers[0], layers[-1]
    s_gy = L9.s_y * 2.0 ** sh["loss"][1] / (T.LOSS_A * sh["loss"][0])
    K = 2 * L0.s_x * (2.0 / 640) * s_gy / (sh["m1"] * 16)

    def f_float(xr):
        acts = [xr[None]]
        for W, b, relu in fm:
            z = acts[-1] @ W.T + b
            acts.append(np.maximum(z, 0) if relu else z)
        return float(np.mean((xr - acts[-1][0]) ** 2))

    def f_int(xq):
        y = A.forward(xq[None], layers)[-1][0]
        return float(np.mean((_ta_real(xq, layers) - (y.astype(np.float64) - L9.zp_y) * L9.s_y) ** 2))
    rf, ri, rif, ratio = [], [], [], []
    for lab, c, xq in _ta_windows(50):
        bq, al = T.baselines("ig", layers, bg, None)
        ai = T.int_xai_reference(layers, xq, bq, al, sh)["attr"].astype(np.float64) * K
        xr, br = _ta_real(xq, layers), _ta_real(bq, layers)
        af = F.float_attr(fm, xr, br, al)
        dfl = f_float(xr) - f_float(br[0])
        din = f_int(xq) - f_int(bq[0])
        rf.append(abs(af.sum() - dfl) / abs(dfl))
        ri.append(abs(ai.sum() - din) / abs(din))
        rif.append(abs(ai.sum() - dfl) / abs(dfl))
        ratio.append(ai.sum() / af.sum())
    print(f"ToyAdmos IG completeness, 100 windows (item 2 windows), baseline = mean normal spectrum, "
          f"f = reconstruction MSE; int8 attribution in real units via K = 2 s_x (2/640) S_gy / (m1 N) = {K:.3e}")
    print(f"  check of K: sum(int8 attr in real units) / sum(float32 attr), median {np.median(ratio):.3f} (1 = same total)")
    for nm, v in (("float32", rf), ("int8 pipeline vs int8 model's own f", ri), ("int8 pipeline vs float32 f", rif)):
        v = np.array(v)
        print(f"  {nm}: median {np.median(v):.4f}, 90th pct {np.percentile(v, 90):.4f}")


# --------------------------------------------------------------------------
# Item 5: upper-clamp saturation
# --------------------------------------------------------------------------
def item5(a) -> None:
    d = _r8_cache(a.images)
    print(f"ResNet-8: fraction of forward activations at +127 / at -128 (zero point), per masked tensor, "
          f"mean over {a.images} images x 16 interpolated samples")
    for m in ("ig", "shap"):
        sat = d[f"{m}_sat"].mean(axis=0)
        n = len(R8_MASKED)
        print(f"  {m}: " + "  ".join(f"{nm} {sat[i] * 100:.3f}%/{sat[n + i] * 100:.1f}%" for i, nm in enumerate(R8_MASKED)))
    for m in ("ig", "shap"):
        r0 = np.array([_r8_metrics(ai, af)[0] for ai, af in zip(d[f"{m}_int"], d[f"{m}_flt"])])
        r1 = np.array([_r8_metrics(ai, af)[0] for ai, af in zip(d[f"{m}_hi"], d[f"{m}_flt"])])
        dl = r1 - r0
        lo, hi = boot_ci(dl)
        print(f"  {m}: map r with +127 also masked {r1.mean():.4f} vs {r0.mean():.4f} now; "
              f"paired diff {dl.mean():+.5f} 95% CI [{lo:+.5f}, {hi:+.5f}]; images changed {int((dl != 0).sum())}")
    E = _ta_setup()
    T, F, layers, sh, bg, fm = E["T"], E["F"], E["layers"], E["sh"], E["bg"], E["fm"]
    sat = {"ig": [], "shap": []}
    rr = {"ig": ([], []), "shap": ([], [])}
    for lab, c, xq in _ta_windows(50):
        for m in ("ig", "shap"):
            bq, al = T.baselines(m, layers, bg, np.random.default_rng(11 + c))
            r0 = ta_int_xai(layers, xq, bq, al, sh)
            r1 = ta_int_xai(layers, xq, bq, al, sh, mask_hi=True)
            sat[m].append([(r0["acts"][i] == 127).mean() for i in range(1, 10)])
            af = F.float_attr(fm, _ta_real(xq, layers), _ta_real(bq, layers), al)
            rr[m][0].append(pearson(r0["attr"], af))
            rr[m][1].append(pearson(r1["attr"], af))
    print("ToyAdmos: fraction of hidden-layer outputs (ReLU layers 1-9) at +127, 100 windows x 16 samples")
    for m in ("ig", "shap"):
        s = np.array(sat[m]).mean(axis=0)
        print(f"  {m}: " + "  ".join(f"L{i + 1} {v * 100:.3f}%" for i, v in enumerate(s)))
    for m in ("ig", "shap"):
        r0, r1 = np.array(rr[m][0]), np.array(rr[m][1])
        dl = r1 - r0
        lo, hi = boot_ci(dl)
        print(f"  {m}: r with +127 masked {r1.mean():.4f} vs {r0.mean():.4f}; paired diff {dl.mean():+.5f} "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]; windows changed {int((dl != 0).sum())}")


# --------------------------------------------------------------------------
# Item 6: launched MACs from the schedule generators
# --------------------------------------------------------------------------
def _macs(launches) -> int:
    return int(sum(L.M * L.N * L.K for L in launches)) * 512


def item6(a) -> None:
    import resnet8 as RR
    import resnet8_int as R
    import resnet8_xai as X
    import resnet8_xai_sched as XS
    import toyadmos as T
    import mobile as MB
    import mobile_int as MI
    E = _ta_setup()
    layers, feats, bg, sh = E["layers"], E["feats"], E["bg"], E["sh"]
    xq, _ = T.select_input(layers, feats)
    ta_useful = int(sum(L.W.size for L in layers))
    rows = []
    p = T.build_inference(layers, xq)
    rows.append(("ToyAdmos inference", _macs([s.launch for s in p.steps if s.launch]), ta_useful, 9_227,
                 sum(s.launch is not None for s in p.steps)))
    bq, al = T.baselines("ig", layers, bg, np.random.default_rng(11))
    p = T.build_xai(layers, xq, bq, al, sh)
    rows.append(("ToyAdmos IG/SHAP (N=16)", _macs([s.launch for s in p.steps if s.launch]),
                 2 * 16 * ta_useful, 91_517, sum(s.launch is not None for s in p.steps)))
    M = R.load_model()
    useful_r8 = 0
    t = R.forward(R.quantize_image(RR.load_image(0)[0], M), M)
    for nm, L in M.convs.items():
        co, kh, kw, ci = L.W.shape
        ho, wo = t[nm].shape[:2]
        useful_r8 += ho * wo * co * kh * kw * ci
    useful_r8 += M.fc.W.size
    img, _ = RR.load_image(0)
    p = RR.build(M, img)
    rows.append(("ResNet-8 inference + Grad-CAM", _macs([s.launch for s in p.steps if s.launch]),
                 useful_r8 + 64 * 64, 57_461, sum(s.launch is not None for s in p.steps)))
    BW = X.backward_weights(M)
    S = X.calibrate(M, BW)
    img1, _ = RR.load_image(1)
    xq1 = R.quantize_image(img1, M)
    bq, al = X.baselines("ig", M, np.random.default_rng(101))
    order = np.argsort(-al, kind="stable")
    p = XS.build(M, BW, S, xq1, bq[order], al[order], 16)
    per = _macs([s.launch for s in p.steps if s.launch])
    rows.append(("ResNet-8 IG/SHAP (N=16)", 16 * per, 16 * 2 * useful_r8, 6_027_558,
                 16 * sum(s.launch is not None for s in p.steps)))
    for task, cyc in (("kws", 53_811), ("vww", 189_788)):
        Mm = MI.load_model(task)
        x, _ = MI.load_inputs(task)
        p = MB.build(Mm, x[0])
        ref = MI.forward(MI.quantize_input(x[0], Mm), Mm)
        u = 0
        for i, L in enumerate(Mm.layers):
            ho, wo = ref[f"l{i}"].shape[:2]
            kh, kw = L.k
            u += ho * wo * L.cout * kh * kw * (1 if L.kind == "dw" else L.W.shape[3])
        last = ref[f"l{len(Mm.layers) - 1}"]
        u += Mm.fc.W.size + last.size                       # FC + Grad-CAM MACs
        rows.append((f"{task.upper()} inference + Grad-CAM", _macs([s.launch for s in p.steps if s.launch]),
                     u, cyc, sum(s.launch is not None for s in p.steps)))
    print("MACs from the schedule generators (sum over launches of M x N x K blocks x 512); "
          "cycles = measured Verilator totals")
    print(f"{'program':34} {'launches':>8} {'launched MACs':>14} {'useful MACs':>12} {'useful/launched':>15} "
          f"{'cycles':>10} {'launched util':>13} {'useful util':>11}")
    for nm, launched, useful, cyc, nl in rows:
        print(f"{nm:34} {nl:>8} {launched:>14,} {useful:>12,} {useful / launched:>15.3f} {cyc:>10,} "
              f"{launched / (cyc * 512):>13.3f} {useful / (cyc * 512):>11.3f}")
    print("useful MACs: ToyAdmos sum(in x out) = %d per row; ResNet-8 conv + FC (+ Grad-CAM 64x64) = %d; "
          "XAI = 16 x (forward + backward) GeMM MACs" % (ta_useful, useful_r8))


# --------------------------------------------------------------------------
# Item 8 (optional): Grad-CAM at the block-2 output (faithfulness only)
# --------------------------------------------------------------------------
def item8(a) -> None:
    import torch
    E = _r8_setup()
    RR, R, X, M, BW, S = E["RR"], E["R"], E["X"], E["M"], E["BW"], E["S"]
    import torch.nn.functional as Fn
    fwd = X.torch_model()
    # float: a2 and d logit / d a2 via autograd on a forward hook-free re-run
    import resnet8_xai as XX
    P = _torch_parts()
    rs = []
    for k in range(200):
        img, _ = RR.load_image(k)
        xq = R.quantize_image(img, M)
        t = R.forward(xq, M)
        pred = int(np.argmax(t["logits"][:10]))
        # integer gradient wrt add2 through block 3 (the first stages of int_input_grad)
        gp = np.clip(M.fc.W[pred, 0, 0, :], -127, 127).astype(np.int8)
        g = XX.mask(np.broadcast_to(gp, (8, 8, 64)).copy(), t["add3"])

        def bconv(name, Gin, Sin, key, in_hw):
            acc = XX.conv_bwd_acc(Gin, BW[name], in_hw)
            return XX.requant_g(acc, Sin * BW[name].s_b / S[key])
        b3 = XX.mask(bconv("b3c2", g, S["add3"], "b3c1", (8, 8)), t["b3c1"])
        m3 = bconv("b3c1", b3, S["b3c1"], "m3", (16, 16))
        s3 = bconv("b3sc", g, S["add3"], "s3", (16, 16))
        g2 = XX.fork_sum(s3, S["s3"], m3, S["m3"], S["add2"])          # d logit / d add2 (pre-mask)
        alpha_i = g2.astype(np.int64).sum(axis=(0, 1))
        cam_i = np.maximum((t["add2"].astype(np.int64) - M.adds[1].zp_y) @ alpha_i, 0).ravel().astype(np.float64)
        x = torch.tensor(((xq.astype(np.float64) - M.zp_in) * M.s_in)[None].transpose(0, 3, 1, 2))
        a2 = P["upto_a2"](x).detach().requires_grad_(True)
        P["from_a2"](a2)[0, pred].backward()
        alpha_f = a2.grad[0].sum(dim=(1, 2)).numpy()
        cam_f = np.maximum(np.einsum("chw,c->hw", a2.detach()[0].numpy(), alpha_f), 0).ravel()
        if cam_i.std() > 0 and cam_f.std() > 0:
            rs.append(pearson(cam_i, cam_f))
    print("ResNet-8 Grad-CAM on the block-2 output (16x16x32), int8 backward through block 3 vs float32, 200 images")
    print("  " + summ("Pearson r", rs))


def _torch_parts():
    import torch.nn.functional as Fn
    import resnet8_xai as X
    import inspect
    fwd = X.torch_model()
    cl = dict(zip(fwd.__code__.co_freevars, [c.cell_contents for c in fwd.__closure__]))
    conv, Wfc, bfc = cl["conv"], cl["Wfc"], cl["bfc"]

    def upto_a2(x):
        c1 = Fn.relu(conv("conv1", x))
        a1 = Fn.relu(c1 + conv("b1c2", Fn.relu(conv("b1c1", c1))))
        return Fn.relu(conv("b2sc", a1) + conv("b2c2", Fn.relu(conv("b2c1", a1))))

    def from_a2(a2):
        a3 = Fn.relu(conv("b3sc", a2) + conv("b3c2", Fn.relu(conv("b3c1", a2))))
        return a3.mean(dim=(2, 3)) @ Wfc.T + bfc
    return {"upto_a2": upto_a2, "from_a2": from_a2}


# --------------------------------------------------------------------------
# Items 9, 10, 11: sample count, baselines, cheap methods (ResNet-8 part)
# --------------------------------------------------------------------------
def _r8_int_attr_gen(xq, bq, alpha, den, pred, E):
    R, X, M, BW, S = E["R"], E["X"], E["M"], E["BW"], E["S"]
    a = alpha[:, None, None, None].astype(np.int64)
    xt = ((a * xq[None].astype(np.int64) + (den - a) * bq.astype(np.int64)) // den).astype(np.int8)
    acc = np.zeros((32, 32, 3), dtype=np.int64)
    for i in range(len(alpha)):
        gx = X.int_input_grad(R.forward(xt[i], M), pred, M, BW, S)[:, :, :3]
        acc += (xq.astype(np.int64) - bq[i].astype(np.int64)) * gx.astype(np.int64)
    return acc


def _r8_float_attr_gen(xr, br, frac, pred, fwd):
    import torch
    xt = br + frac[:, None, None, None] * (xr[None] - br)
    t = torch.tensor(xt.transpose(0, 3, 1, 2), requires_grad=True)
    fwd(t)[:, pred].sum().backward()
    g = t.grad.numpy().transpose(0, 2, 3, 1)
    return ((xr[None] - br) * g).mean(axis=0)


def _r8_sweep_image(k: int) -> Dict[str, float]:
    from scipy.ndimage import gaussian_filter
    E = _r8_setup()
    RR, R, X, M, fwd = E["RR"], E["R"], E["X"], E["M"], E["fwd"]
    img, _ = RR.load_image(k)
    xq = R.quantize_image(img, M)
    pred = int(np.argmax(R.forward(xq, M)["logits"][:10]))
    deq = lambda q: (q.astype(np.float64) - M.zp_in) * M.s_in   # noqa: E731
    xr = deq(xq)
    out = {}
    cs = lambda a: a.sum(axis=2).ravel()                         # noqa: E731
    # references: float32 N=128
    black = R.quantize_image(np.zeros((32, 32, 3)), M)
    ref_ig = _r8_float_attr_gen(xr, np.repeat(deq(black)[None], 128, 0), np.arange(1, 129) / 128, pred, fwd)
    bg = X.cifar_train()
    r = np.random.default_rng(500 + k)
    bidx = r.choice(len(bg), 128, replace=False)
    bref = np.stack([deq(R.quantize_image(bg[i], M)) for i in bidx])
    ref_shap = _r8_float_attr_gen(xr, bref, r.uniform(0, 1, 128), pred, fwd)
    for n in (4, 8, 16, 32):
        den = max(16, n)
        al = np.arange(1, n + 1) * (den // n)
        bq = np.repeat(black[None], n, 0)
        ai = _r8_int_attr_gen(xq, bq, al, den, pred, E)
        af = _r8_float_attr_gen(xr, deq(bq), al / den, pred, fwd)
        out[f"ig{n}_same"] = pearson(cs(ai), cs(af))
        out[f"ig{n}_ref"] = pearson(cs(ai), cs(ref_ig))
        out[f"ig{n}_fref"] = pearson(cs(af), cs(ref_ig))
        rs = np.random.default_rng(100 + k)
        idx = rs.choice(len(bg), n, replace=False)
        als = rs.integers(0, 17, size=n)
        bq = np.stack([R.quantize_image(bg[i], M) for i in idx])
        ai = _r8_int_attr_gen(xq, bq, als, 16, pred, E)
        af = _r8_float_attr_gen(xr, deq(bq), als / 16, pred, fwd)
        out[f"shap{n}_same"] = pearson(cs(ai), cs(af))
        out[f"shap{n}_ref"] = pearson(cs(ai), cs(ref_shap))
        out[f"shap{n}_fref"] = pearson(cs(af), cs(ref_shap))
    # item 10: IG baselines (16 steps)
    mean_img = bg.mean(axis=0)
    blur = gaussian_filter(img.astype(np.float64), sigma=(4, 4, 0))
    al = np.arange(1, 17)
    for nm, b in (("black", np.zeros((32, 32, 3))), ("mean", mean_img), ("blur", blur)):
        bq = np.repeat(R.quantize_image(b, M)[None], 16, 0)
        ai = _r8_int_attr_gen(xq, bq, al, 16, pred, E)
        af = _r8_float_attr_gen(xr, deq(bq), al / 16, pred, fwd)
        out[f"base_{nm}"] = pearson(cs(ai), cs(af))
    # item 11: gradient x input, one sample at x (alpha = 16, black baseline)
    bq = black[None]
    ai = _r8_int_attr_gen(xq, bq, np.array([16]), 16, pred, E)
    af1 = _r8_float_attr_gen(xr, deq(bq), np.array([1.0]), pred, fwd)
    ig16 = _r8_float_attr_gen(xr, np.repeat(deq(black)[None], 16, 0), np.arange(1, 17) / 16, pred, fwd)
    out["gxi_int_vs_ig16"] = pearson(cs(ai), cs(ig16))
    out["gxi_flt_vs_ig16"] = pearson(cs(af1), cs(ig16))
    out["gxi_int_vs_flt"] = pearson(cs(ai), cs(af1))
    out["gxi_int_vs_ig128"] = pearson(cs(ai), cs(ref_ig))
    return out


def item_r8sweep(a) -> None:
    res = pool_map(_r8_sweep_image, list(range(a.images)), a.workers)
    np.savez(os.path.join(CACHE, f"r8sweep_{a.images}.npz"), **{k: np.array([r[k] for r in res]) for k in res[0]})
    print(f"cached -> external/review/r8sweep_{a.images}.npz")


def _ta_sweep() -> Dict[str, List[float]]:
    E = _ta_setup()
    T, F, A, layers, sh, bg, fm = E["T"], E["F"], E["A"], E["layers"], E["sh"], E["bg"], E["fm"]
    out: Dict[str, List[float]] = {}
    add = lambda k, v: out.setdefault(k, []).append(v)        # noqa: E731
    mean_q = A.quantize_input(bg.mean(axis=0)[None], layers)
    zero_q = A.quantize_input(np.zeros((1, 640)), layers)
    for lab, c, xq in _ta_windows(50):
        xr = _ta_real(xq, layers)
        ref_ig = _ta_float_attr(fm, xr, np.repeat(_ta_real(mean_q, layers), 128, 0), np.arange(1, 129) / 128)
        r = np.random.default_rng(500 + c)
        bref = _ta_real(A.quantize_input(bg[r.choice(len(bg), 128, replace=False)], layers), layers)
        ref_shap = _ta_float_attr(fm, xr, bref, r.uniform(0, 1, 128))
        for n in (4, 8, 16, 32):
            den = max(16, n)
            al = np.arange(1, n + 1) * (den // n)
            bq = np.repeat(mean_q, n, 0)
            ai = ta_int_xai(layers, xq, bq, al, sh, den=den)["attr"]
            af = _ta_float_attr(fm, xr, _ta_real(bq, layers), al / den)
            add(f"ig{n}_same", pearson(ai, af))
            add(f"ig{n}_ref", pearson(ai, ref_ig))
            add(f"ig{n}_fref", pearson(af, ref_ig))
            rs = np.random.default_rng(11 + c)
            idx = rs.choice(len(bg), n, replace=False)
            als = rs.integers(0, 17, size=n)
            als[0] = 16
            bq = A.quantize_input(bg[idx], layers)
            ai = ta_int_xai(layers, xq, bq, als, sh)["attr"]
            af = _ta_float_attr(fm, xr, _ta_real(bq, layers), als / 16)
            add(f"shap{n}_same", pearson(ai, af))
            add(f"shap{n}_ref", pearson(ai, ref_shap))
            add(f"shap{n}_fref", pearson(af, ref_shap))
        al = np.arange(1, 17)
        for nm, b in (("mean", mean_q), ("zeros", zero_q)):
            bq = np.repeat(b, 16, 0)
            ai = ta_int_xai(layers, xq, bq, al, sh)["attr"]
            af = _ta_float_attr(fm, xr, _ta_real(bq, layers), al / 16)
            add(f"base_{nm}", pearson(ai, af))
        # item 11: per-feature reconstruction error from the int8 inference
        y = A.forward(xq[None], layers)[-1][0]
        err = (xr - (y.astype(np.float64) - layers[-1].zp_y) * layers[-1].s_y) ** 2
        ig16 = _ta_float_attr(fm, xr, np.repeat(_ta_real(mean_q, layers), 16, 0), al / 16)
        add("err_vs_ig16", pearson(err, ig16))
        add("err_vs_absig16", pearson(err, np.abs(ig16)))
        ai16 = ta_int_xai(layers, xq, np.repeat(mean_q, 16, 0), al, sh)["attr"]
        add("int_ig16_vs_ig16", pearson(ai16, ig16))
    return out


def item9(a) -> None:
    d = dict(np.load(os.path.join(CACHE, f"r8sweep_{a.images}.npz")))
    t = _ta_sweep()
    print(f"Sample-count sweep (emulator). ResNet-8: {a.images} images; ToyAdmos: 100 windows.")
    print("  IG: alpha = 1/N..1 (N = 32 uses a 1/32 grid); SHAP: N baselines + integer alpha 0..16 (/16), "
          "seeds rng(100 + k) (ResNet-8) / rng(11 + clip) (ToyAdmos); float32 N = 128 references: "
          "IG 128 steps, SHAP 128 baselines with uniform alpha, seed rng(500 + k / clip)")
    for model, src in (("ResNet-8", d), ("ToyAdmos", t)):
        print(f"  {model}:")
        print(f"    {'method':6} {'N':>3} | int8 vs float32 same N | int8 vs float32 N=128 | float32 N vs float32 N=128")
        for m in ("ig", "shap"):
            for n in (4, 8, 16, 32):
                cells = []
                for suf in ("same", "ref", "fref"):
                    v = np.asarray(src[f"{m}{n}_{suf}"], dtype=np.float64)
                    lo, hi = boot_ci(v)
                    cells.append(f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]")
                print(f"    {m:6} {n:>3} | {cells[0]:>22} | {cells[1]:>21} | {cells[2]}")


def item10(a) -> None:
    import resnet8_int as R
    import resnet8_xai as X
    import resnet8_xai_sched as XS
    import toyadmos as T
    from gemmx_sched import emit_cfg_code
    d = dict(np.load(os.path.join(CACHE, f"r8sweep_{a.images}.npz")))
    t = _ta_sweep()
    print(f"IG faithfulness (int8 vs float32, same baseline, 16 steps). ResNet-8 {a.images} images, ToyAdmos 100 windows")
    for nm in ("black", "mean", "blur"):
        print("  ResNet-8 " + summ(f"baseline {nm}", d[f"base_{nm}"]))
    for nm in ("mean", "zeros"):
        print("  ToyAdmos " + summ(f"baseline {nm}", t[f"base_{nm}"]))
    # the schedule does not depend on the baseline: compare generated launch code
    E = _r8_setup()
    M, BW, S, RR = E["M"], E["BW"], E["S"], E["RR"]
    from scipy.ndimage import gaussian_filter
    img, _ = RR.load_image(1)
    xq = R.quantize_image(img, M)
    codes = {}
    for nm, b in (("black", np.zeros((32, 32, 3))), ("mean", X.cifar_train().mean(axis=0)),
                  ("blur", gaussian_filter(img.astype(np.float64), sigma=(4, 4, 0)))):
        bq = np.repeat(R.quantize_image(b, M)[None], 16, 0)
        al = np.arange(16, 0, -1)
        p = XS.build(M, BW, S, xq, bq, al, 16)
        codes[nm] = (emit_cfg_code([s.launch for s in p.steps if s.launch], True, "cfg"),
                     [(s.phase, s.blob, s.pre, s.post) for s in p.steps], p.dmas)
    same = all(codes[k] == codes["black"] for k in codes)
    print(f"  ResNet-8 schedule identical for black / mean / blur baselines (launch CSR code, step table, DMA table): {same}")
    Et = _ta_setup()
    layers, feats, bg, sh, A = Et["layers"], Et["feats"], Et["bg"], Et["sh"], Et["A"]
    xq, _ = T.select_input(layers, feats)
    ct = {}
    for nm, b in (("mean", A.quantize_input(bg.mean(axis=0)[None], layers)), ("zeros", A.quantize_input(np.zeros((1, 640)), layers))):
        p = T.build_xai(layers, xq, np.repeat(b, 16, 0), np.arange(1, 17), sh)
        ct[nm] = (emit_cfg_code([s.launch for s in p.steps if s.launch], True, "cfg"),
                  [(s.blob, s.post) for s in p.steps])
    print(f"  ToyAdmos schedule identical for mean / zeros baselines: {ct['mean'] == ct['zeros']}")


def item11(a) -> None:
    d = dict(np.load(os.path.join(CACHE, f"r8sweep_{a.images}.npz")))
    t = _ta_sweep()
    print(f"Cheap methods vs float32 IG. ResNet-8 {a.images} images (gradient x input = one sample at x, black baseline)")
    for k, nm in (("gxi_int_vs_ig16", "int8 grad x input vs float32 IG N=16"),
                  ("gxi_int_vs_ig128", "int8 grad x input vs float32 IG N=128"),
                  ("gxi_flt_vs_ig16", "float32 grad x input vs float32 IG N=16 (method gap alone)"),
                  ("gxi_int_vs_flt", "int8 vs float32 grad x input (int8 gap alone)")):
        print("  " + summ(nm, d[k]))
    print("ToyAdmos, 100 windows: per-feature reconstruction error (x - y)^2 of the int8 inference")
    for k, nm in (("err_vs_ig16", "(x - y)^2 vs float32 IG N=16 (signed)"),
                  ("err_vs_absig16", "(x - y)^2 vs |float32 IG N=16|"),
                  ("int_ig16_vs_ig16", "for comparison: int8 IG N=16 vs float32 IG N=16")):
        print("  " + summ(nm, t[k]))


# --------------------------------------------------------------------------
# Item 12: higher-precision ToyAdmos gradients
# --------------------------------------------------------------------------
def item12(a) -> None:
    E = _ta_setup()
    T, F, layers, sh, bg, fm = E["T"], E["F"], E["layers"], E["sh"], E["bg"], E["fm"]
    _ta_check_copy()
    allk = ["loss"] + [f"bwd{i}" for i in range(1, 11)]
    configs = {
        "int8 everywhere (as built)": {},
        "int16 all stages": {k: "int16" for k in allk},
        "float all stages": {k: "float" for k in allk},
        "int16 output-side 3 (loss, bwd10, bwd9)": {k: "int16" for k in ("loss", "bwd10", "bwd9")},
        "int16 input-side 3 (bwd3, bwd2, bwd1)": {k: "int16" for k in ("bwd3", "bwd2", "bwd1")},
        "float output-side 3 (loss, bwd10, bwd9)": {k: "float" for k in ("loss", "bwd10", "bwd9")},
        "float input-side 3 (bwd3, bwd2, bwd1)": {k: "float" for k in ("bwd3", "bwd2", "bwd1")},
    }
    res = {c: {"ig": [], "shap": []} for c in configs}
    for lab, c, xq in _ta_windows(50):
        for m in ("ig", "shap"):
            bq, al = T.baselines(m, layers, bg, np.random.default_rng(11 + c))
            af = F.float_attr(fm, _ta_real(xq, layers), _ta_real(bq, layers), al)
            for cn, pr in configs.items():
                res[cn][m].append(pearson(ta_int_xai(layers, xq, bq, al, sh, prec=pr)["attr"], af))
    print("ToyAdmos gradient precision (emulator), 100 windows (item 2), r vs float32; masks and forward int8 "
          "as built; int16 = the int8 scale with 8 extra fractional bits, saturating at +-32767")
    for cn in configs:
        cells = []
        for m in ("ig", "shap"):
            v = np.array(res[cn][m])
            lo, hi = boot_ci(v)
            cells.append(f"{m} {v.mean():.4f} [{lo:.4f}, {hi:.4f}] (min {v.min():.3f})")
        print(f"  {cn:42}: " + "  ".join(cells))
    p = T.build_xai(layers, T.select_input(layers, E["feats"])[0], *T.baselines("ig", layers, bg, None), sh)
    nb = sum(1 for s in p.steps if s.launch is not None and s.launch.name.startswith("bwd"))
    print(f"  backward GeMM launches in the built schedule: {nb} (per explanation)")


# --------------------------------------------------------------------------
# Item 13: deletion / insertion
# --------------------------------------------------------------------------
def _r8_delins(k: int) -> Dict[str, float]:
    import torch
    from scipy.ndimage import gaussian_filter
    E = _r8_setup()
    RR, R, M, fwd = E["RR"], E["R"], E["M"], E["fwd"]
    d = _DELINS
    img, _ = RR.load_image(k)
    xq = R.quantize_image(img, M)
    deq = lambda q: (q.astype(np.float64) - M.zp_in) * M.s_in   # noqa: E731
    x = deq(xq)
    blur = deq(R.quantize_image(gaussian_filter(img.astype(np.float64), sigma=(4, 4, 0)), M))
    black = deq(R.quantize_image(np.zeros((32, 32, 3)), M))
    pred = int(d["pred"][k])
    maps = {"ig_int": d["ig_int"][k].sum(axis=2), "ig_flt": d["ig_flt"][k].sum(axis=2),
            "shap_int": d["shap_int"][k].sum(axis=2), "shap_flt": d["shap_flt"][k].sum(axis=2),
            "random": np.random.default_rng(900 + k).standard_normal((32, 32))}
    steps = np.linspace(0, 1024, 33).astype(int)
    out = {}
    for nm, m in maps.items():
        order = np.argsort(-m.ravel(), kind="stable")
        dele, ins = [], []
        for s in steps:
            msk = np.zeros(1024, bool)
            msk[order[:s]] = True
            msk = msk.reshape(32, 32, 1)
            dele.append(np.where(msk, black, x))
            ins.append(np.where(msk, x, blur))
        batch = torch.tensor(np.stack(dele + ins).transpose(0, 3, 1, 2))
        with torch.no_grad():
            p = torch.softmax(fwd(batch), dim=1)[:, pred].numpy()
        out[f"{nm}_del"] = float(np.trapz(p[:33], dx=1 / 32))
        out[f"{nm}_ins"] = float(np.trapz(p[33:], dx=1 / 32))
    return out


_DELINS: Dict[str, np.ndarray] = {}


def item13(a) -> None:
    _DELINS.update(_r8_cache(a.images))
    res = pool_map(_r8_delins, list(range(a.images)), a.workers)
    print(f"ResNet-8 deletion / insertion AUC, {a.images} images, float32 dequantised model, softmax "
          "probability of the predicted class; pixels ranked by the channel-summed attribution "
          "(descending), 32 steps of 32 pixels; deletion fills black, insertion starts from a Gaussian blur "
          "(sigma 4); random map seed rng(900 + k). Lower deletion / higher insertion is better.")
    for nm in ("ig_int", "ig_flt", "shap_int", "shap_flt", "random"):
        dl = np.array([r[f"{nm}_del"] for r in res])
        it = np.array([r[f"{nm}_ins"] for r in res])
        (l1, h1), (l2, h2) = boot_ci(dl), boot_ci(it)
        print(f"  {nm:9}: deletion {dl.mean():.4f} [{l1:.4f}, {h1:.4f}]   insertion {it.mean():.4f} [{l2:.4f}, {h2:.4f}]")
    for m in ("ig", "shap"):
        for kind in ("del", "ins"):
            dl = np.array([r[f"{m}_int_{kind}"] - r[f"{m}_flt_{kind}"] for r in res])
            lo, hi = boot_ci(dl)
            print(f"  {m} int8 - float32 ({kind}): {dl.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")


# --------------------------------------------------------------------------
# Items 14, 15: L3 traffic and 1-bit masks
# --------------------------------------------------------------------------
def item14(a) -> None:
    import resnet8 as RR
    import resnet8_int as R
    import resnet8_xai as X
    import resnet8_xai_sched as XS
    import toyadmos as T
    E = _r8_setup()
    M, BW, S = E["M"], E["BW"], E["S"]
    img, _ = RR.load_image(1)
    xq = R.quantize_image(img, M)
    bq, al = X.baselines("ig", M, np.random.default_rng(101))
    order = np.argsort(-al, kind="stable")
    p = XS.build(M, BW, S, xq, bq[order], al[order], 16)
    X0 = RR.PAct(0, 8, 32, 32)
    cat = dict(weights=0, spill_write=0, mask_read=0, sample_in=0, attr_in=0, interleave_tcdm=0, fill_tcdm=0)
    for s in p.steps:
        if s.blob >= 0:
            cat["weights"] += p.blobs[s.blob].size
        for op in s.pre + s.post:
            if op[0] == "spill":
                cat["spill_write"] += op[3]
            elif op[0] == "mask":
                ct, h, w = op[3] >> 16, op[3] & 0xFFFF, op[4] >> 16
                cat["mask_read"] += ct * h * w * 8
            elif op[0] == "sample_in":
                cat["sample_in"] += 2 * X0.nbytes + 128
            elif op[0] == "attr":
                cat["attr_in"] += 2 * X0.nbytes
            elif op[0] == "dmas":
                cat["interleave_tcdm"] += sum(d[2] * d[5] for d in p.dmas[op[1]:op[2]])
            elif op[0] == "fill":
                cat["fill_tcdm"] += op[2]
    print("ResNet-8 IG, one explanation (16 samples), bytes moved per category (schedule generator; "
          "each sample runs the same step table)")
    tot_l3 = 0
    for k, v in cat.items():
        l3 = "TCDM->TCDM" if k.endswith("tcdm") else "L3<->TCDM"
        if not k.endswith("tcdm"):
            tot_l3 += 16 * v
        print(f"  {k:16} {l3:11} per sample {v:>10,} B   per explanation {16 * v:>12,} B")
    print(f"  total L3<->TCDM per explanation: {tot_l3:,} B (+ one-time TCDM image {p.image.size:,} B)")
    Et = _ta_setup()
    layers, feats, bg, sh = Et["layers"], Et["feats"], Et["bg"], Et["sh"]
    xq, _ = T.select_input(layers, feats)
    bq, al = T.baselines("ig", layers, bg, np.random.default_rng(11))
    pt = T.build_xai(layers, xq, bq, al, sh)
    w = sum(pt.blobs[s.blob].size for s in pt.steps if getattr(s, "blob", -1) >= 0)
    print("ToyAdmos IG, one explanation (16 samples stacked in the mesh rows): weight / bias tiles streamed "
          f"L3->TCDM {w:,} B; inputs x, b_1..b_16 and constants arrive with the TCDM image ({pt.image.size:,} B); "
          "masks, interleaves and gradients stay in the TCDM (no spills).")
    print("DMA busy / stall counters: the simulated cluster exposes no DMA performance counters to software "
          "and the runs record no per-cycle trace (traces were linked to /dev/null for disk space), so the "
          "fraction of time the cores wait on DMA is not measured. The phase timers count DMA-bound "
          "phases together with the core work that overlaps them (see item 14 text).")
    np.savez(os.path.join(CACHE, "item14.npz"), **{k: v for k, v in cat.items()})


def item15(a) -> None:
    import resnet8 as RR
    import resnet8_int as R
    import resnet8_xai as X
    import resnet8_xai_sched as XS
    E = _r8_setup()
    M, BW, S = E["M"], E["BW"], E["S"]
    img, _ = RR.load_image(1)
    xq = R.quantize_image(img, M)
    bq, al = X.baselines("ig", M, np.random.default_rng(101))
    order = np.argsort(-al, kind="stable")
    p = XS.build(M, BW, S, xq, bq[order], al[order], 16)
    elems = 0
    for s in p.steps:
        for op in s.post:
            if op[0] == "mask":
                ct, h, w = op[3] >> 16, op[3] & 0xFFFF, op[4] >> 16
                elems += ct * h * w * 8
    spill = sum(op[3] for s in p.steps for op in s.post if op[0] == "spill")
    tcdm_used = int(p.info.get("tcdm", "0").split()[0]) if "tcdm" in p.info else None
    print("ResNet-8 IG: 1-bit ReLU masks instead of re-reading int8 activations (per sample)")
    print(f"  masked gradient elements per sample: {elems:,}; int8 activation bytes read from L3 now: {elems:,} B; "
          f"activation bytes spilled to L3 now: {spill:,} B")
    print(f"  as 1-bit masks: {elems // 8:,} B per sample (saves {elems - elems // 8:,} B of mask reads and "
          f"replaces {spill:,} B of spills with {elems // 8:,} B)")
    print(f"  TCDM in use by the schedule: {p.image.size:,} B image of {XS.TCDM_LIMIT:,} B limit")


ITEMS = {"1": item1, "r8": item_r8, "r8sweep": item_r8sweep, "2": item2, "3": item3, "4": item4,
         "5": item5, "6": item6, "8": item8, "9": item9, "10": item10, "11": item11, "12": item12,
         "13": item13, "14": item14, "15": item15}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=list(ITEMS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--images", type=int, default=200, help="ResNet-8 images (r8 / 2 / 4 / 5 / 13); 50 for r8sweep / 9 / 10 / 11")
    a = ap.parse_args()
    ITEMS[a.item](a)
