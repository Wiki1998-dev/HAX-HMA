"""ToyAdmos (MLPerf Tiny AD, ad01_int8.tflite) schedules on GeMMX.

Builds, emulates bit-exactly and emits three programs on real DCASE 2020
ToyCar data:

* ``inference``: one 640-feature input vector through the 10 FC layers
  (batch 1 in an 8-row mesh tile).
* ``ig``: Integrated Gradients of the anomaly score (reconstruction MSE),
  16 steps from one baseline (mean normal spectrum).
* ``shap``: Gradient SHAP (expected gradients), 16 baselines drawn from
  normal training vectors with random interpolation factors.

IG and SHAP share one schedule (only the data differs): the 16 samples are
stacked in the mesh rows, so every layer runs once for all samples.

    interp   X~ = [alpha | diag(16 - alpha)] @ [x; b_1..b_16] / 16
    forward  10 layers (layer 1 K-split, layer 10 N-split; weights streamed)
    loss     g_y ~ y - x~ = [y | x~] @ [55 I; -59 I]       (59/55 = s_x/s_y)
    backward g <- (g * relu_mask) @ W_l, l = 10..1          (mask on cores)
    attr     attr[f] = sum_i D[i, f] g0[i, f],  D = x - b_i  (diagonal trick)

Weights and biases stream from L3 in <= 20 kB tiles, double-buffered by the
DMA core. Requantisation shifts of the gradients are calibrated once on
normal training vectors (static).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../../../scripts")))

import ad01_int as A  # noqa: E402
from emit import c_bytes  # noqa: E402
from gemmx_sched import (BLK, Acc, Act, Launch, Tcdm, WB, blockdiag2,  # noqa: E402
                         emit_launches, gemm, pack_act, pack_wb, run_launch)

N_SAMPLES = 16
TILE = 128                    # weight tile side (16 kB int8)
LOSS_A, LOSS_B = 55, 59       # 59 / 55 = 1.072727 ~ s_x / s_y = 1.072755
TCDM_LIMIT = 112 * 1024       # leave room for the core stacks


# --------------------------------------------------------------------------
# Memory + schedule containers
# --------------------------------------------------------------------------
class Alloc:
    """64-byte aligned bump allocator over the TCDM."""

    def __init__(self) -> None:
        self.top = 0

    def take(self, nbytes: int) -> int:
        off = self.top
        self.top = (self.top + nbytes + 63) // 64 * 64
        assert self.top <= TCDM_LIMIT, f"TCDM overflow: {self.top} B"
        return off


@dataclass
class Step:
    """One schedule step: an optional launch, an optional parameter prefetch
    issued while it runs, and an optional core op after it."""

    phase: str
    launch: Optional[Launch] = None
    blob: int = -1                      # parameter blob this launch reads
    # ("mask", g_off, a_off, n) | ("diag", d32_off, dst_off)
    # | ("combine", a0_off, agy_off, out_off, m0, m1)
    post: Optional[Tuple] = None
    scratch: int = -1                   # parameter buffer used as scratch (no prefetch into it)


@dataclass
class Program:
    image: np.ndarray                   # initial TCDM image (static data)
    steps: List[Step]
    blobs: List[np.ndarray]             # L3 parameter blobs
    blob_dst: List[int]                 # TCDM destination per blob
    x_l3: np.ndarray                    # runtime input (640 int8)
    x_dst: int                          # Act whose row 0 receives x
    checks: List[Tuple[str, int, np.ndarray]] = field(default_factory=list)
    attr_off: int = -1
    info: Dict[str, str] = field(default_factory=dict)


def _bias_tile(b: np.ndarray) -> np.ndarray:
    """Bias vector -> 8-row int32 C tile (same bias in every mesh row)."""
    return pack_act(np.tile(b.astype(np.int32), (8, 1)))


def _simd_layer(L: A.Layer) -> dict:
    return dict(shift=L.shift, mult=L.M0, zp=L.zp_y, min=-128, max=127)


def _simd_shift(sh: int, zp: int = 0) -> dict:
    """out = floor(acc / 2^sh) + zp (mult 2, shift sh+1), clipped to int8."""
    return dict(shift=sh + 1, mult=2, zp=zp, min=-128, max=127)


def _simd_ms(ms: Tuple[int, int], zp: int = 0) -> dict:
    """out = floor(acc * mult / 2^shift) + zp, clipped to int8 (GeMMX SIMD)."""
    return dict(shift=ms[1], mult=ms[0], zp=zp, min=-128, max=127)


# --------------------------------------------------------------------------
# Forward pass (shared)
# --------------------------------------------------------------------------
def _forward_steps(layers: List[A.Layer], al: Alloc, rows: int, x_act: Act,
                   pb: Tuple[int, int], blobs: List[np.ndarray],
                   blob_dst: List[int]) -> Tuple[List[Step], List[Act], Act, Acc]:
    """Steps for the 10 FC layers on ``rows`` mesh rows. Returns steps, the
    hidden activations [a1..a9], the output y and the partial-sum buffer."""
    steps: List[Step] = []
    P = Acc(al.take(rows * TILE * 4), rows, TILE)
    acts = []
    cur = x_act
    for li, L in enumerate(layers):
        n_out, n_in = L.W.shape
        WT = L.W.T.copy()                                   # [in x out]
        out = Act(al.take(rows * n_out), rows, n_out)
        name = f"fwd{li + 1}"
        if n_in > TILE:                                     # K-split (layer 1)
            nk = n_in // TILE
            for j in range(nk):
                wt = WT[j * TILE:(j + 1) * TILE]
                buf = pb[len(blobs) % 2]
                parts = [pack_wb(wt)] + ([_bias_tile(L.bias)] if j == 0 else [])
                blobs.append(np.concatenate([p.view(np.uint8) for p in parts]))
                blob_dst.append(buf)
                B = WB(buf, TILE, n_out)
                kw = dict(k_split=(j * TILE // 8, TILE // 8))
                if j == 0:
                    lc = gemm(f"{name}k{j}", cur, B, P, C=Acc(buf + B.nbytes, 8, n_out), **kw)
                elif j < nk - 1:
                    lc = gemm(f"{name}k{j}", cur, B, P, C=P, **kw)
                else:
                    lc = gemm(f"{name}k{j}", cur, B, out, C=P, simd=_simd_layer(L), **kw)
                steps.append(Step("forward", lc, len(blobs) - 1))
        elif n_out > TILE:                                  # N-split (layer 10)
            for j in range(n_out // TILE):
                wt = WT[:, j * TILE:(j + 1) * TILE]
                buf = pb[len(blobs) % 2]
                blobs.append(np.concatenate([pack_wb(wt).view(np.uint8),
                                             _bias_tile(L.bias[j * TILE:(j + 1) * TILE]).view(np.uint8)]))
                blob_dst.append(buf)
                B = WB(buf, n_in, TILE)
                lc = gemm(f"{name}n{j}", cur, B, out, C=Acc(buf + B.nbytes, 8, TILE),
                          n_range=(j * TILE // 8, TILE // 8), simd=_simd_layer(L))
                steps.append(Step("forward", lc, len(blobs) - 1))
        else:
            buf = pb[len(blobs) % 2]
            blobs.append(np.concatenate([pack_wb(WT).view(np.uint8),
                                         _bias_tile(L.bias).view(np.uint8)]))
            blob_dst.append(buf)
            B = WB(buf, n_in, n_out)
            lc = gemm(name, cur, B, out, C=Acc(buf + B.nbytes, 8, n_out), simd=_simd_layer(L))
            steps.append(Step("forward", lc, len(blobs) - 1))
        acts.append(out)
        cur = out
    return steps, acts[:-1], acts[-1], P


# --------------------------------------------------------------------------
# Programs
# --------------------------------------------------------------------------
def build_inference(layers: List[A.Layer], xq: np.ndarray) -> Program:
    al = Alloc()
    pb = (al.take(20480), al.take(20480))
    rows = 8
    X = Act(al.take(rows * 640), rows, 640)
    blobs: List[np.ndarray] = []
    dst: List[int] = []
    steps, _, Y, _ = _forward_steps(layers, al, rows, X, pb, blobs, dst)
    image = np.zeros(al.top, dtype=np.uint8)
    prog = Program(image, steps, blobs, dst, xq.astype(np.int8), X.off)
    t = emulate(prog)
    prog.checks = [("y", Y.off, t.read(Y.off, Y.nbytes, np.uint8))]
    yrow = A.forward(xq[None], layers)[-1][0]
    from gemmx_sched import unpack_act
    y_emu = unpack_act(t.read(Y.off, Y.nbytes), rows, 640)[0]
    assert np.array_equal(y_emu, yrow), "schedule emulation != ad01_int reference"
    prog.info["tcdm"] = f"{al.top} B"
    return prog


def int_xai_reference(layers: List[A.Layer], xq: np.ndarray, bq: np.ndarray,
                      alpha: np.ndarray, sh: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Pure-numpy integer XAI pipeline with the GeMMX semantics of the
    schedule (used for calibration and as an independent cross-check)."""
    def req(acc: np.ndarray, ms: Tuple[int, int], zp: int = 0) -> np.ndarray:
        return np.clip(np.floor_divide(acc.astype(np.int64) * ms[0], 2 ** ms[1]) + zp,
                       -128, 127).astype(np.int8)
    xt = ((alpha[:, None].astype(np.int64) * xq[None].astype(np.int64)
           + (16 - alpha[:, None]).astype(np.int64) * bq.astype(np.int64)) // 16).astype(np.int8)
    acts = A.forward(xt, layers)
    y = acts[-1]
    gacc = LOSS_A * y.astype(np.int64) - LOSS_B * xt.astype(np.int64)
    g = req(gacc, sh["loss"], sh["loss_zp"])
    out = {"xt": xt, "y": y, "acts": acts}
    accs = {}
    for li in range(len(layers) - 1, -1, -1):
        L = layers[li]
        if li < len(layers) - 1:                            # mask through ReLU of layer li+1
            g = np.where(acts[li + 1] != -128, g, 0).astype(np.int8)
        acc = g.astype(np.int64) @ L.W.astype(np.int64)     # [R x out] @ [out x in]
        accs[li] = acc
        g = req(acc, sh[f"bwd{li + 1}"])
    out["g0"] = g
    d = ((xq[None].astype(np.int64) - bq.astype(np.int64)) // 2).astype(np.int8)
    out["d"] = d
    out["gy"] = req(gacc, sh["loss"], sh["loss_zp"])
    out["attr_g0"] = (d.astype(np.int64) * g.astype(np.int64)).sum(axis=0)
    out["attr_gy"] = (d.astype(np.int64) * out["gy"].astype(np.int64)).sum(axis=0)
    # d score / d x~ ~ J^T (y - x~) - (y - x~): combine with the scale ratio
    if "m0" in sh:
        out["attr"] = (sh["m0"] * out["attr_g0"] - sh["m1"] * out["attr_gy"]).astype(np.int64)
    out["accs"] = accs
    out["gacc"] = gacc
    return out


def calibrate(layers: List[A.Layer], bg: np.ndarray, rng: np.random.Generator) -> Dict[str, int]:
    """Static gradient requantisation shifts from normal training vectors:
    exact GeMMX SIMD scale (mult / 2^shift, mult < 2^15) that maps the 99.5th
    percentile |acc| to 127. Returns (mult, shift) per stage, the loss zero
    point and the integer ratio m0 / m1 between the network and direct
    gradient terms."""
    def fit(v: np.ndarray) -> Tuple[int, int]:
        sc = 127.0 / np.percentile(np.abs(v), 99.5)
        sft = int(np.floor(np.log2(2 ** 14 / sc)))
        assert 1 <= sft <= 63
        return int(round(sc * 2 ** sft)), sft
    sh: Dict = {"loss": (2, 1), "loss_zp": 0}
    sh.update({f"bwd{i + 1}": (2, 1) for i in range(len(layers))})
    xs = A.quantize_input(bg[rng.choice(len(bg), 32, replace=False)], layers)
    bs = A.quantize_input(bg[rng.choice(len(bg), 32 * N_SAMPLES, replace=False)], layers)
    keys = ["loss"] + [f"bwd{i}" for i in range(len(layers), 0, -1)]
    for k in keys:                                          # sequential: each depends on previous
        vals = []
        for c in range(len(xs)):
            al = rng.integers(0, 17, size=N_SAMPLES)
            r = int_xai_reference(layers, xs[c], bs[c * N_SAMPLES:(c + 1) * N_SAMPLES], al, sh)
            vals.append(r["gacc"] if k == "loss" else r["accs"][int(k[3:]) - 1])
        sh[k] = fit(np.concatenate([v.reshape(-1) for v in vals]))
        if k == "loss":
            const = LOSS_B * layers[0].zp_x - LOSS_A * layers[-1].zp_y
            sh["loss_zp"] = int(round(const * sh["loss"][0] / 2 ** sh["loss"][1]))
    # g0 = J^T g_y in units of S_gy * prod_l(s_w,l * 2^shift_l / mult_l); the
    # direct term g_y is in units of S_gy. Integer ratio m0 / m1, m <= 2^11.
    r = float(np.prod([L.s_w * 2.0 ** sh[f"bwd{i + 1}"][1] / sh[f"bwd{i + 1}"][0]
                       for i, L in enumerate(layers)]))
    k = 11 - max(0, int(np.ceil(np.log2(max(r, 1.0)))))
    sh["m0"], sh["m1"] = int(round(r * 2 ** k)), 2 ** k
    assert 0 < sh["m0"] <= 2 ** 11 and sh["m1"] <= 2 ** 11, (r, sh["m0"], sh["m1"])
    sh["ratio"] = r
    return sh


def build_xai(layers: List[A.Layer], xq: np.ndarray, bq: np.ndarray, alpha: np.ndarray,
              sh: Dict[str, int]) -> Program:
    R = N_SAMPLES
    al = Alloc()
    pb = (al.take(20480), al.take(20480))
    XB = Act(al.take(24 * 640), 24, 640)                    # [x; b_1..b_16; 0]
    CI = Act(al.take(R * 24), R, 24)
    CD = Act(al.take(R * 24), R, 24)
    BP = WB(al.take(128), 16, 8)
    XT = Act(al.take(R * 640), R, 640)                      # x~, later D
    blobs: List[np.ndarray] = []
    dst: List[int] = []
    steps: List[Step] = []

    ci = np.zeros((R, 24), np.int8)
    cd = np.zeros((R, 24), np.int8)
    for i in range(R):
        ci[i, 0], ci[i, 1 + i] = alpha[i], 16 - alpha[i]
        cd[i, 0], cd[i, 1 + i] = 1, -1
    steps.append(Step("interp", gemm("interp", CI, XB, XT, trans_b_act=True,
                                     simd=_simd_shift(4))))
    fsteps, acts, Y, P = _forward_steps(layers, al, R, XT, pb, blobs, dst)
    steps += fsteps
    # loss gradient, in place over y. x~ sits below y in the TCDM and strides
    # must be non-negative, so the operands are [x~ | y] @ [-59 I; 55 I].
    assert XT.off < Y.off
    steps.append(Step("loss", blockdiag2("loss", XT, Y, BP, Y,
                                         _simd_ms(sh["loss"], sh["loss_zp"]))))
    # Direct term of the score gradient: D = (x - b_i) / 2 (into the x~
    # region, free after the loss) and attr_gy = sum_i D * g_y via the
    # diagonal trick, with parameter buffer 1 as scratch (it is idle between
    # the last forward tile and the second backward tile).
    D = Act(XT.off, R, 640)
    AGY = al.take(640 * 4)
    steps.append(Step("attr", gemm("dmat", CD, XB, D, trans_b_act=True, simd=_simd_shift(1)),
                      scratch=pb[1]))
    DIAG1 = Acc(pb[1], 640, 8)
    steps.append(Step("attr", gemm("diag_gy", D, Y, DIAG1, trans_a=True, trans_b_act=True,
                                   b_advance_with_m=True), post=("diag", DIAG1.off, AGY),
                      scratch=pb[1]))
    # P (forward partial sums) is reused for the backward K-split.
    GA, GB = Act(al.take(R * TILE), R, TILE), Act(al.take(R * TILE), R, TILE)
    ATTR = pb[1] + 640 * 4                                  # parameter buffer 1 is free at the end
    g = Y
    gbufs = [GA, GB]
    for li in range(len(layers) - 1, -1, -1):
        L = layers[li]
        n_out, n_in = L.W.shape
        name = f"bwd{li + 1}"
        if li < len(layers) - 1:
            a = acts[li]                                    # output of layer li+1 (ReLU)
            steps[-1].post = ("mask", g.off, a.off, g.nbytes)
        if n_out > TILE:                                    # layer 10: K-split over 640
            nk = n_out // TILE
            out = Act(gbufs[0].off, R, n_in)
            for j in range(nk):
                buf = pb[len(blobs) % 2]
                blobs.append(pack_wb(L.W[j * TILE:(j + 1) * TILE]).view(np.uint8))
                dst.append(buf)
                B = WB(buf, TILE, n_in)
                kw = dict(k_split=(j * TILE // 8, TILE // 8))
                if j == 0:
                    lc = gemm(f"{name}k{j}", g, B, P, **kw)
                elif j < nk - 1:
                    lc = gemm(f"{name}k{j}", g, B, P, C=P, **kw)
                else:
                    lc = gemm(f"{name}k{j}", g, B, out, C=P, simd=_simd_ms(sh[name]), **kw)
                steps.append(Step("backward", lc, len(blobs) - 1))
        elif n_in > TILE:                                   # layer 1: N-split over 640
            out = Act(Y.off, R, n_in)                       # g0 reuses y / g_y
            for j in range(n_in // TILE):
                buf = pb[len(blobs) % 2]
                blobs.append(pack_wb(L.W[:, j * TILE:(j + 1) * TILE]).view(np.uint8))
                dst.append(buf)
                B = WB(buf, n_out, TILE)
                lc = gemm(f"{name}n{j}", g, B, out, n_range=(j * TILE // 8, TILE // 8),
                          simd=_simd_ms(sh[name]))
                steps.append(Step("backward", lc, len(blobs) - 1))
        else:
            out = Act(gbufs[0].off, R, n_in)
            buf = pb[len(blobs) % 2]
            blobs.append(pack_wb(L.W).view(np.uint8))
            dst.append(buf)
            lc = gemm(name, g, WB(buf, n_out, n_in), out, simd=_simd_ms(sh[name]))
            steps.append(Step("backward", lc, len(blobs) - 1))
        g = out
        gbufs.reverse()
    G0 = g
    # attr_g0 = sum_i D * g0 (parameter buffers are free now), then on the
    # core attr = m0 * attr_g0 - m1 * attr_gy.
    DIAG = Acc(pb[0], 640, 8)                               # 20 kB
    A0 = pb[1]
    steps.append(Step("attr", gemm("diag_g0", D, G0, DIAG, trans_a=True, trans_b_act=True,
                                   b_advance_with_m=True), post=("diag", DIAG.off, A0)))
    steps.append(Step("attr", None, post=("combine", A0, AGY, ATTR, sh["m0"], sh["m1"])))

    t = Tcdm()
    xb = np.zeros((24, 640), np.int8)
    xb[1:1 + R] = bq
    t.write(XB.off, pack_act(xb))
    t.write(CI.off, pack_act(ci))
    t.write(CD.off, pack_act(cd))
    eye = np.eye(8, dtype=np.int64)
    t.write(BP.off, np.concatenate([(-LOSS_B * eye).astype(np.int8).reshape(-1),
                                    (LOSS_A * eye).astype(np.int8).reshape(-1)]))
    image = t.read(0, al.top, np.uint8)
    stop = int(os.environ.get("TA_STOP", "0"))
    if stop:                                                # debug: truncated program
        prog = Program(image, steps[:stop], blobs, dst, xq.astype(np.int8), XB.off, attr_off=ATTR)
        t = emulate(prog)
        prog.checks = [(n, b.off, t.read(b.off, b.nbytes, np.uint8))
                       for n, b in (("y_gy_g0", Y), ("GA", GA), ("GB", GB), ("P", P))]
        prog.info["debug"] = f"stopped after step {stop}: {steps[stop - 1].launch.name}"
        return prog
    prog = Program(image, steps, blobs, dst, xq.astype(np.int8), XB.off, attr_off=ATTR)
    t = emulate(prog)
    ref = int_xai_reference(layers, xq, bq, alpha, sh)
    from gemmx_sched import unpack_act
    assert np.array_equal(unpack_act(t.read(G0.off, G0.nbytes), R, 640), ref["g0"]), "g0 mismatch"
    attr_emu = t.read(ATTR, 640 * 4, np.int32)
    assert np.array_equal(t.read(AGY, 2560, np.int32), ref["attr_gy"]), "attr_gy mismatch"
    assert np.array_equal(attr_emu, ref["attr"]), "attribution mismatch"
    prog.checks = ([(f"a{i + 1}", a.off, t.read(a.off, a.nbytes, np.uint8)) for i, a in enumerate(acts)]
                   + [("D", D.off, t.read(D.off, D.nbytes, np.uint8)),
                      ("attr_gy", AGY, t.read(AGY, 2560, np.uint8)),
                      ("g0", G0.off, t.read(G0.off, G0.nbytes, np.uint8)),
                      ("attr", ATTR, t.read(ATTR, 640 * 4, np.uint8))])
    prog.info["tcdm"] = f"{al.top} B"
    prog.info["shifts"] = str(sh)
    return prog


# --------------------------------------------------------------------------
# Emulation of a whole program (DMA, launches, core ops)
# --------------------------------------------------------------------------
def emulate(prog: Program) -> Tcdm:
    t = Tcdm()
    t.write(0, prog.image)
    for c in range(80):                                     # x into row 0 of its Act
        t.write(prog.x_dst + c * BLK, prog.x_l3[c * 8:(c + 1) * 8])
    for s in prog.steps:
        if s.blob >= 0:
            t.write(prog.blob_dst[s.blob], prog.blobs[s.blob])
        if s.launch is not None:
            run_launch(t, s.launch)
        if s.post and s.post[0] == "mask":
            _, g_off, a_off, n = s.post
            gv, av = t.read(g_off, n), t.read(a_off, n)
            t.write(g_off, np.where(av != -128, gv, 0).astype(np.int8))
        elif s.post and s.post[0] == "diag":
            d = t.read(s.post[1], 80 * 256, np.int32).reshape(80, 8, 8)
            t.write(s.post[2], np.array([d[f // 8, f % 8, f % 8] for f in range(640)], dtype=np.int32))
        elif s.post and s.post[0] == "combine":
            _, a0, agy, o, m0, m1 = s.post
            v = m0 * t.read(a0, 2560, np.int32).astype(np.int64) - m1 * t.read(agy, 2560, np.int32).astype(np.int64)
            assert np.all(np.abs(v) < 2 ** 31), "combine overflow"
            t.write(o, v.astype(np.int32))
    return t


# --------------------------------------------------------------------------
# C emission
# --------------------------------------------------------------------------
PHASES = ["interp", "forward", "loss", "backward", "attr"]


def _verify_prefetch(prog: Program, pre: List[int]) -> None:
    """Replay the DMA plan: every launch finds its blob in its buffer, and no
    prefetch targets a buffer read or used as scratch by the running step."""
    content: Dict[int, int] = {}
    first = next((s.blob for s in prog.steps if s.blob >= 0), -1)
    if first >= 0:
        content[prog.blob_dst[first]] = first
    for i, s in enumerate(prog.steps):
        if s.blob >= 0:
            assert content.get(prog.blob_dst[s.blob]) == s.blob, f"step {i}: blob {s.blob} not loaded"
        if pre[i] >= 0:
            d = prog.blob_dst[pre[i]]
            assert s.scratch != d and not (s.blob >= 0 and prog.blob_dst[s.blob] == d), f"step {i}: prefetch race"
        if s.scratch >= 0:
            content.pop(s.scratch, None)
        if pre[i] >= 0:
            content[prog.blob_dst[pre[i]]] = pre[i]


def emit(prog: Program, title: str) -> str:
    launches = [s.launch for s in prog.steps if s.launch is not None]
    out = ["#include <stdint.h>", '#include "gemmx_exec.h"', "",
           f"// {title}"] + [f"// {k}: {v}" for k, v in prog.info.items()]
    out.append(c_bytes("prog_image", prog.image))
    out.append(f"#define PROG_IMAGE_BYTES {prog.image.size}")
    out.append(c_bytes("prog_x", prog.x_l3))
    out.append(f"#define PROG_X_DST {prog.x_dst}")
    out.append(emit_launches("prog_launches", launches))
    blob_all = np.concatenate(prog.blobs)
    out.append(c_bytes("prog_params", blob_all))
    offs = np.cumsum([0] + [b.size for b in prog.blobs])[:-1]
    out.append("static const int32_t prog_blob_off[] = {" + ", ".join(map(str, offs)) + "};")
    out.append("static const int32_t prog_blob_len[] = {" + ", ".join(str(b.size) for b in prog.blobs) + "};")
    out.append("static const int32_t prog_blob_dst[] = {" + ", ".join(map(str, prog.blob_dst)) + "};")
    # steps: launch index, blob used, next blob to prefetch, post op
    rows, li, rows_pre = [], 0, []
    blob_steps = [i for i, s in enumerate(prog.steps) if s.blob >= 0]
    loaded = prog.steps[blob_steps[0]].blob if blob_steps else -1
    for i, s in enumerate(prog.steps):
        # While step i runs, the DMA core loads the next blob not yet loaded
        # into its buffer, unless step i reads that buffer (blobs alternate
        # buffers) or a step from i up to the blob's use needs it as scratch.
        nxt = [(j, prog.steps[j].blob) for j in blob_steps if j > i]
        pre = -1
        if nxt and nxt[0][1] != loaded:
            j, nb = nxt[0]
            d = prog.blob_dst[nb]
            busy = (s.blob >= 0 and prog.blob_dst[s.blob] == d) or any(
                prog.steps[q].scratch == d for q in range(i, j))
            if not busy:
                pre = loaded = nb
        post, pa, pb_, pn, m0, m1 = 0, 0, 0, 0, 0, 0
        if s.post and s.post[0] == "mask":
            post, pa, pb_, pn = 1, s.post[1], s.post[2], s.post[3]
        elif s.post and s.post[0] == "diag":
            post, pa, pb_ = 2, s.post[1], s.post[2]
        elif s.post and s.post[0] == "combine":
            post, pa, pb_, pn, m0, m1 = 3, s.post[1], s.post[2], s.post[3], s.post[4], s.post[5]
        rows_pre.append(pre)
        lidx = li if s.launch is not None else -1
        rows.append(f"  {{{lidx}, {pre}, {PHASES.index(s.phase)}, {post}, {pa}, {pb_}, {pn}, {m0}, {m1}}}")
        if s.launch is not None:
            li += 1
    _verify_prefetch(prog, rows_pre)
    out.append("typedef struct { int16_t launch, prefetch, phase, post; int32_t a, b, n, m0, m1; } step_t;")
    out.append(f"#define PROG_N_STEPS {len(prog.steps)}")
    out.append("static const step_t prog_steps[] = {\n" + ",\n".join(rows) + "\n};")
    first = prog.steps[blob_steps[0]].blob if blob_steps else -1
    out.append(f"#define PROG_FIRST_BLOB {first}")
    out.append(f"#define PROG_ATTR_OFF {prog.attr_off}")
    out.append('static const char *prog_phase_names[] = {' + ", ".join(f'"{p}"' for p in PHASES) + "};")
    exp = np.concatenate([e.view(np.uint8).reshape(-1) for _, _, e in prog.checks])
    out.append(c_bytes("prog_expected", exp))
    pos = np.cumsum([0] + [e.size for _, _, e in prog.checks])[:-1]
    out.append(f"#define PROG_N_CHECKS {len(prog.checks)}")
    out.append('static const char *prog_check_names[] = {' + ", ".join(f'"{n}"' for n, _, _ in prog.checks) + "};")
    out.append("static const int32_t prog_check_off[] = {" + ", ".join(str(o) for _, o, _ in prog.checks) + "};")
    out.append("static const int32_t prog_check_len[] = {" + ", ".join(str(e.size) for _, _, e in prog.checks) + "};")
    out.append("static const int32_t prog_check_pos[] = {" + ", ".join(map(str, pos)) + "};")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Real-data selection
# --------------------------------------------------------------------------
def load_data():
    """Layers, test features and normal background vectors (cached)."""
    import toyadmos_data as T
    return A.load_layers(), T.load_features(), T.load_background()


def select_input(layers: List[A.Layer], feats: dict) -> Tuple[np.ndarray, str]:
    """The most anomalous test clip (highest GeMMX-semantics anomaly score)
    and, in it, the 5-frame window with the largest reconstruction error."""
    scores_path = os.path.normpath(os.path.join(HERE, "../../../../../external/toycar/gemmx_int_scores.npy"))
    scores = np.load(scores_path)
    anom = np.where(feats["labels"] == 1)[0]
    c = int(anom[np.argmax(scores[anom])])
    xq = A.quantize_input(feats["vectors"][c], layers)
    y = A.forward(xq, layers)[-1]
    L0, L9 = layers[0], layers[-1]
    err = np.mean(((xq.astype(np.float64) - L0.zp_x) * L0.s_x - (y.astype(np.float64) - L9.zp_y) * L9.s_y) ** 2, axis=1)
    v = int(np.argmax(err))
    return xq[v], f"{feats['names'][c]} window {v} (score {scores[c]:.3f})"


def baselines(method: str, layers: List[A.Layer], bg: np.ndarray,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """IG: 16 copies of the mean normal vector, alpha = 1..16 (/16).
    SHAP: 16 random normal training vectors, alpha uniform in 0..16 (/16)."""
    if method == "ig":
        b = A.quantize_input(bg.mean(axis=0)[None], layers)
        return np.repeat(b, N_SAMPLES, axis=0), np.arange(1, N_SAMPLES + 1)
    idx = rng.choice(len(bg), N_SAMPLES, replace=False)
    return A.quantize_input(bg[idx], layers), rng.integers(0, 17, size=N_SAMPLES)


def main(prog_name: str) -> None:
    layers, feats, bg = load_data()
    xq, desc = select_input(layers, feats)
    if prog_name == "inference":
        prog = build_inference(layers, xq)
    else:
        rng = np.random.default_rng(7)
        sh = calibrate(layers, bg, rng)
        bq, alpha = baselines(prog_name, layers, bg, np.random.default_rng(11))
        prog = build_xai(layers, xq, bq, alpha, sh)
    prog.info["input"] = desc
    print(emit(prog, f"ToyAdmos {prog_name} on GeMMX (ad01_int8, DCASE 2020 ToyCar)"))
