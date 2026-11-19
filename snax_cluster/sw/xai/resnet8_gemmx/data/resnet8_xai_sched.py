"""GeMMX schedule for input-level Integrated Gradients / Gradient SHAP of
MLPerf Tiny ResNet-8 (real weights, real CIFAR-10 image), emulated
bit-exactly against resnet8_xai.int_input_grad / int_attr.

One sample = interpolate + forward (as inference) + backward to the input +
attribution multiply-accumulate. The launch table is the same for every
sample: per sample only the interpolation coefficients and the baseline
image change (DMA'd at the start). Sample 0 has alpha = 16 (x itself), so
its forward pass is the inference and gives the explained class.

TCDM slots (padded C8HW8, see resnet8.PAct): S_B < S_C < S_A, 26 kB each.
Forward activations needed for ReLU masks are spilled to L3 and fetched
back in 4 kB chunks during the backward pass.

Backward ops:
  bcast   G_add3[p, c] = W_fc[pred, c]           (ones-row GeMM, 1 launch)
  bconv1  stride-1 transposed conv = conv with the 180-degree rotated,
          per-input-channel requantised kernel, over zero-bordered gradients
  bconv2  stride-2 transposed conv as 4 polyphase launches per 8 channels
          into compact phase buffers + DMA interleave (the D8 writer cannot
          skip pixels); 1x1 stride-2: one compact launch + DMA scatter
  fork    a*G1 + b*G2 block-diagonal add (in place over the higher buffer)
  mask    G * (act != -128), both cores, act chunks fetched from L3
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../toyadmos_gemmx/data")))

import resnet8 as RR  # noqa: E402
import resnet8_int as R  # noqa: E402
import resnet8_xai as X  # noqa: E402
from emit import c_bytes  # noqa: E402
from gemmx_sched import BLK, BLK32, Launch, Tcdm, emit_cfg_code, run_launch  # noqa: E402

TCDM_LIMIT = 122 * 1024
SLOT = 26 * 1024
PBUF = 5 * 1024
MCHUNK = 4096
GMIN, GMAX = -127, 127


class Alloc:
    def __init__(self) -> None:
        self.top = 0

    def take(self, n: int) -> int:
        off = self.top
        self.top = (self.top + n + 63) // 64 * 64
        assert self.top <= TCDM_LIMIT, f"TCDM overflow {self.top}"
        return off


@dataclass
class Step:
    phase: str
    launch: Optional[Launch] = None
    blob: int = -1
    pre: List[Tuple] = field(default_factory=list)    # ops before the launch
    post: List[Tuple] = field(default_factory=list)   # ops after the launch


# --------------------------------------------------------------------------
# Synthetic conv descriptors for backward launches
# --------------------------------------------------------------------------
def _ms_bias(ratio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m0, sh = X.fit_ms(ratio)
    half = np.array([int(round(2 ** (s - 1) / m)) for m, s in zip(m0, sh)], dtype=np.int32)
    return m0, sh, half


def bwd_conv_s1(B: X.BwdConv, ratio: np.ndarray) -> R.Conv:
    """Stride-1 transposed conv as a forward conv: kernel rotated 180 degrees,
    channels swapped (output = forward input channels)."""
    Wr = B.Wb[:, ::-1, ::-1, :].transpose(3, 1, 2, 0)            # (ci, kh, kw, co)
    ci = Wr.shape[0]
    cp = -(-ci // 8) * 8                                       # pad outputs to a full tile
    Wp = np.zeros((cp,) + Wr.shape[1:], dtype=np.int8)
    Wp[:ci] = Wr
    rp = np.ones(cp)
    rp[:ci] = ratio
    m0, sh, half = _ms_bias(rp)
    return R.Conv(Wp, half, m0, sh, 0, 0, 1, Wp.shape[1] // 2, False)


def phase_taps(k: int, p: int) -> List[Tuple[int, int]]:
    """Stride-2 SAME (pad before 0): input row 2q+p receives tap i from
    output row q - (i - p)//2. Returns [(i, dy)] sorted by dy."""
    taps = [(i, -((i - p) // 2)) for i in range(k) if (i - p) % 2 == 0]
    return sorted(taps, key=lambda t: t[1])


def phase_launch(name: str, G: RR.PAct, out_off: int, out_wc: int, B: X.BwdConv, nt: int,
                 py: int, px: int, w_off: int, c_off: int, ratio_cols: np.ndarray,
                 out_hc: int) -> Tuple[Launch, np.ndarray, np.ndarray]:
    """One polyphase launch: input-gradient pixels (2q+py, 2p+px), channels
    nt*8..+7, written compact (out_hc x out_wc). Returns launch, weights, bias."""
    kh, kw = B.Wb.shape[1:3]
    rt, ct_ = phase_taps(kh, py), phase_taps(kw, px)
    dy0, dx0 = rt[0][1], ct_[0][1]
    assert [d for _, d in rt] == list(range(dy0, dy0 + len(rt)))
    assert [d for _, d in ct_] == list(range(dx0, dx0 + len(ct_)))
    K = len(ct_) * len(rt) * G.ct
    M = (out_wc // 8) * out_hc
    a_off = G.addr(0, dy0, dx0)
    L = Launch(name, M, K, 1, a_off, [len(ct_), len(rt), G.ct, 1, out_wc // 8, out_hc],
               [8, G.wp * 8, G.plane, 0, BLK, G.wp * 8], w_off, [K, 1, M], [BLK, 0, 0])
    L.c_en, L.c_off = 1, c_off
    L.c_tb, L.c_ts = [1, 1, M], [BLK32, 0, 0]
    L.bypass = 0
    L.d8_off = out_off
    L.d8_tb, L.d8_ts = [1, out_wc // 8, out_hc], [BLK, BLK, out_wc * 8]
    m0, sh, half = _ms_bias(ratio_cols[nt * 8:nt * 8 + 8])
    L.mult8, L.shift8 = [int(v) for v in m0], [int(v) for v in sh]
    L.zp_out, L.out_min, L.out_max = 0, GMIN, GMAX
    blocks = []
    for ct in range(G.ct):
        for i, _ in rt:
            for j, _ in ct_:
                blocks.append(B.Wb[ct * 8:ct * 8 + 8, i, j, nt * 8:nt * 8 + 8].T)   # [col=ci][k=co]
    w = np.concatenate([np.ascontiguousarray(b).reshape(-1) for b in blocks]).astype(np.int8)
    bias = np.tile(half, (8, 1)).astype(np.int32).reshape(-1)
    return L, w, bias


def ileave_dmas(ph_off: int, hc: int, wc: int, F: RR.PAct, nt: int, py: int, px: int) -> List[Tuple]:
    """2-D DMAs copying a compact phase (hc x wc, one 8-channel tile) to the
    pixels (2q+py, 2p+px) of full-resolution buffer F: (dst, src, size,
    dst_stride, src_stride, repeat)."""
    return [(F.addr(nt, 2 * q + py, px), ph_off + q * wc * 8, 8, 16, 8, wc) for q in range(hc)]


# --------------------------------------------------------------------------
# Program
# --------------------------------------------------------------------------
@dataclass
class Program:
    image: np.ndarray
    steps: List[Step]
    blobs: List[np.ndarray]
    blob_dst: List[int]
    dmas: List[Tuple]                       # 2-D DMA descriptor table
    l3_size: int
    l3_init: Dict[int, np.ndarray]          # read-only L3 data (x, baselines, B pairs)
    samples: int
    xq: np.ndarray
    bq: np.ndarray
    alpha: np.ndarray
    attr_off: int
    info: Dict[str, str] = field(default_factory=dict)
    checks: List[Tuple[str, int, np.ndarray]] = field(default_factory=list)


def build(M: R.Model, BW: Dict[str, X.BwdConv], S: Dict[str, float], xq: np.ndarray,
          bq: np.ndarray, alpha: np.ndarray, n_samples: int) -> Program:
    assert alpha[0] == 16, "sample 0 must be x itself (alpha = 16)"
    _FC["W"] = np.clip(M.fc.W[:, 0, 0, :], -127, 127).astype(np.int8)
    al = Alloc()
    S_B, S_C, S_A = al.take(SLOT), al.take(SLOT), al.take(SLOT)
    pb = (al.take(PBUF), al.take(PBUF))
    PH = (al.take(4096), al.take(4096))                        # compact phase buffers
    MB = al.take(MCHUNK)                                       # mask activation chunk
    ATTR = al.take(32 * 32 * 3 * 4)
    resident: List[Tuple[int, np.ndarray]] = []
    blobs: List[np.ndarray] = []
    blob_dst: List[int] = []
    dmas: List[Tuple] = []
    steps: List[Step] = []

    def param(b: np.ndarray) -> int:
        o = al.take(b.nbytes)
        resident.append((o, b.view(np.uint8).reshape(-1)))
        return o

    def stream(w: np.ndarray, bias: np.ndarray) -> Tuple[int, int, int]:
        buf = pb[len(blobs) % 2]
        blob = np.concatenate([w.view(np.uint8), bias.view(np.uint8)])
        assert blob.size <= PBUF, blob.size
        blobs.append(blob)
        blob_dst.append(buf)
        return buf, buf + w.size, len(blobs) - 1

    # ---- L3 layout: spills (rw), x, baselines, B pairs (ro)
    l3: Dict[str, int] = {}
    l3_top = 0

    def l3_take(name: str, n: int) -> int:
        nonlocal l3_top
        l3[name] = l3_top
        l3_top = (l3_top + n + 63) // 64 * 64
        return l3[name]

    X0 = RR.PAct(S_A, 8, 32, 32)
    XC = RR.PAct(S_C, 8, 32, 32)
    l3_take("x", X0.nbytes)
    for i in range(n_samples):
        l3_take(f"b{i}", X0.nbytes)
        l3_take(f"bp{i}", 128)

    # ---------------- sample start: x -> S_C, b_i -> S_A, interp into S_A
    eye = np.eye(8, dtype=np.int64)
    BPAIR = al.take(128)
    ZC = param(np.zeros(64, dtype=np.int32))                  # zero C tile
    interp = RR.add_launch("interp", XC, X0, X0,
                           R.Add(1, 1, 0, 1, 4, 0), BPAIR, ZC)
    interp.mult, interp.shift = 1, 4                           # floor(acc / 16)
    steps.append(Step("interp", interp, pre=[("sample_in", XC.off, X0.off, BPAIR)]))

    # ---------------- forward (as resnet8.build), with spills
    conv = M.convs

    def fwd_conv(name: str, Xa: RR.PAct, Ya: RR.PAct, L: R.Conv) -> None:
        for nt in range(Ya.ct):
            wb, bias = RR.conv_weights(L, nt, Xa.ct)
            w_off, c_off, bi = stream(wb, bias)
            steps.append(Step("forward", RR.conv_launch(f"{name}n{nt}", Xa, Ya, L, nt, w_off, c_off),
                              blob=bi))

    def spill(P: RR.PAct, name: str) -> None:
        off = l3_take(f"act_{name}", P.nbytes)
        steps[-1].post.append(("spill", P.off, off, P.nbytes))

    def fill(P: RR.PAct, val: int) -> None:
        steps.append(Step("fill", None, pre=[("fill", P.off, P.nbytes, val)]))

    def fwd_add(name: str, X1: RR.PAct, X2: RR.PAct, A: R.Add) -> None:
        bp = param(np.concatenate([(A.a * eye).astype(np.int8).reshape(-1),
                                   (A.b * eye).astype(np.int8).reshape(-1)]))
        ct = param(np.full(64, A.c, dtype=np.int32))
        steps.append(Step("forward", RR.add_launch(name, X1, X2, X2, A, bp, ct)))

    C1 = RR.PAct(S_B, 16, 32, 32)
    fill(C1, -128)                                             # gradient zeros from last sample
    fwd_conv("conv1", X0, C1, conv["conv1"]); spill(C1, "conv1")
    B1 = RR.PAct(S_C, 16, 32, 32)
    fill(B1, -128)
    fwd_conv("b1c1", C1, B1, conv["b1c1"]); spill(B1, "b1c1")
    B2 = RR.PAct(S_A, 16, 32, 32)
    fill(B2, -128)
    fwd_conv("b1c2", B1, B2, conv["b1c2"])
    fwd_add("add1", C1, B2, M.adds[0]); spill(B2, "add1")
    P21 = RR.PAct(S_B, 32, 16, 16); fill(P21, -128)
    fwd_conv("b2c1", B2, P21, conv["b2c1"]); spill(P21, "b2c1")
    P2S = RR.PAct(S_C, 32, 16, 16)
    fwd_conv("b2sc", B2, P2S, conv["b2sc"])
    P22 = RR.PAct(S_A, 32, 16, 16); fill(P22, -128)
    fwd_conv("b2c2", P21, P22, conv["b2c2"])
    fwd_add("add2", P2S, P22, M.adds[1]); spill(P22, "add2")
    P31 = RR.PAct(S_B, 64, 8, 8); fill(P31, -128)
    fwd_conv("b3c1", P22, P31, conv["b3c1"]); spill(P31, "b3c1")
    P3S = RR.PAct(S_C, 64, 8, 8)
    fwd_conv("b3sc", P22, P3S, conv["b3sc"])
    P32 = RR.PAct(S_A, 64, 8, 8)
    fwd_conv("b3c2", P31, P32, conv["b3c2"])
    fwd_add("add3", P3S, P32, M.adds[2]); spill(P32, "add3")
    ones = param(np.ones(64, dtype=np.int8))
    pc = param(np.full(64, 32, dtype=np.int32))
    POOL = al.take(8 * 64)
    steps.append(Step("forward", RR.pool_launch("pool", P32, ones, pc, POOL)))
    from gemmx_sched import pack_wb
    wfc = np.zeros((64, 16), dtype=np.int8)
    wfc[:, :10] = M.fc.W[:, 0, 0, :].T
    fw = param(pack_wb(wfc))
    fb = np.zeros(16, dtype=np.int32)
    fb[:10] = M.fc.bias
    fcb = param(np.tile(fb, (8, 1)).reshape(8, 2, 8).transpose(1, 0, 2).reshape(-1))
    LOG = al.take(8 * 16)
    steps.append(Step("forward", RR.fc_launch("fc", POOL, 8, M.fc, fw, fcb, LOG, 2)))

    # ---------------- backward
    GP = al.take(8 * 64)                                       # g_pool B blocks (placed once)
    ONES1 = np.zeros((8, 8), dtype=np.int8)
    ONES1[:, 0] = 1                                            # A rows [1, 0, ..., 0]
    ones1 = param(ONES1.reshape(-1))
    G3 = RR.PAct(S_A, 64, 8, 8)                                # add3 dead after pool / spill
    fill(G3, 0)
    bc = Launch("bcast", 8, 1, 8, ones1, [1, 8, 8, 1, 1, 1], [0, 0, 0, 0, 0, 0],
                GP, [1, 8, 8], [0, BLK, 0])
    bc.c_tb, bc.c_ts = [1, 8, 8], [BLK32, 0, 0]
    bc.bypass = 0
    bc.d8_off = G3.addr(0, 0, 0)
    bc.d8_tb, bc.d8_ts = [8, 8, 1], [G3.plane, G3.wp * 8, 0]
    bc.mult, bc.shift, bc.zp_out, bc.out_min, bc.out_max = 2, 1, 0, GMIN, GMAX
    steps.append(Step("backward", bc, pre=[("gpool", LOG, GP)]))

    def mask(G: RR.PAct, name: str) -> None:
        # interior only: border gradients are zero and stay zero
        steps[-1].post.append(("mask", G.off, l3[f"act_{name}"], (G.ct << 16) | G.h,
                               (G.w << 16) | G.wp))

    mask(G3, "add3")

    def bconv1(name: str, Gin: RR.PAct, Gout: RR.PAct, Sin: float, key: str) -> None:
        B = BW[name]
        Lc = bwd_conv_s1(B, Sin * B.s_b / S[key])
        for nt in range(Gout.ct):
            wb, bias = RR.conv_weights(Lc, nt, Gin.ct)
            w_off, c_off, bi = stream(wb, bias)
            La = RR.conv_launch(f"bwd_{name}n{nt}", Gin, Gout, Lc, nt, w_off, c_off)
            La.out_min, La.out_max = GMIN, GMAX
            steps.append(Step("backward", La, blob=bi))

    def bconv2(name: str, Gin: RR.PAct, F: RR.PAct, Sin: float, key: str) -> None:
        """Stride-2 transposed conv (3x3 or 1x1) of Gin into full-res F."""
        B = BW[name]
        k = B.Wb.shape[1]
        hc, wc = Gin.h, Gin.w
        ratio = Sin * B.s_b / S[key]
        phases = [(py, px) for py in range(2) for px in range(2)] if k == 3 else [(0, 0)]
        for nt in range(F.ct):
            for (py, px) in phases:
                ph = PH[len([s for s in steps if s.launch is not None and s.launch.name.startswith("ph")]) % 2]
                La, w, bias = phase_launch(f"ph_{name}n{nt}p{py}{px}", Gin, ph, wc, B, nt, py, px,
                                           0, 0, ratio, hc)
                w_off, c_off, bi = stream(w, bias)
                La.b_off, La.c_off = w_off, c_off
                d0 = len(dmas)
                dmas.extend(ileave_dmas(ph, hc, wc, F, nt, py, px))
                steps.append(Step("backward", La, blob=bi, post=[("dmas", d0, len(dmas))]))

    def fork(name: str, X1: RR.PAct, S1: float, X2: RR.PAct, S2: float, key: str) -> None:
        a, b = X.best_pair(S1 / S2)
        m0, sh, half = _ms_bias(np.array([S2 / b / S[key]]))
        A = R.Add(a, b, int(half[0]), int(m0[0]), int(sh[0]), 0)
        bp = param(np.concatenate([(a * eye).astype(np.int8).reshape(-1),
                                   (b * eye).astype(np.int8).reshape(-1)]))
        ct = param(np.full(64, A.c, dtype=np.int32))
        La = RR.add_launch(name, X1, X2, X2, A, bp, ct)
        La.out_min, La.out_max = GMIN, GMAX
        steps.append(Step("backward", La))

    # block 3: add3 -> b3c2' -> b3c1 ; b3c1' (s2) + b3sc' (s2) -> add2
    Gb3 = RR.PAct(S_B, 64, 8, 8)
    fill(Gb3, 0)
    bconv1("b3c2", G3, Gb3, S["add3"], "b3c1"); mask(Gb3, "b3c1")
    S3 = RR.PAct(S_C, 32, 16, 16)
    fill(S3, 0)
    bconv2("b3sc", G3, S3, S["add3"], "s3")                    # G3 dead after this
    M3 = RR.PAct(S_A, 32, 16, 16)
    fill(M3, 0)
    bconv2("b3c1", Gb3, M3, S["b3c1"], "m3")
    fork("fork2", S3, S["s3"], M3, S["m3"], "add2")           # -> M3 (S_A): G_add2
    G2 = M3
    mask(G2, "add2")
    # block 2
    Gb2 = RR.PAct(S_B, 32, 16, 16)
    fill(Gb2, 0)
    bconv1("b2c2", G2, Gb2, S["add2"], "b2c1"); mask(Gb2, "b2c1")
    S2 = RR.PAct(S_C, 16, 32, 32)
    fill(S2, 0)
    bconv2("b2sc", G2, S2, S["add2"], "s2")                    # G2 dead after this
    M2 = RR.PAct(S_A, 16, 32, 32)
    fill(M2, 0)
    bconv2("b2c1", Gb2, M2, S["b2c1"], "m2")
    fork("fork1", S2, S["s2"], M2, S["m2"], "add1")           # -> M2 (S_A): G_add1
    G1 = M2
    mask(G1, "add1")
    # block 1 (identity shortcut)
    Gb1 = RR.PAct(S_B, 16, 32, 32)
    fill(Gb1, 0)
    bconv1("b1c2", G1, Gb1, S["add1"], "b1c1"); mask(Gb1, "b1c1")
    M1 = RR.PAct(S_C, 16, 32, 32)
    fill(M1, 0)
    bconv1("b1c1", Gb1, M1, S["b1c1"], "m1")
    fork("fork0", M1, S["m1"], G1, S["add1"], "conv1")        # -> G1 (S_A): G_conv1
    Gc = G1
    mask(Gc, "conv1")
    GX = RR.PAct(S_B, 8, 32, 32)
    fill(GX, 0)
    bconv1("conv1", Gc, GX, S["conv1"], "x")
    steps[-1].post.append(("attr", GX.off, S_C, S_A, ATTR))  # x -> S_C, b_i -> S_A, MAC

    # ---------------- image, L3, emulation
    t = Tcdm()
    for o, b in resident:
        t.write(o, b)
    image = t.read(0, al.top, np.uint8)
    l3_init = {l3["x"]: RR.pack_pact(xq, X0)}
    for i in range(n_samples):
        l3_init[l3[f"b{i}"]] = RR.pack_pact(bq[i], X0)
        l3_init[l3[f"bp{i}"]] = np.concatenate([(int(alpha[i]) * eye).astype(np.int8).reshape(-1),
                                                ((16 - int(alpha[i])) * eye).astype(np.int8).reshape(-1)]).view(np.uint8)
    prog = Program(image, steps, blobs, blob_dst, dmas, l3_top, l3_init, n_samples, xq, bq,
                   alpha, ATTR)
    prog.info.update(tcdm=f"{al.top} B", l3=f"{l3_top} B", mb=str(MB),
                     launches_per_sample=str(sum(s.launch is not None for s in steps)),
                     dmas_per_sample=str(len(dmas)))
    prog.l3 = l3
    return prog


# --------------------------------------------------------------------------
# Emulation
# --------------------------------------------------------------------------
def emulate(prog: Program, n: Optional[int] = None) -> Tuple[Tcdm, np.ndarray, int]:
    t = Tcdm()
    t.write(0, prog.image)
    L3 = np.zeros(prog.l3_size, dtype=np.uint8)
    for o, b in prog.l3_init.items():
        L3[o:o + b.size] = b.view(np.uint8)
    t.write(prog.attr_off, np.zeros(32 * 32 * 3, dtype=np.int32))
    pred = -1
    l3 = prog.l3
    X0 = RR.PAct(0, 8, 32, 32)
    for smp in range(n or prog.samples):
        for s in prog.steps:
            for op in s.pre:
                if op[0] == "sample_in":
                    _, xc, x0, bp = op
                    t.mem[xc:xc + X0.nbytes] = L3[l3["x"]:l3["x"] + X0.nbytes]
                    t.mem[x0:x0 + X0.nbytes] = L3[l3[f"b{smp}"]:l3[f"b{smp}"] + X0.nbytes]
                    t.mem[bp:bp + 128] = L3[l3[f"bp{smp}"]:l3[f"bp{smp}"] + 128]
                elif op[0] == "fill":
                    t.mem[op[1]:op[1] + op[2]] = np.uint8(op[3] & 0xFF)
                elif op[0] == "gpool":
                    if smp == 0:
                        lg = RR.read_logits(t, op[1])
                        pred = int(np.argmax(lg))
                        blk = np.zeros((8, 8, 8), dtype=np.int8)          # ct, col, k
                        blk[:, :, 0] = R_fc_row(pred).reshape(8, 8)
                        t.write(op[2], blk.reshape(-1))
            if s.blob >= 0:
                t.write(prog.blob_dst[s.blob], prog.blobs[s.blob])
            if s.launch is not None:
                run_launch(t, s.launch)
            for op in s.post:
                if op[0] == "spill":
                    L3[op[2]:op[2] + op[3]] = t.mem[op[1]:op[1] + op[3]]
                elif op[0] == "mask":
                    _, g0, a0, cth, wwp = op
                    ct, h, w, wp = cth >> 16, cth & 0xFFFF, wwp >> 16, wwp & 0xFFFF
                    plane = (h + 2) * wp * 8
                    for c in range(ct):
                        for y in range(h):
                            o = c * plane + (y + 1) * wp * 8 + 64
                            g = t.read(g0 + o, w * 8)
                            a = L3[a0 + o:a0 + o + w * 8].view(np.int8)
                            t.write(g0 + o, np.where(a != -128, g, 0).astype(np.int8))
                elif op[0] == "dmas":
                    for (dst, src, size, ds, ss, rep) in prog.dmas[op[1]:op[2]]:
                        for r in range(rep):
                            t.mem[dst + r * ds:dst + r * ds + size] = t.mem[src + r * ss:src + r * ss + size]
                elif op[0] == "attr":
                    _, gx, xc, x0, attr = op
                    P = RR.PAct(0, 8, 32, 32)
                    t.mem[xc:xc + P.nbytes] = L3[l3["x"]:l3["x"] + P.nbytes]
                    t.mem[x0:x0 + P.nbytes] = L3[l3[f"b{smp}"]:l3[f"b{smp}"] + P.nbytes]
                    g = RR.unpack_pact(t.read(gx, P.nbytes), RR.PAct(gx, 8, 32, 32))[:, :, :3]
                    xv = RR.unpack_pact(t.read(xc, P.nbytes), RR.PAct(xc, 8, 32, 32))[:, :, :3]
                    bv = RR.unpack_pact(t.read(x0, P.nbytes), RR.PAct(x0, 8, 32, 32))[:, :, :3]
                    acc = t.read(attr, 32 * 32 * 3 * 4, np.int32).reshape(32, 32, 3).astype(np.int64)
                    acc += (xv.astype(np.int64) - bv) * g
                    t.write(attr, acc.astype(np.int32))
    return t, t.read(prog.attr_off, 32 * 32 * 3 * 4, np.int32).reshape(32, 32, 3), pred


_FC: Dict[int, np.ndarray] = {}


def R_fc_row(pred: int) -> np.ndarray:
    return _FC["W"][pred]


# --------------------------------------------------------------------------
# C emission
# --------------------------------------------------------------------------
PHASES = ["interp", "forward", "fill", "backward"]
OPS = {"sample_in": 1, "fill": 2, "gpool": 3, "spill": 4, "mask": 5, "dmas": 6, "attr": 7}


def _prefetch_plan(prog: Program) -> List[int]:
    """Prefetch plan for the step table executed sample after sample: while
    step i runs, load the next blob not yet present unless it targets the
    buffer step i reads. Built on the steady state (second pass, after the
    wrap); replayed for the first sample (only blob 0 preloaded) and for the
    steady state to check every launch finds its blob."""
    bsteps = [i for i, s in enumerate(prog.steps) if s.blob >= 0]
    first = prog.steps[bsteps[0]].blob

    def run(content: Dict[int, int], plan: Optional[List[int]]) -> List[int]:
        made = []
        for i, s in enumerate(prog.steps):
            if s.blob >= 0:
                assert content.get(prog.blob_dst[s.blob]) == s.blob, f"step {i}: blob missing"
            if plan is None:
                nxt = [prog.steps[j].blob for j in bsteps if j > i] or [first]
                nb = nxt[0]
                ok = content.get(prog.blob_dst[nb]) != nb and not (
                    s.blob >= 0 and prog.blob_dst[s.blob] == prog.blob_dst[nb])
                made.append(nb if ok else -1)
            else:
                made.append(plan[i])
            if made[-1] >= 0:
                content[prog.blob_dst[made[-1]]] = made[-1]
        return made

    state = {prog.blob_dst[first]: first}
    run(state, None)                       # first pass: reach the steady state
    steady = dict(state)
    plan = run(dict(steady), None)         # plan from the steady state
    run({prog.blob_dst[first]: first}, plan)   # replay: first sample
    run(dict(steady), plan)                    # replay: steady state
    return plan


def emit(prog: Program, title: str) -> str:
    launches = [s.launch for s in prog.steps if s.launch is not None]
    out = ["#include <stdint.h>", "", f"// {title}"] + [f"// {k}: {v}" for k, v in prog.info.items()]
    out.append(c_bytes("prog_image", prog.image))
    out.append(f"#define PROG_IMAGE_BYTES {prog.image.size}")
    out.append(emit_cfg_code(launches, True, "cfg"))
    out.append(c_bytes("prog_params", np.concatenate(prog.blobs)))
    offs = np.cumsum([0] + [b.size for b in prog.blobs])[:-1]
    out.append("static const int32_t prog_blob_off[] = {" + ", ".join(map(str, offs)) + "};")
    out.append("static const int32_t prog_blob_len[] = {" + ", ".join(str(b.size) for b in prog.blobs) + "};")
    out.append("static const int32_t prog_blob_dst[] = {" + ", ".join(map(str, prog.blob_dst)) + "};")
    first = next(s.blob for s in prog.steps if s.blob >= 0)
    out.append(f"#define PROG_FIRST_BLOB {first}")
    out.append("typedef struct { int32_t dst, src, size, dst_stride, src_stride, repeat; } dma2d_t;")
    out.append("static const dma2d_t prog_dmas[] = {\n" + ",\n".join(
        "  {" + ", ".join(map(str, d)) + "}" for d in prog.dmas) + "\n};")
    # L3: one writable image (read-only inputs + spill space)
    l3 = np.zeros(prog.l3_size, dtype=np.uint8)
    for o, b in prog.l3_init.items():
        l3[o:o + b.size] = b.view(np.uint8)
    lines = [", ".join(f"0x{x:02x}" for x in l3[i:i + 24]) for i in range(0, l3.size, 24)]
    out.append(f"static uint8_t prog_l3[{l3.size}] __attribute__((aligned(64))) = {{\n  "
               + ",\n  ".join(lines) + "\n};")
    out.append("static const int8_t prog_fc[640] = {" + ", ".join(str(int(v)) for v in _FC["W"].reshape(-1)) + "};")
    out.append("#define prog_fc_row(p, c) prog_fc[(p) * 64 + (c)]")
    out.append(f"#define PROG_L3_X {prog.l3['x']}")
    out.append(f"#define PROG_SAMPLES {prog.samples}")
    out.append("static const int32_t prog_l3_b[] = {" + ", ".join(str(prog.l3[f'b{i}']) for i in range(prog.samples)) + "};")
    out.append("static const int32_t prog_l3_bp[] = {" + ", ".join(str(prog.l3[f'bp{i}']) for i in range(prog.samples)) + "};")
    out.append(f"#define PROG_X_BYTES {RR.PAct(0, 8, 32, 32).nbytes}")
    out.append(f"#define PROG_MB {prog.info['mb']}")
    out.append(f"#define PROG_MCHUNK {MCHUNK}")
    out.append(f"#define PROG_ATTR_OFF {prog.attr_off}")
    # ops + steps
    ops, rows, li = [], [], 0
    plan = _prefetch_plan(prog)
    for i, s in enumerate(prog.steps):
        p0 = len(ops)
        ops += [(OPS[o[0]],) + tuple(o[1:]) for o in s.pre]
        p1 = len(ops)
        ops += [(OPS[o[0]],) + tuple(o[1:]) for o in s.post]
        p2 = len(ops)
        rows.append(f"  {{{li if s.launch is not None else -1}, {plan[i]}, {PHASES.index(s.phase)}, {p0}, {p1}, {p2}}}")
        li += s.launch is not None
    out.append("typedef struct { int32_t type, a, b, c, d; } op_t;")
    out.append("static const op_t prog_ops[] = {\n" + ",\n".join(
        "  {" + ", ".join(str(int(v)) for v in (list(o) + [0] * (5 - len(o)))) + "}" for o in ops) + "\n};")
    out.append("typedef struct { int16_t launch, prefetch, phase, pre0, pre1, post1; } xstep_t;")
    out.append(f"#define PROG_N_STEPS {len(prog.steps)}")
    out.append("static const xstep_t prog_steps[] = {\n" + ",\n".join(rows) + "\n};")
    out.append('static const char *prog_phase_names[] = {' + ", ".join(f'"{p}"' for p in PHASES) + "};")
    exp = np.concatenate([e.reshape(-1) for _, _, e in prog.checks])
    out.append(c_bytes("prog_expected", exp))
    pos = np.cumsum([0] + [e.size for _, _, e in prog.checks])[:-1]
    out.append(f"#define PROG_N_CHECKS {len(prog.checks)}")
    out.append('static const char *prog_check_names[] = {' + ", ".join(f'"{n}"' for n, _, _ in prog.checks) + "};")
    out.append("static const int32_t prog_check_off[] = {" + ", ".join(str(o) for _, o, _ in prog.checks) + "};")
    out.append("static const int32_t prog_check_len[] = {" + ", ".join(str(e.size) for _, _, e in prog.checks) + "};")
    out.append("static const int32_t prog_check_pos[] = {" + ", ".join(map(str, pos)) + "};")
    return "\n".join(out) + "\n"


def main() -> None:
    method = os.environ.get("R8X_METHOD", "ig")
    n = int(os.environ.get("R8X_N", "16"))
    k = int(os.environ.get("R8_SAMPLE", "1"))
    M = R.load_model()
    BW = X.backward_weights(M)
    S = X.calibrate(M, BW)
    img, label = RR.load_image(k)
    xq = R.quantize_image(img, M)
    bq, al = X.baselines(method, M, np.random.default_rng(100 + k))
    if method == "shap":
        al[0] = 16                                             # sample 0 = x (fused inference)
    order = np.argsort(-al, kind="stable")                     # alpha = 16 first
    bq, al = bq[order], al[order]
    bq, al = bq[:n], al[:n]
    prog = build(M, BW, S, xq, bq, al, n)
    t, attr, pred = emulate(prog)
    ref, pr = X.int_attr(xq, bq, al, M, BW, S)
    assert pr == pred and np.array_equal(attr.astype(np.int64), ref), "schedule != integer reference"
    prog.checks = [("attr", prog.attr_off, attr.view(np.uint8).reshape(-1))]
    prog.info.update(method=method, samples=str(n), image=str(k), label=str(label), predicted=str(pred))
    print(emit(prog, f"ResNet-8 {method.upper()} on GeMMX (pretrainedResnet_quant, CIFAR-10)"))


if __name__ == "__main__":
    main()
