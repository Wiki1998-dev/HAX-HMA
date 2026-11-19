"""MLPerf Tiny ResNet-8 inference + fused Grad-CAM on GeMMX (real
pretrainedResnet_quant.tflite weights, real CIFAR-10 images).

Activations live in padded C8HW8 layouts: (C/8, H+2, W+16, 8) int8, one
zero-point row above / below and an 8-pixel (one block) border left / right,
so every interior block is 64-byte aligned (the D8 writer and B reader reach
bank i only from channel i mod 8) while convolution taps read the border.
Borders hold the consumer's input zero point (-128 for every conv input).

Launches (all bit-exactly emulated by gemmx_sched.run_launch):
  conv     implicit im2col, one launch per 8 output channels (per-channel
           SIMD multipliers / shifts), output written into the next padded
           buffer's interior; stride 2 via the A spatial / temporal strides
  add      block-diagonal [x1 | x2] @ [a I; b I] + c, in place over x2
  pool     ones-block GeMM over the 8x8 map (B transposer), floor((s+32)/64)
  fc       1x1 GeMM + bias, logits
  gradcam  cam = add3 @ alpha (alpha = FC row of the predicted class, placed
           at run time by a 2-D DMA), int32 out; ReLU + Q15 on the core
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

import resnet8_int as R  # noqa: E402
from emit import c_bytes  # noqa: E402
from gemmx_sched import (BLK, BLK32, Launch, Tcdm, emit_cfg_code,  # noqa: E402
                         pack_act, run_launch)

TCDM_LIMIT = 112 * 1024
NEG = -128


@dataclass(frozen=True)
class PAct:
    """Padded C8HW8 activation (C/8, H+2, W+16, 8)."""

    off: int
    c: int
    h: int
    w: int

    @property
    def ct(self) -> int:
        return self.c // 8

    @property
    def hp(self) -> int:
        return self.h + 2

    @property
    def wp(self) -> int:
        return self.w + 16

    @property
    def plane(self) -> int:
        return self.hp * self.wp * 8

    @property
    def nbytes(self) -> int:
        return self.ct * self.plane

    def addr(self, ct: int, y: int, x: int) -> int:
        """Address of pixel (y, x) of channel tile ct; y in [-1, h], x in [-8, w+7]."""
        return self.off + ct * self.plane + (y + 1) * self.wp * 8 + (x + 8) * 8


def pack_pact(t: np.ndarray, P: PAct, fill: int = NEG) -> np.ndarray:
    """(H, W, C) int8 -> padded layout bytes (border = fill)."""
    buf = np.full((P.ct, P.hp, P.wp, 8), fill, dtype=np.int8)
    c = t.shape[2]
    tt = np.zeros((P.h, P.w, P.c), dtype=np.int8)
    tt[:, :, :c] = t
    buf[:, 1:1 + P.h, 8:8 + P.w, :] = tt.reshape(P.h, P.w, P.ct, 8).transpose(2, 0, 1, 3)
    return buf.reshape(-1)


def unpack_pact(b: np.ndarray, P: PAct) -> np.ndarray:
    buf = b.reshape(P.ct, P.hp, P.wp, 8)[:, 1:1 + P.h, 8:8 + P.w, :]
    return buf.transpose(1, 2, 0, 3).reshape(P.h, P.w, P.c)


# --------------------------------------------------------------------------
# Launch builders
# --------------------------------------------------------------------------
def conv_launch(name: str, X: PAct, Y: PAct, L: R.Conv, nt: int, w_off: int,
                c_off: int) -> Launch:
    """Output channels nt*8 .. nt*8+7 of conv L: X -> interior of Y."""
    s = L.stride
    kh, kw = L.W.shape[1:3]
    _, pt, _ = R.same_pad(X.h, kh, s)
    _, pl, _ = R.same_pad(X.w, kw, s)
    ho, wo = Y.h, Y.w
    assert wo % 8 == 0
    K = kw * kh * X.ct
    M = (wo // 8) * ho
    a_off = X.addr(0, -pt, -pl)
    a_tb = [kw, kh, X.ct, 1, wo // 8, ho]
    a_ts = [8, X.wp * 8, X.plane, 0, 8 * 8 * s, X.wp * 8 * s]
    Lc = Launch(name, M, K, 1, a_off, a_tb, a_ts, w_off, [K, 1, M], [BLK, 0, 0])
    Lc.a_sl = 8 * s
    Lc.c_en, Lc.c_off = 1, c_off
    Lc.c_tb, Lc.c_ts = [1, 1, M], [BLK32, 0, 0]
    Lc.bypass = 0
    Lc.d8_off = Y.addr(nt, 0, 0)
    Lc.d8_tb, Lc.d8_ts = [1, wo // 8, ho], [BLK, BLK, Y.wp * 8]
    cols = range(nt * 8, nt * 8 + 8)
    Lc.mult8 = [int(L.m0[c]) for c in cols]
    Lc.shift8 = [int(L.sh[c]) for c in cols]
    Lc.zp_out, Lc.out_min, Lc.out_max = L.zp_y, -128, 127
    return Lc


def conv_weights(L: R.Conv, nt: int, cin_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """B blocks for output tile nt in the A loop's K order (kw inner, kh,
    cin tile outer), and the 8-row bias C tile."""
    kh, kw = L.W.shape[1:3]
    ci = L.W.shape[3]
    W = np.zeros((L.W.shape[0], kh, kw, cin_t * 8), dtype=np.int8)
    W[:, :, :, :ci] = L.W
    blocks = []
    for ct in range(cin_t):
        for i in range(kh):
            for j in range(kw):
                blocks.append(W[nt * 8:nt * 8 + 8, i, j, ct * 8:ct * 8 + 8])   # [col][k]
    bias = np.tile(L.bias[nt * 8:nt * 8 + 8].astype(np.int32), (8, 1))
    return np.concatenate([b.reshape(-1) for b in blocks]), bias.reshape(-1)


def add_launch(name: str, X1: PAct, X2: PAct, out: PAct, A: R.Add, b_off: int,
               c_off: int) -> Launch:
    """out = requant(a*x1 + b*x2 + c), elementwise over the interior."""
    assert X2.off > X1.off and (X1.c, X1.h, X1.w) == (X2.c, X2.h, X2.w)
    N, M_w, M_h = X1.ct, X1.w // 8, X1.h
    a_off = X1.addr(0, 0, 0)
    La = Launch(name, M_w * M_h, 2, N, a_off, [2, N, M_w, M_h, 1, 1],
                [X2.off - X1.off, X1.plane, BLK, X1.wp * 8, 0, 0], b_off, [2, N, M_w * M_h],
                [BLK, 0, 0])
    La.c_en, La.c_off = 1, c_off
    La.c_tb, La.c_ts = [1, N, M_w * M_h], [BLK32, 0, 0]
    La.bypass = 0
    La.d8_off = out.addr(0, 0, 0)
    La.d8_tb, La.d8_ts = [N, M_w, M_h], [out.plane, BLK, out.wp * 8]
    La.mult, La.shift, La.zp_out = A.m0, A.sh, A.zp_y
    return La


def pool_launch(name: str, X: PAct, ones_off: int, c_off: int, out_off: int) -> Launch:
    """pooled[c] = floor((sum_{8x8} x + 32) / 64), all 8 output rows equal."""
    assert X.h == 8 and X.w == 8
    Lp = Launch(name, 1, 8, X.ct, ones_off, [8, X.ct, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0],
                X.addr(0, 0, 0), [8, X.ct, 1], [X.wp * 8, X.plane, 0], trans_b=1)
    Lp.c_en, Lp.c_off = 1, c_off
    Lp.c_tb, Lp.c_ts = [1, X.ct, 1], [BLK32, 0, 0]
    Lp.bypass = 0
    Lp.d8_off = out_off
    Lp.d8_tb, Lp.d8_ts = [1, X.ct, 1], [BLK, BLK, 0]
    Lp.mult, Lp.shift, Lp.zp_out = 1, 6, 0
    return Lp


def fc_launch(name: str, a_off: int, k_t: int, L: R.Conv, w_off: int, c_off: int,
              out_off: int, n_t: int) -> Launch:
    Lf = Launch(name, 1, k_t, n_t, a_off, [k_t, n_t, 1, 1, 1, 1], [BLK, 0, 0, 0, 0, 0],
                w_off, [k_t, n_t, 1], [BLK, k_t * BLK, 0])
    Lf.c_en, Lf.c_off = 1, c_off
    Lf.c_tb, Lf.c_ts = [1, n_t, 1], [BLK32, BLK32, 0]
    Lf.bypass = 0
    Lf.d8_off = out_off
    Lf.d8_tb, Lf.d8_ts = [1, n_t, 1], [BLK, BLK, 0]
    Lf.mult, Lf.shift, Lf.zp_out = int(L.m0[0]), int(L.sh[0]), L.zp_y
    return Lf


def gradcam_launch(name: str, X: PAct, alpha_off: int, c_off: int, out_off: int) -> Launch:
    """cam[h, w] = sum_c (x[h, w, c] - zp) * alpha[c]; rows = 8 pixels of a
    map row, K = channel tiles, one N tile (alpha in column 0), int32 out."""
    Lg = Launch(name, X.h, X.ct, 1, X.addr(0, 0, 0), [X.ct, 1, X.h, 1, 1, 1],
                [X.plane, 0, X.wp * 8, 0, 0, 0], alpha_off, [X.ct, 1, X.h], [BLK, 0, 0])
    Lg.c_en, Lg.c_off = 1, c_off
    Lg.c_tb, Lg.c_ts = [1, 1, X.h], [BLK32, 0, 0]
    Lg.bypass = 1
    Lg.d32_off = out_off
    Lg.d32_tb, Lg.d32_ts = [1, 1, X.h], [BLK32, BLK32, BLK32]
    return Lg


# --------------------------------------------------------------------------
# Program
# --------------------------------------------------------------------------
@dataclass
class Step:
    phase: str
    launch: Optional[Launch] = None
    post: Optional[Tuple] = None     # ("fill", off, n) | ("alpha", ...) | ("cam", ...)
    blob: int = -1                   # streamed parameter blob this launch reads


@dataclass
class Program:
    image: np.ndarray
    steps: List[Step]
    img_l3: np.ndarray               # (32, 32, 3) int8 input
    in_act: PAct
    fc_l3: np.ndarray                # (10, 64) int8 FC weights (alpha source)
    alpha_off: int
    logits_off: int
    cam_off: int
    cam_int_off: int
    blobs: List[np.ndarray] = field(default_factory=list)   # conv weights + bias, in L3
    blob_dst: List[int] = field(default_factory=list)
    checks: List[Tuple[str, int, np.ndarray]] = field(default_factory=list)
    info: Dict[str, str] = field(default_factory=dict)


class Alloc:
    def __init__(self) -> None:
        self.top = 0

    def take(self, n: int) -> int:
        off = self.top
        self.top = (self.top + n + 63) // 64 * 64
        assert self.top <= TCDM_LIMIT, f"TCDM overflow {self.top}"
        return off


def build(M: R.Model, img: np.ndarray) -> Program:
    """Inference + fused Grad-CAM for one image (32, 32, 3) uint8."""
    xq = R.quantize_image(img, M)
    ref = R.forward(xq, M)
    al = Alloc()
    slot = 26 * 1024
    S_B, S_C, S_A = al.take(slot), al.take(slot), al.take(slot)    # x2 slots above x1 slots
    PBUF = 5 * 1024                                            # largest conv tile: 72*64 + 256 B
    pb = (al.take(PBUF), al.take(PBUF))
    params: List[np.ndarray] = []
    blobs: List[np.ndarray] = []
    blob_dst: List[int] = []

    def param(b: np.ndarray) -> int:
        o = al.take(b.nbytes)                                  # bytes, not elements
        params.append((o, b.view(np.uint8)))
        return o

    steps: List[Step] = []
    conv = M.convs

    def do_conv(name: str, X: PAct, Y: PAct, L: R.Conv) -> None:
        # weights + bias of each 8-channel tile stream from L3 (double-buffered)
        for nt in range(Y.ct):
            wb, bias = conv_weights(L, nt, X.ct)
            buf = pb[len(blobs) % 2]
            blob = np.concatenate([wb.view(np.uint8), bias.view(np.uint8)])
            assert blob.size <= PBUF
            blobs.append(blob)
            blob_dst.append(buf)
            steps.append(Step("conv", conv_launch(f"{name}n{nt}", X, Y, L, nt, buf, buf + wb.size),
                              blob=len(blobs) - 1))

    def fill(P: PAct) -> None:
        steps.append(Step("fill", None, ("fill", P.off, P.nbytes)))

    eye = np.eye(8, dtype=np.int64)
    X0 = PAct(S_A, 8, 32, 32)
    C1 = PAct(S_B, 16, 32, 32)
    do_conv("conv1", X0, C1, conv["conv1"])
    B1 = PAct(S_C, 16, 32, 32)
    do_conv("b1c1", C1, B1, conv["b1c1"])
    B2 = PAct(S_A, 16, 32, 32)
    do_conv("b1c2", B1, B2, conv["b1c2"])

    def do_add(name: str, X1: PAct, X2: PAct, A: R.Add) -> None:
        bp = param(np.concatenate([(A.a * eye).astype(np.int8).reshape(-1),
                                   (A.b * eye).astype(np.int8).reshape(-1)]))
        ct = param(np.full(64, A.c, dtype=np.int32))
        steps.append(Step("add", add_launch(name, X1, X2, X2, A, bp, ct)))

    do_add("add1", C1, B2, M.adds[0])                          # add1 in S_A
    A1 = B2
    fill(PAct(S_B, 32, 16, 16))
    P21 = PAct(S_B, 32, 16, 16)
    do_conv("b2c1", A1, P21, conv["b2c1"])
    P2S = PAct(S_C, 32, 16, 16)
    do_conv("b2sc", A1, P2S, conv["b2sc"])
    fill(PAct(S_A, 32, 16, 16))
    P22 = PAct(S_A, 32, 16, 16)
    do_conv("b2c2", P21, P22, conv["b2c2"])
    do_add("add2", P2S, P22, M.adds[1])                        # add2 in S_A
    A2 = P22
    fill(PAct(S_B, 64, 8, 8))
    P31 = PAct(S_B, 64, 8, 8)
    do_conv("b3c1", A2, P31, conv["b3c1"])
    P3S = PAct(S_C, 64, 8, 8)
    do_conv("b3sc", A2, P3S, conv["b3sc"])
    P32 = PAct(S_A, 64, 8, 8)
    do_conv("b3c2", P31, P32, conv["b3c2"])
    do_add("add3", P3S, P32, M.adds[2])                        # add3 in S_A
    A3 = P32
    # pool + fc
    ones = param(np.ones(64, dtype=np.int8))
    pc = param(np.full(64, 32, dtype=np.int32))
    POOL = al.take(8 * 64)
    steps.append(Step("head", pool_launch("pool", A3, ones, pc, POOL)))
    fcL = M.fc
    wfc = np.zeros((64, 16), dtype=np.int8)
    wfc[:, :10] = fcL.W[:, 0, 0, :].T
    from gemmx_sched import pack_wb
    fw = param(pack_wb(wfc))
    fb = np.zeros(16, dtype=np.int32)
    fb[:10] = fcL.bias
    fcb = param(np.tile(fb, (8, 1)).reshape(8, 2, 8).transpose(1, 0, 2).reshape(-1))
    LOG = al.take(8 * 16)
    steps.append(Step("head", fc_launch("fc", POOL, 8, fcL, fw, fcb, LOG, 2)))
    # fused Grad-CAM: alpha = FC row of the predicted class (placed at run time)
    ALPHA = al.take(8 * 64)                                    # B blocks (n=0, kt=0..7), col 0
    zp3 = M.adds[2].zp_y
    CAMC = al.take(256)
    CAM32 = al.take(8 * 256)
    CAMQ = al.take(64 * 4)
    steps.append(Step("gradcam", None, ("alpha", LOG, ALPHA, CAMC, zp3)))
    steps.append(Step("gradcam", gradcam_launch("gradcam", A3, ALPHA, CAMC, CAM32),
                      ("cam", CAM32, CAMQ)))

    # TCDM image: slots filled with -128 (borders), parameters
    t = Tcdm()
    for o in (S_B, S_C, S_A):
        t.write(o, np.full(slot, NEG, dtype=np.int8))
    for o, b in params:
        t.write(o, b)
    image = t.read(0, al.top, np.uint8)
    prog = Program(image, steps, xq, X0, fcL.W[:, 0, 0, :].astype(np.int8), ALPHA, LOG,
                   CAMQ, CAM32, blobs, blob_dst)
    t = emulate(prog)
    # cross-check against the NumPy model
    logits = read_logits(t, LOG)
    if not os.environ.get("R8_NOCHECK"):                       # debug: skip the cross-check
        assert np.array_equal(logits, ref["logits"][:10]), "logits != resnet8_int"
        assert np.array_equal(unpack_pact(t.read(A3.off, A3.nbytes), A3), ref["add3"]), "add3 mismatch"
        assert np.array_equal(t.read(CAMQ, 256, np.int32), gradcam_ref(ref, M)), "cam mismatch"
    prog.checks = [("logits", LOG, t.read(LOG, 2 * BLK, np.uint8)),
                   ("add3", A3.off, t.read(A3.off, A3.nbytes, np.uint8)),
                   ("cam", CAMQ, t.read(CAMQ, 256, np.uint8))]
    prog.info.update(tcdm=f"{al.top} B", predicted=str(int(np.argmax(logits))),
                     launches=str(sum(s.launch is not None for s in steps)))
    return prog


def read_logits(t: Tcdm, off: int) -> np.ndarray:
    """The 10 logits: row 0 of the two 8x8 output blocks of the FC launch."""
    return np.concatenate([t.read(off, 8), t.read(off + BLK, 8)])[:10]


def gradcam_ref(ref: Dict[str, np.ndarray], M: R.Model) -> np.ndarray:
    """Integer Grad-CAM of the predicted class: ReLU + Q15 normalisation."""
    pred = int(np.argmax(ref["logits"][:10]))
    alpha = M.fc.W[pred, 0, 0, :].astype(np.int64)
    zp = M.adds[2].zp_y
    cam = (ref["add3"].astype(np.int64) - zp) @ alpha                # (8, 8)
    cam = np.maximum(cam.reshape(-1), 0)
    mx = int(cam.max())
    if mx == 0:
        return np.zeros(64, dtype=np.int32)
    inv = (1 << 31) // mx
    return ((cam.astype(np.uint64) * np.uint64(inv)) >> np.uint64(16)).astype(np.int32)


def emulate(prog: Program) -> Tcdm:
    t = Tcdm()
    t.write(0, prog.image)
    t.write(prog.in_act.off, pack_pact(prog.img_l3, prog.in_act))
    for s in prog.steps:
        if s.blob >= 0:
            t.write(prog.blob_dst[s.blob], prog.blobs[s.blob])
        if s.post and s.post[0] == "fill":
            t.write(s.post[1], np.full(s.post[2], NEG, dtype=np.int8))
        if s.post and s.post[0] == "alpha":
            _, log_off, a_off, c_off, zp = s.post
            pred = int(np.argmax(read_logits(t, log_off)))
            alpha = prog.fc_l3[pred]
            blk = np.zeros((8, 8, 8), dtype=np.int8)                 # kt, col, k
            blk[:, 0, :] = alpha.reshape(8, 8)
            t.write(a_off, blk.reshape(-1))
            t.write(c_off, np.full(64, -zp * int(alpha.astype(np.int64).sum()), dtype=np.int32))
        if s.launch is not None:
            run_launch(t, s.launch)
        if s.post and s.post[0] == "cam":
            _, c32, cq = s.post
            d = t.read(c32, 8 * 256, np.int32).reshape(8, 8, 8)[:, :, 0]   # row h, pixel w
            cam = np.maximum(d.reshape(-1).astype(np.int64), 0)
            mx = int(cam.max())
            q = (np.zeros(64, np.int32) if mx == 0 else
                 ((cam.astype(np.uint64) * np.uint64((1 << 31) // mx)) >> np.uint64(16)).astype(np.int32))
            t.write(cq, q)
    return t


PHASES = ["conv", "add", "fill", "head", "gradcam"]


def emit(prog: Program, title: str) -> str:
    launches = [s.launch for s in prog.steps if s.launch is not None]
    out = ["#include <stdint.h>", "", f"// {title}"] + [f"// {k}: {v}" for k, v in prog.info.items()]
    out.append(c_bytes("prog_image", prog.image))
    out.append(f"#define PROG_IMAGE_BYTES {prog.image.size}")
    out.append(c_bytes("prog_img", prog.img_l3.reshape(-1)))
    out.append(f"#define PROG_IN_OFF {prog.in_act.off}")
    out.append(f"#define PROG_IN_WP {prog.in_act.wp}")
    out.append(c_bytes("prog_fc", prog.fc_l3.reshape(-1)))
    out.append(emit_cfg_code(launches, True, "cfg"))
    out.append(c_bytes("prog_params", np.concatenate(prog.blobs)))
    offs = np.cumsum([0] + [b.size for b in prog.blobs])[:-1]
    out.append("static const int32_t prog_blob_off[] = {" + ", ".join(map(str, offs)) + "};")
    out.append("static const int32_t prog_blob_len[] = {" + ", ".join(str(b.size) for b in prog.blobs) + "};")
    out.append("static const int32_t prog_blob_dst[] = {" + ", ".join(map(str, prog.blob_dst)) + "};")
    # prefetch plan: while step i runs, load the next blob not yet loaded,
    # unless it goes to the buffer step i reads (blobs alternate buffers)
    blob_steps = [i for i, s in enumerate(prog.steps) if s.blob >= 0]
    loaded = prog.steps[blob_steps[0]].blob
    pre_plan = []
    for i, s in enumerate(prog.steps):
        nxt = [prog.steps[j].blob for j in blob_steps if j > i]
        pre = -1
        if nxt and nxt[0] != loaded and not (s.blob >= 0 and prog.blob_dst[s.blob] == prog.blob_dst[nxt[0]]):
            pre = loaded = nxt[0]
        pre_plan.append(pre)
    content = {prog.blob_dst[prog.steps[blob_steps[0]].blob]: prog.steps[blob_steps[0]].blob}
    for i, s in enumerate(prog.steps):                         # replay: every blob present
        if s.blob >= 0:
            assert content.get(prog.blob_dst[s.blob]) == s.blob, f"step {i}: blob missing"
        if pre_plan[i] >= 0:
            content[prog.blob_dst[pre_plan[i]]] = pre_plan[i]
    out.append(f"#define PROG_FIRST_BLOB {prog.steps[blob_steps[0]].blob}")
    rows, li = [], 0
    for i, s in enumerate(prog.steps):
        post, a, b, c, d = 0, 0, 0, 0, 0
        if s.post and s.post[0] == "fill":
            post, a, b = 1, s.post[1], s.post[2]
        elif s.post and s.post[0] == "alpha":
            post, a, b, c, d = 2, s.post[1], s.post[2], s.post[3], s.post[4]
        elif s.post and s.post[0] == "cam":
            post, a, b = 3, s.post[1], s.post[2]
        rows.append(f"  {{{li if s.launch is not None else -1}, {PHASES.index(s.phase)}, {post}, {pre_plan[i]}, {a}, {b}, {c}, {d}}}")
        li += s.launch is not None
    out.append("typedef struct { int16_t launch, phase, post, prefetch; int32_t a, b, c, d; } rstep_t;")
    out.append(f"#define PROG_N_STEPS {len(prog.steps)}")
    out.append("static const rstep_t prog_steps[] = {\n" + ",\n".join(rows) + "\n};")
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


def load_image(k: int = 0) -> Tuple[np.ndarray, int]:
    """CIFAR-10 test image of the k-th MLPerf performance sample."""
    import pickle
    root = os.path.normpath(os.path.join(HERE, "../../../../../external"))
    d = pickle.load(open(os.path.join(root, "cifar-10-batches-py/test_batch"), "rb"), encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    idx = np.load(os.path.join(root, "tiny/benchmark/training/image_classification/perf_samples_idxs.npy"))
    return X[idx[k]], int(np.array(d[b"labels"])[idx[k]])


if __name__ == "__main__":
    M = R.load_model()
    img, label = load_image(int(os.environ.get("R8_SAMPLE", "0")))
    prog = build(M, img)
    prog.info["label"] = str(label)
    print(emit(prog, "ResNet-8 inference + Grad-CAM on GeMMX (pretrainedResnet_quant, CIFAR-10)"))
