"""MLPerf Tiny keyword spotting (DS-CNN) and visual wake words (MobileNetV1)
inference + fused Grad-CAM on GeMMX, with the real reference int8 models.

Layout: padded C8HW8 activations (C/8, H+2, Wi+16, 8) int8, Wi = W rounded
up to 8 (GeMMX writes 8 pixels per mesh row), one border row above and
below, one 8-pixel block left and right. Borders, and the columns W..Wi-1,
hold -128 (the input zero point of every layer after the first).

Launches (all bit-exactly emulated by gemmx_sched.run_launch):
  conv  implicit im2col, one launch per 8 output channels (per-channel SIMD
        multipliers); stride 2 through the A spatial / temporal strides.
  dw    depthwise 3x3 as a conv whose K loop stays in input-channel tile nt,
        with a diagonal 8x8 weight block per tap (1/8 of the mesh does useful
        work: GeMMX has no depthwise mode).
  pw    pointwise 1x1 = a GeMM over the pixels.
  pool  ones-block GeMM over whole padded rows (borders are -128, removed
        through the bias), exact round(sum / count) in the requantiser.
  fc    1x1 GeMM + bias -> logits.
  gradcam  cam = (map - zp) @ alpha, alpha = the predicted class's FC row
        (Grad-CAM of a linear head), int32 out; ReLU + Q15 on the core.

When W is not a multiple of 8, a layer also writes columns W..Wi-1; one 2-D
DMA per channel tile resets them to -128 before the next layer reads them
as padding. VWW's 96x96x3 input (88 kB padded) does not fit next to the
first conv's output: that conv runs in two bands of 24 output rows whose
input rows the DMA core places from L3 (3-byte pixels into 8-byte slots).
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
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../resnet8_gemmx/data")))

import mobile_int as MI  # noqa: E402
from emit import c_bytes  # noqa: E402
from gemmx_sched import BLK, BLK32, Launch, Tcdm, emit_cfg_code, pack_wb, run_launch  # noqa: E402
from resnet8_int import same_pad  # noqa: E402

TCDM_LIMIT = 122 * 1024
NEG = -128
PBUF = 3072                  # largest streamed tile (KWS conv1: 40 K blocks) + 8x8 int32 bias


def ceil8(n: int) -> int:
    return (n + 7) // 8 * 8


@dataclass(frozen=True)
class PAct:
    """Padded activation (C/8, H+2, Wi+16, 8); off may be virtual (bands)."""

    off: int
    c: int
    h: int
    w: int

    @property
    def ct(self) -> int:
        return -(-self.c // 8)

    @property
    def wi(self) -> int:
        return ceil8(self.w)

    @property
    def wp(self) -> int:
        return self.wi + 16

    @property
    def plane(self) -> int:
        return (self.h + 2) * self.wp * 8

    @property
    def nbytes(self) -> int:
        return self.ct * self.plane

    def addr(self, ct: int, y: int, x: int) -> int:
        return self.off + ct * self.plane + (y + 1) * self.wp * 8 + (x + 8) * 8


def pack_pact(t: np.ndarray, P: PAct, fill: int = NEG) -> np.ndarray:
    buf = np.full((P.ct, P.h + 2, P.wp, 8), fill, dtype=np.int8)
    tt = np.full((P.h, P.w, P.ct * 8), fill, dtype=np.int8)
    tt[:, :, :t.shape[2]] = t
    buf[:, 1:1 + P.h, 8:8 + P.w, :] = tt.reshape(P.h, P.w, P.ct, 8).transpose(2, 0, 1, 3)
    return buf.reshape(-1)


def unpack_pact(b: np.ndarray, P: PAct) -> np.ndarray:
    buf = b.reshape(P.ct, P.h + 2, P.wp, 8)[:, 1:1 + P.h, 8:8 + P.w, :]
    return buf.transpose(1, 2, 0, 3).reshape(P.h, P.w, P.ct * 8)[:, :, :P.c]


# --------------------------------------------------------------------------
# Launch builders
# --------------------------------------------------------------------------
def conv_launch(name: str, X: PAct, Y: PAct, L: MI.Layer, nt: int, w_off: int, c_off: int,
                y0: int = 0, n: Optional[int] = None) -> Launch:
    """Output channels nt*8.. of conv / depthwise L, output rows y0..y0+n-1."""
    n = Y.h if n is None else n
    s = L.stride
    kh, kw = L.k
    _, pt, _ = same_pad(X.h, kh, s)
    _, pl, _ = same_pad(X.w, kw, s)
    dw = L.kind == "dw"
    cin_t = 1 if dw else X.ct
    K = kw * kh * cin_t
    mw = Y.wi // 8
    M = mw * n
    a_off = X.addr(nt if dw else 0, s * y0 - pt, -pl)
    a_tb = [kw, kh, cin_t, 1, mw, n]
    a_ts = [8, X.wp * 8, X.plane, 0, 8 * 8 * s, X.wp * 8 * s]
    Lc = Launch(name, M, K, 1, a_off, a_tb, a_ts, w_off, [K, 1, M], [BLK, 0, 0])
    Lc.a_sl = 8 * s
    Lc.c_en, Lc.c_off = 1, c_off
    Lc.c_tb, Lc.c_ts = [1, 1, M], [BLK32, 0, 0]
    Lc.bypass = 0
    Lc.d8_off = Y.addr(nt, y0, 0)
    Lc.d8_tb, Lc.d8_ts = [1, mw, n], [BLK, BLK, Y.wp * 8]
    cols = [min(c, L.cout - 1) for c in range(nt * 8, nt * 8 + 8)]
    Lc.mult8 = [int(L.m0[c]) for c in cols]
    Lc.shift8 = [int(L.sh[c]) for c in cols]
    Lc.zp_out, Lc.out_min, Lc.out_max = L.zp_y, -128, 127
    return Lc


def conv_weights(L: MI.Layer, nt: int, cin_t: int) -> np.ndarray:
    """Weight blocks of output tile nt in the A loop's K order (kw inner, kh,
    input tile outer) followed by the 8-row int32 bias tile, as bytes."""
    kh, kw = L.k
    blocks = []
    if L.kind == "dw":
        for i in range(kh):
            for j in range(kw):
                blk = np.zeros((8, 8), dtype=np.int8)                 # [col][k]
                for c in range(8):
                    if nt * 8 + c < L.cout:
                        blk[c, c] = L.W[nt * 8 + c, i, j]
                blocks.append(blk)
    else:
        W = np.zeros((L.cout, kh, kw, cin_t * 8), dtype=np.int8)
        W[:, :, :, :L.W.shape[3]] = L.W
        for ct in range(cin_t):
            for i in range(kh):
                for j in range(kw):
                    blk = np.zeros((8, 8), dtype=np.int8)
                    rows = W[nt * 8:nt * 8 + 8, i, j, ct * 8:ct * 8 + 8]
                    blk[:rows.shape[0]] = rows
                    blocks.append(blk)
    b = np.zeros(8, dtype=np.int32)
    seg = L.bias[nt * 8:nt * 8 + 8]
    b[:seg.size] = seg
    bias = np.tile(b, (8, 1))
    return np.concatenate([np.concatenate([x.reshape(-1) for x in blocks]).view(np.uint8),
                           bias.reshape(-1).view(np.uint8)])


def div_multiplier(count: int, vmin: int, vmax: int) -> Tuple[int, int]:
    """(m, sh) with floor(v * m / 2^sh) == floor(v / count) for every v in
    [vmin, vmax] (GeMMX SIMD: int32((v*m) >> (sh-1)) >> 1); checked exhaustively."""
    v = np.arange(vmin, vmax + 1, dtype=np.int64)
    want = np.floor_divide(v, count)
    for sh in range(8, 48):
        m = -(-(1 << sh) // count)
        if m >= 1 << 31:
            break
        if np.array_equal(((v * m) >> (sh - 1)) >> 1, want):
            return m, sh
    raise ValueError(f"no exact multiplier for /{count}")


def pool_launch(name: str, X: PAct, ones_off: int, c_off: int, out_off: int, m: int,
                sh: int) -> Launch:
    """pooled[c] = round(sum / count), summing whole padded rows (row 0..h-1,
    left border to right border are contiguous)."""
    K = X.h * X.wp // 8
    Lp = Launch(name, 1, K, X.ct, ones_off, [K, X.ct, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0],
                X.addr(0, 0, -8), [K, X.ct, 1], [BLK, X.plane, 0], trans_b=1)
    Lp.c_en, Lp.c_off = 1, c_off
    Lp.c_tb, Lp.c_ts = [1, X.ct, 1], [BLK32, 0, 0]
    Lp.bypass = 0
    Lp.d8_off = out_off
    Lp.d8_tb, Lp.d8_ts = [1, X.ct, 1], [BLK, BLK, 0]
    Lp.mult, Lp.shift, Lp.zp_out = m, sh, -128
    return Lp


def fc_launch(name: str, a_off: int, k_t: int, L: MI.Layer, w_off: int, c_off: int,
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
    """cam = sum_c (x - zp) * alpha_c over all Wi/8 x H pixel blocks; K =
    channel tiles, alpha in column 0 of the B blocks; int32 out."""
    mw = X.wi // 8
    M = mw * X.h
    Lg = Launch(name, M, X.ct, 1, X.addr(0, 0, 0), [X.ct, 1, mw, X.h, 1, 1],
                [X.plane, 0, BLK, X.wp * 8, 0, 0], alpha_off, [X.ct, 1, M], [BLK, 0, 0])
    Lg.c_en, Lg.c_off = 1, c_off
    Lg.c_tb, Lg.c_ts = [1, 1, M], [BLK32, 0, 0]
    Lg.bypass = 1
    Lg.d32_off = out_off
    Lg.d32_tb, Lg.d32_ts = [1, 1, M], [BLK32, BLK32, BLK32]
    return Lg


# --------------------------------------------------------------------------
# Program
# --------------------------------------------------------------------------
OP_FILL, OP_ROWS, OP_DMAS, OP_ALPHA, OP_CAM = 1, 2, 3, 4, 5
PHASES = ["input", "conv", "dw", "pw", "fill", "head", "gradcam"]


@dataclass
class Step:
    phase: str
    launch: Optional[Launch] = None
    pre: List[Tuple] = field(default_factory=list)    # ops before the launch
    post: List[Tuple] = field(default_factory=list)   # ops after the launch
    blob: int = -1


@dataclass
class Program:
    task: str
    image: np.ndarray
    steps: List[Step]
    x_l3: np.ndarray                 # quantised input (H, W, C) int8, GeMMX orientation
    fc_l3: np.ndarray                # (classes, C) FC weights (alpha source)
    blobs: List[np.ndarray]
    blob_dst: List[int]
    dmas: List[Tuple[int, int, int, int, int, int]]
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


def build(M: MI.Model, x: np.ndarray) -> Program:
    """Inference + fused Grad-CAM for one float model input."""
    xq = MI.quantize_input(x, M)
    ref = MI.forward(xq, M)
    H, W, C = M.in_shape
    acts = [PAct(0, C, H, W)]
    h, w = H, W
    for L in M.layers:
        h, w = -(-h // L.stride), -(-w // L.stride)
        acts.append(PAct(0, L.cout, h, w))
    slot = max(a.nbytes for a in acts[1:])
    banded = acts[0].nbytes > slot
    al = Alloc()
    S = (al.take(slot), al.take(slot))
    pb = (al.take(PBUF), al.take(PBUF))
    NEG_BYTES = max(a.wp * 8 + 64 + (a.wi - a.w) * 8 for a in acts)
    neg = al.take(NEG_BYTES)                             # -128 source for border DMAs
    params: List[Tuple[int, np.ndarray]] = [(neg, np.full(NEG_BYTES, NEG, dtype=np.int8).view(np.uint8))]

    def param(b: np.ndarray) -> int:
        o = al.take(b.nbytes)
        params.append((o, b.view(np.uint8)))
        return o

    steps: List[Step] = []
    blobs: List[np.ndarray] = []
    blob_dst: List[int] = []
    dmas: List[Tuple[int, int, int, int, int, int]] = []
    in_slot_geo = [None, None]           # geometry last written into each slot (None: all -128)

    def place(P: PAct, si: int) -> PAct:
        return PAct(S[si], P.c, P.h, P.w)

    def need_fill(si: int, P: PAct, pre: List[Tuple]) -> None:
        g = (P.c, P.h, P.w)
        if in_slot_geo[si] is not None and in_slot_geo[si] != g:
            pre.append((OP_FILL, P.off, P.nbytes))           # memset: cheaper than border DMAs
        in_slot_geo[si] = g

    def border_fill(P: PAct) -> Tuple:
        """-128 into the borders and columns W..Wi-1 only (the launch rewrites
        the interior): per channel tile, top row + first left border; right
        side of row y + left border of row y+1; last right side + bottom row."""
        d0 = len(dmas)
        g = (P.wi - P.w) * 8
        for ct in range(P.ct):
            dmas.append((P.addr(ct, -1, -8), neg, P.wp * 8 + 64, 0, 0, 1))
            if P.h > 1:
                dmas.append((P.addr(ct, 0, P.w), neg, g + 128, P.wp * 8, 0, P.h - 1))
            dmas.append((P.addr(ct, P.h - 1, P.w), neg, g + 64 + P.wp * 8, 0, 0, 1))
        return (OP_DMAS, d0, len(dmas))

    def garbage_fix(P: PAct) -> List[Tuple]:
        if P.w == P.wi:
            return []
        # one 2-D DMA over every row of every channel tile, border rows
        # included (they already hold -128), so the row stride is uniform
        d0 = len(dmas)
        dmas.append((P.addr(0, -1, P.w), neg, (P.wi - P.w) * 8, P.wp * 8, 0, P.ct * (P.h + 2)))
        return [(OP_DMAS, d0, len(dmas))]

    def add_conv(name: str, X: PAct, Y: PAct, L: MI.Layer, phase: str,
                 pre: List[Tuple], rows: Optional[Tuple[int, int]] = None,
                 post_last: Optional[List[Tuple]] = None) -> None:
        cin_t = 1 if L.kind == "dw" else X.ct
        for nt in range(Y.ct):
            blob = conv_weights(L, nt, cin_t)
            assert blob.size <= PBUF, (name, blob.size)
            buf = pb[len(blobs) % 2]
            blobs.append(blob)
            blob_dst.append(buf)
            K = L.k[0] * L.k[1] * cin_t
            y0, n = rows if rows else (0, Y.h)
            st = Step(phase, conv_launch(f"{name}n{nt}", X, Y, L, nt, buf, buf + K * BLK, y0, n),
                      pre if nt == 0 else [], blob=len(blobs) - 1)
            steps.append(st)
        if post_last:
            steps[-1].post += post_last

    # ---- input + first conv
    L0 = M.layers[0]
    Y0 = place(acts[1], 1)
    zp_in = M.zp_in
    if not banded:
        X0 = place(acts[0], 0)
        in_slot_geo[0] = (X0.c, X0.h, X0.w)
        rows_op = (OP_ROWS, X0.addr(0, 0, 0), 0, (C << 16) | W, H, X0.wp * 8)
        in_slot_geo[1] = (Y0.c, Y0.h, Y0.w)
        add_conv("l0", X0, Y0, L0, "conv", [rows_op], post_last=garbage_fix(Y0))
        init_fill = {S[0]: (X0, zp_in)}
    else:
        assert L0.stride == 2 and L0.k == (3, 3) and zp_in == NEG
        nb = 2
        band_out = Y0.h // nb
        assert band_out * nb == Y0.h
        in_rows = 2 * band_out + 1
        Xfull = acts[0]
        band_bytes = in_rows * Xfull.wp * 8
        assert Xfull.ct == 1 and band_bytes <= slot
        in_slot_geo[1] = (Y0.c, Y0.h, Y0.w)
        for b in range(nb):
            y0 = b * band_out
            r0 = 2 * y0                                    # first input row (pt = 0)
            Xv = PAct(S[0] - (r0 + 1) * Xfull.wp * 8, Xfull.c, Xfull.h, Xfull.w)
            real = min(in_rows, H - r0)                    # rows past H stay -128 (padding)
            pre = []
            if r0 + in_rows > H:                           # padding row(s) below the image
                d0 = len(dmas)
                for r in range(H, r0 + in_rows):
                    dmas.append((Xv.addr(0, r, -8), neg, Xfull.wp * 8, 0, 0, 1))
                pre.append((OP_DMAS, d0, len(dmas)))
            pre += [(OP_ROWS, Xv.addr(0, r0, 0), r0 * W * C, (C << 16) | W, real, Xfull.wp * 8)]
            add_conv(f"l0b{b}", Xv, Y0, L0, "conv", pre, rows=(y0, band_out))
        steps[-1].post += garbage_fix(Y0)
        in_slot_geo[0] = ("band",)
        init_fill = {}
    # ---- depthwise / pointwise chain
    cur, si = Y0, 1
    for i, L in enumerate(M.layers[1:], start=1):
        so = 1 - si
        Y = place(acts[i + 1], so)
        pre: List[Tuple] = []
        need_fill(so, Y, pre)
        phase = "dw" if L.kind == "dw" else "pw"
        add_conv(f"l{i}", cur, Y, L, phase, pre, post_last=garbage_fix(Y))
        cur, si = Y, so
    last = cur
    # ---- head in the free slot + parameter area
    free = S[1 - si]
    count = M.pool_count
    n_pad = last.h * (last.wp - last.w)
    # sum + c_pool = (sum_valid + 128 * count) + count // 2 >= 0, so the
    # floor division by count is exact; the zero point -128 undoes the offset
    c_pool = 128 * n_pad + 128 * count + count // 2
    pm, psh = div_multiplier(count, count // 2, 255 * count + count // 2)
    ones = param(np.ones(BLK, dtype=np.int8))
    pc = param(np.full(64, c_pool, dtype=np.int32))
    POOL = free
    fill_pre: List[Tuple] = []
    steps.append(Step("head", pool_launch("pool", last, ones, pc, POOL, pm, psh), fill_pre))
    fcL = M.fc
    n_t = -(-M.n_classes // 8)
    wfc = np.zeros((last.ct * 8, n_t * 8), dtype=np.int8)
    wfc[:last.c, :M.n_classes] = fcL.W[:, 0, 0, :].T
    fw = param(pack_wb(wfc))
    fb = np.zeros(n_t * 8, dtype=np.int32)
    fb[:M.n_classes] = fcL.bias
    fcb = param(np.tile(fb, (8, 1)).reshape(8, n_t, 8).transpose(1, 0, 2).reshape(-1))
    LOG = POOL + last.ct * BLK
    steps.append(Step("head", fc_launch("fc", POOL, last.ct, fcL, fw, fcb, LOG, n_t)))
    ALPHA = al.take(last.ct * BLK)                       # zero-initialised B blocks
    CAMC = al.take(BLK32)
    CAM32 = LOG + n_t * BLK
    mw = last.wi // 8
    CAMQ = CAM32 + mw * last.h * BLK32
    assert CAMQ + last.h * last.w * 4 <= free + slot
    zp = M.layers[-1].zp_y
    # alpha = FC row of the predicted class, copied 8 bytes per K tile from the
    # FC weights already in TCDM (layout Nt, Kt, col, k); the C tile is the
    # per-class constant -zp * sum(alpha), precomputed
    csum = param((-zp * fcL.W[:, 0, 0, :].astype(np.int64).sum(axis=1)).astype(np.int32))
    alpha_op = (OP_ALPHA, LOG, ALPHA, CAMC, fw, csum)
    cam_op = (OP_CAM, CAM32, CAMQ, last.h, last.w, mw)
    steps.append(Step("gradcam", gradcam_launch("gradcam", last, ALPHA, CAMC, CAM32),
                      [alpha_op], [cam_op]))

    # ---- TCDM image: slots -128 (KWS input slot: its zero point), parameters
    t = Tcdm()
    for o in S:
        t.write(o, np.full(slot, NEG, dtype=np.int8))
    for o, (P, fillv) in init_fill.items():
        t.write(o, np.full(P.nbytes, fillv, dtype=np.int8))
    for o, b in params:
        t.write(o, b)
    image = t.read(0, al.top, np.uint8)
    prog = Program(M.task, image, steps, xq, fcL.W[:, 0, 0, :].astype(np.int8), blobs,
                   blob_dst, dmas)
    t = emulate(prog)
    logits = read_logits(t, LOG, M.n_classes)
    assert np.array_equal(logits, ref["logits"]), "logits != mobile_int"
    assert np.array_equal(unpack_pact(t.read(last.off, last.nbytes), last),
                          ref[f"l{len(M.layers) - 1}"]), "last feature map mismatch"
    cam = t.read(CAMQ, last.h * last.w * 4, np.int32)
    assert np.array_equal(cam, gradcam_ref(ref, M)), "cam mismatch"
    prog.checks = [("logits", LOG, t.read(LOG, n_t * BLK, np.uint8)),
                   ("fmap", last.off, t.read(last.off, last.nbytes, np.uint8)),
                   ("cam", CAMQ, t.read(CAMQ, last.h * last.w * 4, np.uint8))]
    nl = sum(s.launch is not None for s in steps)
    prog.info.update(tcdm=f"{al.top} B", predicted=str(int(np.argmax(logits))), launches=str(nl),
                     banded=str(banded), map=f"{last.h}x{last.w}x{last.c}")
    return prog


def read_logits(t: Tcdm, off: int, n: int) -> np.ndarray:
    nt = -(-n // 8)
    return np.concatenate([t.read(off + i * BLK, 8) for i in range(nt)])[:n]


def gradcam_ref(ref: Dict[str, np.ndarray], M: MI.Model) -> np.ndarray:
    """Integer Grad-CAM of the predicted class on the last feature map: ReLU
    + Q15 normalisation, (h * w,) int32 in GeMMX orientation."""
    pred = int(np.argmax(ref["logits"]))
    alpha = M.fc.W[pred, 0, 0, :].astype(np.int64)
    fm = ref[f"l{len(M.layers) - 1}"]
    cam = (fm.astype(np.int64) - M.layers[-1].zp_y) @ alpha
    return cam_q15(np.maximum(cam.reshape(-1), 0))


def cam_q15(cam: np.ndarray) -> np.ndarray:
    mx = int(cam.max())
    if mx == 0:
        return np.zeros(cam.size, dtype=np.int32)
    inv = (1 << 31) // mx
    return ((cam.astype(np.uint64) * np.uint64(inv)) >> np.uint64(16)).astype(np.int32)


def _run_op(t: Tcdm, prog: Program, op: Tuple) -> None:
    kind = op[0]
    if kind == OP_FILL:
        t.write(op[1], np.full(op[2], NEG, dtype=np.int8))
    elif kind == OP_ROWS:
        _, dst, src, cw, nrows, dstride = op
        cin, npx = cw >> 16, cw & 0xFFFF
        flat = prog.x_l3.reshape(-1)
        for r in range(nrows):
            row = flat[src + r * cin * npx: src + (r + 1) * cin * npx].reshape(npx, cin)
            for p in range(npx):
                t.write(dst + r * dstride + p * 8, row[p])
    elif kind == OP_DMAS:
        for d in range(op[1], op[2]):
            dst, src, size, ds, ss, rep = prog.dmas[d]
            for r in range(rep):
                t.write(dst + r * ds, t.read(src + r * ss, size))
    elif kind == OP_ALPHA:
        _, log_off, a_off, c_off, fw, csum = op
        n, kt = prog.fc_l3.shape[0], -(-prog.fc_l3.shape[1] // 8)
        pred = int(np.argmax(read_logits(t, log_off, n)))
        for k in range(kt):
            t.write(a_off + k * BLK, t.read(fw + ((pred // 8) * kt + k) * BLK + (pred % 8) * 8, 8))
        t.write(c_off, np.full(64, t.read(csum + 4 * pred, 4, np.int32)[0], dtype=np.int32))
    elif kind == OP_CAM:
        _, c32, cq, h, w, mw = op
        d = t.read(c32, mw * h * BLK32, np.int32).reshape(h, mw, 8, 8)[:, :, :, 0]
        cam = d.reshape(h, mw * 8)[:, :w].reshape(-1).astype(np.int64)
        t.write(cq, cam_q15(np.maximum(cam, 0)))


def emulate(prog: Program) -> Tcdm:
    t = Tcdm()
    t.write(0, prog.image)
    for s in prog.steps:
        for op in s.pre:
            _run_op(t, prog, op)
        if s.blob >= 0:
            t.write(prog.blob_dst[s.blob], prog.blobs[s.blob])
        if s.launch is not None:
            run_launch(t, s.launch)
        for op in s.post:
            _run_op(t, prog, op)
    return t


# --------------------------------------------------------------------------
# C emitter
# --------------------------------------------------------------------------
def _prefetch_plan(prog: Program) -> Tuple[int, List[int]]:
    """While step i runs, load the next blob into the buffer step i does not
    read; replayed to check every launch finds its blob."""
    blob_steps = [i for i, s in enumerate(prog.steps) if s.blob >= 0]
    first = prog.steps[blob_steps[0]].blob
    loaded, plan = first, []
    for i, s in enumerate(prog.steps):
        nxt = [prog.steps[j].blob for j in blob_steps if j > i]
        pre = -1
        if nxt and nxt[0] != loaded and not (s.blob >= 0 and prog.blob_dst[s.blob] == prog.blob_dst[nxt[0]]):
            pre = loaded = nxt[0]
        plan.append(pre)
    content = {prog.blob_dst[first]: first}
    for i, s in enumerate(prog.steps):
        if s.blob >= 0:
            assert content.get(prog.blob_dst[s.blob]) == s.blob, f"step {i}: blob missing"
        if plan[i] >= 0:
            content[prog.blob_dst[plan[i]]] = plan[i]
    return first, plan


def emit(prog: Program, title: str) -> str:
    launches = [s.launch for s in prog.steps if s.launch is not None]
    out = ["#include <stdint.h>", "", f"// {title}"] + [f"// {k}: {v}" for k, v in prog.info.items()]
    out.append(c_bytes("prog_image", prog.image))
    out.append(f"#define PROG_IMAGE_BYTES {prog.image.size}")
    out.append(c_bytes("prog_in", prog.x_l3.reshape(-1)))
    out.append(f"#define PROG_NCLS {prog.fc_l3.shape[0]}")
    out.append(f"#define PROG_KT {-(-prog.fc_l3.shape[1] // 8)}")
    out.append(emit_cfg_code(launches, True, "cfg"))
    out.append(c_bytes("prog_params", np.concatenate(prog.blobs)))
    offs = np.cumsum([0] + [b.size for b in prog.blobs])[:-1]
    out.append("static const int32_t prog_blob_off[] = {" + ", ".join(map(str, offs)) + "};")
    out.append("static const int32_t prog_blob_len[] = {" + ", ".join(str(b.size) for b in prog.blobs) + "};")
    out.append("static const int32_t prog_blob_dst[] = {" + ", ".join(map(str, prog.blob_dst)) + "};")
    first, plan = _prefetch_plan(prog)
    out.append(f"#define PROG_FIRST_BLOB {first}")
    ops, rows, li = [], [], 0
    for i, s in enumerate(prog.steps):
        p0 = len(ops)
        ops += s.pre
        p1 = len(ops)
        ops += s.post
        rows.append(f"  {{{li if s.launch is not None else -1}, {PHASES.index(s.phase)}, "
                    f"{plan[i]}, {p0}, {p1}, {len(ops)}}}")
        li += s.launch is not None
    out.append("typedef struct { int16_t launch, phase, prefetch, pre0, pre1, post1; } mstep_t;")
    out.append("typedef struct { int32_t type, a, b, c, d, e; } op_t;")
    out.append("typedef struct { int32_t dst, src, size, dst_stride, src_stride, repeat; } dma2d_t;")
    out.append(f"#define PROG_N_STEPS {len(prog.steps)}")
    out.append("static const mstep_t prog_steps[] = {\n" + ",\n".join(rows) + "\n};")
    op_rows = [f"  {{{', '.join(str(int(v)) for v in (list(o) + [0] * (6 - len(o))))}}}" for o in ops]
    out.append("static const op_t prog_ops[] = {\n" + ",\n".join(op_rows or ["  {0}"]) + "\n};")
    d_rows = [f"  {{{', '.join(map(str, d))}}}" for d in prog.dmas]
    out.append("static const dma2d_t prog_dmas[] = {\n" + ",\n".join(d_rows or ["  {0}"]) + "\n};")
    out.append(f"#define N_PHASES {len(PHASES)}")
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


if __name__ == "__main__":
    task = os.environ.get("MB_TASK", "kws")
    k = int(os.environ.get("MB_SAMPLE", "0"))
    M = MI.load_model(task)
    x, y = MI.load_inputs(task)
    prog = build(M, x[k])
    prog.info["label"] = str(int(y[k]))
    names = {"kws": "keyword spotting (DS-CNN kws_ref_model, Speech Commands v2)",
             "vww": "visual wake words (MobileNetV1 vww_96_int8, COCO)"}
    print(emit(prog, f"MLPerf Tiny {names[task]} inference + Grad-CAM on GeMMX"))
