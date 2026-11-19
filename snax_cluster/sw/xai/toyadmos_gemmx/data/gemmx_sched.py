"""GeMMX launch schedules: streamer address generation, a bit-exact emulator
of the snax_KUL_cluster GeMMX (8x8x8 int8 mesh, int32 accumulate, SIMD
requantiser) on a byte-array model of the TCDM, and a C emitter for the
generic launch-table executor (src/gemmx_exec.h).

Tensor conventions (all offsets are TCDM byte offsets):

* ``Act``: a matrix [R x C] (R, C multiples of 8) stored as 8x8 int8 blocks,
  block (r, c) at off + (r * C/8 + c) * 64, row-major inside the block.
  This is both GeMMX's A layout (rows x k) and its D8 output layout
  (rows x cols), so a layer's int8 output feeds the next layer directly.
* ``Acc``: the same with int32 elements (256 B blocks): C / D32 layout.
* ``WB``: a weight matrix W[K x N] in GeMMX's B layout, block (n, k) at
  off + (n * K/8 + k) * 64, holding [col][k] (k innermost).

Streamer semantics (snax_KUL_cluster): A has 6 temporal loops, B 3, D8 3,
C/D32 3 (innermost first); A/B/D8 read/write 8 channels x 8 B per step
(spatial stride 8), C/D32 32 channels x 8 B (spatial strides 8 and 64).
GeMMX consumes A and B blocks in lockstep, K blocks per output block, and
emits output blocks in the order the A/B loops visit (m, n) pairs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../../../util/sim/"))
from snax_utils import postprocessing_simd_golden_model  # noqa: E402

BLK = 64        # bytes in an int8 8x8 block
BLK32 = 256     # bytes in an int32 8x8 block


# --------------------------------------------------------------------------
# Tensors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Act:
    """int8 [rows x cols] matrix in 8x8 block layout."""

    off: int
    rows: int
    cols: int

    @property
    def rt(self) -> int:
        return self.rows // 8

    @property
    def ct(self) -> int:
        return self.cols // 8

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols

    def blk(self, r: int, c: int) -> int:
        return self.off + (r * self.ct + c) * BLK


@dataclass(frozen=True)
class Acc:
    """int32 [rows x cols] matrix in 8x8 block layout."""

    off: int
    rows: int
    cols: int

    @property
    def ct(self) -> int:
        return self.cols // 8

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * 4


@dataclass(frozen=True)
class WB:
    """Weight matrix W[K x N] in GeMMX B layout."""

    off: int
    k: int
    n: int

    @property
    def kt(self) -> int:
        return self.k // 8

    @property
    def nbytes(self) -> int:
        return self.k * self.n


def pack_act(m: np.ndarray) -> np.ndarray:
    """Dense [R x C] -> Act byte layout (R, C multiples of 8)."""
    r, c = m.shape
    return m.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3).reshape(-1)


def unpack_act(b: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return b.reshape(rows // 8, cols // 8, 8, 8).transpose(0, 2, 1, 3).reshape(rows, cols)


def pack_wb(w: np.ndarray) -> np.ndarray:
    """Dense W[K x N] -> B layout (Nt, Kt, col, k)."""
    k, n = w.shape
    return w.reshape(k // 8, 8, n // 8, 8).transpose(2, 0, 3, 1).reshape(-1)


# --------------------------------------------------------------------------
# Launch description
# --------------------------------------------------------------------------
@dataclass
class Launch:
    """All CSR values for one GeMMX launch (mirrors gemmx_launch_t in C)."""

    name: str
    M: int
    K: int
    N: int
    a_off: int
    a_tb: List[int]
    a_ts: List[int]
    b_off: int
    b_tb: List[int]
    b_ts: List[int]
    trans_a: int = 0
    trans_b: int = 0
    c_off: int = 0
    c_en: int = 0
    c_tb: List[int] = field(default_factory=lambda: [1, 1, 1])
    c_ts: List[int] = field(default_factory=lambda: [0, 0, 0])
    d32_off: int = 0
    d32_tb: List[int] = field(default_factory=lambda: [0, 0, 0])
    d32_ts: List[int] = field(default_factory=lambda: [0, 0, 0])
    d8_off: int = 0
    d8_tb: List[int] = field(default_factory=lambda: [0, 0, 0])
    d8_ts: List[int] = field(default_factory=lambda: [0, 0, 0])
    bypass: int = 1           # 1: int32 D32 out, 0: SIMD int8 D8 out
    shift: int = 0            # SIMD total right shift
    mult: int = 0             # SIMD multiplier
    zp_out: int = 0
    out_min: int = -128
    out_max: int = 127


def _loops(bounds: Sequence[int], strides: Sequence[int]) -> List[int]:
    """Offsets visited by nested temporal loops (innermost first)."""
    offs = [0]
    for b, s in zip(bounds, strides):
        offs = [o + i * s for i in range(b) for o in offs]
    return offs


def gemm(name: str, A: Act, B, out, *, trans_a: bool = False,
         trans_b_act: bool = False, k_tiles: Optional[int] = None,
         k_split: Optional[Tuple[int, int]] = None, n_range: Optional[Tuple[int, int]] = None,
         C: Optional[Acc] = None, simd: Optional[dict] = None,
         a_k_concat: Optional[Tuple[Act, Act]] = None, b_advance_with_m: bool = False) -> Launch:
    """Build one launch computing out = A @ B (+ C).

    A: Act (rows = M, cols = K); with ``trans_a`` the Act holds A^T
       (rows = K, cols = M) and blocks are transposed by the A datapath.
    B: WB (weights, K x N) or Act holding B (rows = K, cols = N) with
       ``trans_b_act`` (blocks transposed by the B datapath).
    out: Act (int8, SIMD on) or Acc (int32, bypass).
    k_split: (k0, kn) tile range of the K dimension (partial sums).
    n_range: (n0, nn) tile range of the N dimension written to ``out``.
    a_k_concat: K = [A1 | A2] (same shape), read through a 2-level K loop.
    b_advance_with_m: diagonal trick, B's tile index follows m (N = 1).
    """
    k0, kn = k_split if k_split else (0, None)
    n0, nn = n_range if n_range else (0, None)
    # --- dimensions
    if a_k_concat:
        A1, A2 = a_k_concat
        mt, kt_half = A1.rt, A1.ct
        kt = 2 * kt_half
    elif trans_a:
        mt, kt = A.ct, A.rt
    else:
        mt, kt = A.rt, A.ct
    if kn is None:
        kn = kt - k0
    if isinstance(B, WB):
        nt_all = B.n // 8
    else:
        nt_all = 1 if b_advance_with_m else B.ct
    if nn is None:
        nn = nt_all - n0
    M, K, N = mt, kn, nn

    # --- A loops: [k, n, m] (+ split for concat)
    if a_k_concat:
        assert k0 == 0 and kn == kt
        a_off = A1.blk(0, 0)
        a_tb = [kt_half, 2, N, M, 1, 1]
        a_ts = [BLK, A2.off - A1.off, 0, A1.ct * BLK, 0, 0]
    elif trans_a:
        a_off = A.blk(k0, 0)
        a_tb = [K, N, M, 1, 1, 1]
        a_ts = [A.ct * BLK, 0, BLK, 0, 0, 0]
    else:
        a_off = A.blk(0, k0)
        a_tb = [K, N, M, 1, 1, 1]
        a_ts = [BLK, 0, A.ct * BLK, 0, 0, 0]

    # --- B loops: [k, n, m]. A streamed weight tile holds only the K / N
    # range of this launch: index it from 0 when its size equals the range.
    if isinstance(B, WB):
        bk0 = 0 if B.kt == kn else k0
        bn0 = 0 if B.n // 8 == nn else n0
        b_off = B.off + (bn0 * B.kt + bk0) * BLK
        b_tb, b_ts = [K, N, M], [BLK, B.kt * BLK, 0]
        tb = 0
    else:
        tb = 1
        if b_advance_with_m:
            b_off = B.blk(k0, 0)
            b_tb, b_ts = [K, 1, M], [B.ct * BLK, 0, BLK]
        else:
            b_off = B.blk(k0, n0)
            b_tb, b_ts = [K, N, M], [B.ct * BLK, BLK, 0]

    L = Launch(name, M, K, N, a_off, a_tb, a_ts, b_off, b_tb, b_ts,
               trans_a=int(trans_a), trans_b=tb)
    # --- C. Even with its channels disabled, the C reader runs its temporal
    # loops and hands GeMMX one (zero) block per output block, so its bounds
    # must always cover the M x N output blocks (else the launch never ends).
    L.c_tb, L.c_ts = [1, N, M], [BLK32, 0, 0]
    if C is not None:
        # C is either a full [M x N] accumulator (partial sums) or a single
        # 8-row bias tile reused for every m (stride 0).
        L.c_en = 1
        L.c_off = C.off + (0 if C.ct == nn else n0) * BLK32
        m_stride = 0 if C.rows == 8 else C.ct * BLK32
        L.c_tb, L.c_ts = [1, N, M], [BLK32, BLK32, m_stride]
    # --- outputs
    if isinstance(out, Acc):
        L.bypass = 1
        L.d32_off = out.off + n0 * BLK32
        L.d32_tb, L.d32_ts = [1, N, M], [BLK32, BLK32, out.ct * BLK32]
    else:
        assert simd is not None, "int8 output needs SIMD parameters"
        L.bypass = 0
        L.d8_off = out.off + n0 * BLK
        L.d8_tb, L.d8_ts = [1, N, M], [BLK, BLK, out.ct * BLK]
        L.shift, L.mult = simd["shift"], simd["mult"]
        L.zp_out = simd.get("zp", 0)
        L.out_min, L.out_max = simd.get("min", -128), simd.get("max", 127)
    return L


def blockdiag2(name: str, A1: Act, A2: Act, Bpair: WB, out: Act, simd: dict) -> Launch:
    """out = A1 @ diag(B0) + A2 @ diag(B1), elementwise per 8x8 block.

    Bpair holds two 8x8 B blocks (K = 16, N = 8), e.g. 55*I and -59*I, so each
    output block is A1_blk @ B0^T + A2_blk @ B1^T. A1, A2: same shape. One
    launch, K = 2 per output block, B reused for every (m, n)."""
    assert (A1.rows, A1.cols) == (A2.rows, A2.cols) == (out.rows, out.cols)
    M, N = A1.rt, A1.ct
    L = Launch(name, M, 2, N, A1.off, [2, N, M, 1, 1, 1],
               [A2.off - A1.off, BLK, A1.ct * BLK, 0, 0, 0],
               Bpair.off, [2, N, M], [BLK, 0, 0])
    L.c_tb, L.c_ts = [1, N, M], [BLK32, 0, 0]
    L.bypass = 0
    L.d8_off = out.off
    L.d8_tb, L.d8_ts = [1, N, M], [BLK, BLK, out.ct * BLK]
    L.shift, L.mult = simd["shift"], simd["mult"]
    L.zp_out = simd.get("zp", 0)
    L.out_min, L.out_max = simd.get("min", -128), simd.get("max", 127)
    return L


# --------------------------------------------------------------------------
# Emulator
# --------------------------------------------------------------------------
class Tcdm:
    """Byte-array TCDM model."""

    def __init__(self, size: int = 128 * 1024):
        self.mem = np.zeros(size, dtype=np.uint8)

    def write(self, off: int, data: np.ndarray) -> None:
        b = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.mem[off:off + b.size] = b

    def read(self, off: int, nbytes: int, dtype=np.int8) -> np.ndarray:
        return self.mem[off:off + nbytes].view(dtype).copy()


def run_launch(t: Tcdm, L: Launch) -> None:
    """Execute one launch on the TCDM model, exactly as GeMMX would."""
    a_seq = _loops(L.a_tb, L.a_ts)
    b_seq = _loops(L.b_tb, L.b_ts)
    nout = L.M * L.N
    assert len(a_seq) == len(b_seq) == nout * L.K, (L.name, len(a_seq), len(b_seq), nout, L.K)

    def blk8(off: int, trans: int) -> np.ndarray:
        x = t.read(off, BLK).reshape(8, 8).astype(np.int64)
        return x.T if trans else x

    accs = []
    for j in range(nout):
        acc = np.zeros((8, 8), dtype=np.int64)
        for kk in range(L.K):
            i = j * L.K + kk
            A = blk8(L.a_off + a_seq[i], L.trans_a)       # [row][k]
            Bm = blk8(L.b_off + b_seq[i], L.trans_b)      # [col][k]
            acc += A @ Bm.T
        accs.append(acc)
    if L.c_en:
        c_seq = _loops([L.c_tb[1], L.c_tb[2]], [L.c_ts[1], L.c_ts[2]])
        for j in range(nout):
            accs[j] += t.read(L.c_off + c_seq[j], BLK32, np.int32).reshape(8, 8)
    out_seq = (_loops([L.d32_tb[1], L.d32_tb[2]], [L.d32_ts[1], L.d32_ts[2]]) if L.bypass
               else _loops([L.d8_tb[1], L.d8_tb[2]], [L.d8_ts[1], L.d8_ts[2]]))
    for j in range(nout):
        a = accs[j]
        assert np.all(np.abs(a) < 2 ** 31), f"{L.name}: int32 overflow"
        if L.bypass:
            t.write(L.d32_off + out_seq[j], a.astype(np.int32))
        else:
            q = postprocessing_simd_golden_model(a.astype(np.int32), 0, L.zp_out, L.shift,
                                                 L.out_max, L.out_min, 0, L.mult)
            t.write(L.d8_off + out_seq[j], q.astype(np.int8))


# --------------------------------------------------------------------------
# C emitter
# --------------------------------------------------------------------------
def _arr(v: Sequence[int]) -> str:
    return "{" + ", ".join(str(int(x)) for x in v) + "}"


def emit_launches(name: str, launches: List[Launch]) -> str:
    """C initialiser for a gemmx_launch_t table."""
    rows = []
    for L in launches:
        # GeMMX streamer strides are unsigned: a negative stride (e.g. two
        # operands placed in decreasing address order) breaks the launch.
        for nm in ("a_ts", "b_ts", "c_ts", "d32_ts", "d8_ts"):
            assert min(getattr(L, nm)) >= 0, f"{L.name}: negative {nm} {getattr(L, nm)}"
        packed = (L.shift & 0xFF) * 0x01010101
        rows.append(
            f"    /* {L.name} */ {{{L.M}, {L.K}, {L.N}, "
            f"{L.a_off}, {_arr(L.a_tb)}, {_arr(L.a_ts)}, "
            f"{L.b_off}, {_arr(L.b_tb)}, {_arr(L.b_ts)}, "
            f"{L.c_off}, {L.c_en}, {_arr(L.c_tb)}, {_arr(L.c_ts)}, "
            f"{L.d32_off}, {_arr(L.d32_tb)}, {_arr(L.d32_ts)}, "
            f"{L.d8_off}, {_arr(L.d8_tb)}, {_arr(L.d8_ts)}, "
            f"{L.trans_a}, {L.trans_b}, {L.bypass}, {packed}, {L.mult}, "
            f"{L.zp_out}, {L.out_min}, {L.out_max}}}")
    return (f"static const gemmx_launch_t {name}[{len(launches)}] = {{\n"
            + ",\n".join(rows) + "\n};\n")
