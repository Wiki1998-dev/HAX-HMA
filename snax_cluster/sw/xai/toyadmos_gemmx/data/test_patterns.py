#!/usr/bin/env python3
"""Hardware check of every GeMMX launch pattern used by the ToyAdmos
pipeline: plain, K-split with in-place C chaining + SIMD, N-split column
range, transposed A, transposed B, K-concat, diagonal trick, bias tile.
Emits a header with the TCDM image, launch table and emulated expected
outputs; the simulator must reproduce them byte for byte."""

from __future__ import annotations

import numpy as np

from emit import emit_schedule_header
from gemmx_sched import Acc, Act, Tcdm, WB, blockdiag2, gemm, pack_act, pack_wb, run_launch


def build():
    rng = np.random.default_rng(1)
    r8 = lambda *s: rng.integers(-20, 20, size=s).astype(np.int8)  # noqa: E731
    t = Tcdm()
    X, W, X2, G = r8(16, 32), r8(32, 24), r8(16, 32), r8(32, 24)
    Wc, Dm, Gm = r8(64, 24), r8(16, 24), r8(16, 24)
    bias = rng.integers(-1000, 1000, size=24).astype(np.int32)

    A, A2 = Act(0, 16, 32), Act(512, 16, 32)
    B, Bc = WB(1024, 32, 24), WB(1792, 64, 24)
    XT, GA = Act(3328, 32, 16), Act(3840, 32, 24)
    DA, GM = Act(4608, 16, 24), Act(4992, 16, 24)
    Cb = Acc(5376, 8, 24)
    D1, D2, D3, D4, D5, D6, D7 = (Acc(6144, 16, 24), Acc(7680, 16, 24), Acc(9216, 16, 24),
                                  Acc(10752, 16, 24), Acc(12288, 16, 24), Acc(13824, 24, 8),
                                  Acc(14592, 16, 24))
    O1, O2 = Act(16128, 16, 24), Act(16512, 16, 24)
    P = Acc(16896, 16, 24)                      # in-place K-split accumulator
    Y, XL, BP = Act(18432, 16, 24), Act(18816, 16, 24), WB(19200, 16, 8)
    used = 19328

    for T, M in ((A, X), (A2, X2), (XT, X.T.copy()), (GA, G), (DA, Dm), (GM, Gm)):
        t.write(T.off, pack_act(M))
    t.write(B.off, pack_wb(W)); t.write(Bc.off, pack_wb(Wc))
    t.write(Cb.off, pack_act(np.tile(bias, (8, 1)).astype(np.int32)))
    t.write(Y.off, pack_act(r8(16, 24))); t.write(XL.off, pack_act(r8(16, 24)))
    eye = np.eye(8, dtype=np.int64)
    t.write(BP.off, np.concatenate([(55 * eye).astype(np.int8).reshape(-1),
                                    (-59 * eye).astype(np.int8).reshape(-1)]))
    image = t.read(0, used, np.uint8)

    simd = dict(shift=4, mult=2)
    launches = [
        gemm("plain", A, B, D1),
        gemm("ksplit0", A, B, D2, k_split=(0, 2)),
        gemm("ksplit1_simd", A, B, O1, k_split=(2, 2), C=D2, simd=simd),
        gemm("nsplit_simd", A, B, O2, n_range=(1, 2), simd=simd),
        gemm("trans_a", XT, B, D3, trans_a=True),
        gemm("trans_b", A, GA, D4, trans_b_act=True),
        gemm("concat", None, Bc, D5, a_k_concat=(A, A2)),
        gemm("diag", DA, GM, D6, trans_a=True, trans_b_act=True, b_advance_with_m=True),
        gemm("bias", A, B, D7, C=Cb),
        gemm("inplace_k0", A, B, P, k_split=(0, 1), C=Cb),
        gemm("inplace_k1", A, B, P, k_split=(1, 1), C=P),
        gemm("inplace_k2", A, B, P, k_split=(2, 1), C=P),
        gemm("inplace_k3", A, B, P, k_split=(3, 1), C=P),
        blockdiag2("loss_inplace", Y, XL, BP, Y, dict(shift=6, mult=2, zp=-3)),
    ]
    for L in launches:
        run_launch(t, L)
    checks = [("plain", D1.off, t.read(D1.off, D1.nbytes, np.uint8)),
              ("ksplit_simd", O1.off, t.read(O1.off, O1.nbytes, np.uint8)),
              ("nsplit_simd", O2.off + 8 * 8, None),
              ("trans_a", D3.off, t.read(D3.off, D3.nbytes, np.uint8)),
              ("trans_b", D4.off, t.read(D4.off, D4.nbytes, np.uint8)),
              ("concat", D5.off, t.read(D5.off, D5.nbytes, np.uint8)),
              ("diag", D6.off, t.read(D6.off, D6.nbytes, np.uint8)),
              ("bias", D7.off, t.read(D7.off, D7.nbytes, np.uint8)),
              ("inplace_acc", P.off, t.read(P.off, P.nbytes, np.uint8)),
              ("loss_inplace", Y.off, t.read(Y.off, Y.nbytes, np.uint8))]
    # N-split wrote only column tiles 1..2 of O2: check those blocks per row tile.
    ns = [c for c in checks if c[0] != "nsplit_simd"]
    for r in range(O2.rt):
        off = O2.blk(r, 1)
        ns.append((f"nsplit_r{r}", off, t.read(off, 2 * 64, np.uint8)))
    return image, launches, ns


if __name__ == "__main__":
    image, launches, checks = build()
    print("#include <stdint.h>\n#include \"gemmx_exec.h\"\n")
    print(emit_schedule_header("tp", image, launches, checks))
    from gemmx_sched import emit_cfg_code
    print(emit_cfg_code(launches, False, "cfgfull"))
    print(emit_cfg_code(launches, True, "cfgdelta"))
