#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Data generator for Grad-CAM on the SNAX GeMMX accelerator (int8 in, int32 out).
#
# Both Grad-CAM matrix products run on GeMMX:
#   GeMM1 (backward): grad[1 x K]  = d_logit[1 x C] @ W^T[C x K]
#   GeMM2 (wsum):     cam[HW x 1]  = fmaps[HW x K]  @ alpha[K x 1]
# The cluster cores are rv32ima (no FPU), so the whole pipeline is integer:
# alpha = grad shifted into int8 range, ReLU, then Q15 normalisation with one
# reciprocal. The golden output here is bit-exact with the C kernel.
#
# GeMMX block layout (see util/sim/snax_utils.block_gemm_golden_model):
#   A: (Mt, Kt, row=8, k=8)   B: (Nt, Kt, col=8, k=8)   D32: (Mt, Nt, row=8, col=8)

import argparse
import math
import os
import pathlib
import sys

import hjson
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import (  # noqa: E402
    emit_license,
    format_scalar_definition,
    format_vector_definition,
)
from snax_utils import (  # noqa: E402
    align_wide_addr,
    block_gemm_golden_model,
    postprocessing_simd_golden_model,
)

BANK_WIDTH = 64       # bits
IN_W = 8              # int8 inputs
OUT_W = 32            # int32 outputs
Q8_OUT_W = 8          # SIMD-quantised output (unused: bypassSIMD = 1)
Q15 = 15


def tiles(n: int, t: int) -> int:
    return max(1, math.ceil(n / t))


def pack_a(mat: np.ndarray, mt: int, kt: int, row: int, size: int) -> np.ndarray:
    """Dense (m x k) int8 matrix -> padded A block layout (Mt, Kt, row, size)."""
    pad = np.zeros((mt * row, kt * size), dtype=np.int8)
    pad[: mat.shape[0], : mat.shape[1]] = mat
    return pad.reshape(mt, row, kt, size).transpose(0, 2, 1, 3).reshape(-1)


def pack_b(mat: np.ndarray, nt: int, kt: int, col: int, size: int) -> np.ndarray:
    """Dense (k x n) int8 matrix -> padded B block layout (Nt, Kt, col, size)."""
    pad = np.zeros((kt * size, nt * col), dtype=np.int8)
    pad[: mat.shape[0], : mat.shape[1]] = mat
    return pad.reshape(kt, size, nt, col).transpose(2, 0, 3, 1).reshape(-1)


def unpack_d(d: np.ndarray, mt: int, nt: int, row: int, col: int) -> np.ndarray:
    """D32 block layout (Mt, Nt, row, col) -> dense (Mt*row x Nt*col)."""
    return d.reshape(mt, nt, row, col).transpose(0, 2, 1, 3).reshape(mt * row, nt * col)


def streamer_defs(p: str, mt: int, kt: int, nt: int, row: int, size: int,
                  col: int, c32_ser: int, d8_ser: int, base: int,
                  d8_offset: int = -1):
    """Streamer CSR values for one (Mt x Kt) @ (Kt x Nt) block GeMM.

    Stride formulas follow apps/snax-gemmx-matmul/data/datagen.py. Buffers
    are placed from TCDM offset ``base``; returns (defines, next_free_offset).
    """
    d = []

    def s(name, val):
        d.append(format_scalar_definition("int32_t", p + name, int(val)))

    s("M", mt); s("K", kt); s("N", nt)
    # A
    s("Aslstride0", BANK_WIDTH / 8)
    a_bounds = [(kt, IN_W * size * row / 8), (nt, 0), (mt, kt * IN_W * size * row / 8),
                (1, 0), (1, 0), (1, 0)]
    for i, (b, st) in enumerate(a_bounds):
        s(f"Atlbound{i}", b); s(f"Atlstride{i}", st)
    # B
    s("Bslstride0", BANK_WIDTH / 8)
    b_bounds = [(kt, IN_W * size * col / 8), (nt, kt * IN_W * size * col / 8), (mt, 0)]
    for i, (b, st) in enumerate(b_bounds):
        s(f"Btlbound{i}", b); s(f"Btlstride{i}", st)
    # C, D32 (same shape)
    for x in ("C", "D32"):
        s(f"{x}slstride0", BANK_WIDTH / 8)
        bounds = [(OUT_W * row * col / c32_ser, 8 * BANK_WIDTH / 8),
                  (nt, OUT_W * row * col / 8),
                  (mt, nt * OUT_W * row * col / 8), (1, 0)]
        for i, (b, st) in enumerate(bounds):
            s(f"{x}tlbound{i}", b); s(f"{x}tlstride{i}", st)
    # D8
    s("D8slstride0", BANK_WIDTH / 8)
    bounds = [(Q8_OUT_W * row * col / d8_ser, 8 * BANK_WIDTH / 8),
              (nt, Q8_OUT_W * row * col / 8),
              (mt, nt * Q8_OUT_W * row * col / 8), (1, 0)]
    for i, (b, st) in enumerate(bounds):
        s(f"D8tlbound{i}", b); s(f"D8tlstride{i}", st)

    # TCDM placement
    a_bytes = mt * kt * row * size
    b_bytes = nt * kt * size * col
    d_bytes = mt * nt * row * col * OUT_W // 8
    off_a = align_wide_addr(base)
    off_b = align_wide_addr(off_a + a_bytes)
    off_c = align_wide_addr(off_b + b_bytes)       # C disabled; own region anyway
    off_d = align_wide_addr(off_c + d_bytes)
    nxt = align_wide_addr(off_d + d_bytes)
    s("delta_local_a", off_a); s("delta_local_b", off_b)
    s("delta_local_c", off_c); s("delta_local_d32", off_d)
    s("delta_local_d8", off_d if d8_offset < 0 else d8_offset)
    s("a_bytes", a_bytes); s("b_bytes", b_bytes); s("d_bytes", d_bytes)
    return d, nxt


def requant_shift(v: np.ndarray) -> int:
    """Smallest right shift that brings every |v| into int8 range."""
    m = int(np.max(np.abs(v))) if v.size else 0
    sh = 0
    while (m >> sh) > 127:
        sh += 1
    return sh


def golden_int(fm_i8, w_i8, d_i8, sh):
    """Bit-exact integer Grad-CAM pipeline (matches main.c).

    ``sh`` is the static alpha requantisation shift (see emit_header).
    """
    grad = d_i8.astype(np.int32) @ w_i8.astype(np.int32).T          # (K,)
    alpha = np.clip(grad >> sh, -128, 127).astype(np.int8)
    cam = fm_i8.astype(np.int32) @ alpha.astype(np.int32)          # (HW,)
    cam = np.maximum(cam, 0)
    mx = int(cam.max())
    if mx > 0:
        inv = (1 << 31) // mx
        camq = ((cam.astype(np.uint64) * np.uint64(inv)) >> np.uint64(16)).astype(np.int32)
    else:
        camq = np.zeros_like(cam)
    return grad, alpha, cam, camq


def golden_float(fm, w, target):
    alpha = w[:, target]
    cam = np.maximum(fm.reshape(-1, fm.shape[-1]) @ alpha, 0.0)
    return cam / cam.max() if cam.max() > 0 else cam


def emit_header(p: dict, hw: dict) -> str:
    acc = hw["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0]
    row, size, col = acc["snax_gemmx_mesh_row"], acc["snax_gemmx_tile_size"], acc["snax_gemmx_mesh_col"]
    c32_ser, d8_ser = acc["snax_gemmx_serial_c32_d32_width"], acc["snax_gemmx_serial_d8_width"]
    h, w, K, C, tgt = p["h"], p["w"], p["K"], p["C"], p["target_class"]
    HW = h * w
    rng = np.random.default_rng(p.get("seed", 42))

    fm = rng.random((h, w, K), dtype=np.float32)
    wfc = rng.random((K, C), dtype=np.float32)
    fm_scale = float(fm.max()) / 127.0
    w_scale = float(np.abs(wfc).max()) / 127.0
    fm_i8 = np.clip(np.round(fm / fm_scale), -128, 127).astype(np.int8).reshape(HW, K)
    w_i8 = np.clip(np.round(wfc / w_scale), -128, 127).astype(np.int8)       # (K, C)
    d_i8 = np.zeros(C, dtype=np.int8); d_i8[tgt] = 1

    # d_logit is one-hot with value 1, so grad is a column of W for any
    # target class: the int8 range of alpha is a static property of W.
    alpha_shift = requant_shift(w_i8.astype(np.int32))
    grad, alpha, cam, camq = golden_int(fm_i8, w_i8, d_i8, alpha_shift)
    camf = golden_float(fm, wfc, tgt)
    q_err = float(np.max(np.abs(camq / 32768.0 - camf)))

    # GeMM1: A = d_logit (1 x C), B = W^T (C x K)
    m1, k1, n1 = tiles(1, row), tiles(C, size), tiles(K, col)
    a1 = pack_a(d_i8.reshape(1, C), m1, k1, row, size)
    b1 = pack_b(w_i8.T, n1, k1, col, size)
    # GeMM2: A = fmaps (HW x K), B = alpha (K x 1)
    m2, k2, n2 = tiles(HW, row), tiles(K, size), tiles(1, col)
    a2 = pack_a(fm_i8, m2, k2, row, size)
    b2 = pack_b(alpha.reshape(K, 1), n2, k2, col, size)

    # Self-check packing against the GeMMX golden model.
    z1 = np.zeros(m1 * n1 * row * col, dtype=np.int32)
    d1 = unpack_d(block_gemm_golden_model(m1, k1, n1, row, size, col, a1, b1, 0, 0, z1), m1, n1, row, col)
    assert np.array_equal(d1[0, :K], grad), "GeMM1 packing mismatch"
    d1_raw = block_gemm_golden_model(m1, k1, n1, row, size, col, a1, b1, 0, 0, z1)
    d8 = postprocessing_simd_golden_model(d1_raw, 0, 0, alpha_shift + 1, 127, -128, 0, 2)
    assert np.array_equal(d8.astype(np.int8), b2), "GeMM1 SIMD output != GeMM2 B layout"
    z2 = np.zeros(m2 * n2 * row * col, dtype=np.int32)
    d2 = unpack_d(block_gemm_golden_model(m2, k2, n2, row, size, col, a2, b2, 0, 0, z2), m2, n2, row, col)
    assert np.array_equal(np.maximum(d2[:HW, 0], 0), cam), "GeMM2 packing mismatch"

    out = [emit_license(), "#include <stdint.h>"]
    out.append(f"// fmaps {h}x{w}x{K}, C={C}, target={tgt}; GeMM1 tiles M{m1} K{k1} N{n1}, "
               f"GeMM2 tiles M{m2} K{k2} N{n2}")
    out.append(f"// Host check: max |cam_q15/32768 - cam_float32| = {q_err:.6f}")
    for name, val in (("H", h), ("W", w), ("K_CH", K), ("N_CLASSES", C),
                      ("TARGET_CLASS", tgt), ("MESH_ROW", row), ("TILE", size),
                      ("MESH_COL", col)):
        out.append(f"#define {name} {val}")  # compile-time: index math -> shifts
    # GeMM1's SIMD int8 output (D8, layout (Mt,Nt,row,col)) has the same
    # byte layout as GeMM2's B operand (Nt=1,Kt,col,k) when Kt2 == Nt1: row 0
    # is alpha (column 0 of B2) and rows 1..7 are zero padding. So GeMM1
    # writes D8 straight into B2 and no core glue is needed between them.
    assert k2 == n1
    d2defs, nxt = streamer_defs("g2_", m2, k2, n2, row, size, col, c32_ser, d8_ser, 0)
    b2_off = int([x for x in d2defs if "g2_delta_local_b " in x][0].split("=")[1].strip(" ;"))
    d1defs, nxt = streamer_defs("g1_", m1, k1, n1, row, size, col, c32_ser, d8_ser, nxt,
                                d8_offset=b2_off)
    out += d1defs + d2defs
    out.append(f"#define ALPHA_SHIFT {alpha_shift}")
    # SIMD requant out = (x * mult) >> shift (floor); mult=2, shift=s+1 -> x >> s
    sh8 = alpha_shift + 1
    packed = (sh8 << 24) | (sh8 << 16) | (sh8 << 8) | sh8
    out.append("int32_t g1_shift_packed[] = { " + ", ".join([str(packed)] * (col // 4)) + " };")
    out.append("int32_t g1_multiplier[] = { " + ", ".join(["2"] * col) + " };")
    out.append(format_scalar_definition("int32_t", "tcdm_used", nxt))
    n_en = int(math.ceil(row * col * OUT_W / BANK_WIDTH / 32))
    out.append("int32_t channel_en_C[] = { " + ", ".join(["0"] * n_en) + " };")
    out.append("int32_t shared_bitpacked_shift[] = { " + ", ".join(["0"] * (col // 4)) + " };")
    out.append("int32_t shared_multiplier[] = { " + ", ".join(["0"] * col) + " };")
    out.append(format_vector_definition("int8_t", "A1", a1))
    out.append(format_vector_definition("int8_t", "B1", b1))
    out.append(format_vector_definition("int8_t", "A2", a2))
    # Dense copies for the scalar reference path on the same core.
    out.append(format_vector_definition("int8_t", "d_logit_i8", d_i8))
    out.append(format_vector_definition("int8_t", "w_fc_i8", w_i8.reshape(-1)))       # (K, C)
    out.append(format_vector_definition("int8_t", "fmaps_i8", fm_i8.reshape(-1)))     # (HW, K)
    out.append(format_vector_definition("int32_t", "cam_golden_q15", camq))
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM GeMMX data generator")
    ap.add_argument("--swcfg", type=pathlib.Path, required=True)
    ap.add_argument("--hwcfg", type=pathlib.Path, required=True)
    a = ap.parse_args()
    with a.swcfg.open() as f:
        p = hjson.loads(f.read())
    with a.hwcfg.open() as f:
        hw = hjson.loads(f.read())
    print(emit_header(p, hw), end="")


if __name__ == "__main__":
    main()
