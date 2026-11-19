#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Data generator for Gradient SHAP on the SNAX GeMMX accelerator (int8 in,
# int32 out), GAP+FC model, hoisted form (Phase 2b).
#
# For a linear head the gradient g is the same for every sample, so
#   sum_i (x - b_i) * g  =  g * (N*x - sum_i b_i)
# and the per-sample accumulation loop collapses to one GeMM:
#   S[1 x E] = [-1 ... -1, N][1 x (N+1)] @ [b_1; ...; b_N; x][(N+1) x E]
#            = N*x - sum_i b_i
# followed by one elementwise pass on the core: attr = g[k] * S.
# The baselines are static (background set) and pre-packed; the runtime input
# x is scattered into row N of B by the DMA core while core 0 configures
# GeMMX, so B's x row is emitted as zeros here.
# The 1/N and the quantisation scales are applied on the host, so the C
# kernel output (int32) is bit-exact with the golden output here.

import argparse
import importlib.util
import math
import os
import pathlib
import sys

import hjson
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../../../util/sim/"))
from data_utils import (  # noqa: E402
    emit_license,
    format_scalar_definition,
    format_vector_definition,
)
from snax_utils import block_gemm_golden_model  # noqa: E402

# Reuse the block packing + streamer CSR helpers of the Grad-CAM GeMMX app.
_spec = importlib.util.spec_from_file_location(
    "gradcam_gemmx_datagen", os.path.join(HERE, "../../gradcam_gemmx/data/datagen.py"))
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)
tiles, pack_a, pack_b, unpack_d, streamer_defs = (
    _g.tiles, _g.pack_a, _g.pack_b, _g.unpack_d, _g.streamer_defs)
BANK_WIDTH, OUT_W = _g.BANK_WIDTH, _g.OUT_W


def golden_int(x_i8: np.ndarray, b_i8: np.ndarray, g_i8: np.ndarray) -> np.ndarray:
    """Bit-exact integer hoisted SHAP (matches main.c).

    Args:
        x_i8: (H*W, K) int8 input feature map.
        b_i8: (N, H*W*K) int8 baselines.
        g_i8: (K,) int8 gradient (column ``target`` of the FC weights).

    Returns:
        (H*W*K,) int32 unscaled attribution ``g * (N*x - sum_i b_i)``.
    """
    n = b_i8.shape[0]
    diff = n * x_i8.reshape(-1).astype(np.int32) - b_i8.astype(np.int32).sum(axis=0)
    return (diff.reshape(x_i8.shape) * g_i8.astype(np.int32)[None, :]).reshape(-1)


def golden_float(x: np.ndarray, b: np.ndarray, w: np.ndarray, target: int) -> np.ndarray:
    """Float32 Gradient SHAP for the GAP+FC model (same as sw/xai/shap)."""
    h, wd, k = x.shape
    grad = w[:, target] / (h * wd)
    return ((x[None] - b).mean(axis=0) * grad[None, None, :]).reshape(-1)


def emit_header(p: dict, hw: dict) -> str:
    """Return the C header for one parameter set."""
    acc = hw["snax_streamer_gemmX_core_template"]["snax_acc_cfg"][0]
    row, size, col = (acc["snax_gemmx_mesh_row"], acc["snax_gemmx_tile_size"],
                      acc["snax_gemmx_mesh_col"])
    c32_ser, d8_ser = acc["snax_gemmx_serial_c32_d32_width"], acc["snax_gemmx_serial_d8_width"]
    h, w, K, C, tgt, N = p["h"], p["w"], p["K"], p["C"], p["target_class"], p["n_samples"]
    HW, E = h * w, h * w * K
    assert K % col == 0, "post loop assumes K is a multiple of MESH_COL"
    assert N <= 127, "A operand holds N as an int8 coefficient"
    rng = np.random.default_rng(p.get("seed", 42))

    # Same distributions as sw/xai/shap/data/datagen.py.
    x = rng.random((h, w, K), dtype=np.float32)
    wfc = (rng.random((K, C), dtype=np.float32) * 0.2 - 0.1).astype(np.float32)
    b = (rng.standard_normal((N, h, w, K)).astype(np.float32) * 0.1).astype(np.float32)

    # x and the baselines share one scale (their difference is formed in int).
    s_x = float(max(np.abs(x).max(), np.abs(b).max())) / 127.0
    s_w = float(np.abs(wfc).max()) / 127.0
    x_i8 = np.clip(np.round(x / s_x), -128, 127).astype(np.int8).reshape(HW, K)
    b_i8 = np.clip(np.round(b / s_x), -128, 127).astype(np.int8).reshape(N, E)
    w_i8 = np.clip(np.round(wfc / s_w), -128, 127).astype(np.int8)       # (K, C)
    g_i8 = w_i8[:, tgt]

    attr_q = golden_int(x_i8, b_i8, g_i8)
    attr_f = golden_float(x, b, wfc, tgt)
    scale = s_x * s_w / (HW * N)              # attr_float ~= attr_q * scale
    q_err = float(np.max(np.abs(attr_q * scale - attr_f)) / np.max(np.abs(attr_f)))

    # GeMM: A = [-1 x N, N] (1 x N+1), B = [baselines; x] (N+1 x E) -> S.
    m, kt, nt = tiles(1, row), tiles(N + 1, size), tiles(E, col)
    a_row = np.append(-np.ones(N, dtype=np.int8), np.int8(N)).reshape(1, N + 1)
    a = pack_a(a_row, m, kt, row, size)
    bx = np.vstack([b_i8, x_i8.reshape(1, E)])
    bp_full = pack_b(bx, nt, kt, col, size)
    bp = pack_b(np.vstack([b_i8, np.zeros((1, E), np.int8)]), nt, kt, col, size)
    # The kernel scatters x[e] to B offset
    #   ((e / MESH_COL * KT + N / TILE) * MESH_COL + e % MESH_COL) * TILE + N % TILE.
    e = np.arange(E)
    x_off = ((e // col * kt + N // size) * col + e % col) * size + N % size
    chk = bp.copy(); chk[x_off] = x_i8.reshape(-1)
    assert np.array_equal(chk, bp_full), "x scatter offsets mismatch"
    z = np.zeros(m * nt * row * col, dtype=np.int32)
    d_raw = block_gemm_golden_model(m, kt, nt, row, size, col, a, bp_full, 0, 0, z)
    s_row = unpack_d(d_raw, m, nt, row, col)[0, :E]
    s_ref = N * x_i8.reshape(-1).astype(np.int32) - b_i8.astype(np.int32).sum(axis=0)
    assert np.array_equal(s_row, s_ref), "GeMM packing mismatch"
    # The kernel reads S[e] at d[(e / MESH_COL) * MESH_ROW * MESH_COL + e % MESH_COL].
    assert np.array_equal(d_raw[(e // col) * row * col + e % col], s_row)

    defs, nxt = streamer_defs("g_", m, kt, nt, row, size, col, c32_ser, d8_ser, 0)

    out = [emit_license(), "#include <stdint.h>"]
    out.append(f"// x {h}x{w}x{K}, C={C}, target={tgt}, N={N}; GeMM tiles M{m} K{kt} N{nt}")
    out.append(f"// Host check: max |attr_q*scale - attr_float32| / max|attr_float32| = {q_err:.6f}")
    out.append(f"// Dequantise: attr_float = attr_q * {scale:.9e} (s_x*s_w/(H*W*N))")
    for name, val in (("H", h), ("W", w), ("K_CH", K), ("N_CLASSES", C),
                      ("TARGET_CLASS", tgt), ("N_SAMPLES", N), ("E_SIZE", E),
                      ("MESH_ROW", row), ("MESH_COL", col), ("TILE", size),
                      ("KT", kt)):
        out.append(f"#define {name} {val}")
    out += defs
    out.append(format_scalar_definition("int32_t", "tcdm_used", nxt))
    n_en = int(math.ceil(row * col * OUT_W / BANK_WIDTH / 32))
    out.append("int32_t channel_en_C[] = { " + ", ".join(["0"] * n_en) + " };")
    out.append("int32_t shared_bitpacked_shift[] = { " + ", ".join(["0"] * (col // 4)) + " };")
    out.append("int32_t shared_multiplier[] = { " + ", ".join(["0"] * col) + " };")
    out.append(format_vector_definition("int8_t", "A_coef", a))
    out.append(format_vector_definition("int8_t", "B_base", bp))
    out.append(format_vector_definition("int8_t", "x_i8", x_i8.reshape(-1)))          # (HW*K)
    out.append(format_vector_definition("int8_t", "w_fc_i8", w_i8.reshape(-1)))       # (K, C)
    out.append(format_vector_definition("int8_t", "base_i8", b_i8.reshape(-1)))       # (N, E)
    out.append(format_vector_definition("int32_t", "attr_golden_q", attr_q))
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Gradient SHAP GeMMX data generator")
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
