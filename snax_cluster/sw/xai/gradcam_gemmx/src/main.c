// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Grad-CAM on the SNAX GeMMX accelerator (snax_KUL_cluster).
//
// Both matrix products run on GeMMX (int8 x int8):
//   GeMM1: alpha[1 x K] = requant(d_logit[1 x C] @ W^T[C x K])   (SIMD on)
//   GeMM2: cam[HW x 1]  = fmaps[HW x K] @ alpha[K x 1]           (int32 out)
// GeMM1's SIMD unit requantises grad -> int8 (>> ALPHA_SHIFT) and writes it
// straight into GeMM2's B operand: the D8 block layout matches B's, with
// alpha in column 0 and zero padding elsewhere. No core work between GeMMs.
// The cluster cores are rv32ima (no FPU): ReLU + Q15 normalisation are integer.
//
// The same integer pipeline also runs as scalar loops on the same core, so
// the accelerator speedup is measured like-for-like; both results are
// checked bit-exactly against the datagen golden output.

#include <stdint.h>

#include "data.h"
#include "snax-gemmx-lib.h"
#include "snrt.h"

// Q15 normalisation by the maximum (one integer divide, then multiply-shift).
static inline void norm_q15(int32_t *cam, uint32_t n, int32_t mx) {
    if (mx > 0) {
        uint32_t inv = (1u << 31) / (uint32_t)mx;
        for (uint32_t i = 0; i < n; i++)
            cam[i] = (int32_t)(((uint64_t)(uint32_t)cam[i] * inv) >> 16);
    } else {
        for (uint32_t i = 0; i < n; i++) cam[i] = 0;
    }
}

// ReLU + Q15 normalisation of a dense vector.
static inline void relu_norm_q15(int32_t *cam, uint32_t n) {
    int32_t mx = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (cam[i] < 0) cam[i] = 0;
        if (cam[i] > mx) mx = cam[i];
    }
    norm_q15(cam, n, mx);
}

static void run_gemmx(int32_t *Aslstride, int32_t *Atlbound, int32_t *Atlstride,
                      int32_t *Bslstride, int32_t *Btlbound, int32_t *Btlstride,
                      int32_t *D8slstride, int32_t *D8tlbound,
                      int32_t *D8tlstride, int32_t *Cslstride,
                      int32_t *Ctlbound, int32_t *Ctlstride,
                      int32_t *D32slstride, int32_t *D32tlbound,
                      int32_t *D32tlstride, int32_t da, int32_t db,
                      int32_t dd8, int32_t dc, int32_t dd32, int32_t m,
                      int32_t k, int32_t n, int32_t bypass, int32_t *shift,
                      int32_t *mult) {
    set_gemmx_streamer_csr(Aslstride, Atlbound, Atlstride, 0, Bslstride,
                           Btlbound, Btlstride, 0, D8slstride, D8tlbound,
                           D8tlstride, 0, Cslstride, Ctlbound, Ctlstride, 0,
                           D32slstride, D32tlbound, D32tlstride, 0, da, db,
                           dd8, dc, dd32, bypass, /*transpose_A=*/0,
                           /*transpose_B=*/0, channel_en_C, /*broadcast_C=*/0);
    set_gemmx_csr(k, n, m, gen_subtraction_config(0, 0),
                  gen_csr0_config(0, 0, 127, (uint8_t)-128),
                  gen_csr1_config(0), shift, mult, m * n, bypass);
    set_gemmx_streamer_start();
    set_gemmx_start();
    wait_gemmx_and_streamer();
}

// C/D32 have 2-D spatial bounds [8, 4]: the library writes two spatial
// strides (8 B between channels, 64 B between groups of 8 channels).
#define STREAMER_ARGS(p)                                                     \
    (int32_t[]){p##Aslstride0},                                              \
        (int32_t[]){p##Atlbound0, p##Atlbound1, p##Atlbound2, p##Atlbound3,  \
                    p##Atlbound4, p##Atlbound5},                             \
        (int32_t[]){p##Atlstride0, p##Atlstride1, p##Atlstride2,             \
                    p##Atlstride3, p##Atlstride4, p##Atlstride5},            \
        (int32_t[]){p##Bslstride0},                                          \
        (int32_t[]){p##Btlbound0, p##Btlbound1, p##Btlbound2},               \
        (int32_t[]){p##Btlstride0, p##Btlstride1, p##Btlstride2},            \
        (int32_t[]){p##D8slstride0},                                         \
        (int32_t[]){p##D8tlbound0, p##D8tlbound1, p##D8tlbound2,             \
                    p##D8tlbound3},                                          \
        (int32_t[]){p##D8tlstride0, p##D8tlstride1, p##D8tlstride2,          \
                    p##D8tlstride3},                                         \
        (int32_t[]){p##Cslstride0, 64},                                          \
        (int32_t[]){p##Ctlbound0, p##Ctlbound1, p##Ctlbound2, p##Ctlbound3}, \
        (int32_t[]){p##Ctlstride0, p##Ctlstride1, p##Ctlstride2,             \
                    p##Ctlstride3},                                          \
        (int32_t[]){p##D32slstride0, 64},                                        \
        (int32_t[]){p##D32tlbound0, p##D32tlbound1, p##D32tlbound2,          \
                    p##D32tlbound3},                                         \
        (int32_t[]){p##D32tlstride0, p##D32tlstride1, p##D32tlstride2,       \
                    p##D32tlstride3},                                        \
        p##delta_local_a, p##delta_local_b, p##delta_local_d8,               \
        p##delta_local_c, p##delta_local_d32, p##M, p##K, p##N

int main() {
    const uint32_t HW = H * W;
    uint8_t *l1 = (uint8_t *)snrt_l1_next();

    int8_t *a1 = (int8_t *)(l1 + g1_delta_local_a);
    int8_t *b1 = (int8_t *)(l1 + g1_delta_local_b);
    int8_t *a2 = (int8_t *)(l1 + g2_delta_local_a);
    int32_t *d2 = (int32_t *)(l1 + g2_delta_local_d32);

    // Scalar-path buffers after the accelerator region.
    uint8_t *p = l1 + tcdm_used;
    int8_t *s_d = (int8_t *)p;   p += (N_CLASSES + 7) & ~7;
    int8_t *s_w = (int8_t *)p;   p += (K_CH * N_CLASSES + 7) & ~7;
    int8_t *s_fm = (int8_t *)p;  p += (HW * K_CH + 7) & ~7;
    int32_t *grad = (int32_t *)p; p += K_CH * sizeof(int32_t);
    int8_t *alpha = (int8_t *)p; p += (K_CH + 7) & ~7;
    int32_t *cam_acc = (int32_t *)p; p += HW * sizeof(int32_t);
    int32_t *cam_sc = (int32_t *)p; p += HW * sizeof(int32_t);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a1, A1, g1_a_bytes);
        snrt_dma_start_1d(b1, B1, g1_b_bytes);
        snrt_dma_start_1d(a2, A2, g2_a_bytes);
        snrt_dma_start_1d(s_d, d_logit_i8, N_CLASSES);
        snrt_dma_start_1d(s_w, w_fc_i8, K_CH * N_CLASSES);
        snrt_dma_start_1d(s_fm, fmaps_i8, HW * K_CH);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t err = 0;
    if (snrt_global_core_idx() == 0) {
      // Iteration 0 runs with a cold I-cache, iteration 1 warm; both checked.
      for (uint32_t it = 0; it < 2; it++) {
        const char *tag = it ? "warm" : "cold";
        // ------------------------------------------------ GeMMX path
        uint32_t t0 = snrt_mcycle();
        // alpha = requant(d @ W^T), written by GeMMX directly into B2.
        run_gemmx(STREAMER_ARGS(g1_), /*bypass=*/0, g1_shift_packed,
                  g1_multiplier);
        uint32_t t1 = snrt_mcycle();
        uint32_t perf1 = read_gemmx_perf_counter();
        uint32_t t2 = snrt_mcycle();                      // no glue

        run_gemmx(STREAMER_ARGS(g2_), /*bypass=*/1, shared_bitpacked_shift,
                  shared_multiplier);                     // cam = fm @ alpha
        uint32_t t3 = snrt_mcycle();
        uint32_t perf2 = read_gemmx_perf_counter();

        // Gather column 0 of the blocked D32 result, fused with ReLU + max.
        // GeMM2 has a single N tile (alpha is one column), so element (r, 0)
        // of the (Mt, 1, row, col) layout sits at r * MESH_COL.
        int32_t mx = 0;
        for (uint32_t r = 0; r < HW; r++) {
            int32_t v = d2[r * MESH_COL];
            v = v < 0 ? 0 : v;
            cam_acc[r] = v;
            if (v > mx) mx = v;
        }
        norm_q15(cam_acc, HW, mx);
        uint32_t t4 = snrt_mcycle();

        printf("[GEMMX %s] Cycles: gemm1=%u glue=%u gemm2=%u post=%u total=%u "
               "(acc busy: %u + %u)\n",
               tag, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0, perf1, perf2);

        // ------------------------------------------------ scalar int path
        uint32_t s0 = snrt_mcycle();
        for (uint32_t k = 0; k < K_CH; k++) {           // grad = d @ W^T
            int32_t acc = 0;
            for (uint32_t c = 0; c < N_CLASSES; c++)
                acc += (int32_t)s_d[c] * (int32_t)s_w[k * N_CLASSES + c];
            grad[k] = acc;
        }
        uint32_t s1 = snrt_mcycle();
        for (uint32_t k = 0; k < K_CH; k++) {
            int32_t a = grad[k] >> ALPHA_SHIFT;
            alpha[k] = (int8_t)(a > 127 ? 127 : (a < -128 ? -128 : a));
        }
        uint32_t s2 = snrt_mcycle();
        for (uint32_t r = 0; r < HW; r++) {             // cam = fm @ alpha
            int32_t acc = 0;
            for (uint32_t k = 0; k < K_CH; k++)
                acc += (int32_t)s_fm[r * K_CH + k] * (int32_t)alpha[k];
            cam_sc[r] = acc;
        }
        uint32_t s3 = snrt_mcycle();
        relu_norm_q15(cam_sc, HW);
        uint32_t s4 = snrt_mcycle();

        printf("[SCALAR-INT %s] Cycles: gemm1=%u glue=%u gemm2=%u post=%u "
               "total=%u\n",
               tag, s1 - s0, s2 - s1, s3 - s2, s4 - s3, s4 - s0);

        // ------------------------------------------------ BIST (bit-exact)
        uint32_t e_acc = 0, e_sc = 0;
        for (uint32_t r = 0; r < HW; r++) {
            if (cam_acc[r] != cam_golden_q15[r]) {
                if (e_acc < 4)
                    printf("MISMATCH gemmx cam[%u]: got %d expected %d\n", r,
                           cam_acc[r], cam_golden_q15[r]);
                e_acc++;
            }
            if (cam_sc[r] != cam_golden_q15[r]) e_sc++;
        }
        printf("GEMMX %s: %u/%u Errors\n", tag, e_acc, HW);
        printf("SCALAR-INT %s: %u/%u Errors\n", tag, e_sc, HW);
        err += e_acc + e_sc;
      }
    }
    snrt_cluster_hw_barrier();
    return err;
}
