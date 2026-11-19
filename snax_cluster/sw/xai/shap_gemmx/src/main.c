// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Gradient SHAP (GAP+FC head, hoisted form) on the SNAX GeMMX accelerator.
//
// For a linear head the gradient g is identical for every sample, so
//   attr = sum_i (x - b_i) * g = g * (N*x - sum_i b_i)
// and the N-sample accumulation collapses to one GeMM on GeMMX:
//   S[1 x E] = [-1 ... -1, N] @ [b_1; ...; b_N; x]     (S = N*x - sum_i b_i)
// followed by one elementwise pass on the core: attr[e] = g[k] * S[e].
// The baselines are static and pre-packed into GeMMX's B layout; the runtime
// input x is scattered into B's last row by the DMA core while core 0 writes
// the GeMMX CSRs (~3k cycles), so the scatter is off the critical path.
// The cluster cores are rv32ima (no FPU): the output is the unscaled int32
// attribution; the 1/N and quantisation scales are applied on the host.
//
// The same integer pipeline also runs as scalar loops on the same core, and
// both results are checked bit-exactly against the datagen golden output.

#include <stdint.h>

#include "data.h"
#include "snax-gemmx-lib.h"
#include "snrt.h"

// C/D32 have 2-D spatial bounds [8, 4]: the library writes two spatial
// strides (8 B between channels, 64 B between groups of 8 channels).
#define STREAMER_ARGS(p)                                                     \
    (int32_t[]){p##Aslstride0},                                              \
        (int32_t[]){p##Atlbound0, p##Atlbound1, p##Atlbound2, p##Atlbound3,  \
                    p##Atlbound4, p##Atlbound5},                             \
        (int32_t[]){p##Atlstride0, p##Atlstride1, p##Atlstride2,             \
                    p##Atlstride3, p##Atlstride4, p##Atlstride5},            \
        0, (int32_t[]){p##Bslstride0},                                       \
        (int32_t[]){p##Btlbound0, p##Btlbound1, p##Btlbound2},               \
        (int32_t[]){p##Btlstride0, p##Btlstride1, p##Btlstride2}, 0,         \
        (int32_t[]){p##D8slstride0},                                         \
        (int32_t[]){p##D8tlbound0, p##D8tlbound1, p##D8tlbound2,             \
                    p##D8tlbound3},                                          \
        (int32_t[]){p##D8tlstride0, p##D8tlstride1, p##D8tlstride2,          \
                    p##D8tlstride3},                                         \
        0, (int32_t[]){p##Cslstride0, 64},                                       \
        (int32_t[]){p##Ctlbound0, p##Ctlbound1, p##Ctlbound2, p##Ctlbound3}, \
        (int32_t[]){p##Ctlstride0, p##Ctlstride1, p##Ctlstride2,             \
                    p##Ctlstride3},                                          \
        0, (int32_t[]){p##D32slstride0, 64},                                     \
        (int32_t[]){p##D32tlbound0, p##D32tlbound1, p##D32tlbound2,          \
                    p##D32tlbound3},                                         \
        (int32_t[]){p##D32tlstride0, p##D32tlstride1, p##D32tlstride2,       \
                    p##D32tlstride3},                                        \
        0, p##delta_local_a, p##delta_local_b, p##delta_local_d8,            \
        p##delta_local_c, p##delta_local_d32

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    int8_t *a = (int8_t *)(l1 + g_delta_local_a);
    int8_t *b = (int8_t *)(l1 + g_delta_local_b);
    const int32_t *d = (const int32_t *)(l1 + g_delta_local_d32);

    // Core-side buffers after the accelerator region.
    uint8_t *p = l1 + tcdm_used;
    int8_t *x = (int8_t *)p;      p += (E_SIZE + 7) & ~7;
    int8_t *w = (int8_t *)p;      p += (K_CH * N_CLASSES + 7) & ~7;
    int8_t *base = (int8_t *)p;   p += (N_SAMPLES * E_SIZE + 7) & ~7;
    int32_t *g = (int32_t *)p;    p += K_CH * sizeof(int32_t);
    int32_t *attr_acc = (int32_t *)p; p += E_SIZE * sizeof(int32_t);
    int32_t *attr_sc = (int32_t *)p;  p += E_SIZE * sizeof(int32_t);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(a, A_coef, g_a_bytes);
        snrt_dma_start_1d(b, B_base, g_b_bytes);
        snrt_dma_start_1d(x, x_i8, E_SIZE);
        snrt_dma_start_1d(w, w_fc_i8, K_CH * N_CLASSES);
        snrt_dma_start_1d(base, base_i8, N_SAMPLES * E_SIZE);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t err = 0;
    const int is_compute = snrt_global_core_idx() == 0;
    // Iteration 0 runs with a cold I-cache, iteration 1 warm; both checked.
    for (uint32_t it = 0; it < 2; it++) {
        const char *tag = it ? "warm" : "cold";
        uint32_t t0 = 0, t1 = 0, t2 = 0, perf = 0, x0 = 0, x1 = 0;
        snrt_cluster_hw_barrier();
        // ------------------------------------------------ GeMMX path
        if (is_compute) {
            t0 = snrt_mcycle();
            // Hoisted backward: g = column TARGET_CLASS of W (1/(H*W) on host).
            for (uint32_t k = 0; k < K_CH; k++)
                g[k] = w[k * N_CLASSES + TARGET_CLASS];
            t1 = snrt_mcycle();
            set_gemmx_streamer_csr(STREAMER_ARGS(g_), /*bypassSIMD=*/1,
                                   /*transpose_A=*/0, /*transpose_B=*/0,
                                   channel_en_C, /*broadcast_C=*/0);
            set_gemmx_csr(g_K, g_N, g_M, gen_subtraction_config(0, 0),
                          gen_csr0_config(0, 0, 127, (uint8_t)-128),
                          gen_csr1_config(0), shared_bitpacked_shift,
                          shared_multiplier, g_M * g_N, /*bypassSIMD=*/1);
        } else if (snrt_is_dm_core()) {
            // Scatter x[e] into row N_SAMPLES of the blocked B operand.
            x0 = snrt_mcycle();
            for (uint32_t e0 = 0; e0 < E_SIZE; e0 += MESH_COL) {
                int8_t *dst = b + ((e0 / MESH_COL) * KT + N_SAMPLES / TILE) *
                                      (MESH_COL * TILE) +
                              N_SAMPLES % TILE;
                for (uint32_t c = 0; c < MESH_COL; c++) dst[c * TILE] = x[e0 + c];
            }
            x1 = snrt_mcycle();
        }
        snrt_cluster_hw_barrier();  // x is in B before GeMMX starts
        if (!is_compute) {
            // Printed after the barrier so it does not delay the GeMM start.
            if (snrt_is_dm_core())
                printf("[DMA-core %s] x scatter: %u cycles (overlaps CSR "
                       "config)\n", tag, x1 - x0);
            continue;
        }

        set_gemmx_streamer_start();
        set_gemmx_start();
        wait_gemmx_and_streamer();
        t2 = snrt_mcycle();
        perf = read_gemmx_perf_counter();

        // attr = g * S. S[e] is row 0 of the blocked D32 result:
        // d[(e / MESH_COL) * MESH_ROW * MESH_COL + e % MESH_COL], e = hw*K + k.
        // k-outer keeps g[k] in a register.
        uint32_t t3 = snrt_mcycle();
        for (uint32_t k = 0; k < K_CH; k++) {
            const int32_t gk = g[k];
            const int32_t *s =
                d + (k / MESH_COL) * (MESH_ROW * MESH_COL) + k % MESH_COL;
            int32_t *o = attr_acc + k;
            for (uint32_t hw = 0; hw < H * W; hw++) {
                o[hw * K_CH] = gk * s[hw * (K_CH / MESH_COL) * MESH_ROW * MESH_COL];
            }
        }
        uint32_t t4 = snrt_mcycle();

        printf("[GEMMX %s] Cycles: bwd=%u gemm=%u post=%u total=%u "
               "(acc busy: %u)\n",
               tag, t1 - t0, t2 - t1, t4 - t3, (t4 - t3) + (t2 - t0), perf);

        // ------------------------------------------------ scalar int path
        uint32_t s0 = snrt_mcycle();
        for (uint32_t k = 0; k < K_CH; k++) g[k] = w[k * N_CLASSES + TARGET_CLASS];
        uint32_t s1 = snrt_mcycle();
        for (uint32_t e = 0; e < E_SIZE; e++) {
            int32_t acc = (int32_t)N_SAMPLES * x[e];
            for (uint32_t i = 0; i < N_SAMPLES; i++) acc -= base[i * E_SIZE + e];
            attr_sc[e] = g[e % K_CH] * acc;
        }
        uint32_t s2 = snrt_mcycle();

        printf("[SCALAR-INT %s] Cycles: bwd=%u accum=%u total=%u\n", tag,
               s1 - s0, s2 - s1, s2 - s0);

        // ------------------------------------------------ BIST (bit-exact)
        uint32_t e_acc = 0, e_sc = 0;
        for (uint32_t e = 0; e < E_SIZE; e++) {
            if (attr_acc[e] != attr_golden_q[e]) {
                if (e_acc < 4)
                    printf("MISMATCH gemmx attr[%u]: got %d expected %d\n", e,
                           attr_acc[e], attr_golden_q[e]);
                e_acc++;
            }
            if (attr_sc[e] != attr_golden_q[e]) e_sc++;
        }
        printf("GEMMX %s: %u/%u Errors\n", tag, e_acc, E_SIZE);
        printf("SCALAR-INT %s: %u/%u Errors\n", tag, e_sc, E_SIZE);
        err += e_acc + e_sc;
    }
    snrt_cluster_hw_barrier();
    return err;
}
