// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// One ResNet-8 inference layer on the SNAX GeMMX accelerator, timed.
//
// Data (int8 NHWC input, int8 weights, int8 requantised output through the
// SIMD unit) and the golden result come from the stock KU Leuven conv data
// generator (apps/snax-gemmx-conv/data/datagen.py), driven by one params file
// per layer. The conv itself follows apps/snax-gemmx-conv (implicit im2col in
// the streamers). Added here: per-phase timing (CSR configuration vs
// accelerator run), a cold and a warm iteration, and an optional scalar GAP
// measurement for the classifier head.
//
// Fix vs the stock example: in snax_KUL_cluster the C, D8 and D32 streamers
// have 3 temporal dimensions, but the stock generator emits 4 (bound3 =
// out_height). The 4th loop is dropped by the hardware, the writers expect
// fewer blocks than GeMMX produces and the run never finishes (the stock
// snax-gemmx-conv app hangs on this cluster; it is not in upstream CI). The
// two outer M loops are contiguous (stride3 == bound2 * stride2), so they are
// merged into one loop here. Also: the library writes two spatial strides
// for C/D32 (2-D spatial bounds) where the stock app passes one, and the
// bias matrix C must be copied to TCDM (channel_en_C = 1).

#include "data.h"
#include "snax-gemmx-lib.h"
#include "snax-gemmx-params.h"
#include "snrt.h"

#ifdef MEASURE_GAP
// Global average pooling of the 8x8x64 block3_b output on the core (int8 in,
// int32 sums; the 1/64 is folded into the FC requantisation).
#define GAP_HW 64
#define GAP_C 64
// Word-wise: 4 channels per 32-bit load, bias to unsigned (x ^ 0x80) and
// accumulate two 16-bit lanes per register (64 * 255 < 2^16, exact).
static void gap_i8(const int8_t *in, int32_t *out) {
    for (uint32_t c = 0; c < GAP_C; c += 4) {
        uint32_t ae = 0, ao = 0;
        const uint32_t *w = (const uint32_t *)(in + c);
        for (uint32_t p = 0; p < GAP_HW; p++) {
            uint32_t v = w[p * (GAP_C / 4)] ^ 0x80808080u;
            ae += v & 0x00FF00FFu;
            ao += (v >> 8) & 0x00FF00FFu;
        }
        out[c + 0] = (int32_t)(ae & 0xFFFF) - 128 * GAP_HW;
        out[c + 1] = (int32_t)(ao & 0xFFFF) - 128 * GAP_HW;
        out[c + 2] = (int32_t)(ae >> 16) - 128 * GAP_HW;
        out[c + 3] = (int32_t)(ao >> 16) - 128 * GAP_HW;
    }
}
static uint32_t gap_check(const int8_t *in, const int32_t *out) {
    uint32_t e = 0;
    for (uint32_t c = 0; c < GAP_C; c++) {
        int32_t s = 0;
        for (uint32_t p = 0; p < GAP_HW; p++) s += in[p * GAP_C + c];
        e += s != out[c];
    }
    return e;
}
#endif

int main() {
    int err = 0;
    int8_t *local_a = (int8_t *)(snrt_l1_next() + delta_local_a);
    int8_t *local_b = (int8_t *)(snrt_l1_next() + delta_local_b);
    int8_t *local_d8 = (int8_t *)(snrt_l1_next() + delta_local_d8);
    int32_t *local_d32 = (int32_t *)(snrt_l1_next() + delta_local_d32);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A,
                          Nbatch * (H + 2 * pad_h) * (W + 2 * pad_w) * Cin *
                              sizeof(int8_t));
        snrt_dma_start_1d(local_b, B, Cout * Kh * Kw * Cin * sizeof(int8_t));
        // int32 bias matrix C (channel_en_C = 1)
        snrt_dma_start_1d((int32_t *)(snrt_l1_next() + delta_local_c), C,
                          M * N * meshRow * meshCol * sizeof(int32_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // The stock generator hard-codes the A spatial stride to 8 B (the
    // stride-aware line is commented out there); with stride_w > 1 the 8
    // output pixels of one mesh block are stride_w input pixels apart.
    int32_t Aslstride[] = {Aslstride0 * stride_w};
    int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                          Atlbound3, Atlbound4, Atlbound5};
    int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                           Atlstride3, Atlstride4, Atlstride5};
    int32_t Bslstride[] = {Bslstride0};
    int32_t Btlbound[] = {Btlbound0, Btlbound1, Btlbound2};
    int32_t Btlstride[] = {Btlstride0, Btlstride1, Btlstride2};
    // Merge the two outer (M) loops of C / D8 / D32 into one (see header).
    // C/D32 have 2-D spatial bounds [8, 4] (32 x 64-bit channels): the
    // library writes two spatial strides. The stock apps pass a 1-element
    // array (out-of-bounds read); the second stride is 8 channels x 8 B = 64.
    int32_t Cslstride[] = {Cslstride0, 64};
    int32_t Ctlbound[] = {Ctlbound0, Ctlbound1, Ctlbound2 * Ctlbound3};
    int32_t Ctlstride[] = {Ctlstride0, Ctlstride1, Ctlstride2};
    int32_t D32slstride[] = {D32slstride0, 64};
    int32_t D32tlbound[] = {D32tlbound0, D32tlbound1, D32tlbound2 * D32tlbound3};
    int32_t D32tlstride[] = {D32tlstride0, D32tlstride1, D32tlstride2};
    int32_t D8slstride[] = {D8slstride0};
    int32_t D8tlbound[] = {D8tlbound0, D8tlbound1, D8tlbound2 * D8tlbound3};
    int32_t D8tlstride[] = {D8tlstride0, D8tlstride1, D8tlstride2};
    if (snrt_global_core_idx() == 0 &&
        ((Ctlbound3 > 1 && Ctlstride3 != Ctlbound2 * Ctlstride2) ||
         (D32tlbound3 > 1 && D32tlstride3 != D32tlbound2 * D32tlstride2) ||
         (D8tlbound3 > 1 && D8tlstride3 != D8tlbound2 * D8tlstride2))) {
        printf("ERROR: outer M loops not contiguous, cannot merge\n");
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        // Iteration 0 runs with a cold I-cache, iteration 1 warm.
        for (uint32_t it = 0; it < 2; it++) {
            const char *tag = it ? "warm" : "cold";
            uint32_t t0 = snrt_mcycle();
            set_gemmx_streamer_csr(
                Aslstride, Atlbound, Atlstride, set_addr_remap_index_A,
                Bslstride, Btlbound, Btlstride, set_addr_remap_index_B,
                D8slstride, D8tlbound, D8tlstride, set_addr_remap_index_D8,
                Cslstride, Ctlbound, Ctlstride, set_addr_remap_index_C,
                D32slstride, D32tlbound, D32tlstride, set_addr_remap_index_D32,
                delta_local_a, delta_local_b, delta_local_d8, delta_local_c,
                delta_local_d32, bypassSIMD, transposed_A, transposed_B,
                channel_en_C, broadcast_C);
            set_gemmx_csr(K, N, M, gen_subtraction_config(subtraction_a,
                                                          subtraction_b),
                          gen_csr0_config(input_zp_i, output_zp_i, max_int_i,
                                          min_int_i),
                          gen_csr1_config(double_round_i),
                          shared_bitpacked_shift, shared_multiplier, M * N,
                          bypassSIMD);
            uint32_t t1 = snrt_mcycle();
            set_gemmx_streamer_start();
            set_gemmx_start();
            wait_gemmx_and_streamer();
            uint32_t t2 = snrt_mcycle();
            uint32_t busy = read_gemmx_perf_counter();

            uint32_t e = bypassSIMD
                             ? check_gemmx_result_D32(local_d32, D32, Batch, M, N, false)
                             : check_gemmx_result_D8(local_d8, D8, Batch, M, N, false);
            err += e;
            printf("[LAYER %s] Cycles: config=%u run=%u total=%u (acc busy: %u, "
                   "M=%d K=%d N=%d) Errors: %u\n",
                   tag, t1 - t0, t2 - t1, t2 - t0, busy, M, K, N, e);
#ifdef MEASURE_GAP
            // Timing is data-independent: read 4 kB from the start of TCDM as
            // the 8x8x64 int8 feature map; sums at 8 kB, past that window
            // and past this layer's buffers (fc uses < 4 kB).
            int32_t *sums = (int32_t *)(snrt_l1_next() + 8192);
            uint32_t g0 = snrt_mcycle();
            gap_i8(local_a, sums);
            uint32_t g1 = snrt_mcycle();
            uint32_t ge = gap_check(local_a, sums);
            err += ge;
            printf("[GAP %s] Cycles: %u (8x8x64 int8 on the core) Errors: %u\n",
                   tag, g1 - g0, ge);
#endif
        }
    }
    snrt_cluster_hw_barrier();
    return err;
}
