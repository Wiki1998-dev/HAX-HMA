// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Generic GeMMX launch-table executor (snax_KUL_cluster). Each table entry
// holds every streamer / GeMM / SIMD CSR value of one launch; the values are
// produced, and the whole schedule is emulated bit-exactly, by
// data/gemmx_sched.py. Offsets are TCDM byte offsets from snrt_l1_next().

#pragma once
#include <stdint.h>

#include "snax-gemmx-lib.h"
#include "snrt.h"

typedef struct {
    int32_t M, K, N;
    int32_t a_off, a_tb[6], a_ts[6];
    int32_t b_off, b_tb[3], b_ts[3];
    int32_t c_off, c_en, c_tb[3], c_ts[3];
    int32_t d32_off, d32_tb[3], d32_ts[3];
    int32_t d8_off, d8_tb[3], d8_ts[3];
    int32_t trans_a, trans_b, bypass;
    int32_t shift_packed;  // SIMD shift, same for all 8 columns (4 x 8 bit)
    int32_t mult;          // SIMD multiplier, same for all 8 columns
    int32_t zp_out, out_min, out_max;
} gemmx_launch_t;

// Write all CSRs of one launch (does not start it).
static inline void gemmx_configure(const gemmx_launch_t *L) {
    int32_t sl8[1] = {8};
    // C/D32 have 2-D spatial bounds [8, 4]: two spatial strides.
    int32_t sl32[2] = {8, 64};
    int32_t ch_en[1] = {L->c_en ? (int32_t)0xFFFFFFFF : 0};
    int32_t shift[2] = {L->shift_packed, L->shift_packed};
    int32_t mult[8];
    for (int i = 0; i < 8; i++) mult[i] = L->mult;
    set_gemmx_streamer_csr(
        sl8, (int32_t *)L->a_tb, (int32_t *)L->a_ts, 0,
        sl8, (int32_t *)L->b_tb, (int32_t *)L->b_ts, 0,
        sl8, (int32_t *)L->d8_tb, (int32_t *)L->d8_ts, 0,
        sl32, (int32_t *)L->c_tb, (int32_t *)L->c_ts, 0,
        sl32, (int32_t *)L->d32_tb, (int32_t *)L->d32_ts, 0,
        L->a_off, L->b_off, L->d8_off, L->c_off, L->d32_off, L->bypass,
        L->trans_a, L->trans_b, ch_en, /*broadcast_C=*/0);
    set_gemmx_csr(L->K, L->N, L->M, gen_subtraction_config(0, 0),
                  gen_csr0_config(0, (uint8_t)L->zp_out, (uint8_t)L->out_max,
                                  (uint8_t)L->out_min),
                  gen_csr1_config(0), shift, mult, L->M * L->N, L->bypass);
}

// Start the configured launch and wait for it to finish.
static inline void gemmx_run(void) {
    set_gemmx_streamer_start();
    set_gemmx_start();
    wait_gemmx_and_streamer();
}
