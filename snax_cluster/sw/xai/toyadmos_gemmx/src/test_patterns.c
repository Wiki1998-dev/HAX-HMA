// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the launch-pattern table from data/test_patterns.py on GeMMX and
// compares every output region byte for byte with the emulator.

#include <stdint.h>

#include "data.h"
#include "gemmx_exec.h"
#include "snrt.h"

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1, tp_image, TP_IMAGE_BYTES);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t err = 0;
    if (snrt_global_core_idx() == 0) {
        for (int i = 0; i < TP_N_LAUNCHES; i++) {
            uint32_t t0 = snrt_mcycle();
            gemmx_configure(&tp_launches[i]);
            uint32_t t1 = snrt_mcycle();
            gemmx_run();
            uint32_t t2 = snrt_mcycle();
            printf("launch %d: config=%u run=%u\n", i, t1 - t0, t2 - t1);
        }
        for (int c = 0; c < TP_N_CHECKS; c++) {
            const uint8_t *got = l1 + tp_check_off[c];
            const uint8_t *exp = tp_expected + tp_check_pos[c];
            uint32_t e = 0;
            for (int j = 0; j < tp_check_len[c]; j++) e += got[j] != exp[j];
            printf("check %-12s %s (%u/%d bytes differ)\n", tp_check_names[c],
                   e ? "FAIL" : "OK", e, tp_check_len[c]);
            err += e;
        }
    }
    snrt_cluster_hw_barrier();
    return err;
}
