// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the launch-pattern table from data/test_patterns.py on GeMMX and
// compares every output region byte for byte with the emulator, once per
// configuration mode:
//   0 library   set_gemmx_streamer_csr / set_gemmx_csr (switch per CSR)
//   1 full      generated straight-line writes, all 75 registers
//   2 delta     generated writes of registers that changed since the
//               previous launch (CSR-manager registers persist)
// A fourth mode, writing launch i+1's registers while launch i runs, hangs
// the cluster (a CSR write during a run stalls the interface), so it is not
// used: delta writes already cost ~77 cycles per launch.

#include <stdint.h>

#include "data.h"
#include "gemmx_exec.h"
#include "snrt.h"

static uint32_t check(uint8_t *l1) {
    uint32_t err = 0;
    for (int c = 0; c < TP_N_CHECKS; c++) {
        const uint8_t *got = l1 + tp_check_off[c];
        const uint8_t *exp = tp_expected + tp_check_pos[c];
        uint32_t e = 0;
        for (int j = 0; j < tp_check_len[c]; j++) e += got[j] != exp[j];
        if (e) printf("  check %-12s FAIL (%u/%d bytes differ)\n", tp_check_names[c], e,
                      tp_check_len[c]);
        err += e;
    }
    return err;
}

int main() {
    uint8_t *l1 = (uint8_t *)snrt_l1_next();
    const uint32_t base = (uint32_t)l1;
    uint32_t err = 0;
    for (int mode = 0; mode < 3; mode++) {
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(l1, tp_image, TP_IMAGE_BYTES);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        if (snrt_global_core_idx() == 0) {
            uint32_t t0 = snrt_mcycle(), cfg = 0;
            for (int i = 0; i < TP_N_LAUNCHES; i++) {
                uint32_t a = snrt_mcycle();
                if (mode == 0) gemmx_configure(&tp_launches[i]);
                else if (mode == 1) cfgfull_fns[i](base);
                else if (mode == 2) cfgdelta_fns[i](base);
                cfg += snrt_mcycle() - a;
                set_gemmx_streamer_start();
                set_gemmx_start();
                wait_gemmx_and_streamer();
            }
            uint32_t e = check(l1);
            printf("mode %d: total=%u config=%u  %s\n", mode, snrt_mcycle() - t0, cfg,
                   e ? "FAIL" : "all checks OK");
            err += e;
        }
        snrt_cluster_hw_barrier();
    }
    return err;
}
