#define BIST
// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Phase 4b — Neuro-symbolic rule evaluation kernel for SNAX cluster.
//
// Loops over N_TEST_INPUTS pre-scaled fixed-point inputs and calls the
// auto-generated `symbolic_predict()` function (see data.h, emitted by
// data/datagen.py from src/xai/symbolic/rule_to_c.py). Times the
// per-input cost with snrt_mcycle() and BIST-compares against the Python
// reference predictions.
//
// Integer-only — no FPU is touched at any point.

#include <stdint.h>

#include "data.h"
#include "snrt.h"

int main() {
    uint32_t inputs_size = N_TEST_INPUTS * N_FEATURES * sizeof(int32_t);
    uint32_t outputs_size = N_TEST_INPUTS * sizeof(int32_t);

    // Allocate L1 SPM buffers
    void *ptr = (void *)snrt_l1_next();
    int32_t *local_inputs = ptr;
    ptr += inputs_size;
    int32_t *local_outputs = ptr;

    // DMA in test inputs
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_inputs, test_inputs, inputs_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Compute on core 0 only — deterministic cycle counting
    if (snrt_cluster_core_idx() == 0) {
        uint32_t t0 = snrt_mcycle();

        for (uint32_t i = 0; i < N_TEST_INPUTS; i++) {
            const int32_t *x = &local_inputs[i * N_FEATURES];
            local_outputs[i] = symbolic_predict(x);
        }

        uint32_t t1 = snrt_mcycle();

        uint32_t total = t1 - t0;
        uint32_t per_input = total / N_TEST_INPUTS;
        printf("Cycles: total=%u per_input=%u n_inputs=%u\n",
               total, per_input, N_TEST_INPUTS);
    }

    snrt_cluster_hw_barrier();

    // DMA out predictions
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(test_outputs, local_outputs, outputs_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

#ifdef BIST
    // Verify against golden predictions from Python rule reference
    if (snrt_cluster_core_idx() == 0) {
        uint32_t errors = 0;
        for (uint32_t i = 0; i < N_TEST_INPUTS; i++) {
            if (local_outputs[i] != golden_predictions[i]) {
                printf("MISMATCH input[%u]: got %d expected %d\n", i,
                       local_outputs[i], golden_predictions[i]);
                errors++;
            }
        }
        printf("%u/%u Errors\n", errors, N_TEST_INPUTS);
        return errors;
    }
#endif

    return 0;
}
