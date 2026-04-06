// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Gradient SHAP test harness for SNAX cluster.
// DMA in feature maps + FC weights + baselines, compute SHAP attributions,
// DMA out result. Measures per-step and total cycles.

#include <math.h>
#include <stdint.h>

#include "data.h"
#include "shap.h"
#include "snrt.h"

int main() {
    // Buffer sizes (from data.h compile-time constants)
    uint32_t spatial_size = H * W * K_CH;
    uint32_t fmaps_bytes = spatial_size * sizeof(float);
    uint32_t wfc_bytes = K_CH * N_CLASSES * sizeof(float);
    uint32_t baselines_bytes = N_SAMPLES * spatial_size * sizeof(float);
    uint32_t alphas_bytes = N_SAMPLES * sizeof(float);
    uint32_t attr_bytes = spatial_size * sizeof(float);
    // Scratch: interp_buf(H*W*K) + pooled(K) + logits(C) + grad(H*W*K)
    uint32_t scratch_bytes =
        (2 * spatial_size + K_CH + N_CLASSES) * sizeof(float);

    // Allocate L1 SPM buffers
    void *ptr = (void *)snrt_l1_next();
    float *local_fmaps = ptr;
    ptr += fmaps_bytes;
    float *local_wfc = ptr;
    ptr += wfc_bytes;
    float *local_baselines = ptr;
    ptr += baselines_bytes;
    float *local_alphas = ptr;
    ptr += alphas_bytes;
    float *local_attr = ptr;
    ptr += attr_bytes;
    float *local_scratch = ptr;

    // DMA in: feature maps, FC weights, baselines, alphas
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_fmaps, feature_maps, fmaps_bytes);
        snrt_dma_start_1d(local_wfc, w_fc, wfc_bytes);
        snrt_dma_start_1d(local_baselines, baselines, baselines_bytes);
        snrt_dma_start_1d(local_alphas, alphas, alphas_bytes);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Compute on core 0
    if (snrt_cluster_core_idx() == 0) {
        uint32_t t_start = snrt_mcycle();

        shap_gradient_full(local_fmaps, local_wfc, local_baselines,
                           local_alphas, local_attr, local_scratch, H, W,
                           K_CH, N_CLASSES, N_SAMPLES, TARGET_CLASS);

        uint32_t t_end = snrt_mcycle();
        printf("SHAP Cycles: total=%u (N=%u samples, H=%u W=%u K=%u C=%u)\n",
               t_end - t_start, N_SAMPLES, H, W, K_CH, N_CLASSES);
    }

    snrt_cluster_hw_barrier();

    // DMA out: attributions
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(attr_out, local_attr, attr_bytes);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

#ifdef BIST
    // Verify against golden reference
    if (snrt_cluster_core_idx() == 0) {
        uint32_t errors = 0;
        float max_err = 0.0f;
        for (uint32_t i = 0; i < spatial_size; i++) {
            float err = fabsf(local_attr[i] - attr_golden[i]);
            if (err > max_err) max_err = err;
            if (err > TOLERANCE) {
                if (errors < 8) {
                    printf("MISMATCH attr[%u]: got %f expected %f (err=%f)\n",
                           i, local_attr[i], attr_golden[i], err);
                }
                errors++;
            }
        }
        printf("%u/%u Errors (max_err=%f, tol=%f)\n", errors, spatial_size,
               max_err, TOLERANCE);
        return errors;
    }
#endif

    return 0;
}
