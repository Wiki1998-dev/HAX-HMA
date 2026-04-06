// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Grad-CAM test harness for SNAX cluster.
// DMA in feature maps + FC weights, compute saliency map, DMA out result.

#include <math.h>
#include <stdint.h>

#include "data.h"
#include "gradcam.h"
#include "snrt.h"

int main() {
    // Buffer sizes (from data.h compile-time constants)
    uint32_t fmaps_size = H * W * K_CH * sizeof(float);
    uint32_t wfc_size = K_CH * N_CLASSES * sizeof(float);
    uint32_t wfc_t_size = N_CLASSES * K_CH * sizeof(float);
    uint32_t dlogit_size = N_CLASSES * sizeof(float);
    uint32_t grad_size = K_CH * sizeof(float);
    uint32_t alpha_size = K_CH * sizeof(float);
    uint32_t cam_size = H * W * sizeof(float);

    // Allocate L1 SPM buffers
    void *ptr = (void *)snrt_l1_next();
    float *local_fmaps = ptr;
    ptr += fmaps_size;
    float *local_wfc = ptr;
    ptr += wfc_size;
    float *local_wfc_t = ptr;
    ptr += wfc_t_size;
    float *local_dlogit = ptr;
    ptr += dlogit_size;
    float *local_grad = ptr;
    ptr += grad_size;
    float *local_alpha = ptr;
    ptr += alpha_size;
    float *local_cam = ptr;

    // DMA in: feature maps, FC weights, one-hot d_logit
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_fmaps, feature_maps, fmaps_size);
        snrt_dma_start_1d(local_wfc, w_fc, wfc_size);
        snrt_dma_start_1d(local_dlogit, d_logit, dlogit_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Compute on non-DM core
    if (snrt_cluster_core_idx() == 0) {
        uint32_t t0, t1, t2, t3, t4;

        // Step 0: transpose W_fc (K×C) → W_fc_T (C×K)
        t0 = snrt_mcycle();
        transpose_2d_fp32(local_wfc, local_wfc_t, K_CH, N_CLASSES);

        // Step 1: backward GeMM — grad[K] = d_logit[C] @ W_fc_T[C×K]
        t1 = snrt_mcycle();
        gradcam_backward_gemm(local_dlogit, local_wfc_t, local_grad,
                              N_CLASSES, K_CH);

        // Step 2: global average pool → alpha[K]
        t2 = snrt_mcycle();
        gradcam_gap_weights(local_grad, local_alpha, H, W, K_CH);

        // Step 3: weighted sum → cam[H×W]
        t3 = snrt_mcycle();
        gradcam_weighted_sum(local_fmaps, local_alpha, local_cam, H, W, K_CH);

        // Step 4: ReLU + normalize
        t4 = snrt_mcycle();
        gradcam_relu_normalize(local_cam, H, W);

        uint32_t t_end = snrt_mcycle();
        printf("Cycles: transpose=%u gemm=%u gap=%u wsum=%u relu=%u total=%u\n",
               t1 - t0, t2 - t1, t3 - t2, t4 - t3, t_end - t4, t_end - t0);
    }

    snrt_cluster_hw_barrier();

    // DMA out: cam
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(cam_out, local_cam, cam_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

#ifdef BIST
    // Verify against golden reference
    if (snrt_cluster_core_idx() == 0) {
        uint32_t errors = 0;
        for (uint32_t i = 0; i < H * W; i++) {
            if (fabsf(local_cam[i] - cam_golden[i]) > 1e-3f) {
                printf("MISMATCH cam[%u]: got %f expected %f\n", i,
                       local_cam[i], cam_golden[i]);
                errors++;
            }
        }
        printf("%u/%u Errors\n", errors, H * W);
        return errors;
    }
#endif

    return 0;
}
