// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Grad-CAM kernel for SNAX cluster.
// Computes saliency map from last-conv feature maps and FC weights.
//
// The SNAX GeMM accelerator operates on int8 block-tiled data, so the
// float32 Grad-CAM backward pass runs on the RISC-V compute core.
// For quantized (int8) models, the backward GeMM can be offloaded to
// the SNAX GeMM accelerator via snax-gemmx-lib — see gradcam_backward_gemm_i8.

#pragma once

#include <stdint.h>

#include "snrt.h"

// Transpose a row-major float matrix: dst[j][i] = src[i][j]
static inline void transpose_2d_fp32(const float *src, float *dst,
                                     uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Backward GeMM: grad[1×K] = d_logit[1×C] @ W_fc^T[C×K]
// Caller must provide W_fc_T already transposed (C×K layout).
// Runs on RISC-V core (float32 — SNAX GeMM is int8 only).
static inline void gradcam_backward_gemm(const float *d_logit,
                                         const float *w_fc_t, float *grad,
                                         uint32_t C, uint32_t K) {
    for (uint32_t k = 0; k < K; k++) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < C; c++) {
            sum += d_logit[c] * w_fc_t[c * K + k];
        }
        grad[k] = sum;
    }
}

// Global average pool of gradients over spatial dimensions.
// grad: (h*w, K) layout  →  alpha: (K,)
// alpha[k] = (1/(h*w)) * sum_{i,j} grad[i*K + k]
static inline void gradcam_gap_weights(const float *grad, float *alpha,
                                       uint32_t h, uint32_t w, uint32_t K) {
    float inv_hw = 1.0f / (float)(h * w);
    for (uint32_t k = 0; k < K; k++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < h * w; i++) {
            sum += grad[i * K + k];
        }
        alpha[k] = sum * inv_hw;
    }
}

// Weighted sum of feature maps: cam[ij] = sum_k alpha[k] * fmaps[ij*K + k]
// fmaps: (h*w, K) layout, alpha: (K,), cam: (h*w,)
static inline void gradcam_weighted_sum(const float *fmaps, const float *alpha,
                                        float *cam, uint32_t h, uint32_t w,
                                        uint32_t K) {
    for (uint32_t ij = 0; ij < h * w; ij++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            sum += alpha[k] * fmaps[ij * K + k];
        }
        cam[ij] = sum;
    }
}

// ReLU + normalize cam to [0, 1].
static inline void gradcam_relu_normalize(float *cam, uint32_t h, uint32_t w) {
    float max_val = 0.0f;
    for (uint32_t i = 0; i < h * w; i++) {
        if (cam[i] < 0.0f) cam[i] = 0.0f;
        if (cam[i] > max_val) max_val = cam[i];
    }
    if (max_val > 0.0f) {
        float inv_max = 1.0f / max_val;
        for (uint32_t i = 0; i < h * w; i++) {
            cam[i] *= inv_max;
        }
    }
}
