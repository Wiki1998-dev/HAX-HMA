// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Grad-CAM kernel for SNAX cluster.
// Computes saliency map from last-conv feature maps and FC weights.
//
// All functions here run on the RISC-V compute core. The SNAX GeMM
// accelerator (int8, block-tiled) is not used yet; gradcam_backward_gemm_i8
// is a scalar-core INT8-weight variant, not an accelerator offload.

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

// ---------------------------------------------------------------------------
// INT8-weight backward GeMM (scalar core)
// ---------------------------------------------------------------------------
//
// Computes: grad[K] = d_logit[C] @ w_fc_i8^T[C x K] * scale
// w_fc_i8 is (K x C) row-major, symmetric per-tensor quantized
// (w_fc ~= w_fc_i8 * scale). Each product is fixed-point (Q7) truncated and
// accumulated in int32.
//
// An SSR+frep variant was removed: it used vfdotpex.s.b, which is an FP8
// (minifloat) dot product, so it misread int8 weights and fp32 d_logit; the
// rtl-generic runtime does not expose the SSR API and the toolchain targets
// generic rv32imafdzfh, so it also did not compile.

static inline void gradcam_backward_gemm_i8(
    const float *d_logit,
    const int8_t *w_fc_i8,
    float *grad,
    uint32_t C,
    uint32_t K,
    float scale)
{
    for (uint32_t k = 0; k < K; k++) {
        int32_t acc = 0;
        const int8_t *row = w_fc_i8 + k * C;
        for (uint32_t c = 0; c < C; c++) {
            acc += (int32_t)((float)row[c] * d_logit[c] * 128.0f);
        }
        grad[k] = ((float)acc / 128.0f) * scale;
    }
}
