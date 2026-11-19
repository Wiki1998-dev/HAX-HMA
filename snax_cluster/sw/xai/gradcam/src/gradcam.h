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

// ---------------------------------------------------------------------------
// INT8 backward GeMM via SSR + frep
// ---------------------------------------------------------------------------
//
// Computes: grad[K] = d_logit[C] @ w_fc_i8^T[C x K] * scale
//
// For C=10, K=16 (default params), uses scalar path since C%8 != 0.
// The _ssr variant requires C%8==0 and K%8==0 for the frep loop.

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

// SSR+frep variant: requires C%8==0 and K%8==0.
static inline void gradcam_backward_gemm_i8_ssr(
    const float *d_logit,
    const int8_t *w_fc_i8,
    float *grad,
    uint32_t C,
    uint32_t K,
    float scale)
{
    typedef float v2f32 __attribute__((vector_size(8)));
    const uint32_t unroll = 8;
    const uint32_t n_frep = C / 8 - 1;

    {
        uint32_t ssr0_b[3] = {unroll, C / 8, K / unroll};
        uint32_t ssr0_i[3] = {0, sizeof(float) * 8, 0};
        snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[0], ssr0_b[1], ssr0_b[2],
                         ssr0_i[0], ssr0_i[1], ssr0_i[2]);
        snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
    }
    {
        uint32_t ssr1_b[4] = {unroll, C / 8, K / unroll, 1};
        uint32_t ssr1_i[4] = {sizeof(int8_t) * C,
                              sizeof(int8_t) * 8,
                              sizeof(int8_t) * C * unroll,
                              0};
        snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2],
                         ssr1_b[3], ssr1_i[0], ssr1_i[1], ssr1_i[2],
                         ssr1_i[3]);
    }

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, d_logit);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, w_fc_i8);
    snrt_ssr_enable();

    const register float zero = 0.0f;

    for (uint32_t k0 = 0; k0 < K / unroll; k0++) {
        v2f32 c0, c1, c2, c3, c4, c5, c6, c7;
        v2f32 r0, r1, r2, r3, r4, r5, r6, r7;

        asm volatile(
            "vfcpka.s.s %[c0], %[z], %[z]\n"
            "vfcpka.s.s %[c1], %[z], %[z]\n"
            "vfcpka.s.s %[c2], %[z], %[z]\n"
            "vfcpka.s.s %[c3], %[z], %[z]\n"
            "vfcpka.s.s %[c4], %[z], %[z]\n"
            "vfcpka.s.s %[c5], %[z], %[z]\n"
            "vfcpka.s.s %[c6], %[z], %[z]\n"
            "vfcpka.s.s %[c7], %[z], %[z]\n"
            "frep.o  %[n_frep], 8, 0, 0\n"
            "vfdotpex.s.b %[c0], ft1, ft0\n"
            "vfdotpex.s.b %[c1], ft1, ft0\n"
            "vfdotpex.s.b %[c2], ft1, ft0\n"
            "vfdotpex.s.b %[c3], ft1, ft0\n"
            "vfdotpex.s.b %[c4], ft1, ft0\n"
            "vfdotpex.s.b %[c5], ft1, ft0\n"
            "vfdotpex.s.b %[c6], ft1, ft0\n"
            "vfdotpex.s.b %[c7], ft1, ft0\n"
            "vfcpka.s.s %[r0], %[z], %[z]\n"
            "vfcpka.s.s %[r1], %[z], %[z]\n"
            "vfcpka.s.s %[r2], %[z], %[z]\n"
            "vfcpka.s.s %[r3], %[z], %[z]\n"
            "vfcpka.s.s %[r4], %[z], %[z]\n"
            "vfcpka.s.s %[r5], %[z], %[z]\n"
            "vfcpka.s.s %[r6], %[z], %[z]\n"
            "vfcpka.s.s %[r7], %[z], %[z]\n"
            "vfsum.s %[r0], %[c0]\n"
            "vfsum.s %[r1], %[c1]\n"
            "vfsum.s %[r2], %[c2]\n"
            "vfsum.s %[r3], %[c3]\n"
            "vfsum.s %[r4], %[c4]\n"
            "vfsum.s %[r5], %[c5]\n"
            "vfsum.s %[r6], %[c6]\n"
            "vfsum.s %[r7], %[c7]\n"
            : [c0]"+f"(c0),[c1]"+f"(c1),[c2]"+f"(c2),[c3]"+f"(c3),
              [c4]"+f"(c4),[c5]"+f"(c5),[c6]"+f"(c6),[c7]"+f"(c7),
              [r0]"+f"(r0),[r1]"+f"(r1),[r2]"+f"(r2),[r3]"+f"(r3),
              [r4]"+f"(r4),[r5]"+f"(r5),[r6]"+f"(r6),[r7]"+f"(r7)
            : [n_frep]"r"(n_frep), [z]"f"(zero)
            : "ft0", "ft1", "ft2");

        uint32_t base = k0 * unroll;
        grad[base + 0] = ((float*)&r0)[0] * scale;
        grad[base + 1] = ((float*)&r1)[0] * scale;
        grad[base + 2] = ((float*)&r2)[0] * scale;
        grad[base + 3] = ((float*)&r3)[0] * scale;
        grad[base + 4] = ((float*)&r4)[0] * scale;
        grad[base + 5] = ((float*)&r5)[0] * scale;
        grad[base + 6] = ((float*)&r6)[0] * scale;
        grad[base + 7] = ((float*)&r7)[0] * scale;
    }

    snrt_ssr_disable();
}
