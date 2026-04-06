// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Gradient SHAP kernel for SNAX cluster.
// Computes SHAP attribution values by averaging gradient × (input - baseline)
// over N random baseline samples.
//
// Phase 2a: float32 on RISC-V scalar core (GeMM accelerator is INT8 only).
// Phase 2b (future): INT8 forward passes on GeMM accelerator with async dispatch.

#pragma once

#include <stdint.h>

#include "snrt.h"

// Interpolate between baseline and input:
//   out[i] = baseline[i] + alpha * (input[i] - baseline[i])
static inline void shap_interpolate(const float *input, const float *baseline,
                                    float alpha, float *out, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        out[i] = baseline[i] + alpha * (input[i] - baseline[i]);
    }
}

// Simple forward pass: GAP over spatial dims, then FC.
//   fmaps: (H*W, K) → pooled: (K,) → logits: (C,)
//   pooled[k] = (1/(H*W)) * sum_{ij} fmaps[ij*K + k]
//   logits[c] = sum_k pooled[k] * w_fc[k*C + c]
static inline void shap_forward_fc(const float *fmaps, const float *w_fc,
                                   float *logits, float *pooled_buf,
                                   uint32_t H, uint32_t W, uint32_t K,
                                   uint32_t C) {
    float inv_hw = 1.0f / (float)(H * W);

    // Global average pool
    for (uint32_t k = 0; k < K; k++) {
        float sum = 0.0f;
        for (uint32_t ij = 0; ij < H * W; ij++) {
            sum += fmaps[ij * K + k];
        }
        pooled_buf[k] = sum * inv_hw;
    }

    // FC layer
    for (uint32_t c = 0; c < C; c++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            sum += pooled_buf[k] * w_fc[k * C + c];
        }
        logits[c] = sum;
    }
}

// Backward pass for GAP+FC model.
// For target_class c:
//   d logit[c] / d fmaps[ij, k] = w_fc[k, c] / (H*W)
// Gradient is constant (linear model), output to grad buffer.
//   grad: (H*W*K,) — broadcast of w_fc[:, target_class] / (H*W)
static inline void shap_backward_fc(const float *w_fc, float *grad,
                                    uint32_t target_class, uint32_t H,
                                    uint32_t W, uint32_t K, uint32_t C) {
    float inv_hw = 1.0f / (float)(H * W);
    for (uint32_t ij = 0; ij < H * W; ij++) {
        for (uint32_t k = 0; k < K; k++) {
            grad[ij * K + k] = w_fc[k * C + target_class] * inv_hw;
        }
    }
}

// Accumulate SHAP attribution for one sample:
//   attr_accum[i] += (input[i] - baseline[i]) * grad[i]
static inline void shap_accumulate(const float *input, const float *baseline,
                                   const float *grad, float *attr_accum,
                                   uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        attr_accum[i] += (input[i] - baseline[i]) * grad[i];
    }
}

// Finalize: divide accumulated attributions by N_SAMPLES.
static inline void shap_normalize(float *attr, uint32_t size,
                                  uint32_t n_samples) {
    float inv_n = 1.0f / (float)n_samples;
    for (uint32_t i = 0; i < size; i++) {
        attr[i] *= inv_n;
    }
}

// Cycle breakdown counters for profiling.
typedef struct {
    uint32_t zero;        // zeroing accumulator
    uint32_t interpolate; // total across all samples
    uint32_t forward;     // total across all samples
    uint32_t backward;    // total across all samples
    uint32_t accumulate;  // total across all samples
    uint32_t normalize;   // final division
    uint32_t total;       // end-to-end
} shap_profile_t;

// Full Gradient SHAP computation (sequential, single core).
// Processes all N samples end-to-end.
//
// Args:
//   fmaps:      input feature maps, (H*W*K,) float32
//   w_fc:       FC weights, (K*C,) float32 row-major
//   baselines:  (N*H*W*K,) float32 — N baselines concatenated
//   alphas:     (N,) float32 — interpolation factors
//   attr_out:   (H*W*K,) float32 — output SHAP attributions
//   scratch:    workspace, >= (2*H*W*K + K + C) floats
//   profile:    if non-NULL, filled with per-stage cycle counts
//   H, W, K, C, N, target_class: dimensions
static inline void shap_gradient_full(
    const float *fmaps, const float *w_fc, const float *baselines,
    const float *alphas, float *attr_out, float *scratch, uint32_t H,
    uint32_t W, uint32_t K, uint32_t C, uint32_t N,
    uint32_t target_class, shap_profile_t *profile) {
    uint32_t spatial_size = H * W * K;

    // Partition scratch buffer
    float *interp_buf = scratch;              // H*W*K
    float *pooled_buf = scratch + spatial_size; // K
    float *logits_buf = pooled_buf + K;       // C
    float *grad_buf = logits_buf + C;         // H*W*K

    uint32_t t0, t1, t2, t3, t4, t5;
    uint32_t cyc_interp = 0, cyc_fwd = 0, cyc_bwd = 0, cyc_accum = 0;

    t0 = snrt_mcycle();

    // Zero output accumulator
    for (uint32_t i = 0; i < spatial_size; i++) {
        attr_out[i] = 0.0f;
    }

    t1 = snrt_mcycle();

    for (uint32_t n = 0; n < N; n++) {
        const float *baseline = baselines + n * spatial_size;
        float alpha = alphas[n];

        // Step 1: interpolate
        t2 = snrt_mcycle();
        shap_interpolate(fmaps, baseline, alpha, interp_buf, spatial_size);
        t3 = snrt_mcycle();
        cyc_interp += t3 - t2;

        // Step 2: forward pass (to validate — gradient is independent of input
        //         for linear model, but we compute it for generality)
        shap_forward_fc(interp_buf, w_fc, logits_buf, pooled_buf, H, W, K, C);
        t4 = snrt_mcycle();
        cyc_fwd += t4 - t3;

        // Step 3: backward pass
        shap_backward_fc(w_fc, grad_buf, target_class, H, W, K, C);
        t5 = snrt_mcycle();
        cyc_bwd += t5 - t4;

        // Step 4: accumulate attribution
        shap_accumulate(fmaps, baseline, grad_buf, attr_out, spatial_size);
        uint32_t t6 = snrt_mcycle();
        cyc_accum += t6 - t5;
    }

    uint32_t t_prenorm = snrt_mcycle();

    // Step 5: normalize by N
    shap_normalize(attr_out, spatial_size, N);

    uint32_t t_end = snrt_mcycle();

    if (profile) {
        profile->zero = t1 - t0;
        profile->interpolate = cyc_interp;
        profile->forward = cyc_fwd;
        profile->backward = cyc_bwd;
        profile->accumulate = cyc_accum;
        profile->normalize = t_end - t_prenorm;
        profile->total = t_end - t0;
    }
}
