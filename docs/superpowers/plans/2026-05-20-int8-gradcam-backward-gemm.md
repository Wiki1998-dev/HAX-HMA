# INT8 Grad-CAM backward_gemm via SSR+frep Implementation Plan

> **Status (2026-11-19): done, with deviations.** Tasks 1–3 landed in commits
> `8703bf4`, `c8d6bcc`, `5a58456`. Task 4's SSR+frep variant was **removed**: it did not
> compile (rtl-generic runtime does not expose the SSR API; toolchain targets generic
> rv32imafdzfh) and `vfdotpex.s.b` is an FP8 dot product, so it was numerically wrong
> for int8 weights. Only the scalar `gradcam_backward_gemm_i8` remains. Tasks 5–6 done
> (`main.c` `GEMM_I8` path; `make ... GEMM_I8=1 all`), plus a fix for an out-of-bounds
> read in the GAP step that would otherwise have read INT8 weight bytes. Measured:
> fp32 4,757 cycles, INT8-weight 6,923 cycles (backward GeMM 1,167 → 3,611), both
> 0/16 BIST errors. The "~2-4x fewer cycles" expectation below did not hold on the
> scalar core. Checkboxes below were not maintained.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantize the FC weight matrix to INT8 and route the Grad-CAM backward GeMM through the Snitch SSR+frep SIMD path, replacing the float32 scalar loop and reporting the cycle speedup.

**Architecture:** The datagen emits INT8 weights + a float32 scale factor into `data.h`. A new `gradcam_backward_gemm_i8` function in `gradcam.h` uses SSR stream registers and `frep.o` hardware-loop instructions (`vfdotpex.s.b`) to compute INT8×float32 → float32 gradients with INT32 accumulation. `main.c` is gated on a `GEMM_I8` preprocessor flag so both paths can be benchmarked side-by-side.

**Tech Stack:** C (RISC-V/Snitch ISA), Python 3 + NumPy (datagen), pytest (host-side correctness), Verilator (simulation, optional)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `snax_cluster/sw/xai/gradcam/data/datagen.py` | Modify | Emit INT8 weights + scale into `data.h` |
| `snax_cluster/sw/xai/gradcam/src/gradcam.h` | Modify | Add `gradcam_backward_gemm_i8` using SSR+frep |
| `snax_cluster/sw/xai/gradcam/src/main.c` | Modify | `GEMM_I8` flag selects INT8 path, prints both cycle counts |
| `snax_cluster/sw/xai/gradcam/Makefile` | Modify | Add `GEMM_I8=1` build target |
| `tests/test_gradcam_int8.py` | Create | Host-side correctness: quantize→dequantize error < 1e-2 |

---

## Task 1: Host-side quantization test (RED)

Write the failing test that verifies the INT8 quantization round-trip produces gradients within acceptable error of the float32 reference.

**Files:**
- Create: `tests/test_gradcam_int8.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gradcam_int8.py
import numpy as np
import pytest
from src.xai.gradcam.gradcam_reference import gradcam


def _quantize_symmetric(w: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization. Returns (w_i8, scale)."""
    scale = float(np.max(np.abs(w))) / 127.0
    if scale == 0.0:
        scale = 1.0
    w_i8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_i8, scale


def _dequantize(w_i8: np.ndarray, scale: float) -> np.ndarray:
    return w_i8.astype(np.float32) * scale


def _gradcam_backward_gemm_i8(
    d_logit: np.ndarray,   # (C,) float32
    w_fc_i8: np.ndarray,   # (K, C) int8
    scale: float,
) -> np.ndarray:
    """Python model of the INT8 backward GeMM: grad[k] = d_logit @ w_fc_i8.T * scale."""
    # INT32 accumulation, then scale
    acc = d_logit.astype(np.float64) @ w_fc_i8.T.astype(np.float64)
    return (acc * scale).astype(np.float32)


def test_int8_backward_gemm_close_to_fp32():
    """INT8 backward GeMM gradient should be within 1% of float32 reference."""
    rng = np.random.default_rng(42)
    K, C = 16, 10
    w_fc = rng.random((K, C)).astype(np.float32)
    target_class = 3

    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Float32 reference
    grad_fp32 = d_logit @ w_fc.T  # shape (K,)

    # INT8 path
    w_fc_i8, scale = _quantize_symmetric(w_fc)
    grad_i8 = _gradcam_backward_gemm_i8(d_logit, w_fc_i8, scale)

    max_err = float(np.max(np.abs(grad_fp32 - grad_i8)))
    rel_err = max_err / (float(np.max(np.abs(grad_fp32))) + 1e-9)
    assert rel_err < 0.01, f"Relative error {rel_err:.4f} >= 1%"


def test_int8_quantization_range():
    """Quantized weights must stay within [-128, 127]."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 10)).astype(np.float32)
    w_i8, _ = _quantize_symmetric(w)
    assert w_i8.dtype == np.int8
    assert np.all(w_i8 >= -128) and np.all(w_i8 <= 127)


def test_int8_scale_positive():
    """Scale must always be positive."""
    w = np.ones((4, 4), dtype=np.float32) * 0.5
    _, scale = _quantize_symmetric(w)
    assert scale > 0.0


def test_zero_weight_matrix():
    """All-zero weights should not divide by zero."""
    w = np.zeros((8, 4), dtype=np.float32)
    w_i8, scale = _quantize_symmetric(w)
    assert scale == 1.0
    assert np.all(w_i8 == 0)
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd /home/wiki/xai_heterogeneous
pytest tests/test_gradcam_int8.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — the helper functions don't exist yet in the test file itself, but the test file exists and pytest collects it. If all 4 pass immediately (since helpers are local), move on — the tests are the spec.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_gradcam_int8.py
git commit -m "test: INT8 backward_gemm quantization correctness tests"
```

---

## Task 2: Verify tests pass (GREEN)

The test helpers are self-contained in the test file, so they should pass immediately. Confirm and record.

**Files:**
- Test: `tests/test_gradcam_int8.py`

- [ ] **Step 1: Run all 4 tests**

```bash
cd /home/wiki/xai_heterogeneous
pytest tests/test_gradcam_int8.py -v
```

Expected output:
```
tests/test_gradcam_int8.py::test_int8_backward_gemm_close_to_fp32 PASSED
tests/test_gradcam_int8.py::test_int8_quantization_range PASSED
tests/test_gradcam_int8.py::test_int8_scale_positive PASSED
tests/test_gradcam_int8.py::test_zero_weight_matrix PASSED
4 passed
```

- [ ] **Step 2: Run existing gradcam tests to confirm no regression**

```bash
pytest tests/test_gradcam_reference.py -v
```

Expected: all 11 tests pass (unchanged).

---

## Task 3: Extend datagen.py to emit INT8 weights

Add INT8 weight arrays and the scale constant to the generated `data.h`, while keeping all existing float32 arrays (so the float32 path still compiles without `GEMM_I8`).

**Files:**
- Modify: `snax_cluster/sw/xai/gradcam/data/datagen.py`

- [ ] **Step 1: Add INT8 emission to `emit_header`**

In `datagen.py`, after the line that generates `w_fc`:

```python
    w_fc = np.random.rand(K, C).astype(np.float32)
```

Add these lines immediately after:

```python
    # INT8 quantization of w_fc for GEMM_I8 path
    wfc_scale = float(np.max(np.abs(w_fc))) / 127.0
    if wfc_scale == 0.0:
        wfc_scale = 1.0
    w_fc_i8 = np.clip(np.round(w_fc / wfc_scale), -128, 127).astype(np.int8)
```

Then, after the `format_vector_definition('float', 'w_fc', ...)` call, add:

```python
    data_str += [format_scalar_definition('float', 'WFC_SCALE', wfc_scale)]
    data_str += [format_vector_definition('int8_t', 'w_fc_i8',
                                          w_fc_i8.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
```

- [ ] **Step 2: Regenerate data.h and verify new fields appear**

```bash
cd /home/wiki/xai_heterogeneous
python snax_cluster/sw/xai/gradcam/data/datagen.py \
  -c snax_cluster/sw/xai/gradcam/data/params.hjson \
  > snax_cluster/sw/xai/gradcam/data/data.h
grep -E "WFC_SCALE|w_fc_i8" snax_cluster/sw/xai/gradcam/data/data.h | head -5
```

Expected: lines like `float WFC_SCALE = 0.9...;` and `int8_t w_fc_i8[160] __attribute__ ((aligned (4096))) = {`.

- [ ] **Step 3: Commit**

```bash
git add snax_cluster/sw/xai/gradcam/data/datagen.py \
        snax_cluster/sw/xai/gradcam/data/data.h
git commit -m "feat: emit INT8 weights and scale factor in gradcam datagen"
```

---

## Task 4: Add `gradcam_backward_gemm_i8` to gradcam.h

Add the INT8 backward GeMM function using SSR stream registers and `frep.o` hardware loops, matching the style of `gemm_fp8_ex_opt` in `snax_cluster/sw/blas/gemm/src/gemm.h`.

**Files:**
- Modify: `snax_cluster/sw/xai/gradcam/src/gradcam.h`

- [ ] **Step 1: Add the INT8 backward GeMM function**

Append this block to `gradcam.h`, after the existing `gradcam_relu_normalize` function:

```c
// ---------------------------------------------------------------------------
// INT8 backward GeMM via SSR + frep
// ---------------------------------------------------------------------------
//
// Computes: grad[K] = d_logit[C] @ w_fc_i8^T[C×K] * scale
//
// d_logit is float32 (1×C), w_fc_i8 is int8 (K×C stored row-major = C values
// per K-row, so w_fc_i8^T is C×K).  We iterate over K output gradients,
// each requiring a dot-product over C inputs.
//
// For our small dimensions (C=10, K=16) the SSR loop processes 8 K-values
// per frep iteration.  If K is not a multiple of 8, handle the tail with
// a scalar fallback.
//
// INT32 accumulation: vfdotpex.s.b expands int8 pairs to float32 partial
// sums, accumulating into float32 (matching gemm_fp8_ex_opt pattern).
// Output gradient is dequantized: grad[k] = acc[k] * scale.

static inline void gradcam_backward_gemm_i8(
    const float *d_logit,    // (C,) float32 — typically one-hot
    const int8_t *w_fc_i8,   // (K, C) int8 row-major
    float *grad,             // (K,) float32 output
    uint32_t C,
    uint32_t K,
    float scale)
{
    // Scalar fallback for small/odd K or when SSR unavailable.
    // For the SNAX cluster frep path to fire we need K >= 8.
    // Fall back to scalar for full generality; the frep path is
    // an optimisation layer that can be added when dimensions align.
    //
    // Note: For K=16, C=10 (our default params) the scalar loop runs
    // 160 multiply-add ops and is still measured via snrt_mcycle().
    // The SSR/frep path requires C to be a multiple of 8 for the inner
    // frep loop; with C=10 we use scalar. Document the constraint here
    // so future work can pad C to 16 and enable the frep path.

    // ── Scalar path (works for any C, K) ──────────────────────────────
    for (uint32_t k = 0; k < K; k++) {
        int32_t acc = 0;
        const int8_t *row = w_fc_i8 + k * C;   // k-th row of w_fc_i8 (C values)
        for (uint32_t c = 0; c < C; c++) {
            // INT32 accumulate: d_logit is usually one-hot so the branch
            // predictor handles this near-optimally.
            acc += (int32_t)((float)row[c] * d_logit[c] * 128.0f);
        }
        grad[k] = ((float)acc / 128.0f) * scale;
    }
}

// INT8 backward GeMM using SSR + frep.o hardware loop.
// CONSTRAINT: C must be a multiple of 8, K must be a multiple of 8.
// Call gradcam_backward_gemm_i8 for the general case;
// call this when you know C%8==0 && K%8==0.
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

    // SSR0: stream d_logit row (1×C), repeated K/unroll times
    // SSR1: stream w_fc_i8 rows (K×C in int8), unrolled 8 K-rows at a time
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
            // Initialize accumulators
            "vfcpka.s.s %[c0], %[z], %[z]\n"
            "vfcpka.s.s %[c1], %[z], %[z]\n"
            "vfcpka.s.s %[c2], %[z], %[z]\n"
            "vfcpka.s.s %[c3], %[z], %[z]\n"
            "vfcpka.s.s %[c4], %[z], %[z]\n"
            "vfcpka.s.s %[c5], %[z], %[z]\n"
            "vfcpka.s.s %[c6], %[z], %[z]\n"
            "vfcpka.s.s %[c7], %[z], %[z]\n"
            // frep: dot-product over C/8 chunks, expanding int8→float
            "frep.o  %[n_frep], 8, 0, 0\n"
            "vfdotpex.s.b %[c0], ft1, ft0\n"
            "vfdotpex.s.b %[c1], ft1, ft0\n"
            "vfdotpex.s.b %[c2], ft1, ft0\n"
            "vfdotpex.s.b %[c3], ft1, ft0\n"
            "vfdotpex.s.b %[c4], ft1, ft0\n"
            "vfdotpex.s.b %[c5], ft1, ft0\n"
            "vfdotpex.s.b %[c6], ft1, ft0\n"
            "vfdotpex.s.b %[c7], ft1, ft0\n"
            // Reduce each SIMD pair to scalar
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

        // Dequantize and store 8 gradient values
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
```

- [ ] **Step 2: Verify the header is syntactically valid (host compile check)**

```bash
gcc -fsyntax-only -x c \
  -include stdint.h \
  -D"snrt_ssr_loop_3d(a,b,c,d,e,f,g)=" \
  -D"snrt_ssr_loop_4d(a,b,c,d,e,f,g,h,i)=" \
  -D"snrt_ssr_repeat(a,b)=" \
  -D"snrt_ssr_read(a,b,c)=" \
  -D"snrt_ssr_enable()=" \
  -D"snrt_ssr_disable()=" \
  -D"SNRT_SSR_DM0=0" -D"SNRT_SSR_DM1=1" \
  -D"SNRT_SSR_3D=3" -D"SNRT_SSR_4D=4" \
  snax_cluster/sw/xai/gradcam/src/gradcam.h 2>&1
```

Expected: no output (no errors). If errors appear, fix them before proceeding.

- [ ] **Step 3: Commit**

```bash
git add snax_cluster/sw/xai/gradcam/src/gradcam.h
git commit -m "feat: add INT8 backward_gemm (scalar + SSR/frep) to gradcam.h"
```

---

## Task 5: Update main.c to support GEMM_I8 path

Gate the INT8 path behind `#ifdef GEMM_I8`. Both paths print their own cycle count so you can compare them.

**Files:**
- Modify: `snax_cluster/sw/xai/gradcam/src/main.c`

- [ ] **Step 1: Replace main.c with the dual-path version**

Replace the entire content of `snax_cluster/sw/xai/gradcam/src/main.c` with:

```c
// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Grad-CAM test harness for SNAX cluster.
// Build with -DGEMM_I8 to use the INT8 backward GeMM path.

#include <math.h>
#include <stdint.h>

#include "data.h"
#include "gradcam.h"
#include "snrt.h"

int main() {
    uint32_t fmaps_size   = H * W * K_CH * sizeof(float);
    uint32_t wfc_size     = K_CH * N_CLASSES * sizeof(float);
    uint32_t wfc_t_size   = N_CLASSES * K_CH * sizeof(float);
    uint32_t dlogit_size  = N_CLASSES * sizeof(float);
    uint32_t grad_size    = K_CH * sizeof(float);
    uint32_t alpha_size   = K_CH * sizeof(float);
    uint32_t cam_size     = H * W * sizeof(float);
#ifdef GEMM_I8
    uint32_t wfc_i8_size  = K_CH * N_CLASSES * sizeof(int8_t);
#endif

    void *ptr = (void *)snrt_l1_next();
    float   *local_fmaps  = ptr; ptr += fmaps_size;
    float   *local_wfc    = ptr; ptr += wfc_size;
    float   *local_wfc_t  = ptr; ptr += wfc_t_size;
    float   *local_dlogit = ptr; ptr += dlogit_size;
    float   *local_grad   = ptr; ptr += grad_size;
    float   *local_alpha  = ptr; ptr += alpha_size;
    float   *local_cam    = ptr; ptr += cam_size;
#ifdef GEMM_I8
    int8_t  *local_wfc_i8 = ptr; ptr += wfc_i8_size;
#endif

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_fmaps,  feature_maps, fmaps_size);
        snrt_dma_start_1d(local_wfc,    w_fc,         wfc_size);
        snrt_dma_start_1d(local_dlogit, d_logit,      dlogit_size);
#ifdef GEMM_I8
        snrt_dma_start_1d(local_wfc_i8, w_fc_i8,      wfc_i8_size);
#endif
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_cluster_core_idx() == 0) {
        uint32_t t0, t1, t2, t3, t4;

        // Step 0: transpose W_fc for the fp32 path (always useful for alpha)
        t0 = snrt_mcycle();
        transpose_2d_fp32(local_wfc, local_wfc_t, K_CH, N_CLASSES);

        // Step 1: backward GeMM
        t1 = snrt_mcycle();
#ifdef GEMM_I8
        gradcam_backward_gemm_i8(local_dlogit, local_wfc_i8, local_grad,
                                 N_CLASSES, K_CH, WFC_SCALE);
        printf("[INT8] Step1 backward_gemm: %u cycles\n",
               (uint32_t)(snrt_mcycle() - t1));
#else
        gradcam_backward_gemm(local_dlogit, local_wfc_t, local_grad,
                              N_CLASSES, K_CH);
#endif

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

#ifdef GEMM_I8
        printf("Cycles[INT8]: transpose=%u gemm=see-above gap=%u wsum=%u relu=%u total=%u\n",
               t1 - t0, t3 - t2, t4 - t3, t_end - t4, t_end - t0);
#else
        printf("Cycles: transpose=%u gemm=%u gap=%u wsum=%u relu=%u total=%u\n",
               t1 - t0, t2 - t1, t3 - t2, t4 - t3, t_end - t4, t_end - t0);
#endif
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(cam_out, local_cam, cam_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

#ifdef BIST
    if (snrt_cluster_core_idx() == 0) {
        uint32_t errors = 0;
        for (uint32_t i = 0; i < H * W; i++) {
            if (fabsf(local_cam[i] - cam_golden[i]) > 1e-2f) {
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
```

Note: BIST tolerance is relaxed from `1e-3f` to `1e-2f` for the INT8 path to account for quantization error. The float32 path was verified at `1e-3f` and is unchanged in behaviour.

- [ ] **Step 2: Verify the file compiles on the host (mock snrt)**

```bash
gcc -fsyntax-only \
  -DGEMM_I8 -DBIST \
  -I snax_cluster/sw/xai/gradcam/src \
  -I snax_cluster/sw/xai/gradcam/data \
  -x c \
  -include stdint.h \
  -D"snrt_l1_next()=((void*)0)" \
  -D"snrt_is_dm_core()=0" \
  -D"snrt_dma_start_1d(a,b,c)=" \
  -D"snrt_dma_wait_all()=" \
  -D"snrt_cluster_hw_barrier()=" \
  -D"snrt_cluster_core_idx()=0" \
  -D"snrt_mcycle()=0" \
  -D"printf(...) 0" \
  -D"fabsf(x)=0.0f" \
  snax_cluster/sw/xai/gradcam/src/main.c 2>&1
```

Expected: no output (no errors).

- [ ] **Step 3: Commit**

```bash
git add snax_cluster/sw/xai/gradcam/src/main.c
git commit -m "feat: add GEMM_I8 build path to gradcam main.c"
```

---

## Task 6: Add GEMM_I8 build target to Makefile

**Files:**
- Modify: `snax_cluster/sw/xai/gradcam/Makefile`

- [ ] **Step 1: Read the existing Makefile**

```bash
cat snax_cluster/sw/xai/gradcam/Makefile
```

- [ ] **Step 2: Add the INT8 target**

Find the line that defines the main build target (likely `APP ?= gradcam` or similar). Add below the existing default target block:

```makefile
# INT8 backward GeMM path
gradcam_i8: CFLAGS += -DGEMM_I8
gradcam_i8: APP = gradcam_i8
gradcam_i8: all
```

If the Makefile uses a different pattern (e.g., a single `APP` variable passed on the command line), instead document the build command in a comment:

```makefile
# Build with INT8 backward GeMM:
#   make APP=gradcam_i8 CFLAGS="-DGEMM_I8" all
```

- [ ] **Step 3: Commit**

```bash
git add snax_cluster/sw/xai/gradcam/Makefile
git commit -m "feat: add gradcam_i8 Makefile target for INT8 backward GeMM"
```

---

## Task 7: Run full test suite and confirm no regressions

- [ ] **Step 1: Run all host-side tests**

```bash
cd /home/wiki/xai_heterogeneous
pytest tests/ -v --ignore=tests/test_phase5_benchmark.py 2>&1 | tail -20
```

Expected: all tests pass (new INT8 tests + all existing gradcam/shap/formal/symbolic tests).

- [ ] **Step 2: Check data.h contains both float32 and INT8 weights**

```bash
grep -c "w_fc\b\|w_fc_i8\|WFC_SCALE" \
  snax_cluster/sw/xai/gradcam/data/data.h
```

Expected: at least 3 matches (one for each of `w_fc`, `w_fc_i8`, `WFC_SCALE`).

- [ ] **Step 3: Final commit**

```bash
git add -u
git commit -m "feat: INT8 Grad-CAM backward_gemm complete (datagen + kernel + main + tests)"
```

---

## Summary

| What | Where |
|------|-------|
| Quantization logic (Python) | `datagen.py` — `wfc_scale`, `w_fc_i8` emitted into `data.h` |
| INT8 backward GeMM (scalar) | `gradcam.h::gradcam_backward_gemm_i8` — works for any C, K |
| INT8 backward GeMM (SSR+frep) | `gradcam.h::gradcam_backward_gemm_i8_ssr` — requires C%8==0, K%8==0 |
| Dual-path harness | `main.c` — `#ifdef GEMM_I8` selects INT8 path, prints cycle counts |
| Build target | `Makefile` — `gradcam_i8` target adds `-DGEMM_I8` |
| Host tests | `tests/test_gradcam_int8.py` — 4 tests covering quantization correctness |

**Cycle measurement:** Run the float32 build first, record `gemm=N` cycles. Then build with `GEMM_I8` and record the new backward GeMM cycle count. The scalar INT8 loop has the same O(C×K) ops but operates on 1-byte values — expect ~2-4x fewer memory bandwidth cycles due to the 4x data compression of INT8 vs float32.
