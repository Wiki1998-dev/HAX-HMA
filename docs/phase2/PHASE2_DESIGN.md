# Phase 2 Design: Gradient SHAP Async Dispatch on SNAX

## Overview

Gradient SHAP computes feature attributions by averaging `gradient × (input - baseline)`
over N random baseline samples. Each sample requires an independent forward pass,
making SHAP naturally parallelizable on the SNAX multi-accelerator platform.

## Algorithm

```
For i = 1..N:
    x'_i = random baseline
    alpha_i ~ U(0, 1)
    x_interp = x'_i + alpha_i * (x - x'_i)
    grad_i = d f(x_interp) / d x         # requires forward + backward pass
    phi += (x - x'_i) * grad_i
phi /= N
```

## SNAX Dispatch Architecture

### Sequential Baseline (scalar core only)
```
┌─────────────────────────────────────────────────────┐
│ Core 0: for each sample i:                          │
│   1. Generate masked input (prepare x_interp)       │
│   2. Forward pass (GAP + FC matmul)                 │
│   3. Backward pass (gradient computation)           │
│   4. Accumulate attribution                         │
└─────────────────────────────────────────────────────┘
```

### Async Dispatch (overlap compute + GeMM accelerator)
```
┌──────────────────────────┐  ┌──────────────────────┐
│ Core 0 (scalar, float32) │  │ GeMM Acc (INT8)      │
│                          │  │                      │
│ Prepare mask[0]  ───────────→ Launch GeMM[0]      │
│ Prepare mask[1]          │  │ ▓▓▓ Computing[0] ▓▓▓│
│              ────────────────→ Launch GeMM[1]      │
│ Accumulate result[0] ←───── │ Done[0]              │
│ Prepare mask[2]          │  │ ▓▓▓ Computing[1] ▓▓▓│
│              ────────────────→ Launch GeMM[2]      │
│ Accumulate result[1] ←───── │ Done[1]              │
│ ...                      │  │ ...                  │
└──────────────────────────┘  └──────────────────────┘
```

### Double-Buffering Strategy
```
TCDM L1 (128 kB):
┌──────────────┬──────────────┬──────────────┬──────────┐
│ input_buf_A  │ input_buf_B  │ result_buf   │ weights  │
│ (interp[i])  │ (interp[i+1])│ (gradients)  │ (W_fc)   │
│ H*W*K*4B     │ H*W*K*4B     │ K*C*4B       │ K*C*4B   │
└──────────────┴──────────────┴──────────────┴──────────┘
```

While GeMM processes `input_buf_A`, scalar core writes `input_buf_B`.
After GeMM finishes, swap buffers.

## Current Implementation (Phase 2a): Float32 Scalar Core

Since the GeMM accelerator is INT8-only and our model uses float32 weights,
Phase 2a runs entirely on the RISC-V scalar core. This establishes:
1. Correct SHAP computation matching the Python reference
2. Baseline cycle count for comparison with accelerated version
3. Multi-core distribution (N samples across available compute cores)

### Phase 2a Kernel Structure (shap.h)
```c
// Step 0: Generate interpolated input for sample i
shap_interpolate(input, baseline, alpha, interp_buf, size);

// Step 1: Forward pass (GAP + FC)
shap_forward_fc(interp_buf, w_fc, logits, H, W, K, C);

// Step 2: Backward pass (gradient of target class w.r.t. input)
shap_backward_fc(w_fc, grad, target_class, H, W, K, C);

// Step 3: Accumulate attribution
shap_accumulate(input, baseline, grad, attr_accum, H, W, K);
```

### Cycle Budget: Initial Estimate vs Measured

**Initial estimate** assumed 1 cycle/FLOP (pipelined FPU):

| Step | FLOPs/sample | Est. cycles/sample | Est. Total (N=16) |
|------|-------------|-------------------|-------------|
| Interpolate | 3*H*W*K = 768 | ~800 | ~12,800 |
| Forward FC | H*W*K + K*C = 416 | ~500 | ~8,000 |
| Backward FC | K*C = 160 | ~200 | ~3,200 |
| Accumulate | 3*H*W*K = 768 | ~800 | ~12,800 |
| **Total** | | **~2,300** | **~36,800** |

**Measured: 182,297 cycles total** (4.95x over estimate).

The gap is explained by:
1. **Loop-carried dependencies**: Accumulation loops (`sum += ...`) create
   serial FP chains. The Snitch FPU has multi-cycle latency even with 1-cycle
   throughput, but dependent ops stall. Each accumulate iteration takes ~5-8
   cycles instead of 1.
2. **`fdiv.s` latency**: `inv_hw = 1.0f / (float)(H*W)` in both `forward_fc`
   and `backward_fc` costs ~25 cycles per call. With 2 divisions × 16 samples
   = 32 fdiv operations = ~800 cycles (minor but adds up).
3. **Loop overhead**: Branch, index increment, comparison add ~3-5 cycles per
   iteration. With 256 elements × 4 steps × 16 samples = ~16,384 iterations,
   this contributes ~50-80k cycles.
4. **Function call overhead**: Inline expansion helps, but the compiler may
   not fully eliminate call/return sequences for all helpers.

### Measured Per-Stage Breakdown (Verilator, 175,402 total cycles)

| Stage | Total cycles | Per-sample | % of Total |
|-------|-------------|-----------|-----------|
| Zero accumulator | 795 | — | 0.5% |
| Interpolate | 48,625 | 3,039 | 27.7% |
| Forward FC (GAP+matmul) | 40,201 | 2,512 | 22.9% |
| Backward FC (gradient) | 30,427 | 1,901 | 17.3% |
| Accumulate | 53,136 | 3,321 | 30.3% |
| Normalize (÷N) | 1,816 | — | 1.0% |
| **Total** | **175,402** | **10,963/sample** | |

**Profile**: No single bottleneck — cost is spread across memory-bound loops.
Interpolate + accumulate together = 58% (both are 256-element loops with
2 loads + FP op + store per element). The actual compute (forward+backward)
is only 40.2% of total.

**Optimization targets for Phase 2b**:
1. Multi-core: distribute N samples across 8 compute cores → ~22k cycles
2. Loop fusion: merge interpolate + accumulate into one pass per sample
3. Hoist backward: gradient is input-independent (linear model) — compute once
4. GeMM offload: forward FC matmul to accelerator (INT8 path)

## Phase 2b: INT8 Accelerated (Future)

When using a quantized INT8 model:
1. Forward pass becomes INT8 GeMM → offload to SNAX accelerator
2. Each of N=16 passes dispatched via CSR fire-and-forget
3. Gradient computed from INT32 accumulator output (bypassSIMD=1)
4. Expected speedup: ~50-100x on the forward pass (from 152x on conv)

### CSR Dispatch Pattern (from snax-gemmx-lib)
```c
for (int i = 0; i < N_SAMPLES; i++) {
    // Prepare masked input in buffer[i % 2]
    prepare_masked_input(buf[i % 2], ...);

    // Configure streamer for this sample
    set_gemmx_streamer_csr(..., delta_local_a[i % 2], ...);
    set_gemmx_csr(K, N, M, ...);

    // Fire!
    set_gemmx_streamer_start();
    set_gemmx_start();

    // While GeMM runs, prepare next mask
    if (i > 0) {
        // Collect result from previous sample
        accumulate_attribution(result_buf, ...);
    }

    // Wait for this sample
    wait_gemmx_and_streamer();
}
```

## Test Parameters
- Feature map: H=4, W=4, K=16 channels
- FC layer: K=16 → C=10 classes
- Target class: 3
- N_SAMPLES: 16 (start), scale to 32
- Tolerance: 1e-2 (relaxed due to floating-point accumulation over N samples)

## Verification
- BIST mode: compare C output against Python golden reference (datagen.py)
- Metric: per-element absolute error < tolerance for all H*W*K attributions
- Additional check: sum of attributions ≈ f(x) - E[f(baselines)] (completeness)
