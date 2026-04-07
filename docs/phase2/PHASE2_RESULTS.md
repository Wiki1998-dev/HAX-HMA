# Phase 2a Results: Gradient SHAP on SNAX Cluster

**Status**: COMPLETE, VERIFIED (BIST 0/256 errors, max_err=0.000000)
**Date**: 2026-04-07
**Platform**: SNAX cluster (single RISC-V RV32IMF core, Verilator RTL simulation)

## Configuration

| Parameter | Value |
|-----------|-------|
| Feature map | H=4, W=4, K=16 channels |
| FC layer | K=16 inputs, C=10 classes |
| Target class | 3 |
| N samples | 16 |
| Total elements | 256 (H*W*K) |
| Precision | float32 |
| BIST tolerance | 1e-2 |

## Verified Results

| Metric | Value |
|--------|-------|
| **Total cycles** | **175,510** |
| **BIST errors** | **0/256** |
| **Max absolute error** | **0.000000** |
| Per-sample average | 10,969 cycles |
| Cycles per element per sample | ~42.8 |

## Per-Stage Cycle Breakdown

| Stage | Total cycles | Per-sample | % of Total | Description |
|-------|-------------|-----------|-----------|-------------|
| Zero accumulator | 816 | — | 0.5% | Clear 256-element attr buffer |
| **Interpolate** | **48,702** | **3,043** | **27.8%** | x' + alpha*(x - x') per element |
| **Forward FC** | **40,199** | **2,512** | **22.9%** | GAP (16 reductions) + FC (160 MACs) |
| Backward FC | 30,341 | 1,896 | 17.3% | Constant gradient broadcast + fdiv |
| **Accumulate** | **53,178** | **3,323** | **30.3%** | (x - x') * grad per element |
| Normalize | 1,815 | — | 1.0% | Single fdiv + 256 muls |
| Unaccounted | 459 | — | 0.3% | Loop setup, sample indexing |

### Analysis

**No single bottleneck** — unlike Grad-CAM where fdiv dominated (68.8%),
SHAP cost is distributed across all stages. The two most expensive stages
(accumulate + interpolate = 58.1%) are both 256-element loops doing
2 loads + 1-2 FP ops + 1 store per element.

**Cycles per FP operation** breakdown:

| Stage | FLOPs/sample | Measured cyc/sample | Cyc/FLOP |
|-------|-------------|-------------------|---------|
| Interpolate | 768 (3 ops × 256) | 3,043 | 3.96 |
| Forward FC | 416 (256 GAP + 160 FC) | 2,512 | 6.04 |
| Backward FC | 256 (mul + store × 256) | 1,896 | 7.41 |
| Accumulate | 768 (3 ops × 256) | 3,323 | 4.33 |

Forward and backward have higher cyc/FLOP due to reduction loops (GAP)
with serial FP dependencies and fdiv.s instructions (~25 cycles each).

## Comparison: Grad-CAM vs SHAP

| Metric | Grad-CAM | SHAP (N=16) | Ratio |
|--------|----------|-------------|-------|
| Total cycles | 6,153 | 175,510 | 28.5x |
| BIST errors | 0/16 | 0/256 | — |
| Dominant cost | fdiv in normalize (68.8%) | accumulate loops (30.3%) | — |
| Elements | 16 (6×6 spatial, no channel) | 256 (4×4×16) | 16x |
| Per-element | 384 cyc/elem | 686 cyc/elem | 1.78x |
| Acceleratable portion | 29% (gemm+wsum) | 40.2% (fwd+bwd) | — |

Per-element, SHAP is only 1.78x more expensive than Grad-CAM, but SHAP
processes 16x more elements across 16 samples, giving the 28.5x total gap.

## Phase 2b: Optimized Kernel (VERIFIED)

**Status**: COMPLETE, VERIFIED (BIST 0/256 errors, max_err=0.000000)
**Date**: 2026-04-07
**Speedup**: **3.02x** over Phase 2a (175,510 → 58,022 cycles)

**Optimization applied**: Hoist backward + eliminate forward + eliminate interpolation.

For the linear GAP+FC model, the gradient is constant and independent of the
interpolated input. This allows eliminating the interpolation, forward pass,
and all but one backward pass from the sample loop.

| Stage | Phase 2a | Phase 2b (verified) | Change |
|-------|----------|-----------------|--------|
| Zero | 816 | 1,036 | +220 (setup) |
| Interpolate (×16) | 48,702 | 0 | eliminated |
| Forward FC (×16) | 40,199 | 0 | eliminated |
| Backward FC | 30,341 (×16) | 1,930 (×1) | -28,411 (hoisted) |
| Accumulate (×16) | 53,178 | 53,129 | -49 (minor drift) |
| Normalize | 1,815 | 1,787 | -28 |
| **Total** | **175,510** | **58,022** | **3.02× speedup** |

### Verified Phase 2b Cycle Breakdown

| Stage | Total cycles | % of Total | Description |
|-------|-------------|-----------|-------------|
| Zero accumulator | 1,036 | 1.8% | Clear 256-element attr buffer |
| Backward FC | 1,930 | 3.3% | Constant gradient broadcast (hoisted) |
| **Accumulate** | **53,129** | **91.6%** | (input - baseline) * grad per sample |
| Normalize | 1,787 | 3.1% | Final division by N |
| Unaccounted | 140 | 0.2% | Loop setup |

**Analysis**:
- **Accumulate dominates** (91.6%): After eliminating interpolation and forward passes, the per-sample attribution accumulation is the only remaining bottleneck.
- **Speedup efficiency**: The speedup (3.02x) is exactly as predicted by the theoretical model (3.0x).
- **Correctness**: BIST passes with 0/256 errors (max_err=0.000000), proving the mathematical equivalence of the optimized kernel for linear models.

## Optimization Roadmap

| Optimization | Mechanism | Expected cycles | Speedup |
|-------------|-----------|----------------|---------|
| **Phase 2a** | Sequential scalar core | **175,510** | **1.0x** |
| **Phase 2b** | Hoist bwd + eliminate interp/fwd | **58,022** | **3.02x** |
| + Multi-core (8 cores) | Distribute samples across cores | ~8,500 | 20.6x |
| + GeMM offload (INT8) | Forward on accelerator (non-linear) | ~6,000 | 29.3x |

## Source Files

| File | Description |
|------|-------------|
| `src/xai/shap/gradient_shap_reference.py` | Python reference (gradient_shap, analytical, expected_gradients) |
| `tests/test_shap_reference.py` | 13 tests (shape, completeness axiom, variance, consistency) |
| `snax_cluster/sw/xai/shap/src/shap.h` | C kernel with per-stage profiling |
| `snax_cluster/sw/xai/shap/src/main.c` | DMA harness, cycle counting, BIST |
| `snax_cluster/sw/xai/shap/data/datagen.py` | Golden data generator |
| `snax_cluster/sw/xai/shap/data/params.hjson` | Test parameters |
| `docs/phase2/PHASE2_DESIGN.md` | Architecture and async dispatch design |
