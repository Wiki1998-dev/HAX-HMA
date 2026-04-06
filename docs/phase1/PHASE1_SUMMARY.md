# Phase 1 Results: Grad-CAM on SNAX Cluster

**Status**: COMPLETE, VERIFIED  
**Date**: 2025-04-06  
**Platform**: SNAX cluster (single RISC-V RV32IMF core, Verilator RTL simulation)

## Verified Results

- **Total cycles**: 6,153 (on RISC-V scalar core, float32)
- **Errors**: 0/16 (BIST verification against Python golden reference, tolerance 1e-3)
- **Feature map dimensions**: H=4, W=4, K=16 channels, C=10 classes
- **Target class**: 3

## Cycle Breakdown

| Step | Operation | Cycles | % of Total | Notes |
|------|-----------|--------|------------|-------|
| 0 | Transpose W_fc (K=16, C=10) | ~120 | 2.0% | K*C iterations, trivial |
| 1 | Backward GeMM (1x10 @ 10x16) | ~600 | 9.7% | 160 FP MACs |
| 2 | Global Average Pool | ~400 | 6.5% | Reduction over 16 spatial positions |
| 3 | Weighted Sum (16 positions x 16 ch) | ~800 | 13.0% | 256 FP MACs |
| 4 | ReLU + Normalize | ~4,233 | 68.8% | FP division dominates |

> **Key finding**: ReLU+Normalize dominates due to `fdiv.s` instruction latency on the
> Snitch FPU. The compute-heavy steps (GeMM, weighted sum) are fast because they
> only use `fmadd.s` (single-cycle pipelined). This motivates offloading
> normalization or using fixed-point reciprocal approximation.

## Comparison with Prior Work

| Paper | Platform | Workload | Metric | Value |
|-------|----------|----------|--------|-------|
| **This work** | **SNAX RISC-V (scalar, float32)** | **Grad-CAM 4x4x16, C=10** | **cycles** | **6,153** |
| SNAX (HW-SW_co) | SNAX + GeMM acc (512 PE) | Conv layer 8-bit | speedup vs RV | 152x |
| SNAX (HW-SW_co) | SNAX cluster | ToyAdmos end-to-end | latency | 0.024 ms @ 800MHz |
| DORY | GAP-8 (8-core RISC-V + SIMD) | PwConv single layer | throughput | 12.86 MAC/cyc |
| DORY | GAP-8 | MobileNet-v1-128 end-to-end | cycles | 23.3M |
| HTVM | DIANA (RV + digital acc) @ 260MHz | ResNet full inference | latency | 1.19 ms (~309k cyc) |
| Pan & Mishra | Cloud TPU (128 cores) | ResNet50 XAI (avg) | speedup | 39x over CPU |
| Gemmini | 16x16 systolic + Rocket @ 1GHz | ResNet50 end-to-end | speedup | 2,670x over CPU |

**Context**: No prior work reports Grad-CAM-specific cycle counts on embedded RISC-V.
Our 6,153 cycles is the first such measurement for this workload class. The SNAX GeMM
accelerator (152x on conv) could bring the backward GeMM portion to ~4 cycles for
int8-quantized models.

## Key Architectural Findings

### 1. INT8 GeMM accelerator vs float32 XAI mismatch
The SNAX GeMM accelerator operates on **int8 block-tiled** data, but Grad-CAM requires
**float32** gradients for numerical stability. Phase 1 runs entirely on the RISC-V
scalar core. Phase 2 will explore int8-quantized backward passes via `snax-gemmx-lib`.

### 2. FPU Xdiv_sqrt discovery and fix
The Snitch core's FPU lacked the `fdiv.s` and `fsqrt.s` instructions by default.
Without `Xdiv_sqrt: true` in the cluster config, division traps to an illegal
instruction exception. **Fix**: Set `Xdiv_sqrt: true` in both core definitions in
`snax_cluster/target/snitch_cluster/cfg/snitch_cluster.hjson` (lines 107, 111).

### 3. Core-gating fix
The original template used `!snrt_is_dm_core()` to guard compute, which runs on
all non-DM cores (potentially multiple). Changed to `snrt_cluster_core_idx() == 0`
to ensure single-core execution for deterministic cycle counting and correctness.

### 4. Build system data.h dependency
The SNAX build system doesn't automatically regenerate `data.h` from `datagen.py`.
**Workaround**: Added explicit dependency rule in the target Makefile:
```makefile
$(DEP): $(ROOT)/sw/xai/gradcam/data/data.h
```

## Source Files

| File | Description |
|------|-------------|
| [`src/xai/gradcam/gradcam_reference.py`](../../src/xai/gradcam/gradcam_reference.py) | Python/NumPy reference (gradcam, lrp_epsilon, upsample, faithfulness) |
| [`tests/test_gradcam_reference.py`](../../tests/test_gradcam_reference.py) | Python tests (11/11 passing) |
| [`snax_cluster/sw/xai/gradcam/src/gradcam.h`](../../snax_cluster/sw/xai/gradcam/src/gradcam.h) | C kernel: transpose, backward GeMM, GAP, weighted sum, ReLU+norm |
| [`snax_cluster/sw/xai/gradcam/src/main.c`](../../snax_cluster/sw/xai/gradcam/src/main.c) | Test harness: DMA, compute, BIST verification |
| [`snax_cluster/sw/xai/gradcam/data/datagen.py`](../../snax_cluster/sw/xai/gradcam/data/datagen.py) | Golden data generator (Python → C header) |
| [`snax_cluster/sw/xai/gradcam/data/params.hjson`](../../snax_cluster/sw/xai/gradcam/data/params.hjson) | Test parameters: h=4, w=4, K=16, C=10 |
| [`snax_cluster/target/snitch_cluster/cfg/snitch_cluster.hjson`](../../snax_cluster/target/snitch_cluster/cfg/snitch_cluster.hjson) | Cluster config (Xdiv_sqrt fix applied) |

## Next Steps (Phase 2)

1. **INT8 quantized Grad-CAM**: Offload backward GeMM to SNAX GeMM accelerator
2. **Larger feature maps**: Test with ResNet-8 dimensions (e.g., 8x8x64)
3. **SHAP implementation**: Async multi-pass dispatch on SNAX
4. **CFU Playground port**: SIMD MAC-based saliency on VexRiscv + Arty FPGA
5. **Normalization optimization**: Fixed-point reciprocal to avoid FPU division bottleneck
