# Changelog

## Phase 2: Gradient SHAP on SNAX Cluster (2025-04)

### Python Reference Implementation
- Implemented `gradient_shap()`, `gradient_shap_analytical()`, `expected_gradients()`,
  `shap_interaction_values()` in `src/xai/shap/gradient_shap_reference.py`
- All 13 pytest tests passing (`tests/test_shap_reference.py`)
- Test classes: TestGradientSHAP (6), TestAnalyticalSHAP (3), TestExpectedGradients (4)
- Verified SHAP completeness axiom: sum(attr) ≈ f(x) - E[f(baselines)]

### SNAX C Kernel Development
- Created `snax_cluster/sw/xai/shap/src/shap.h` — header-only kernel library:
  - `shap_interpolate()` — baseline-input interpolation
  - `shap_forward_fc()` — GAP + FC forward pass
  - `shap_backward_fc()` — gradient computation (constant for linear model)
  - `shap_accumulate()` — per-sample attribution accumulation
  - `shap_normalize()` — final division by N
  - `shap_gradient_full()` — orchestrator with per-stage cycle profiling
- Created `snax_cluster/sw/xai/shap/src/main.c` — test harness with DMA,
  per-stage cycle breakdown printing, and BIST verification
- Created `snax_cluster/sw/xai/shap/data/datagen.py` — generates baselines,
  alphas, and golden SHAP output
- Parameters: h=4, w=4, K=16, C=10, n_samples=16, tolerance=1e-2

### Verification Results
- **Total cycles**: 182,297 on RISC-V scalar core (float32, N=16 samples)
- **Per-sample**: ~11,394 cycles average
- **5x over initial estimate** — explained by loop-carried FP dependencies,
  fdiv.s latency, and loop overhead (see PHASE2_DESIGN.md)
- **BIST**: pending per-stage breakdown from instrumented kernel

### Design Documentation
- `docs/phase2/PHASE2_DESIGN.md`: async dispatch architecture, double-buffering
  strategy, TCDM memory layout, CSR fire-and-forget pattern

---

## Phase 1: Grad-CAM on SNAX Cluster (2025-03 to 2025-04)

### Python Reference Implementation
- Implemented `gradcam()`, `lrp_epsilon()`, `upsample_saliency()`, `faithfulness_score()`
  in `src/xai/gradcam/gradcam_reference.py`
- All 11 pytest tests passing (`tests/test_gradcam_reference.py`)
- Test classes: TestGradCAM (5), TestLRP (2), TestUpsample (2), TestFaithfulness (2)

### SNAX C Kernel Development
- Created `snax_cluster/sw/xai/gradcam/src/gradcam.h` — header-only kernel library:
  - `transpose_2d_fp32()` — row-major matrix transpose
  - `gradcam_backward_gemm()` — gradient via transpose-GeMM (float32 on RISC-V)
  - `gradcam_gap_weights()` — global average pooling of gradients
  - `gradcam_weighted_sum()` — weighted combination of feature maps
  - `gradcam_relu_normalize()` — ReLU + normalize to [0,1]
- Created `snax_cluster/sw/xai/gradcam/src/main.c` — test harness with DMA
  transfers, cycle counting per step, and BIST verification mode
- Created `snax_cluster/sw/xai/gradcam/data/datagen.py` — generates `data.h` with
  random test inputs and golden Grad-CAM output from Python reference
- Created `snax_cluster/sw/xai/gradcam/data/params.hjson` — h=4, w=4, K=16, C=10
- Created Makefiles for kernel build and target integration

### Verilator RTL Simulation
- Full Verilator build of SNAX Snitch cluster (~66 minutes)
- Simulation target: `snax_cluster/target/snitch_cluster/`

### Bug Fixes and Patches

#### FPU Xdiv_sqrt discovery and fix
- **Problem**: `fdiv.s` instruction trapped as illegal — Snitch FPU extension not enabled
- **Fix**: Set `Xdiv_sqrt: true` for both core entries in
  `snax_cluster/target/snitch_cluster/cfg/snitch_cluster.hjson` (lines 107, 111)
- **Impact**: Without this, any floating-point division or square root crashes the program

#### Core-gating fix
- **Problem**: `!snrt_is_dm_core()` guard ran compute on all non-DM cores
- **Fix**: Changed to `snrt_cluster_core_idx() == 0` for single-core execution
- **Impact**: Ensures deterministic cycle counting and avoids race conditions on shared buffers

#### Build system data.h dependency
- **Problem**: `data.h` not regenerated when `datagen.py` or `params.hjson` change
- **Fix**: Added explicit dependency in target Makefile:
  `$(DEP): $(ROOT)/sw/xai/gradcam/data/data.h`

### Verification Results
- **BIST**: 0/16 errors (all 16 spatial positions match golden reference within 1e-3)
- **Total cycles**: 6,153 on RISC-V scalar core (float32)
- **Cycle breakdown**: transpose=~120, gemm=~600, gap=~400, wsum=~800, relu_norm=~4233
- **Key insight**: FPU division in normalize step dominates (68.8% of total)
