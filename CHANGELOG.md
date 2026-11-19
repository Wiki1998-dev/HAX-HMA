# Changelog

## 2026-11-19: Reviewer experiments

### Added
- `docs/REVIEW_EXPERIMENTS.md`: the 15 reviewer items, each with command, seeds and raw output
  (`docs/review_raw/`), plus a summary table.
  Harness `scripts/review_experiments.py`; ToyAdmos forward-vs-backward diagnosis `scripts/review_ta_diag.py`.
- ToyAdmos 8-sample IG / SHAP programs (`PROG=toyadmos_ig_n8|toyadmos_shap_n8`).
  Measured: 60,689 cycles, 66% of N=16. The 16-sample output is byte-identical.

### Found
- ResNet-8 on all 10,000 CIFAR-10 test images: GeMMX 0.8704 vs TFLite 0.8702.
- ToyAdmos SHAP r is 0.707 with the current code (docs said 0.700).
- ResNet-8 inference launches 13.77M MACs.
- ToyAdmos attribution error comes from the int8 forward pass; int16 gradients do not help.

## 2026-11-19: Keyword spotting and visual wake words on GeMMX

### Added
- `snax_cluster/sw/xai/mobile_gemmx/`: MLPerf Tiny DS-CNN (KWS) and MobileNetV1 (VWW) inference + fused
  Grad-CAM, bit-exact in Verilator.
  - KWS: 48,024 + 5,787 cycles (+12.0%).
  - VWW: 188,250 + 1,538 cycles (+0.8%).
  - Depthwise convs run as diagonal-block conv launches.
- `scripts/kws_vww_data.py` (Speech Commands MFCC in NumPy, COCO vww01 images), `scripts/tflite_eval.py`,
  `scripts/mobile_gradcam_fidelity.py`.
- Accuracy: KWS 0.9180 (TFLite 0.9180), VWW 0.848 (TFLite 0.850).
- Grad-CAM r vs float32: 0.994 / 0.997.

## 2026-11-19: ResNet-8 input-level Integrated Gradients and Gradient SHAP on GeMMX

### Added
- `resnet8_xai.py`: integer backward pass in GeMMX semantics, plus a float32 reference
  (PyTorch autograd); static calibration.
- `resnet8_xai_sched.py`: schedule with interpolation, forward with L3 activation spills,
  rotated-kernel stride-1 and polyphase stride-2 transposed convs, DMA interleaves,
  block-diagonal fork sums, chunked interior ReLU masks and attribution MAC.
  Emulated bit-exactly.
- `resnet8_xai_run.c`.
- Measured, bit-exact: IG and Gradient SHAP 6,027,558 cycles each (16 samples,
  inference fused).
- Fidelity vs float32 over 50 images: r 0.946 (IG), 0.916 (SHAP).
- `scripts/resnet8_xai_fidelity.py`, `docs/gemmx/RESNET8_XAI_GEMMX.md`.

## 2026-11-19: Real ResNet-8 + Grad-CAM on GeMMX; GeMMX configuration 30x cheaper

### Added
- `snax_cluster/sw/xai/resnet8_gemmx/`: real MLPerf ResNet-8 on GeMMX, with
  `resnet8_int.py` (GeMMX integer semantics, top-1 0.870 = TFLite) and `resnet8.py`
  (schedule: implicit-im2col convs, one launch per 8 channels for per-channel requant,
  block-diagonal residual adds, ones-GeMM average pool, fused Grad-CAM).
- Measured, bit-exact: inference ≈ 53.6k cycles, Grad-CAM +3,841 (7.2%), GeMMX busy 67%.
- Grad-CAM int8 vs float32 over 200 images: Pearson r 0.997.
- Generated straight-line delta CSR configuration (`gemmx_sched.emit_cfg_code`):
  ~77 vs ~2,300 cycles per launch. Results:
  - ToyAdmos inference 47,189 → 9,227 cycles;
  - IG / SHAP with fused inference 181,188 → 91,517 cycles.
- `docs/gemmx/RESNET8_GEMMX.md`, `docs/gemmx/CONFIG_OVERHEAD.md`,
  `scripts/resnet8_gradcam_fidelity.py`.

### Fixed
- TF `SAME` padding for stride-2 convs is asymmetric (0 before, 1 after). Symmetric
  padding had dropped accuracy to 0.595.

## 2026-11-19: ToyAdmos inference + IG + Gradient SHAP on GeMMX (real model and data)

### Added
- `snax_cluster/sw/xai/toyadmos_gemmx/`: a GeMMX launch-schedule generator with a bit-exact
  emulator, a generic launch-table executor, and the MLPerf `ad01_int8` model in GeMMX
  integer semantics.
- Real DCASE 2020 ToyCar data: GeMMX AUC 0.8406 vs TFLite 0.8416.
- Measured, bit-exact:
  - inference 47,189 cycles;
  - Integrated Gradients and Gradient SHAP 181,188 cycles each (3.84× inference,
    41 launches, weights streamed from L3).
- int8 attribution vs float32 (100 windows): Pearson r 0.73 (IG), 0.70 (SHAP).
- `scripts/toyadmos_data.py`, `scripts/toyadmos_xai_fidelity.py`;
  `docs/gemmx/TOYADMOS_XAI_GEMMX.md`.

## 2026-11-19: ResNet-8 inference measured on GeMMX

### Added
- `snax_cluster/sw/xai/inference_gemmx/` + `scripts/run_inference_gemmx.sh`: all 8 ResNet-8
  layers on GeMMX, each verified with 0 errors. Total **71,798 cycles** (11 launches, ~2,530
  config cycles each, 44.5% accelerator busy). The cost model predicted 772k (10.8× too high).
- Grad-CAM on GeMMX = **11.1%** of measured ResNet-8 inference.
- `docs/gemmx/INFERENCE_GEMMX.md`.
- ToyAdmos inference: 10 FC layers on GeMMX (1×1 convs, batch 1), 0 errors. Total
  **30,402 cycles**, 84% of it launch configuration, 15.2% accelerator busy
  (`MODEL=toyadmos scripts/run_inference_gemmx.sh`).

### Fixed
- Stock GeMMX conv flow on `snax_KUL_cluster`:
  - it emitted 4 temporal loops for 3-D streamers, which hung the run;
  - it passed 1 of 2 C/D32 spatial strides;
  - the A spatial stride for stride-2 layers was wrong.
- Grad-CAM/SHAP GeMMX apps now pass both C/D32 spatial strides; re-verified with 0 errors.

## 2026-11-19: Gradient SHAP on the SNAX GeMMX accelerator

### Added
- `snax_cluster/sw/xai/shap_gemmx/`: hoisted Gradient SHAP as one GeMMX launch.
  `N·x − Σᵢ bᵢ = [−1…−1, N] @ [b₁;…;b_N; x]`, with baselines pre-packed into the B layout.
  The DMA core scatters `x` into B while core 0 writes the GeMMX CSRs; the core then
  computes `g[k]·S[e]`. Scalar int8 reference path on the same core; bit-exact BIST.
- Results (GAP+FC, N=16, warm): **5,824 vs 22,673 cycles** scalar int8 (**3.9×**), 0/256
  errors on both paths. Exact for linear heads only.
- `docs/gemmx/SHAP_GEMMX.md`; paper §4.6 extended.

## 2026-11-19: Grad-CAM on the SNAX GeMMX accelerator

### Added
- `snax_cluster/sw/xai/gradcam_gemmx/`: both Grad-CAM GeMMs on GeMMX (int8). GeMM1's SIMD
  requantiser writes alpha straight into GeMM2's B operand (no core glue); integer-only
  (KUL cores have no FPU); scalar int8 reference path on the same core; bit-exact BIST.
- Results (warm): ResNet-8 hook layer 8,043 vs 68,978 cycles scalar int8 (**8.6×**);
  GAP+FC 6,602 vs 3,862 (1.7× slower). ~3,000-cycle launch overhead per GeMM dominates.
- Built the `snax_KUL_cluster` Verilator simulator; `scripts/snax_cluster_mode.sh` switches
  between plain and GeMMX clusters (verified round trip: Phase 1 still 4,757 cycles, 0/16).

### Fixed
- `target/common/common.mk` local patch now only adds `snitch_cluster_wrapper.sv` when bender
  lists no cluster wrapper, so SNAX configs build.


## 2026-11-19: INT8 Grad-CAM measured, cost model calibrated, doc corrections

### Fixed
- `gradcam.h` did not compile: the INT8 SSR variant used SSR APIs the rtl-generic
  runtime does not expose and `vfdotpex.s.b`, an FP8 (not INT8) instruction.
  Removed it; the scalar INT8-weight path remains.
- Out-of-bounds read in Grad-CAM: `gradcam_gap_weights` was called with H×W
  spatial positions on a K-element gradient buffer. BIST passed only because the
  over-read L1 was zero. Now called with 1×1 (gradient is spatially uniform).

### Added
- `GEMM_I8` build path (`main.c`, `Makefile`: `make ... GEMM_I8=1 all`).
- Verilator results, both 0/16 BIST errors: fp32 **4,757** cycles (was 6,153 with
  the bug), INT8-weight **6,923** cycles (backward GeMM 3.1× slower on the scalar core).
- Cost model restructured: XAI methods priced on the scalar core per stage,
  matching the C kernels; within 0.2% of all GAP+FC measurements (Grad-CAM,
  SHAP 2a, SHAP 2b, symbolic). 3 xfail calibration tests now pass at 5% tolerance.
  Hoisted SHAP falls back to naive for non-linear heads (ToyAdmos).

### Corrected docs
- Phase 1 breakdown (68.8% fdiv) was estimated, not measured; replaced with counters.
- Symbolic rule cost: 47 cycles measured (not ~15–25).
- GAP+FC inference (12.0k) is a planning budget, not a measurement.
- Paper: wrong ECQx and SNAX citations fixed; novelty claim flagged for a literature search;
  26× SHAP scaling claim corrected to ~2.8×.


## Phase 4: Neuro-Symbolic Rule Extraction (2026-04)

### Python Reference Implementation
- Added `src/xai/symbolic/rule_extractor.py` — saliency-guided decision tree distillation
- Added `src/xai/symbolic/rule_to_c.py` — C header exporter (inline + table styles)
- Added `src/xai/symbolic/fidelity_metrics.py` — fidelity, coverage, saliency agreement
- Added `tests/test_rule_extraction.py` — 33 tests (all passing, incl. gcc compile+run)
- Added `docs/phase4/PHASE4_DESIGN.md` — Design document

### Key Features
- Depth-bounded decision tree extraction (default depth ≤ 3) using sklearn backend
- Saliency-based feature selection: tree only splits on top-K salient features
- Two C export formats: nested if/else (best for depth ≤ 3) and flat node table
- Integer-only C output: no FPU required, single-instruction comparisons on RISC-V
- Fixed-point threshold quantization (configurable scale, default 1024)
- Fidelity metrics: overall, per-class, leaf coverage, saliency agreement (Jaccard)
- End-to-end verified: extracted rules compiled with gcc and round-trip tested

### Research Narrative Closed
Phase 1 Grad-CAM (WHERE) → Phase 2 SHAP (WHY) → Phase 3 QVIP (SAFE) →
**Phase 4 Rules (WHAT)** — distill the black-box into ~8 auditable IF-THEN
rules that run in 47 cycles (measured) on RISC-V, cheaper than any other XAI method.

---

## Phase 3: Formal Verification of Quantized Neural Networks (2026-04)

### Python Reference Implementation
- Added `src/xai/formal/quantization.py` — INT8 quantization (symmetric + QVIP-style uniform)
- Added `src/xai/formal/bound_propagation.py` — Interval arithmetic bound propagation
- Added `src/xai/formal/qvip_verifier.py` — ILP-based robustness verification
- Added `tests/test_formal_verification.py` — 33 tests (all passing)
- Added `docs/phase3/PHASE3_DESIGN.md` — Design document

### Key Features
- QVIP encoding: QNN → ILP via piecewise constant functions for ReLU
- Symmetric INT8 quantization matching SNAX GeMM scheme
- Interval analysis prunes infeasible ReLU intervals (22-85% constraint reduction)
- LP relaxation with triangle relaxation for crossing ReLU neurons
- Maximum robustness radius via binary search (QVIP Algorithm 1)
- XAI integration: saliency-guided verification of high-importance regions

### Verification Properties
- Local robustness: given input x ± r, does argmax(output) stay the same?
- Quantization safety: does INT8 model agree with float32 on bounded inputs?
- Counterexample generation: finds adversarial input if not robust

### Research Narrative
Phase 1 (Grad-CAM) → WHERE the model looks →
Phase 3 (Verification) → SAFE after INT8 quantization →
Phase 2 (SHAP) → WHY features matter →
Deploy on SNAX with runtime XAI + formal safety guarantees

---

## Phase 2b: Optimized SHAP Kernel (VERIFIED)

### Verification Results (BIST PASSED)
- **Total cycles**: **58,022** on RISC-V scalar core (float32, N=16 samples)
- **Speedup**: **3.02×** over Phase 2a (175,510 cycles)
- **BIST**: 0/256 errors (max_err=0.000000)
- **Cycle breakdown**:
  - zero=1,036, bwd=1,930, accum=53,129, norm=1,787
  - **Accumulate dominates (91.6%)** as expected after hoisting
- **Key optimizations**:
  - Hoisted backward pass (computed once instead of N times)
  - Eliminated interpolation and forward passes (exact for linear models)
  - Reduced scratch buffer requirement to `H*W*K` floats

---

## Phase 2a: Gradient SHAP on SNAX Cluster (2025-04)

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

### Verification Results (BIST PASSED)
- **Total cycles**: 175,510 on RISC-V scalar core (float32, N=16 samples)
- **BIST**: 0/256 errors (max_err=0.000000, tolerance=1e-2)
- **Per-sample**: 10,969 cycles average
- **Cycle breakdown**:
  - zero=816, interp=48,702, fwd=40,199, bwd=30,341, accum=53,178, norm=1,815
  - Accumulate dominates (30.3%), interpolate second (27.8%)
  - Forward+backward = 40.2% — these are Phase 2b acceleration targets
- **4.77x over initial estimate** — loop-carried FP stalls + load/store overhead

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
- **Total cycles**: 6,153 on RISC-V scalar core (float32) — *superseded, see 2026-11-19 entry*
- **Cycle breakdown**: transpose=~120, gemm=~600, gap=~400, wsum=~800, relu_norm=~4233 — *these were estimates; the simulator's counters were 1124/1167/1516/2015/331*
- **Key insight**: FPU division in normalize step dominates (68.8% of total) — *incorrect, normalize was ~5%*
