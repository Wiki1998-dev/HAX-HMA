# Phase 5 Status: Benchmark Framework and Cost Model

**Status:** in progress (2026-11-19)

## What exists
- `src/xai/benchmark/models.py` — 4 model specs: GAP+FC, ResNet-8, ToyAdmos, MobileBERT-tiny
- `src/xai/benchmark/model_runners.py` — synthetic-weight runners for all 4 phases × 4 models
- `src/xai/benchmark/cycle_model.py` — SNAX cycle cost model
- `src/xai/benchmark/runner.py` — comparison matrix (`run_phase5_benchmark()`, markdown/CSV)
- Tests: `test_phase5_benchmark.py` (108), `test_benchmark_models.py` (80, 3 skipped as slow)

## Cost model calibration
XAI kernels are priced on the RISC-V **scalar core**, where all measured kernels run,
as a per-stage sum mirroring each C kernel. Constants are fitted to Verilator per-stage
counters on GAP+FC (E=256, K=16, C=10, N=16):

| Method | Predicted | Measured | Error |
|--------|----------:|---------:|------:|
| Grad-CAM (fp32) | 4,752 | 4,757 | −0.1% |
| SHAP naive (2a) | 175,616 | 175,510 | +0.06% |
| SHAP hoisted (2b) | 57,984 | 58,022 | −0.07% |
| Symbolic | 47 | 47 | 0% |

`_verify_calibration()` enforces 5% in the test suite (`MEASURED_GAP_FC` holds the targets).

## Comparison matrix (current)

| Model | Inference | Grad-CAM | SHAP naive | SHAP hoisted | QVIP | Symbolic |
|-------|----------:|---------:|-----------:|-------------:|-----:|---------:|
| GAP+FC (XAI measured) | 12.0k† | 4.8k (40%) | 175.5k | 58.0k (484%) | 12.4s | 47 |
| ResNet-8 (est.) | 772.4k | 41.9k (5.4%) | 2.58M | 927.7k (120%) | 3600s | 47 |
| ToyAdmos (est.) | 20.6k | 1.98M | 31.8M | = naive (non-linear head) | 234s | 47 |
| MobileBERT-tiny (est.) | 870.9k | 35.6k (4.1%) | 2.53M | 927.7k (107%) | 3600s | 47 |

† planning budget, not a measurement.

## Known limits / open work
1. **Inference is uncalibrated.** No GeMM inference kernel has been simulated; GeMM throughput and launch constants are assumptions. Every overhead percentage inherits this.
2. **Only GAP+FC is measured.** Larger-model rows extrapolate per-element constants.
3. **XAI on the scalar core vs inference on the accelerator.** ToyAdmos' huge overheads come from walking a 132k-MAC decoder head on the scalar core; offloading XAI to the GeMM accelerator is the fix and is not implemented.
4. **Hoisting is exact only for linear heads**; the model falls back to naive for ToyAdmos.
5. Calibrate inference by simulating a GeMM inference kernel; then re-derive the matrix.
6. **Measured GeMMX launch cost is ~3,000 cycles** per GeMM (Grad-CAM on GeMMX, `docs/gemmx/GRADCAM_GEMMX.md`), vs `gemm_launch_overhead = 400` in the model, so inference estimates (one launch per layer) are likely optimistic. Not yet folded into the model.
7. **Hoisted SHAP on GeMMX is measured** (`docs/gemmx/SHAP_GEMMX.md`): 5,824 cycles vs 22,673 for the same integer algorithm on the same core. The cost model still prices SHAP on the fp32 scalar core only.
