# ResNet-8 Inference on the SNAX GeMMX Accelerator

**Status:** all 8 layers measured, every layer verified against the golden model with 0 errors (2026-11-19)
**Cluster:** `snax_KUL_cluster` (8×8×8 int8 GeMMX, controller + DMA core, rv32ima, 128 kB TCDM)

Each conv layer runs as one GeMMX implicit-im2col convolution: int8 NHWC input, int8 weights, int32 bias, and int8 output through the SIMD requantiser. Data and golden output come from the stock KU Leuven conv generator. GAP runs on the core; FC is a 1×1 conv on GeMMX.

## Results (Verilator, warm I-cache, cycles)

| Layer | Shape | Launches | Config / launch | Run / launch | Layer total | Accelerator busy |
|---|---|---:|---:|---:|---:|---:|
| conv1 | 32×32×3→16 (Cin padded to 8) | 2 | 2,545 | 1,661 | 8,412 | 3,270 |
| block1_a | 32×32×16→16 | 2 | 2,530 | 3,065 | 11,190 | 6,072 |
| block1_b | 32×32×16→16 | 2 | 2,530 | 3,065 | 11,190 | 6,072 |
| block2_a | 32×32×16→16×16×32, stride 2 | 1 | 2,530 | 2,841 | 5,371 | 2,813 |
| block2_b | 16×16×32→32 | 1 | 2,530 | 5,733 | 8,263 | 5,706 |
| block3_a | 16×16×32→8×8×64, stride 2 | 1 | 2,530 | 2,769 | 5,299 | 2,741 |
| block3_b | 8×8×64→64 | 1 | 2,530 | 5,261 | 7,791 | 5,234 |
| GAP | 8×8×64 int8, core | — | — | — | 11,702 | 0 |
| FC | 64→10 (1×1 conv) | 1 | 2,530 | 50 | 2,580 | 34 |
| **Total** | | **11** | | | **71,798** | **31,942 (44.5%)** |

- The 32×32 layers' outputs don't fit the 128 kB TCDM in one launch, so they run as 2 H-tiles. One tile was measured and counted twice; both tiles have identical shape and timing.
- Configuration costs about 2,530 cycles per launch: 27.8k of the 71.8k cycles (39%).
- While running, GeMMX streams at close to full throughput: run cycles ≈ busy cycles + ~30.
- GAP was first written as one byte load plus a read-modify-write per element (53,665 cycles). It now uses word-wise loads with two 16-bit lanes (11,702 cycles, exact).

## ToyAdmos (MLPerf Tiny anomaly detection autoencoder)

The 10 fully-connected layers run as 1×1 convs on GeMMX, batch 1: one of the 8 mesh rows holds the real input vector. All layers have 0 errors.

| Layer(s) | Shape | Count | Config | Run | Total each | Busy each |
|---|---|---:|---:|---:|---:|---:|
| enc1 | 640→128 | 1 | 2,545 | 1,341 | 3,886 | 1,315 |
| enc2–4, dec2–4 | 128→128 | 6 | 2,545 | 321 | 2,866 | 292 |
| bottleneck | 128→8 | 1 | 2,557 | 74 | 2,631 | 33 |
| dec1 | 8→128 | 1 | 2,554 | 165 | 2,719 | 137 |
| recon | 128→640 | 1 | 2,545 | 1,425 | 3,970 | 1,396 |
| **Total** | | **10** | | | **30,402** | **4,633 (15.2%)** |

- **Configuration dominates: 25,471 of 30,402 cycles (84%).** The layers are tiny (about 264k MACs in total), so each launch's ~2,550-cycle setup outweighs its compute.
- The cost model predicted 20,584 cycles, 32% too low. It assumed 400 cycles of launch overhead; the measured figure is about 2,550.
- **A batch of 8 costs about the same as one input.** The 7 idle mesh rows could take 7 more input vectors, giving ~3.8k cycles per input. This is not measured as a batch-8 run; the run cycles would grow slightly with more rows.

## XAI overhead against this inference (measured, same cluster)

| XAI | Cycles | % of inference (71,798) |
|---|---:|---:|
| Grad-CAM on GeMMX (ResNet-8 hook layer) | 7,996 | **11.1%** |

The cost model predicted 772,388 cycles for this inference, 10.8× too high. It assumed 16 MAC/cycle; the measured rate is about 450 MAC/cycle while GeMMX runs.

## Not included yet

- Work between layers: re-padding the int8 output into the next layer's padded input, ReLU (the requantiser can clamp it), and the residual adds.
- DMA of weights and activations from L3. Everything was preloaded in TCDM.
- Batch size 1 only.

## Fixes to the stock `snax-gemmx-conv` flow (it hangs on `snax_KUL_cluster`; not in upstream CI)

1. **Temporal loop count.** C/D8/D32 have 3 temporal dimensions in this cluster, but the generator emits 4 (`bound3 = out_height`). The hardware drops the 4th loop, the writers expect fewer blocks and the run never finishes. Fix: merge the two contiguous M loops.
2. **Spatial strides.** The C/D32 spatial bounds are 2-D (`[8, 4]`), so the library writes 2 spatial strides, but apps pass a 1-element array. Fix: pass `{8, 64}`. The Grad-CAM and SHAP apps had the same latent bug; they now pass both strides and still give 0 errors (warm totals changed by <1%: Grad-CAM 6,573 / 7,996, SHAP 5,826).
3. **Stride-2 layers.** The A spatial stride is hard-coded to 8 B; it must be `8 × stride_w`.
4. **Bias.** C must be DMA'd to TCDM when `channel_en_C = 1`. This was missing from my first version of the kernel, not a stock bug.

## Reproduce

```bash
scripts/snax_cluster_mode.sh gemmx
scripts/run_inference_gemmx.sh              # all layers; or: scripts/run_inference_gemmx.sh block3_b
scripts/snax_cluster_mode.sh plain
```

## Files

| File | Purpose |
|---|---|
| `snax_cluster/sw/xai/inference_gemmx/params/*.hjson` | one params file per layer (`launches_per_layer` in the header); `stock.hjson` = stock example sanity check |
| `snax_cluster/sw/xai/inference_gemmx/src/main.c` | timed conv kernel with the fixes above; GAP on the core for the FC build |
| `snax_cluster/sw/xai/inference_gemmx/Makefile` | `LAYER=<name>`, `EXTRA_CFLAGS` |
| `scripts/run_inference_gemmx.sh` | build + simulate all layers |
