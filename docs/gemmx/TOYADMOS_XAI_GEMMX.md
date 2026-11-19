# ToyAdmos Inference + XAI on GeMMX (real model, real data)

**Status:** measured, bit-exact (2026-11-19)
**Model:** MLPerf Tiny `ad01_int8.tflite` (10 FC layers, per-tensor int8)
**Data:** DCASE 2020 Task 2 ToyCar test set (2,459 clips: 1,400 normal, 1,059 anomalous); MLPerf log-mel features (640 per 5-frame window)
**Cluster:** `snax_KUL_cluster` (GeMMX 8×8×8 int8, controller core + DMA core, 128 kB TCDM)

## Model fidelity on GeMMX

GeMMX's requantiser rounds once; TFLite rounds twice. With half an LSB folded into the bias, GeMMX layer outputs differ from TFLite by ±1 LSB on ~0.5% of activations. End to end on all 2,459 test clips:

| | mean AUC | mean pAUC |
|---|---:|---:|
| TFLite int8 reference | 0.8416 | 0.7224 |
| GeMMX integer semantics | 0.8406 | 0.7208 |

Anomaly scores differ by a median of 0.06%.

## Measured cycles (Verilator, all outputs bit-exact against the emulator)

Explained input: the most anomalous test clip (`anomaly_id_01_00000194.wav`) and its window with the largest reconstruction error. Weights and biases stream from L3 in ≤20 kB tiles, double-buffered by the DMA core.

| Program | Cycles | Launches | Config | GeMMX run | Core ops |
|---|---:|---:|---:|---:|---:|
| Inference (batch 1) | **47,189** | 18 | 40,880 | 4,983 | — |
| Integrated Gradients (16 steps) | **181,188** | 41 | 95,162 | 23,534 | masks 39,901, gather + combine 18,483 |
| Gradient SHAP (16 baselines) | **181,188** | 41 | 95,162 | 23,534 | same |

IG and SHAP run the same schedule with different data, so their cycle counts are identical. XAI = **3.84× inference** when run serially. GeMMX is busy for 13% of the XAI time.

IG phase breakdown: interp 4,172; forward 51,250; loss 3,606; backward 90,881 (includes masks); attribution 27,171.

## After the configuration fix + fused inference (current)

Configuration via generated delta CSR writes (`CONFIG_OVERHEAD.md`). The XAI pass also produces the inference: the sample with alpha = 16/16 *is* x, and its output row is copied out by a 2-D DMA (`y_inf`, bit-exact with standalone inference).

| Program | Cycles | Config | GeMMX run | Core ops | GeMMX busy |
|---|---:|---:|---:|---:|---:|
| Inference | **9,227** | 2,631 | 5,059 | — | 55% |
| IG / Gradient SHAP **including the inference** | **91,517** | 5,023 | 23,835 | masks 39,863, gather + combine 18,737 | 26% |

The remaining XAI time is core work: ReLU masks (44%) and attribution gather / combine (20%).

## Schedule (all matrix work on GeMMX)

| Step | Formulation |
|---|---|
| Interpolate 16 samples | `X~ = [alpha │ diag(16−alpha)] @ [x; b₁..b₁₆] / 16`; x placed by a 2-D DMA |
| Forward | 16 samples stacked in mesh rows; layer 1 K-split ×5, layer 10 N-split ×5; ReLU via the requantiser clamp |
| Loss gradient | `g_y ∝ y − x~ = [x~ │ y] @ [−59·I; 55·I]` (59/55 = s_x/s_y to 0.001%), in place |
| Backward | `g ← (g ⊙ mask) @ W_l` (TFLite's (out, in) weights are already GeMMX's B layout); masks on both cores |
| Attribution | `Σᵢ D⊙(Jᵀg)` and `Σᵢ D⊙g_y` via the diagonal trick (A and B transposers); `attr = m₀·net − m₁·direct` on the core |

The score's input gradient is `Jᵀ(y − x~) − (y − x~)`: the direct term is required.

## Attribution fidelity (int8 GeMMX vs float32 on the dequantised model)

100 windows (the max-error window of 50 anomalous + 50 normal test clips); the integer pipeline is the one the simulator reproduces bit-exactly.

| Method | Pearson r | Spearman ρ | top-32 overlap | sign agreement (top-32) |
|---|---:|---:|---:|---:|
| Integrated Gradients | 0.731 | 0.678 | 0.675 | 0.909 |
| Gradient SHAP | 0.700 | 0.668 | 0.588 | 0.869 |

- **Precision limit.** The direct term matches float almost exactly (r = 0.998). The loss comes from the 10-layer int8 backward chain: per-element gradient correlation falls from 0.995 at the loss to 0.73 at the input layer.
- **Calibration.** Exact SIMD multipliers at the 99.5th percentile beat power-of-two shifts at the 99.9th percentile (IG 0.77 → 0.81, SHAP 0.59 → 0.74 on a 50-window check).
- **Weak windows.** A few windows correlate poorly, mostly well-reconstructed normal windows where the attribution signal is tiny.

## Hardware rules found on the way (all now handled by `gemmx_sched.py`)

- The C reader runs its temporal loops even when disabled: its bounds must always cover M×N.
- Streamer strides are unsigned: place K-concatenated operands in increasing address order.
- C and D32 can be the same buffer (in-place accumulation); D8 can overwrite its own A input.
- A core-side op that reads a buffer must be followed by a barrier before the DMA core prefetches into that buffer.

## Reproduce

```bash
python scripts/toyadmos_data.py                 # features + TFLite AUC (once)
scripts/snax_cluster_mode.sh gemmx
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/toyadmos_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson PROG=toyadmos_ig all
cd target/snitch_cluster && ./bin/snitch_cluster.vlt sw/apps/xai/toyadmos_gemmx/build/toyadmos_gemmx_toyadmos_ig.elf
# PROG=test_patterns | toyadmos_inference | toyadmos_ig | toyadmos_shap
python scripts/toyadmos_xai_fidelity.py --clips 50
```
