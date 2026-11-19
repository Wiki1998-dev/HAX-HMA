# ResNet-8 Inference + Grad-CAM on GeMMX (real model, real data)

**Status:** measured, bit-exact (2026-11-19)
**Model:** MLPerf Tiny `pretrainedResnet_quant.tflite`: 9 convolutions with per-channel int8 weights (including 2 strided 1×1 shortcuts), 3 residual ADDs with fused ReLU, 8×8 average pool, FC
**Data:** CIFAR-10 test set, the 200 MLPerf performance samples
**Cluster:** `snax_KUL_cluster`

## Model fidelity

| | top-1 on the 200 MLPerf samples |
|---|---:|
| TFLite int8 reference | 0.870 |
| GeMMX integer semantics | **0.870** (98% identical predictions) |
| dequantised float32 | 0.870 |

The model's real structure differs from the simplified `models.py` ResNet-8: 9 convs (plus the shortcuts) and 3 residual adds, with TF `SAME` padding that is asymmetric for stride 2 (0 rows before, 1 after).

## Measured cycles (Verilator; logits, final feature map and CAM bit-exact with the emulator)

| Phase | Cycles |
|---|---:|
| 9 convs, 42 launches (one per 8 output channels) | 40,449 |
| 3 residual ADDs (block-diagonal launches, in place) | 4,069 |
| Border refills (DMA memset when a buffer changes shape) | 2,080 |
| Pool + FC | 382 |
| **Inference** | **≈ 53,600** |
| **Grad-CAM (fused: argmax, alpha placement, 1 launch, ReLU + Q15)** | **3,841** |
| Total | 57,461 |

- Grad-CAM adds **7.2%** to inference, inside the project's 20% target.
- GeMMX is busy for **67%** of the time. Configuration (generated delta CSR writes) costs 6,420 cycles over 48 launches.
- Weights stream from L3 in ≤5 kB tiles, double-buffered by the DMA core; the memory footprint is 96.8 kB.

## Grad-CAM fidelity (int8 GeMMX vs float32 Grad-CAM of the dequantised model)

All 200 MLPerf samples, predicted class:

| Pearson r (8×8 maps) | top-1 cell agreement | top-8 cell overlap |
|---:|---:|---:|
| **0.997** (min 0.989) | 0.945 | 0.952 |

## Mapping

| Op | GeMMX |
|---|---|
| CONV_2D | Implicit im2col: A walks the padded input with 6 temporal loops; stride via the A spatial stride (8·s) and temporal strides. **One launch per 8 output channels**: the SIMD requantiser has 8 per-column multiplier / shift slots, which fit per-channel quantisation exactly. The output is written straight into the next layer's padded buffer interior. |
| ADD (+ReLU) | `[x₁ │ x₂] @ [a·I; b·I] + c`, with a/b ≈ s₁/s₂ (≤127) and c = zero points + rounding; in place over x₂ |
| AVERAGE_POOL 8×8 | ones-block GeMM over the map (B transposer), floor((sum + 32)/64) |
| FC | 1×1 GeMM + bias |
| Grad-CAM | linear head, so alpha = FC row of the predicted class. The core places it at run time; one launch `cam = (A₃ − zp) @ alpha`; ReLU + Q15 on the core |

**Layout.** Activations are padded C8HW8, `(C/8, H+2, W+16, 8)`: one zero-point row above and below, and an 8-pixel border left and right. The D8 writer and B reader connect channel *i* only to banks ≡ *i* mod 8, so every output block must be 64-byte aligned. Borders hold −128, the input zero point of every conv. Three 26 kB activation slots are reused across the network.

## Reproduce

```bash
scripts/snax_cluster_mode.sh gemmx
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/resnet8_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson R8_SAMPLE=1 all
cd target/snitch_cluster && ./bin/snitch_cluster.vlt sw/apps/xai/resnet8_gemmx/build/resnet8_gemmx_s1.elf
python scripts/resnet8_gradcam_fidelity.py
```
