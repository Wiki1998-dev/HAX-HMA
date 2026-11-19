# ResNet-8 Input-Level Integrated Gradients + Gradient SHAP on GeMMX

**Status:** measured, bit-exact (2026-11-19)
**Model / data:** MLPerf Tiny `pretrainedResnet_quant.tflite`, CIFAR-10 (MLPerf performance samples)
**Explained score:** logit of the predicted class
**Methods (16 samples each):**
- IG: 16 steps from a black image
- Gradient SHAP: 16 baselines drawn from CIFAR-10 training images, random alpha

Sample 0 is x itself, so the inference falls out of the XAI pass.

## Measured (Verilator, 16 samples, attribution bit-exact against the emulator)

| | IG | Gradient SHAP |
|---|---:|---:|
| **Total cycles** | **6,027,558** | **6,027,558** |
| per sample | ~377k | ~377k |
| vs inference (~53.6k) | 112× | 112× |

IG and SHAP run the same schedule, so their cycle counts are identical. Where the time goes (both methods):

| | Cycles | Share |
|---|---:|---:|
| ReLU masks (both cores, interior rows, activations fetched from L3) | 2,867,330 | 48% |
| GeMMX compute | 1,272,970 | 21% |
| Attribution multiply-accumulate (cores) | 603,610 | 10% |
| DMA interleave / scatter (polyphase transposed convs) | 526,178 | 9% |
| Configuration (generated delta CSR writes, 99 launches / sample) | 185,485 | 3% |

Each sample is interpolate (1 launch), forward (48 launches, 7 activations spilled to L3), backward to the input (50 launches, 320 2-D DMAs), then the attribution MAC.

## Attribution fidelity (int8 GeMMX semantics vs float32 PyTorch autograd, dequantised model)

50 images, channel-summed 32×32 maps:

| Method | Pearson r | Spearman ρ | top-64 pixel overlap | full 32×32×3 r |
|---|---:|---:|---:|---:|
| Integrated Gradients | **0.946** (min 0.874) | 0.938 | 0.760 | 0.948 |
| Gradient SHAP | **0.916** (min 0.868) | 0.898 | 0.701 | 0.917 |

CNN input gradients survive the int8 backward pass far better than the ToyAdmos autoencoder's (r ≈ 0.7).

## Backward mapping

| Forward op | Backward on GeMMX |
|---|---|
| avgpool + FC | `G = W_fc[pred]` broadcast over the 8×8 map (one ones-row launch) |
| ReLU | `G ⊙ (act ≠ −128)` on both cores; act fetched from L3 in 2-D chunks, interior only |
| ADD | identity: both branches get the same tensor |
| 3×3 conv, stride 1 | conv launch with the 180°-rotated kernel, requantised per input channel offline (int8), over zero-bordered gradients |
| 3×3 conv, stride 2 | **polyphase**: 4 sub-conv launches per 8 channels into compact buffers, then DMA interleave (the D8 writer cannot skip pixels) |
| 1×1 conv, stride 2 | compact launch + DMA scatter to even pixels |
| fork (residual) | block-diagonal add `a·G₁ + b·G₂`, `a/b ≈ S₁/S₂` |
| attribution | `acc += (x − bᵢ) ⊙ gₓ` on both cores |

Gradient scales are per tensor and calibrated statically: exact SIMD multipliers at the 99.5th percentile, on interpolated CIFAR-10 training images.

## Reproduce

```bash
scripts/snax_cluster_mode.sh gemmx
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/resnet8_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson R8_SAMPLE=1 R8_METHOD=ig R8X_N=16 all
# simulate with the per-hart traces linked to /dev/null (6M cycles would write GBs):
mkdir -p run/logs && ln -sf /dev/null run/logs/trace_chip_00_hart_00000.dasm && ln -sf /dev/null run/logs/trace_chip_00_hart_00001.dasm
cd run && ../target/snitch_cluster/bin/snitch_cluster.vlt ../target/snitch_cluster/sw/apps/xai/resnet8_gemmx/build/resnet8_ig_n16_s1.elf
python scripts/resnet8_xai_fidelity.py --images 50
```
