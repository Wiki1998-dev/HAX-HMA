# Keyword Spotting (DS-CNN) and Visual Wake Words (MobileNetV1) on GeMMX

**Status:** measured, bit-exact (2026-11-19)
**Models:** MLPerf Tiny `kws_ref_model.tflite` and `vww_96_int8.tflite`, unchanged
**Data:**
- Speech Commands v0.02 test set (4,890 clips). MFCC features re-implemented in NumPy from the MLPerf reference pipeline.
- The 1,000 COCO val2017 images of the MLPerf `vww01` set, resized to 96×96 bilinear.

**Workload:** inference + fused Grad-CAM of the predicted class on the last feature map.

## Measured (Verilator, sample 0, every output bit-exact against the emulator)

| | Keyword spotting | Visual wake words |
|---|---:|---:|
| **Total cycles (inference + Grad-CAM)** | **53,811** | **189,788** |
| Inference | 48,024 | 188,250 |
| Grad-CAM | 5,787 (**+12.0%**) | 1,538 (**+0.8%**) |
| GeMMX launches | 75 | 346 |
| GeMMX run / configuration | 24,364 / 9,933 | 65,307 / 38,834 |
| Input placement (DMA) | 1,232 | 19,128 |
| Border / column refills (DMA) | 2,894 | 21,873 |
| TCDM used | 51.8 kB | 113.5 kB |

Per-layer-type time (VWW): depthwise 50,810, pointwise 58,692, first conv 3,758.

## Accuracy and Grad-CAM fidelity

`python scripts/mobile_gradcam_fidelity.py`

| | Keyword spotting | Visual wake words |
|---|---:|---:|
| top-1, GeMMX integer model | 0.9180 | 0.848 |
| top-1, TFLite int8 reference | 0.9180 | 0.850 |
| same prediction as TFLite | 99.35% | 99.4% |
| Grad-CAM Pearson r vs float32 (500 inputs) | 0.994 (min 0.936) | 0.997 (min 0.677, median 1.000) |
| Grad-CAM top-1 cell agreement | 0.832 | 0.984 |

The GeMMX requantiser rounds once where TFLite rounds twice. Layer 0 differs from TFLite on 0.1–0.2% of values, by ±1. Through 9 / 27 layers this grows to ±10–15 on some activations, which still changes only 0.6% of the predictions.

## Mapping

| Op | On GeMMX |
|---|---|
| CONV_2D (first layer) | implicit-im2col conv, one launch per 8 output channels |
| DEPTHWISE_CONV_2D 3×3 | conv launch whose K loop stays in input-channel tile nt, with a diagonal 8×8 weight block per tap. GeMMX has no depthwise mode, so 1/8 of the mesh does useful work. |
| CONV_2D 1×1 | GeMM over the pixels |
| AVERAGE_POOL_2D (125 / 9 pixels) | ones-block GeMM over whole padded rows. The borders are a known −128 constant, removed through the bias. Division by the count is exact (multiplier checked exhaustively over every possible sum). |
| FULLY_CONNECTED | 1×1 GeMM |
| Grad-CAM | `cam = (map − zp) @ alpha`, with alpha = the predicted class's FC row, copied 8 bytes per K tile from the FC weights already in TCDM. ReLU + Q15 normalisation on the core. |

**Pixel groups.** GeMMX writes 8 pixels per mesh row, so map widths that are not a multiple of 8 (KWS 25, VWW 12 / 6 / 3) also produce extra columns. One 2-D DMA per layer, covering every row of every channel tile, resets them to −128 before the next layer reads them as padding.

**KWS orientation.** KWS runs transposed (time along the width): 25 of 32 pixel slots used instead of 5 of 8.

**Pruned VWW channels.** About 100 VWW channels have all-zero weights and a ±2^30 bias. Their constant output is computed with TFLite's fixed-point arithmetic and re-encoded to fit int32.

**VWW input.** The 96×96 input (88 kB padded) does not fit beside the first conv's output. That conv runs in two bands of 24 output rows; the DMA core places each band's input rows from L3.

## Reproduce

```bash
python scripts/kws_vww_data.py                 # needs external/speech_commands_test, external/vww01_coco
python scripts/tflite_eval.py kws ; python scripts/tflite_eval.py vww
scripts/snax_cluster_mode.sh gemmx
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/mobile_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson MB_TASK=kws MB_SAMPLE=0 all
cd run && ../target/snitch_cluster/bin/snitch_cluster.vlt ../target/snitch_cluster/sw/apps/xai/mobile_gemmx/build/mobile_kws_s0.elf
```
