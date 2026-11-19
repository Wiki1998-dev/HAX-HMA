# End-to-End XAI Benchmark on SNAX GeMMX (measured, real MLPerf Tiny models)

Every number below comes from a Verilator simulation of the `snax_KUL_cluster`: 1 GeMMX controller core + 1 DMA core (rv32ima), an 8×8×8 int8 GeMMX and 128 kB of TCDM. Every run uses the real MLPerf Tiny int8 models and real data, and its outputs are bit-exact against a Python emulator of the accelerator. Weights stream from L3. Configuration uses generated delta CSR writes (`CONFIG_OVERHEAD.md`).

## Results

| Model | Workload | Cycles | vs inference | GeMMX busy | Faithfulness |
|---|---|---:|---:|---:|---|
| ToyAdmos `ad01_int8` | inference (1 window of 640 features) | **9,227** | 1.00× | 55% | AUC 0.8406 (TFLite 0.8416), 2,459 clips |
| ToyAdmos | **Integrated Gradients**, 16 steps, inference fused | **91,517** | 9.9× | 26% | r = 0.73 vs float32 IG (100 windows) |
| ToyAdmos | **Gradient SHAP**, 16 baselines, inference fused | **91,517** | 9.9× | 26% | r = 0.70 vs float32 SHAP |
| ResNet-8 `pretrainedResnet_quant` | inference (1 CIFAR-10 image) | **≈ 53,600** | 1.00× | 67%* | top-1 0.870 (TFLite 0.870), 200 images |
| ResNet-8 | **Grad-CAM**, fused | **+3,841** | **1.07×** | — | r = 0.997 vs float32 Grad-CAM (200 images) |
| ResNet-8 | **Integrated Gradients** (input level, 16 steps), inference fused | **6,027,558** | 112× | 21% | r = 0.946 vs float32 IG (50 images) |
| ResNet-8 | **Gradient SHAP** (input level, 16 baselines), inference fused | **6,027,558** | 112× | 21% | r = 0.916 vs float32 SHAP |
| Keyword spotting `kws_ref_model` (DS-CNN) | inference (1 Speech Commands clip) | **48,024** | 1.00× | 45%† | top-1 0.9180 (TFLite 0.9180), 4,890 clips |
| Keyword spotting | **Grad-CAM**, fused | **+5,787** | **1.12×** | — | r = 0.994 vs float32 Grad-CAM (500 clips) |
| Visual wake words `vww_96_int8` (MobileNetV1) | inference (1 COCO image, 96×96) | **188,250** | 1.00× | 34%† | top-1 0.848 (TFLite 0.850), 1,000 images |
| Visual wake words | **Grad-CAM**, fused | **+1,538** | **1.008×** | — | r = 0.997 vs float32 Grad-CAM (500 images) |

*Busy share of the 57,461-cycle inference + Grad-CAM run. †Busy share of the inference + Grad-CAM run (53,811 / 189,788 cycles); depthwise layers use 1/8 of the mesh (`MOBILE_GEMMX.md`).

## Against the project targets

- **XAI overhead < 20% of inference.**
  - Met by Grad-CAM on all three CNNs: ResNet-8 +7.2%, keyword spotting +12.0%, visual wake words +0.8%. With a linear head, Grad-CAM is one launch on the feature map inference already produced.
  - Not met by the gradient-sampling methods on ToyAdmos (+890%). They need 16 samples × (forward + backward) through a 10-layer network. Stacking the samples into the mesh rows keeps this at 41 launches instead of 320, and the inference comes for free inside the XAI pass.
  - Against the real-time budget of the anomaly detector (one window per 32 ms hop: 3.2M cycles at 100 MHz), the full IG / SHAP explanation is 2.9%.
- **Accelerator utilisation > 90%.**
  - Not met: 55–67% for inference, 26% for ToyAdmos XAI.
  - Configuration is no longer the bottleneck (~77 cycles per launch after the fix, from ~2,300).
  - The remaining XAI time is core-side work: ReLU masks of the backward pass (ToyAdmos 44%, ResNet-8 48%) and attribution reductions (20% / 10%).
- **Faithfulness of int8 explanations.**
  - Grad-CAM: near-exact (r = 0.997).
  - Gradient-sampling methods: r ≈ 0.95 / 0.92 through the ResNet-8 CNN, but r ≈ 0.7 through the ToyAdmos 10-layer fully-connected autoencoder. There the direct gradient term alone is r = 0.998, so the loss comes from the deep int8 backward chain.

## Not yet measured

- Time between frames: feature extraction (log-mel) for ToyAdmos, image capture for ResNet-8.
- IG / SHAP on keyword spotting and visual wake words (only Grad-CAM so far).
- MobileBERT is out of scope: it has no integer softmax / layernorm kernels.

## Sources

`TOYADMOS_XAI_GEMMX.md`, `RESNET8_GEMMX.md`, `MOBILE_GEMMX.md`, `CONFIG_OVERHEAD.md`, `INFERENCE_GEMMX.md` (per-layer, simplified ResNet-8), `GRADCAM_GEMMX.md`, `SHAP_GEMMX.md` (GAP+FC test model).
