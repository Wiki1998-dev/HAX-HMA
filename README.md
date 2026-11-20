# Explainable AI on a Tiny Edge Accelerator

**In one sentence:** this project tests whether a small, low-power chip, the kind used in sensors and wearables, can explain its own AI decisions, not just make them. It also measures how fast, how efficient and how accurate those explanations are.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Wiki1998-dev/HAX-HMA)

---

## 1. The problem

Neural networks increasingly run directly on small devices ("edge AI" or "TinyML"): a camera that recognises objects, or a microphone that detects a machine about to fail. These devices make decisions but cannot say **why**.

**Explainable AI (XAI)** methods answer the "why". They produce a map showing which parts of the input (pixels of an image, or frequencies of a sound) drove the decision. Today these explanations are normally computed afterwards on a large computer or GPU, not on the device itself.

Computing explanations on the device would help with:
- **Trust and certification:** a safety auditor can check what the model looked at.
- **Debugging in the field:** see why a sensor raised a false alarm.
- **Privacy:** the raw data never has to leave the device to be explained.

The obstacle is cost. Most explanation methods need extra passes through the network, often many of them, and small devices have very little compute, memory and energy.

## 2. The research question

> Can explanations be computed on a tiny AI accelerator cheaply enough, efficiently enough and accurately enough to be practical?

We made this concrete with three targets:

| # | Target | Plain meaning |
|---|---|---|
| 1 | **XAI overhead < 20% of inference** | Explaining a decision should cost at most a fifth of the time it takes to make the decision. |
| 2 | **Accelerator utilisation > 90%** | The AI hardware should be busy doing useful maths most of the time, not waiting. |
| 3 | **Faithful int8 explanations** | The chip uses 8-bit integers to save energy. Its explanations should match the "ideal" ones computed in full precision on a PC. |

## 3. What we used

**The hardware: SNAX.** SNAX is an open-source research chip design from KU Leuven, with:
- a small RISC-V processor that controls everything;
- a second RISC-V processor that moves data;
- **GeMMX**, a matrix-multiplication accelerator. It multiplies 8×8 blocks of 8-bit numbers very quickly and does almost all of a neural network's work;
- 128 kB of fast on-chip memory, about the size of one small photo. Everything else has to be streamed in from slower external memory.

There is no physical chip. We run the exact hardware design in **Verilator**, a cycle-accurate simulator. Every number below is a measured clock-cycle count from that simulator, not an estimate.

**The models and data: MLPerf Tiny,** the industry-standard benchmark suite for tiny AI devices:

| Model | Task | Data |
|---|---|---|
| **ResNet-8** | Recognise the object in a 32×32 image (10 classes) | CIFAR-10 images |
| **ToyAdmos autoencoder** | Detect abnormal sounds from a toy-car motor | DCASE 2020 ToyCar recordings |

We use the official pre-trained 8-bit models, unchanged.

**The explanation methods:**

| Method | What it does | Cost |
|---|---|---|
| **Grad-CAM** | Heat-map of *where in the image* the model looked, from the last layer. | Cheap: one extra small step. |
| **Integrated Gradients (IG)** | Per-pixel importance. Fades the input in from a black image over 16 steps and adds up how sensitive the output is at each step. | Expensive: 16 forward **and** backward passes. |
| **Gradient SHAP** | Like IG, but compares against 16 real "reference" examples instead of a black image. | Same cost as IG. |

## 4. What we built

The accelerator was designed to *run* networks (the forward pass). Explanations also need the **backward pass**: working out how sensitive the output is to each input value. We mapped every piece of that backward pass onto the same matrix accelerator:

- **Backward convolutions** run as ordinary convolutions with the filter flipped.
- **Stride-2 layers** (the ones that halve the image size) are split into 4 smaller convolutions and re-interleaved by the data-mover core, because the hardware cannot write every other pixel directly.
- **The 16 IG/SHAP samples** are processed in sequence, and the first one *is* the normal prediction, so inference comes for free.
- **Data movement:** activations that don't fit in the 128 kB memory are saved to external memory and fetched back when needed.
- **Accelerator setup:** we replaced the vendor library's setup code (~2,300 cycles per accelerator call) with generated code that only writes settings that changed (~77 cycles), about 30× cheaper.

**How we know it's correct:** a Python model reproduces the accelerator's arithmetic exactly. Every simulator run is compared byte for byte against it, and all runs match with 0 differences. A separate full-precision PyTorch model provides the "ideal" explanations for the accuracy comparison.

## 5. Results (all measured)

Clock cycles in the simulator. At 100 MHz, 1 million cycles = 10 ms.

| Model | What was run | Cycles | Compared with inference | Accuracy of the result |
|---|---|---:|---:|---|
| ToyAdmos | Inference (anomaly score) | 9,227 | 1× | Detection quality (AUC) 0.841, same as the official model (0.842) |
| ToyAdmos | Integrated Gradients | 91,517 | 9.9× | Correlation with the ideal explanation r = 0.73 |
| ToyAdmos | Gradient SHAP | 91,517 | 9.9× | r = 0.70 |
| ResNet-8 | Inference (classify an image) | ~53,600 | 1× | 87.0% correct, same as the official model |
| ResNet-8 | Grad-CAM | +3,841 | **+7.2%** | r = **0.997** (nearly identical) |
| ResNet-8 | Integrated Gradients | 6,027,558 | 112× | r = 0.95 |
| ResNet-8 | Gradient SHAP | 6,027,558 | 112× | r = 0.92 |

*r is the correlation between the chip's explanation and the full-precision one: 1.0 is identical, 0 is unrelated.*

## 6. What this means

| Target | Result |
|---|---|
| **1. Overhead < 20%** | **Met by Grad-CAM** (+7.2%).<br> **Not met by IG / SHAP** (10× to 112×). They need 16 full forward + backward passes, so they cannot fit in 20%.<br>IG / SHAP may still be practical where time allows: for the audio detector, a full explanation uses about 3% of the real-time budget. |
| **2. Utilisation > 90%** |  **Not met.** The accelerator is busy 55–67% of the time during inference and 21–26% during IG / SHAP.<br>The main bottleneck is no longer the accelerator but work it cannot do: applying the ReLU "on/off" masks in the backward pass runs on the small processors and takes about half the time. |
| **3. Faithful 8-bit explanations** |  **Grad-CAM: near perfect** (r = 0.997).<br>**Image model IG / SHAP: good** (r = 0.92–0.95).<br> **Audio model: weaker** (r ≈ 0.7). Its 10 stacked fully-connected layers lose precision at each 8-bit backward step. |

**Takeaways:**
1. Cheap explanations like Grad-CAM can run on a tiny accelerator almost for free.
2. Detailed per-pixel explanations (IG, SHAP) work and are accurate on the image model, but they are about 100× more expensive than inference.
3. To make them efficient, future accelerators need hardware support for backward-pass masking, not just faster matrix maths.

## 7. Not done yet

- **Two other MLPerf Tiny models** (keyword spotting, visual wake words) are not implemented on the accelerator.
- **The ReLU-mask bottleneck** is not yet optimised.
- **16-bit gradients** for the audio model, which would likely improve its explanation accuracy.
- **Earlier work (Phases 1–5 below)** ran explanations on the RISC-V processor alone, or on the host PC. It is kept for comparison.

---

## Repository guide

| Path | What's there |
|---|---|
| `snax_cluster/sw/xai/resnet8_gemmx/` | ResNet-8 inference, Grad-CAM, IG and SHAP on the accelerator. C program plus the Python scheduler that generates it. |
| `snax_cluster/sw/xai/toyadmos_gemmx/` | ToyAdmos inference, IG and SHAP. `data/gemmx_sched.py` is the shared core: it plans accelerator calls, emulates them exactly and generates the setup code. |
| `snax_cluster/sw/xai/{gradcam,shap,shap_gemmx,inference_gemmx}/` | Earlier, smaller experiments. |
| `scripts/` | Dataset preparation, fidelity measurements, cluster-mode switching. |
| `src/xai/` | Python reference implementations: Grad-CAM, SHAP, formal verification, rule extraction, cost model. |
| `tests/` | Python test suite (279 tests). |
| `docs/gemmx/BENCHMARK.md` | **Start here for results.** Links to the detailed write-up of each experiment. |
| `docs/RESEARCH_PAPER_DRAFT.md` | Paper draft. |
| `CHANGELOG.md` | What changed and when. |

## How to reproduce

```bash
# Python environment and tests
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pytest scikit-learn hjson
pytest tests/ -v

# Datasets (git-ignored, under external/):
#   MLPerf Tiny repo -> external/tiny, CIFAR-10 -> external/cifar-10-batches-py,
#   DCASE ToyCar zip -> external/; then:
python scripts/toyadmos_data.py

# Simulator: needs Verilator and pixi
cd snax_cluster && pixi install && cd ..
scripts/snax_cluster_mode.sh gemmx          # select the accelerator configuration

# Example: ResNet-8 Integrated Gradients on image 1 (see docs/gemmx/RESNET8_XAI_GEMMX.md)
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/resnet8_gemmx \
    CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson R8_SAMPLE=1 R8_METHOD=ig R8X_N=16 all
```

Each document in `docs/gemmx/` ends with the exact commands for its experiment.

## Glossary

| Term | Meaning |
|---|---|
| **Inference** | Running the network once to get a prediction. |
| **Forward / backward pass** | Forward computes the output from the input. Backward computes how much each input value affected the output (the gradient). Explanations need backward. |
| **int8 / quantised** | Numbers stored as 8-bit integers instead of 32-bit floats: 4× smaller and much cheaper, but less precise. |
| **Accelerator / GeMMX** | Dedicated hardware for matrix multiplication, the core operation of neural networks. |
| **Cycle** | One tick of the chip's clock; our unit of time. |
| **Utilisation** | Fraction of time the accelerator is doing useful work. |
| **Bit-exact** | The simulator's output matches the reference model in every byte. |
| **Pearson r** | Similarity score between two explanations: 1 = identical, 0 = unrelated. |
| **AUC** | Anomaly-detection quality: 1.0 = perfect, 0.5 = random guessing. |
| **ReLU mask** | The network's ReLU layers zero out negative values. In the backward pass, gradients must be zeroed at the same positions. |

## Earlier phases (processor-only and host-side work)

| Phase | Method | Where it runs | Result |
|---|---|---|---|
| 1 | Grad-CAM | RISC-V processor (float32) | 4,757 cycles on a small 4×4×16 test layer |
| 2 | Gradient SHAP | RISC-V processor (float32) | 58,022 cycles (3× faster than the naive version) |
| 3 | Formal verification (QVIP) | Host PC | Proves robustness of 8-bit networks; 33 tests |
| 4 | Rule extraction | Host + 47-cycle C code | Turns explanations into small readable decision trees; 33 tests |
| 5 | Cost model | Host | Planning estimates; superseded by the measured results above |

Details: `docs/phase1/` … `docs/phase5/`.

## References

- R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.
- M. Sundararajan, A. Taly, Q. Yan, "Axiomatic Attribution for Deep Networks" (Integrated Gradients), ICML 2017.
- S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS 2017.
- C. Banbury et al., "MLPerf Tiny Benchmark," NeurIPS Datasets and Benchmarks Track, 2021.
- R. A. Antonio, J. Dumoulin, X. Yi, J. Van Delm, Y. Deng, G. Paim, M. Verhelst, "An Open-Source HW-SW Co-Development Framework Enabling Efficient Multi-Accelerator Systems," ISLPED 2025, arXiv:2508.14582.
- Z. Pan and P. Mishra, "Hardware Acceleration of Explainable Artificial Intelligence," arXiv:2305.04887, 2023.
- Z. Zhang et al., "QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks," ASE 2022.
- D. Becking, M. Dreyer, W. Samek, K. Müller, S. Lapuschkin, "ECQx: Explainability-Driven Quantization for Low-Bit and Sparse DNNs," arXiv:2109.04236, 2022.
- S. Bach et al., "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation," PLOS ONE, 2015.
- H. Genc et al., "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration," DAC 2021.
- B. Fresz, V. P. Göbels, S. Omri, D. Brajovic, A. Aichele, J. Kutz, J. Neuhüttler, M. F. Huber, "The Contribution of XAI for the Safe Development and Certification of AI: An Expert-Based Analysis," arXiv:2408.02379, 2024.

## License

Apache 2.0. Kernel code follows KU Leuven SNAX licensing.
