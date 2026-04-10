# XAI on Heterogeneous Multi-Accelerator Platforms

Hardware-accelerated Explainable AI (XAI) methods on heterogeneous embedded platforms. Implements Grad-CAM, LRP, Gradient SHAP, QVIP formal verification, and neuro-symbolic rule extraction targeting the **SNAX cluster** (RISC-V + GeMM accelerator) and **CFU Playground** (FPGA + VexRiscv).

## Platforms

| Platform | Architecture | Memory | XAI Role |
|----------|-------------|--------|----------|
| **SNAX Cluster** | RISC-V Snitch core + GeMM/MaxPool accelerators, TCDM SPM | 128 kB L1 | Gradient-based XAI via transpose-GeMM (Grad-CAM backward, SHAP masked passes) |
| **CFU Playground** | VexRiscv + custom function units on Arty FPGA | 256 MB DDR | SIMD saliency computation alongside TFLite Micro inference |

## Implementation Status

| Phase | Method | Platform | Status | Key Result |
|-------|--------|----------|--------|------------|
| 1 | **Grad-CAM** | SNAX (RV32IMF scalar) | Complete | 6,153 cycles, 0/16 BIST errors |
| 2a | **Gradient SHAP** (naive) | SNAX (RV32IMF scalar) | Complete | 175,510 cycles, 0/256 errors |
| 2b | **Gradient SHAP** (hoisted) | SNAX (RV32IMF scalar) | Complete | 58,022 cycles (3.02x speedup) |
| 3 | **QVIP Formal Verification** | Host (Python + scipy ILP) | Complete | 33/33 tests, INT8 robustness proofs |
| 4 | **Neuro-Symbolic Rules** | Host + RV32IMF (generated C) | Complete | 33/33 tests, ~47 cycles per rule walk |
| 5 | **Benchmark & Scaling** | Host (cycle cost model) | In Progress | 4 models, 4 methods, comparison matrix |

## Benchmark Models

All XAI methods are evaluated across four MLPerf Tiny benchmark architectures:

| Model | Task | Input Shape | Hook Layer | Feature Elements |
|-------|------|-------------|------------|------------------|
| **ResNet-8** | Image classification (CIFAR-10) | 32x32x3 | block3_b (8x8x64) | 4,096 |
| **ToyAdmos** | Anomaly detection (audio) | 640-dim log-mel | bottleneck (8-dim) | 8 |
| **MobileBERT-tiny** | NLP (2-block transformer) | 32x128 | final layernorm | 4,096 |
| **GAP+FC** | Calibration model | 4x4x16 | conv output | 256 |

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pytest scikit-learn hjson

# Run all tests (Phases 1-5)
pytest tests/ -v

# Individual phase tests
pytest tests/test_gradcam_reference.py -v       # Phase 1: Grad-CAM + LRP
pytest tests/test_shap_reference.py -v          # Phase 2: Gradient SHAP
pytest tests/test_formal_verification.py -v     # Phase 3: QVIP verification
pytest tests/test_rule_extraction.py -v         # Phase 4: Rule extraction
pytest tests/test_phase5_benchmark.py -v        # Phase 5: Benchmark framework
pytest tests/test_benchmark_models.py -v        # Phase 5: All methods x all models

# SNAX simulation (requires Verilator + pixi)
cd snax_cluster && pixi install
make -C target/snitch_cluster/sw/apps/xai/gradcam all
make -C target/snitch_cluster/sw/apps/xai/shap all
```

## Repository Structure

```
xai_heterogeneous/
├── src/xai/                    # All XAI implementations
│   ├── gradcam/                # Grad-CAM + LRP reference & kernels
│   │   ├── gradcam_reference.py    # NumPy reference (perturbation gradient)
│   │   └── gradcam_kernel.c        # Embedded C for SNAX
│   ├── shap/                   # Gradient SHAP reference & kernels
│   │   ├── gradient_shap_reference.py  # Naive + analytical + expected gradients
│   │   └── shap_kernel.c              # Embedded C for SNAX
│   ├── formal/                 # QVIP formal verification
│   │   ├── quantization.py         # INT8 symmetric quantization
│   │   ├── bound_propagation.py    # Interval analysis + ReLU pruning
│   │   └── qvip_verifier.py        # ILP robustness verification engine
│   ├── symbolic/               # Neuro-symbolic rule extraction
│   │   ├── rule_extractor.py       # Saliency-guided decision tree distillation
│   │   ├── rule_to_c.py            # C header export (inline + table)
│   │   └── fidelity_metrics.py     # Faithfulness, coverage, agreement
│   ├── benchmark/              # Phase 5 benchmark framework
│   │   ├── models.py               # Model catalogue (4 architectures)
│   │   ├── model_runners.py        # Synthetic model factories for benchmarking
│   │   ├── cycle_model.py          # SNAX cycle cost model (calibrated)
│   │   ├── hoisted_shap.py         # Backward-hoisted SHAP reference
│   │   ├── ecq_filter.py           # ECQx saliency-guided quantization
│   │   ├── topk_filter.py          # Top-K feature filter for rule distillation
│   │   └── runner.py               # Benchmark matrix builder + formatters
│   └── utils/                  # Shared utilities
│       ├── buffer_planner.py       # 128 kB SPM static layout calculator
│       └── metrics.py              # Faithfulness, localization metrics
├── tests/                      # Test suite (193+ tests)
│   ├── test_gradcam_reference.py
│   ├── test_shap_reference.py
│   ├── test_formal_verification.py
│   ├── test_rule_extraction.py
│   ├── test_phase5_benchmark.py
│   └── test_benchmark_models.py
├── snax_cluster/               # SNAX hardware platform
│   └── sw/xai/                 # XAI C kernels for SNAX
│       ├── gradcam/                # Grad-CAM kernel + data generator
│       └── shap/                   # SHAP kernel + data generator
├── CFU-Playground/             # CFU FPGA platform
├── docs/                       # Design docs & results per phase
│   ├── phase1/PHASE1_SUMMARY.md
│   ├── phase2/PHASE2_DESIGN.md
│   ├── phase2/PHASE2_RESULTS.md
│   ├── phase3/PHASE3_DESIGN.md
│   └── phase4/PHASE4_DESIGN.md
├── papers/                     # Reference papers
├── zigzag/                     # ZigZag cost estimation tool
└── dory/                       # DORY memory tiling framework
```

## Key Results

### Phase 1: Grad-CAM on SNAX

Saliency map computation on a single RISC-V RV32IMF core (Snitch, float32):

| Metric | Value |
|--------|-------|
| Feature map | 4x4 spatial, 16 channels, 10 classes |
| Total cycles | **6,153** |
| BIST errors | 0/16 |
| Bottleneck | FPU division in normalize (68.8%) |

### Phase 2: Gradient SHAP on SNAX

Backward-hoisted SHAP with N=16 baseline samples:

| Variant | Cycles | Errors | Speedup |
|---------|--------|--------|---------|
| Phase 2a (naive) | 175,510 | 0/256 | 1.0x |
| Phase 2b (hoisted) | **58,022** | 0/256 | **3.02x** |

The hoisting optimization caches backbone feature maps and re-runs only the lightweight head for each SHAP sample, reducing cost from O(N x model) to O(backbone + N x head).

### Phase 3: QVIP Formal Verification

ILP-based robustness verification for quantized neural networks:

- Symmetric INT8 quantization matching SNAX GeMM scheme
- Interval analysis with ReLU neuron pruning (reduces ILP constraint count)
- LP relaxation with triangle relaxation for 2-layer networks
- Saliency-guided verification: focus ILP work on high-importance regions
- **33/33 tests passing**

### Phase 4: Neuro-Symbolic Rule Extraction

Saliency-guided decision tree distillation:

- Uses top-K most salient features from Grad-CAM/SHAP
- Depth-bounded trees (depth <= 3) for auditability
- Two C export formats: nested if/else and flat node table
- Integer-only inference (no FPU) — ~47 cycles on RV32IMF
- **33/33 tests passing**

### Phase 5: Benchmark Comparison Matrix

Estimated XAI overhead across benchmark models (cycle cost model):

| Model | Inference | Grad-CAM | SHAP (hoisted) | Symbolic | QVIP (host) |
|-------|----------:|----------:|---------------:|---------:|------------:|
| GAP+FC (cal.) | 12.0k | 6.2k (51%) | 58.0k (483%) | 47 (0.4%) | 12.4s |
| ResNet-8 | 772.4k | 4.8k (0.6%) | 949.4k (123%) | 47 (0.0%) | 3600s |
| ToyAdmos | 20.6k | 11.0k (54%) | 359.8k (1748%) | 47 (0.2%) | 234s |
| MobileBERT | 870.9k | 4.8k (0.6%) | 949.4k (109%) | 47 (0.0%) | 3600s |

*Source: gap_fc row is measured (Phases 1/2b/4); other rows are cycle-model estimates.*

## XAI Method Summary

### Data Flow (SNAX)

```
Input → [Backbone: 7-layer Conv/FC] → Feature Maps → [Head: GAP+FC] → Logits
                                            │
                          ┌─────────────────┤
                          ▼                 ▼
                    Grad-CAM:          Hoisted SHAP:
                    grad = ∂y/∂A       cache features,
                    α = GAP(grad)      re-run head N times
                    cam = ReLU(Σ αA)   φ = E[(x-x')·∇f]
                          │                 │
                          ▼                 ▼
                    Saliency Map ──→ Top-K Filter ──→ Rule Tree
                          │                              │
                          ▼                              ▼
                    ECQx Masking              C Code (47 cycles)
                    (8-bit critical,
                     4-bit rest)
                          │
                          ▼
                    QVIP Verification
                    (ILP on critical slice)
```

### Hardware Mapping

| XAI Operation | Hardware Unit | Strategy |
|--------------|---------------|----------|
| Grad-CAM backward (∂y/∂A) | SNAX GeMM (transpose streamer) | Single transpose-GeMM call |
| SHAP masked forward | SNAX GeMM | Fire-and-forget async dispatch |
| SHAP accumulation | RISC-V scalar | Element-wise in SPM |
| Rule tree evaluation | RISC-V scalar | Integer branch cascade (~47 cycles) |
| QVIP ILP solving | Host CPU | scipy.optimize.linprog |

## Papers

| File | Topic | Used In |
|------|-------|---------|
| `HW-SW_co.pdf` | SNAX architecture + MLIR compiler | Phases 1-2 |
| `2201_01863v3.pdf` | CFU interface & deploy-profile-optimize loop | CFU port |
| `Hardware Acceleration of XAI.pdf` | XAI-to-matrix-op mapping | All phases |
| `QVIP: An ILP-based Formal Verification.pdf` | ILP bounds verification | Phase 3 |
| `ECQx.pdf` | Saliency-guided quantization | Phase 5 (ECQx filter) |
| `ZigZag_...pdf` | Cost estimation for SNAX | Cycle model |
| `DORY.pdf` | Memory tiling for 128 kB SPM | Buffer planner |
| `HTVM.pdf` | Heterogeneous TinyML deployment | Architecture reference |
| `Gemmini.pdf` | Systolic array benchmarking | Baseline comparison |

## License

Apache 2.0 (kernel code follows KU Leuven SNAX licensing)
