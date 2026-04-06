# XAI on Heterogeneous Multi-Accelerator Platforms

Explainable AI (XAI) methods — Grad-CAM, LRP, SHAP, formal verification,
neuro-symbolic reasoning — implemented on heterogeneous embedded platforms.

## Platforms

| Platform | Architecture | Memory | Use Case |
|----------|-------------|--------|----------|
| **SNAX Cluster** | RISC-V + GeMM/MaxPool accelerators, TCDM SPM | 128 kB L1 | Gradient-based XAI via transpose-GeMM |
| **CFU Playground** | VexRiscv + custom function units on FPGA | 256 MB DDR | SIMD saliency alongside TFLite Micro |

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Grad-CAM on SNAX (Python ref + C kernel + Verilator sim) | **COMPLETE** - 6,153 cycles, 0/16 errors |
| 2 | INT8 Grad-CAM via SNAX GeMM accelerator | Planned |
| 3 | SHAP async dispatch on SNAX | Planned |
| 4 | CFU Playground XAI port | Planned |
| 5 | Formal verification (QVIP-style) | Planned |

## Quick Start

```bash
# Python reference tests
source .venv/bin/activate
pytest tests/ -v

# SNAX simulation (requires Verilator + pixi)
cd snax_cluster && pixi install
make -C target/snitch_cluster/sw/apps/xai/gradcam all
```

## Repository Structure

```
xai_heterogeneous/
├── src/xai/              # XAI implementations (Python reference + C kernels)
│   ├── gradcam/          # Grad-CAM: reference, kernel, compiler pass
│   ├── shap/             # SHAP: reference, kernel, async dispatcher
│   ├── formal/           # QVIP formal verification
│   └── symbolic/         # Neuro-symbolic rule extraction
├── tests/                # Python test suite
├── snax_cluster/         # SNAX hardware platform (submodule)
│   └── sw/xai/gradcam/   # Grad-CAM C kernel for SNAX
├── CFU-Playground/       # CFU FPGA platform (submodule)
├── papers/               # Reference papers (see CLAUDE.md for index)
├── docs/phase1/          # Phase 1 results and documentation
├── zigzag/               # ZigZag cost estimation tool
└── dory/                 # DORY memory tiling framework
```

## Key Results (Phase 1)

Grad-CAM saliency computation on a single RISC-V RV32IMF core (Snitch, float32):

| Metric | Value |
|--------|-------|
| Feature map | 4x4 spatial, 16 channels, 10 classes |
| Total cycles | 6,153 |
| BIST errors | 0/16 |
| Bottleneck | FPU division in normalize (68.8%) |

See [`docs/phase1/PHASE1_SUMMARY.md`](docs/phase1/PHASE1_SUMMARY.md) for full analysis
and comparison with prior work.

## Papers

| File | Topic |
|------|-------|
| `HW-SW_co.pdf` | SNAX architecture + MLIR compiler |
| `Hardware Acceleration of Explainable Artificial Intelligence.pdf` | XAI-to-matrix mapping |
| `DORY.pdf` | Memory tiling for 128 kB SPM |
| `HTVM.pdf` | Heterogeneous TinyML deployment |
| `Gemmini.pdf` | Systolic array benchmarking |

## License

Apache 2.0 (kernel code follows KU Leuven SNAX licensing)
