# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Implement Explainable AI (XAI) methods — Grad-CAM, LRP, SHAP, formal verification, neuro-symbolic reasoning — on heterogeneous multi-accelerator platforms: **SNAX cluster** (RISC-V + GeMM + MLIR) and **CFU Playground** (FPGA + VexRiscv + custom function units).

## Common Commands

```bash
# XAI Python tests (host-side reference implementations)
pytest tests/ -v
pytest tests/test_gradcam_reference.py -v          # single test file

# SNAX simulation (requires Verilator)
cd snax_cluster && pixi install                     # install SNAX deps
make -C snax_cluster/target/snax-gemm-cluster all   # build + simulate GeMM
pytest tests/ -v --snax-sim                         # tests needing Verilator

# CFU Playground
make -C CFU-Playground/proj/xai_cfu TARGET=arty_35t prog   # build + flash
cd CFU-Playground && make TARGET=<proj> simulate            # simulate

# Dependency setup
cd zigzag && pip install -e . --break-system-packages
cd dory && pip install -e . --break-system-packages
```

## Environment

- Python 3.10 in `.venv/` (numpy, scipy, pytest, hjson)
- Simulation: Verilator (installed) + Questasim (optional)
- SNAX toolchain: managed by `pixi` (see `snax_cluster/pixi.toml`)
- CFU toolchain: see `CFU-Playground/environment/`
- Custom pytest flags: `--snax-sim` (enable Verilator tests), `--tolerance` (float comparison)

## Architecture

### Two-platform approach
Each XAI method follows: **Python reference** → **embedded C kernel** → **compiler pass/hardware integration**.

- `src/xai/` — All XAI implementations. Each subdirectory (`gradcam/`, `shap/`, `formal/`, `symbolic/`) contains a `*_reference.py` (NumPy ground truth), a `*_kernel.c` (embedded target), and a compiler/dispatch layer.
- `src/xai/utils/` — Shared utilities: `buffer_planner.py` (static SPM layout for 128 kB), `zigzag_wrapper.py` (cost estimation), `metrics.py` (faithfulness, localization).
- `snax_cluster/` — SNAX hardware platform. CSR-programmed accelerators, TCDM scratchpad. See `snax_cluster/CLAUDE.md` for CSR patterns and simulation commands.
- `CFU-Playground/` — FPGA TinyML framework. CFU projects live in `proj/`. See `CFU-Playground/CLAUDE.md` for interface details and build flow.
- `src/xai/CLAUDE.md` — Detailed XAI implementation plan, formulas, and testing protocol.

### Key data flow (SNAX)
RISC-V core writes CSR registers to configure streamers + GeMM accelerator → fires kernel → barrier wait. XAI gradient ops (Grad-CAM backward, SHAP masked passes) are mapped to transpose-GeMM on the same accelerator. All buffer sizes must be statically known (128 kB SPM budget).

### Key data flow (CFU Playground)
C code calls `cfu_op(funct7, funct3, a, b)` which maps to a custom RISC-V R-format instruction executed by the CFU hardware (Amaranth HDL in `proj/<name>/cfu.py`). XAI saliency uses SIMD MAC instructions running alongside TFLite Micro inference.

## Papers (`papers/` directory)

Read these when working on related code — they contain the formulations and hardware mappings used in implementation:

| File | Use When |
|------|----------|
| `HW-SW_co.pdf` | Touching `snax_cluster/` or `snax-mlir/` — SNAX architecture + MLIR compiler |
| `2201_01863v3.pdf` | Touching `CFU-Playground/` — CFU interface, deploy-profile-optimize loop |
| `Hardware Acceleration of Explainable Artificial Intelligence.pdf` | Implementing `src/xai/` — maps XAI ops to matrix/conv for hardware acceleration |
| `QVIP: An ILP-based Formal Verification Approach.pdf` | `src/xai/formal/` — ILP bounds verification |
| `ECQx.pdf` | Quantization-aware XAI (saliency-guided bit-width) |
| `ZigZag_...pdf` | Cost estimation for XAI compute on SNAX |
| `DORY.pdf` | Memory allocation strategy for 128 kB SPM |
| `HTVM.pdf` | End-to-end heterogeneous TinyML deployment reference |
| `Gemmini.pdf` | Benchmarking baseline |

## Coding Standards

- **Python**: type hints required, docstrings on all public functions
- **C (SNAX kernels)**: follow `snax_cluster/sw/snax-gemm/src/` style; no dynamic memory allocation
- **HDL (CFU)**: Amaranth HDL preferred
- All XAI buffer sizes must be compile-time constants or MLIR-computed

## Do NOT Touch

- `snax_cluster/hw/` — generated RTL; edit `.hjson` config files in `snax_cluster/util/cfg/` instead
- `CFU-Playground/third_party/` — upstream submodules
- `dory/dory/DORY_network/` — auto-generated; run `network_generate.py` to regenerate
- Any `*.lock` or `pixi.lock` files

## Phase 1 Status: COMPLETE, VERIFIED

Grad-CAM on SNAX cluster (RISC-V scalar core, float32):
- **6,153 cycles**, 0/16 BIST errors, H=4 W=4 K=16 C=10
- Python reference: 11/11 tests passing
- Key patches applied: `Xdiv_sqrt: true` in cluster config, core-gating fix (`snrt_cluster_core_idx() == 0`)
- Full results: `docs/phase1/PHASE1_SUMMARY.md`

## Phase 2 Status: COMPLETE, VERIFIED

Gradient SHAP on SNAX cluster (RISC-V scalar core, float32, N=16 samples):
- **58,022 cycles** (Optimized Phase 2b), 3.02x speedup over 2a (175,510)
- BIST: 0/256 errors (max_err=0.000000)
- Per-stage: zero=1,036 bwd=1,930 accum=53,129 norm=1,787
- Design doc: `docs/phase2/PHASE2_DESIGN.md`, results: `docs/phase2/PHASE2_RESULTS.md`

### Known Working Commands
```bash
# Generate test data
python snax_cluster/sw/xai/gradcam/data/datagen.py -c snax_cluster/sw/xai/gradcam/data/params.hjson > snax_cluster/sw/xai/gradcam/data/data.h
python snax_cluster/sw/xai/shap/data/datagen.py -c snax_cluster/sw/xai/shap/data/params.hjson > snax_cluster/sw/xai/shap/data/data.h

# Build + simulate (from snax_cluster/)
make -C target/snitch_cluster/sw/apps/xai/gradcam all
make -C target/snitch_cluster/sw/apps/xai/shap all

# Check results in simulation log
grep "Cycles:\|Errors\|SHAP" target/snitch_cluster/logs/*.log
```

## Phase 3 Status: COMPLETE

Formal verification of quantized neural networks (host-side Python):
- **33/33 tests passing** across quantization, bound propagation, ILP verification
- QVIP-style ILP verification: local robustness + maximum robustness radius
- Symmetric INT8 quantization matching SNAX GeMM scheme
- Interval analysis for constraint reduction (ReLU neuron pruning)
- LP relaxation with triangle relaxation for 2-layer networks
- XAI integration: saliency-guided verification of high-importance regions
- Design doc: `docs/phase3/PHASE3_DESIGN.md`

### Key Files
- `src/xai/formal/quantization.py` — INT8 quantization (symmetric + QVIP uniform)
- `src/xai/formal/bound_propagation.py` — Layer-by-layer interval bounds
- `src/xai/formal/qvip_verifier.py` — ILP robustness verification engine
- `tests/test_formal_verification.py` — 33 tests

## Benchmarks

- MLPerf Tiny v1.0: ToyAdmos (anomaly detection), ResNet-8 (image classification)
- Target: XAI overhead < 20% of inference latency at >90% accelerator utilization
