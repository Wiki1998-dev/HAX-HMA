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
- **4,757 cycles** fp32, 0/16 BIST errors, H=4 W=4 K=16 C=10 (published 6,153 included a GAP-step out-of-bounds read, fixed 2026-11-19)
- INT8-weight variant (`GEMM_I8=1`): **6,923 cycles**, 0/16 errors — backward GeMM is 3.1× slower on the scalar core (GeMMX offload: see below)
- Build with BIST (from `snax_cluster/`): `RISCV_CFLAGS=-DBIST pixi run make -C target/snitch_cluster/sw/apps/xai/gradcam [GEMM_I8=1] all`, run `target/snitch_cluster/bin/snitch_cluster.vlt <elf>`. Do not `make clean` — it deletes the tracked `data.h`
- Python reference: 11/11 tests passing
- Key patches applied: `Xdiv_sqrt: true` in cluster config, core-gating fix (`snrt_cluster_core_idx() == 0`)
- Full results: `docs/phase1/PHASE1_SUMMARY.md`

## End-to-End Benchmark on GeMMX (real MLPerf Tiny models): MEASURED

Summary in `docs/gemmx/BENCHMARK.md`. Real int8 reference models + real data, bit-exact in Verilator:
- ToyAdmos (`ad01_int8`, DCASE ToyCar): inference 9,227 cycles; IG / Gradient SHAP 91,517 incl. fused inference; AUC 0.8406 (TFLite 0.8416); attribution r 0.73 / 0.70 vs float32
- ResNet-8 (`pretrainedResnet_quant`, CIFAR-10): inference ~53.6k; Grad-CAM +3,841 (7.2%); top-1 0.870 = TFLite; Grad-CAM r 0.997 vs float32
- ResNet-8 input-level IG / Gradient SHAP (16 samples): 6,027,558 cycles each; r 0.946 / 0.916 vs float32 (`R8_METHOD=ig|shap R8X_N=16`; link sim traces to /dev/null for long runs)
- Apps: `snax_cluster/sw/xai/toyadmos_gemmx/` (PROG=test_patterns|toyadmos_inference|toyadmos_ig|toyadmos_shap), `snax_cluster/sw/xai/resnet8_gemmx/` (R8_SAMPLE=k)
- `toyadmos_gemmx/data/gemmx_sched.py`: launch builder + bit-exact GeMMX emulator + delta CSR codegen (~77 vs ~2,300 cycles/launch, see `docs/gemmx/CONFIG_OVERHEAD.md`)
- Data: `external/` (git-ignored): MLPerf `tiny` clone, CIFAR-10 symlink, `toycar/` features (run `scripts/toyadmos_data.py` once)
- GeMMX hardware rules: unsigned strides; C reader always runs its loops; D8/B blocks 64 B aligned (granular banks); C/D32 two spatial strides; 3 temporal dims for C/D8/D32; no CSR writes during a run

## GeMMX Accelerator Grad-CAM: WORKING, VERIFIED

Grad-CAM with both matrix products on the SNAX GeMMX accelerator (`snax_KUL_cluster`, int8), see `docs/gemmx/GRADCAM_GEMMX.md`:
- ResNet-8 hook layer (8×8×64): **8,043 cycles vs 68,978 scalar int8 on the same core (8.6×)**; GAP+FC (4×4×16): 6,602 vs 3,862 (1.7× slower — launch overhead)
- Bit-exact BIST (0 errors) for both sizes, cold and warm
- **ResNet-8 inference on GeMMX** (`docs/gemmx/INFERENCE_GEMMX.md`, `scripts/run_inference_gemmx.sh`): 71,798 cycles measured, all layers 0 errors; Grad-CAM = 11.1% of it. Stock `snax-gemmx-conv` hangs on this cluster: use `sw/xai/inference_gemmx` (fixes documented there)
- **SHAP on GeMMX** (`docs/gemmx/SHAP_GEMMX.md`, app `sw/xai/shap_gemmx/`): hoisted SHAP's sample loop as one GeMM `[−1…−1,N] @ [b;x]`; GAP+FC N=16: **5,824 vs 22,673 cycles scalar int8 (3.9×)**, bit-exact; linear heads only
- ~3,000 cycles CSR config per GeMM launch dominates; accelerator busy <2% of the time
- Cluster cores are rv32ima (no FPU): kernel is integer-only (Q15 output)
- Switch clusters: `scripts/snax_cluster_mode.sh plain|gemmx|status` (plain = Phases 1/2/4; rebuild apps after switching)
- SNAX files are git-ignored by the outer repo: add new ones with `git add -f`

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

## Phase 4 Status: COMPLETE

Neuro-symbolic rule extraction (host-side Python + generated C):
- **33/33 tests passing**, including real gcc compile + run round-trip
- Saliency-guided decision tree distillation (sklearn backend, depth ≤ 3)
- Uses only top-K most-salient features from Grad-CAM/LRP/SHAP
- Two C export formats: nested if/else (inline) and flat node table
- Integer-only inference (no FPU) — thresholds quantized to fixed-point
- Measured embedded cycle count: **47 cycles** on SNAX (≈100× cheaper than Grad-CAM)
- Fidelity metrics: overall, per-class, coverage, saliency agreement
- Design doc: `docs/phase4/PHASE4_DESIGN.md`

### Key Files
- `src/xai/symbolic/rule_extractor.py` — distillation + TreeNode data model
- `src/xai/symbolic/rule_to_c.py` — C header exporter
- `src/xai/symbolic/fidelity_metrics.py` — faithfulness + coverage metrics
- `tests/test_rule_extraction.py` — 33 tests

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

## Phase 5 Status: IN PROGRESS

Benchmark framework + cycle cost model (host-side Python), see `docs/phase5/PHASE5_STATUS.md`:
- Cost model prices XAI kernels on the **scalar core** per stage; within 0.2% of all GAP+FC measurements (enforced at 5% by `_verify_calibration`)
- Inference cycles (GeMM path) are **uncalibrated**; GAP+FC inference 12.0k is a planning budget
- Non-GAP+FC rows are extrapolated estimates
- Tests: 279 passed, 3 skipped (slow), 0 xfail

## Benchmarks

- MLPerf Tiny v1.0: ToyAdmos (anomaly detection), ResNet-8 (image classification)
- Target: XAI overhead < 20% of inference latency at >90% accelerator utilization

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
