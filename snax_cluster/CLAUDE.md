# SNAX Cluster — Submodule Context

## What This Repo Is
Open-source multi-accelerator compute cluster template from KU Leuven MICAS-ESAT.
Paper: arXiv 2508.14582. Full context in ../papers/HW-SW_co.pdf.

## Directory Map
```
snax_cluster/
├── hw/
│   ├── chisel/         ← Chisel-generated RTL (DO NOT EDIT)
│   └── ip/             ← IP blocks: streamer, TCDM interconnect, DMA, CSR buffer
├── sw/
│   ├── snax-gemm/      ← GeMM accelerator C runtime + CSR driver
│   │   └── src/        ← main kernel files: gemm.c, streamer_csr.c
│   ├── snax-maxpool/   ← MaxPool accelerator runtime
│   └── runtime/        ← Shared: barrier.h, sync.h, csr.h macros
├── target/
│   ├── snax-gemm-cluster/    ← simulation target (Verilator)
│   └── snax-multi-cluster/   ← multi-accelerator target
└── util/
    └── cfg/            ← .hjson config files — THIS is how you customize HW
```

## How to Add a New Accelerator (from paper Section IV)
1. Edit `util/cfg/snax_cluster.hjson` — add accelerator entry with:
   - `num_cores`: how many RISC-V cores control it
   - `tcdm_ports`: data interface bandwidth
   - `streamer_params`: loop depth, FIFO size
2. Run `make -C target/snax-gemm-cluster hw` to regenerate RTL
3. Write CSR driver in `sw/<your-acc>/src/`
4. Write MLIR dialect description for the compiler

## CSR Programming Pattern (from paper Section IV-A)
```c
// Standard pattern for any SNAX accelerator
#include "snax_util.h"

void launch_kernel(uint32_t *a, uint32_t *b, uint32_t *c, int M, int N, int K) {
    // 1. Configure data streamers via CSR
    write_csr(STREAMER_A_BASE, (uint32_t)a);
    write_csr(STREAMER_A_LOOP0, M);   // outer loop
    write_csr(STREAMER_A_LOOP1, K);   // inner loop
    // 2. Configure compute kernel
    write_csr(GEMM_M, M);
    write_csr(GEMM_N, N);
    write_csr(GEMM_K, K);
    // 3. Fire-and-forget
    write_csr(GEMM_START, 1);
    // 4. Wait (barrier)
    snax_barrier();
}
```

## XAI Integration Points in This Repo

### Grad-CAM backward pass
Gradient of loss w.r.t. final conv feature map = element-wise multiply then
GeMM. Add second kernel in `sw/snax-gemm/src/gradcam.c`:
- Forward pass: existing GeMM
- Gradient: transpose-GeMM (reuse GeMM with transposed streamer config)
- Global avg pool: SIMD on RISC-V (simple loop, no accelerator needed)
- Weighted sum + ReLU: SIMD on RISC-V

### SHAP async dispatch
Use the loosely-coupled async model: dispatch N masked forward passes
back-to-back without waiting, then barrier once at the end.
Pattern: `for (int i=0; i<N_samples; i++) { launch_kernel(...); }  snax_barrier();`

### ZigZag cost estimation
Run `../zigzag/main.py` with the SNAX accelerator description file to get
predicted cycle counts for XAI ops BEFORE implementing them.
Reference accelerator YAML: `util/zigzag/snax_gemm.yaml` (create this).

## Key Files to Read Before Editing
- `sw/snax-gemm/src/gemm.c` — reference for CSR programming style
- `sw/runtime/include/snax_util.h` — barrier, CSR read/write macros
- `target/snax-gemm-cluster/Makefile` — how to build + simulate

## Simulation Commands
```bash
# Build and simulate GeMM kernel
make -C target/snax-gemm-cluster all

# Run specific test
make -C target/snax-gemm-cluster run TEST=tiled_matmul

# Check cycle count in output
grep "Cycles:" target/snax-gemm-cluster/logs/*.log
```

## Memory Budget for XAI (128 kB SPM)
```
Model activations:    ~40 kB  (ResNet-8 largest layer)
Model weights (L1):   ~32 kB  (tiled from external DRAM)
Gradient buffer:      ~20 kB  (same shape as activations)
SHAP baseline store:  ~16 kB  (N_samples × input tile)
Symbolic rules:        ~4 kB  (RISC-V scratchpad)
OS/runtime overhead:  ~16 kB
TOTAL:               128 kB   ← tight! use double-buffering carefully
```

## Phase 1 Status: COMPLETE, VERIFIED

Grad-CAM kernel verified on Snitch cluster via Verilator:
- **6,153 cycles** on single RISC-V core (float32), 0/16 BIST errors
- Source: `sw/xai/gradcam/` (kernel + data generation)
- Target: `target/snitch_cluster/sw/apps/xai/gradcam/`
- Config: `target/snitch_cluster/cfg/snitch_cluster.hjson` (Xdiv_sqrt enabled)

### Patches Applied
- `Xdiv_sqrt: true` on both core definitions (lines 107, 111) — enables FPU div/sqrt
- Compute guard: `snrt_cluster_core_idx() == 0` instead of `!snrt_is_dm_core()`
- Build dep: `$(DEP): $(ROOT)/sw/xai/gradcam/data/data.h` in target Makefile

## Do Not Edit
- `hw/` — regenerated from .hjson by hardware generator
- `Bender.yml` / `Bender.local` — package manager config
- `pixi.lock` — environment lockfile
