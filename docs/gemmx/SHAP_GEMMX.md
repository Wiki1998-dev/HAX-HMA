# Gradient SHAP on the SNAX GeMMX Accelerator

**Status:** working, verified bit-exact (2026-11-19)
**Cluster:** `snax_KUL_cluster` (same simulator as `GRADCAM_GEMMX.md`): GeMMX-controller core + DMA core, both `rv32ima` (no FPU).

## Idea: the sample loop is a GeMM

The optimised (Phase 2b) SHAP hoists the backward pass: for a linear head (GAP+FC) the gradient `g` is the same for every sample. Then

```
attr = (1/N) Σᵢ (x − bᵢ) ⊙ g  =  (1/N) · g ⊙ (N·x − Σᵢ bᵢ)
```

and `N·x − Σᵢ bᵢ` is one matrix product:

```
S[1 × E] = [−1 … −1, N][1 × (N+1)]  @  [b₁; …; b_N; x][(N+1) × E]
```

So the N-sample accumulation loop (53k of the 58k fp32 Phase 2b cycles) becomes **one GeMMX launch**, plus one elementwise pass on the core: `attr[e] = g[k] · S[e]`.

| Step | Where |
|------|-------|
| `g` = column `target` of W (hoisted backward) | core (16 loads) |
| scatter runtime input `x` into B's last row | **DMA core**, overlapped with core 0's CSR writes |
| `S = A @ B` (int8 in, int32 out) | GeMMX |
| `attr = g[k] · S[e]` | core |

The baselines are a static background set, so they are pre-packed offline into GeMMX's B block layout. Only `x` changes at runtime. The output is the unscaled int32 attribution; `1/N`, `1/(H·W)` and the quantisation scales are one host-side constant. Quantisation error vs float32: 0.54% of max |attr|.

## Results (Verilator, GAP+FC 4×4×16, N=16, warm I-cache, cycles)

| Path | bwd | GeMM / accum | post | **total** |
|------|----:|-------------:|-----:|----------:|
| GeMMX | 137 | 3,178 | 2,509 | **5,824** |
| scalar int8, same core | 140 | 22,533 | — | **22,673** |

- **3.9× faster** than the same integer algorithm on the same core. Both paths have 0/256 BIST errors (bit-exact), cold and warm.
- The DMA core's `x` scatter takes 1,381 cycles and is fully hidden behind the ~3,000-cycle CSR configuration on core 0.
- The accelerator is busy for 117 cycles; launch overhead is still most of the GeMM time.
- For reference, Phase 2b fp32 on the plain cluster is 58,022 cycles. That run uses a different core with an FPU, float arithmetic and the per-sample loop, so it is not like-for-like.

Optimisation history (warm total): 7,927 (GeMM computes `−Σbᵢ` and the core adds `N·x`; post 18 cycles/element) → 5,824 (`x` folded into the GeMM via the DMA core; k-outer post loop keeps `g[k]` in a register; post 9.8 cycles/element).

## Limits

- **Linear heads only.** With a non-linear head the gradient depends on each sample, so `Σᵢ gᵢ ⊙ (x − bᵢ)` is no longer a matrix product. ToyAdmos falls into this case.
- Post is now 43% of the total. It's bounded by the core: one load at a time (`num_int_outstanding_loads: 1`) and a 4-cycle `mul`. A per-element `g[k]` scale doesn't fit GeMMX's per-column SIMD multiplier when K > 8.
- Only the GAP+FC size is simulated. At the ResNet-8 hook size (8×8×64) the int8 baselines alone are 64 kB, so they would need tiling within the 128 kB TCDM.

## Reproduce

```bash
scripts/snax_cluster_mode.sh gemmx
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/shap_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson all
cd target/snitch_cluster && ./bin/snitch_cluster.vlt sw/apps/xai/shap_gemmx/build/shap_gemmx.elf   # exit code = BIST errors
cd ../../.. && scripts/snax_cluster_mode.sh plain
```

## Files

| File | Purpose |
|------|---------|
| `snax_cluster/sw/xai/shap_gemmx/data/datagen.py` | quantisation, B packing with x-scatter offsets, bit-exact golden; self-checks against `block_gemm_golden_model`; reuses the Grad-CAM GeMMX packing helpers |
| `snax_cluster/sw/xai/shap_gemmx/src/main.c` | GeMMX path (DMA-core scatter, GeMM, post) + scalar int8 path, cold/warm timing, BIST |
| `snax_cluster/target/snitch_cluster/sw/apps/xai/shap_gemmx/Makefile` | target app (links `snax-gemmx-lib.o`) |
