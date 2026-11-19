# Grad-CAM on the SNAX GeMMX Accelerator

**Status:** working, verified bit-exact (2026-11-19)
**Cluster:** `snax_KUL_cluster` — 1 GeMMX-controller core + 1 DMA core, both `rv32ima` (no FPU); GeMMX 8×8×8 int8 mesh, int32 accumulate, SIMD requantiser.

## Mapping

| Step | Operation | Where |
|------|-----------|-------|
| GeMM1 | `alpha[1×K] = requant(d_logit[1×C] @ Wᵀ[C×K])` | GeMMX, SIMD on (int32 → int8, `>> ALPHA_SHIFT`) |
| GeMM2 | `cam[HW×1] = fmaps[HW×K] @ alpha[K×1]` | GeMMX, int32 out |
| post | ReLU + Q15 normalise (one integer divide) | core |

**No core work between the GeMMs.** GeMM1's SIMD int8 output (block layout `(Mt,Nt,row,col)`) has the same bytes as GeMM2's B operand (`(Nt=1,Kt,col,k)`): row 0 is alpha (column 0 of B2) and rows 1–7 are zero padding from the padded `d_logit`. GeMM1 writes straight into B2.

**Static requantisation shift.** `d_logit` is one-hot with value 1, so `grad` is a column of W for any target class; its int8 range is a property of W, computed once in datagen (`ALPHA_SHIFT`). SIMD config `mult=2, shift=ALPHA_SHIFT+1` gives exactly `x >> ALPHA_SHIFT`.

**Integer only.** The cores have no FPU, so the pipeline is int8/int32 end to end; the datagen golden output is bit-exact with the kernel. Quantisation error vs the float32 golden: ≤ 0.27% (GAP+FC), ≤ 0.20% (ResNet-8).

Weights are pre-transposed and pre-packed into GeMMX block layout offline (they are static), so GeMMX pays no runtime transpose. The fp32 scalar kernel transposes at runtime (1,124 cycles).

## Results (Verilator, warm I-cache, cycles)

The same integer pipeline also runs as plain scalar loops on the same core for a like-for-like comparison. Both paths: 0 BIST errors, cold and warm.

**GAP+FC test model** (4×4×16, C=10):

| Path | GeMM1 | glue | GeMM2 | post | **total** |
|------|------:|-----:|------:|-----:|----------:|
| GeMMX | 3,116 | 7 | 2,889 | 590 | **6,602** |
| scalar int8 | 1,167 | 120 | 2,094 | 481 | **3,862** |

GeMMX is **1.7× slower**: the problem is too small to amortise the launches.

**ResNet-8 hook layer** (8×8×64, C=10):

| Path | GeMM1 | glue | GeMM2 | post | **total** |
|------|------:|-----:|------:|-----:|----------:|
| GeMMX | 3,051 | 20 | 3,009 | 1,963 | **8,043** |
| scalar int8 | 4,482 | 714 | 62,023 | 1,759 | **68,978** |

GeMMX is **8.6× faster**; the weighted-sum GeMM alone is **20.6× faster** (62,023 → 3,009).

## Where the GeMMX cycles go

| | GAP+FC | ResNet-8 |
|---|---:|---:|
| accelerator busy (GeMM1 + GeMM2) | 26 + 10 | 74 + 77 |
| per-launch cost (config + start + wait) | ~3,000 | ~3,000 |
| accelerator utilisation in the GeMMX path | 0.5% | 1.9% |

**Launch overhead dominates.** Each launch is ~3,000 cycles of CSR configuration (dozens of streamer/GeMM CSR writes, many in loops, via `set_gemmx_streamer_csr` / `set_gemmx_csr`) regardless of size; the mesh itself is busy for tens of cycles. The project target of >90% accelerator utilisation needs launch amortisation: larger or batched GeMMs (e.g. Grad-CAM for several classes or images per launch), or overlapping the next configuration with the running GeMM.

Optimisation history (ResNet-8, warm total): 20,734 (glue on core, runtime divides) → 9,371 (GeMM1 SIMD writes B2 directly) → 8,043 (stride-8 gather fused with ReLU/max).

## Caveats

- Scalar int8 on this core costs ~15 cycles/MAC (62,023 cycles / 4,096 MACs) vs ~7.5 cycles/MAC for fp32 `fmadd` on the plain cluster's FPU cores. Comparing GeMMX against the plain-cluster fp32 kernel is not like-for-like; the scalar int8 path on the same core is.
- The plain-cluster fp32 Grad-CAM at ResNet-8 size has not been simulated (the `gradcam_resnet8` app was never built); the cost model estimates ~41.9k cycles.
- Inference is still not simulated on GeMMX. The measured ~3,000-cycle launch cost is far above the cost model's `gemm_launch_overhead = 400`, so the model's inference estimates are likely optimistic.

## Reproduce

```bash
scripts/snax_cluster_mode.sh gemmx          # switch sim binary + runtime to snax_KUL_cluster
cd snax_cluster
pixi run make -C target/snitch_cluster/sw/apps/xai/gradcam_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson [SIZE=resnet8] all
cd target/snitch_cluster
./bin/snitch_cluster.vlt sw/apps/xai/gradcam_gemmx/build/gradcam_gemmx[_resnet8].elf   # exit code = BIST errors
cd ../../.. && scripts/snax_cluster_mode.sh plain   # back to the plain cluster for Phases 1/2/4
```

Building the GeMMX simulator from scratch (needs ~10 GB disk, ~6 GB RAM):
```bash
pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson
# If generated/bender_targets.tmp exists from another config, regenerate it:
pixi run util/snaxgen/snaxgen.py --cfg_path=$(realpath target/snitch_cluster/cfg/snax_KUL_cluster.hjson) --get_bender_targets > target/snitch_cluster/generated/bender_targets.tmp
pixi run make CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson -C target/snitch_cluster bin/snitch_cluster.vlt -j2
cp target/snitch_cluster/bin/snitch_cluster.vlt target/snitch_cluster/bin/snitch_cluster.kul.vlt
```

## Files

| File | Purpose |
|------|---------|
| `snax_cluster/sw/xai/gradcam_gemmx/data/datagen.py` | block packing, streamer CSR values, static shift, bit-exact golden; self-checks against `snax_utils` golden models |
| `snax_cluster/sw/xai/gradcam_gemmx/data/params*.hjson` | GAP+FC and ResNet-8 shapes |
| `snax_cluster/sw/xai/gradcam_gemmx/src/main.c` | GeMMX path + scalar int8 path, cold/warm timing, bit-exact BIST |
| `snax_cluster/target/snitch_cluster/sw/apps/xai/gradcam_gemmx/Makefile` | target app (links `snax-gemmx-lib.o`) |
| `snax_cluster/target/common/common.mk` | local wrapper patch made conditional so SNAX configs build |
| `scripts/snax_cluster_mode.sh` | switch between plain and GeMMX clusters |
