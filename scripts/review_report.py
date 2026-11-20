#!/usr/bin/env python3
"""Assemble docs/REVIEW_EXPERIMENTS.md from the raw outputs in
docs/review_raw/ (written by scripts/review_experiments.py and friends).

    python scripts/review_report.py
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "docs/review_raw")
PY = "python"   # .venv/bin/python, run as: env -u PYTHONHOME -u PYTHONPATH .venv/bin/python


def raw(name: str) -> str:
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return f"(missing: docs/review_raw/{name})"
    return open(p).read().rstrip()


def block(name: str) -> str:
    return "```\n" + raw(name) + "\n```\n"


ITEMS = []


def item(title: str, cmd: str, seeds: str, body: str, files) -> None:
    ITEMS.append((title, cmd, seeds, body, files))


item("1. ResNet-8 accuracy on the full CIFAR-10 test set",
     f"{PY} scripts/review_experiments.py 1 --workers 2",
     "none (all 10,000 test images, deterministic)",
     """- **Top-1:** GeMMX integer semantics 0.8704, 95% Wilson CI [0.8637, 0.8768]. TFLite int8 0.8702, CI [0.8635, 0.8766].
- **Differences:** 187 of 10,000 predictions differ. Of these, GeMMX is right on 77 and TFLite on 75.
- **Logits vs softmax:** TFLite's argmax over the FC logits and over the softmax output never disagree, so int8 softmax ties play no role.
""", ["item1.txt"])

item("2. Faithfulness with confidence intervals",
     f"{PY} scripts/review_experiments.py r8 --images 200 --workers 2   # per-image data, cached\n"
     f"{PY} scripts/review_experiments.py 2 --images 200\n"
     f"{PY} scripts/resnet8_gradcam_fidelity.py\n"
     f"{PY} scripts/toyadmos_xai_fidelity.py --clips 50\n"
     f"{PY} scripts/resnet8_xai_fidelity.py --images 200",
     "ResNet-8 baselines rng(100 + k) for image k; ToyAdmos clips rng(3), baselines rng(11 + clip), calibration "
     "rng(7) (the scripts' own seeds); bootstrap rng(0), 10,000 resamples over inputs",
     """The harness recomputes each script's loop with the scripts' functions and seeds, keeping per-input values for the bootstrap. The scripts themselves were also rerun unchanged; their output is at the end of this item.

**Check against the paper:**

| Paper value | Now: mean [95% CI] | Matches? |
|---|---|---|
| Grad-CAM r 0.997 | 0.9974 [0.9971, 0.9977] | yes |
| ResNet-8 IG r 0.946 | 0.9444 [0.9413, 0.9475] | yes |
| ResNet-8 SHAP r 0.916 | 0.9126 [0.9093, 0.9158] | yes, at the edge of the CI |
| ToyAdmos IG r 0.731 | 0.7315 [0.6684, 0.7919] | yes |
| ToyAdmos SHAP r 0.700 | 0.7072 [0.6428, 0.7695] | the paper's value is inside the CI |

- **ResNet-8 IG / SHAP:** the paper's numbers came from the first 50 images; these are the first 200, which include those 50.
- **ToyAdmos SHAP:** the current code gives a mean of 0.707 on the same 100 windows, not the 0.700 in the paper and docs. The unchanged script, rerun now, also prints 0.707.
- **Unchanged scripts:** all three print the same means as this harness (Grad-CAM 0.997, ToyAdmos 0.731 / 0.707, ResNet-8 at 200 images 0.944 / 0.913).
- **Minima:**
  - ToyAdmos IG r −0.264 and SHAP r −0.342. Some windows get an attribution anti-correlated with float32.
  - ResNet-8 IG r 0.852, SHAP r 0.835, Grad-CAM r 0.989.
""", ["item2.txt", "item2_unchanged_scripts.txt"])

item("3. ToyAdmos calibration ablation on held-out normal training windows",
     f"{PY} scripts/review_experiments.py 3",
     "calibration rng(7) (as the repo); held-out windows rng(21) from the 19,056 training windows the calibration "
     "does not touch; SHAP baselines rng(1000 + j) from the remaining unused windows; bootstrap rng(0)",
     """**Setup:**
- The four cells re-run the repo's calibration code with the percentile and the multiplier form as parameters. The (exact, 99.5) cell is asserted equal to the repo's calibration.
- All 100 evaluation windows are normal *training* windows that the calibration did not use, so no test data enters the choice.

**Result:** the configuration the repo uses (exact multiplier, 99.5th percentile) is still the best cell for both IG (r 0.743) and SHAP (r 0.703).
- **Power-of-two shift at 99.5:** statistically indistinguishable (paired difference IG −0.003 [−0.011, +0.004], SHAP −0.002 [−0.013, +0.009]).
- **Exact at 99.9:** significantly worse (IG −0.033 [−0.059, −0.010], SHAP −0.041).
- **Power-of-two at 99.9:** significantly worse still (−0.10).

The choice made on test windows therefore holds on held-out training data.
""", ["item3.txt"])

item("4. IG completeness",
     f"{PY} scripts/review_experiments.py 4 --images 200",
     "same inputs and seeds as item 2 (ResNet-8: 200 images, black baseline; ToyAdmos: the 100 windows, mean "
     "normal spectrum baseline)",
     """Relative error |Σ attr − (f(x) − f(b))| / |f(x) − f(b)|:

| | Median | 90th pct |
|---|---:|---:|
| ResNet-8, float32 | 0.276 | 2.51 |
| ResNet-8, int8 pipeline (vs its own logits) | 0.415 | 2.60 |
| ToyAdmos, float32 | 0.091 | 0.70 |
| ToyAdmos, int8 pipeline (vs its own f) | 0.044 | 0.12 |

**ResNet-8.** The float32 error is already large, and int8 adds 0.14 at the median. The tail is not a quantisation effect (`item4_diag.txt`):
- The 38 images with relative error above 1 have a small |f(x) − f(b)| (median 0.57, against 4.39 for the rest).
- In 42 images the black baseline scores *higher* than x.
- In absolute terms the float32 error is a median of 1.0 logit, and Σ attr / Δf has a median of 1.07.
- A 16-step right-endpoint Riemann sum overshoots on this network.

**ToyAdmos.** The int8 pipeline is closer to its own Δf than float32 is.
- The integer attribution was converted to real units analytically: K = 2 s_x · (2/640) · S_gy / (m1 · N).
- That conversion is consistent: the median ratio of the int8 to the float32 attribution sums is 0.944.
- The int8 Riemann sum uses the int8 model's own path, which is flatter (clipped activations), so the 16-step sum fits better. This says nothing about agreement with float32 (item 2).
""", ["item4.txt", "item4_diag.txt"])

item("5. Upper-clamp saturation",
     f"{PY} scripts/review_experiments.py 5 --images 200",
     "same inputs as item 2",
     """**Finding:** saturation at +127 at masked positions is essentially absent, but not zero.
- **ToyAdmos** (100 windows x 16 samples x 9 ReLU layers): never reaches +127.
- **ResNet-8 IG** (200 images x 16 samples x 7 masked tensors): never reaches +127.
- **ResNet-8 SHAP:** reaches +127 in 3 of 200 images, only in the first residual add, at most 0.0008% of its activations. That is why the per-tensor means print as 0.000%.

**Effect of also masking +127:** negligible. The r values are identical to 4 decimals. ToyAdmos: 0 windows change. ResNet-8 IG: 0 images change. ResNet-8 SHAP: the 3 images above change by less than 1e-5 in r.

For contrast, the zero-point fraction (gradient masked) is 25–73% per tensor.
""", ["item5.txt"])

item("6. Utilisation check (MACs launched)",
     f"{PY} scripts/review_experiments.py 6",
     "none (schedule generators; ToyAdmos input = toyadmos.select_input, ResNet-8 image 0 (inference) / 1 (XAI), "
     "KWS / VWW sample 0)",
     """**MACs launched** (Σ M·N·K blocks × 512, over every launch the generators emit):

| Program | Reviewer | Measured | Useful / launched |
|---|---|---:|---:|
| ResNet-8 inference + Grad-CAM | 14.82M | **13.77M** | 0.908 |
| ResNet-8 IG / SHAP (16 samples) | — | 444.7M | 0.900 |
| ToyAdmos inference | — | 2.11M (8 mesh rows for batch 1) | 0.125 |
| ToyAdmos IG / SHAP | — | 9.27M | 0.912 |
| KWS inference + Grad-CAM | — | 9.06M | 0.294 (depthwise diagonal blocks) |
| VWW inference + Grad-CAM | — | 18.33M | 0.409 |

**Useful-MAC utilisation** (useful MACs / (cycles × 512)): the reviewer's analytic figures are confirmed.

| Program | Reviewer | Measured |
|---|---:|---:|
| ToyAdmos inference | 5.6% | 5.6% |
| ToyAdmos XAI | 18% | 18.0% |
| ResNet-8 inference + Grad-CAM | 43% | 42.5% |
| ResNet-8 XAI | 13% | 13.0% |
| KWS inference + Grad-CAM | — | 9.7% |
| VWW inference + Grad-CAM | — | 7.7% |

**What "GeMMX busy" measures.** Core 0 reads `mcycle` after the generated CSR writes, then calls `set_gemmx_streamer_start(); set_gemmx_start(); wait_gemmx_and_streamer();` and reads `mcycle` again (`run` in the printouts). "Busy" is the sum of these windows over the total.
- It counts every cycle the streamers and the array are active for a launch: pipeline fill and drain, TCDM bank conflicts and streamer stalls.
- So it is an upper bound on MAC-array activity, not a MAC count.
- Example: ResNet-8 inference + Grad-CAM is 67% busy, but only 46.8% of the cycles would be needed at one 8×8×8 block per cycle for the launched blocks, and 42.5% for the useful ones.
""", ["item6.txt"])

item("7. Data independence of the core-side loops",
     "inspection of snax_cluster/sw/xai/*/src/*.c and of the compiled binary "
     "(target/.../resnet8_gemmx/build/resnet8_shap_n16_s1.dump)",
     "n/a",
     """**No data-dependent branches** in the ReLU mask (SWAR), the attribution MAC or the interleave (static 2-D DMA descriptors). In the disassembly, the only branches in these loops compare a counter or pointer against a constant bound.

**Data-dependent steps that do exist**, once per explanation:
- the argmax that picks the class;
- the Grad-CAM finish (ReLU, running max, one division).

This is consistent with IG and SHAP measuring identical cycle counts on different data.
""", ["item7.txt"])

item("8. (Optional) Grad-CAM at an intermediate layer",
     f"{PY} scripts/review_experiments.py 8",
     "none (first 200 MLPerf samples)",
     """**Faithfulness: measured.** Grad-CAM on the block-2 output (16×16×32):
- The int8 backward pass goes through block 3, using the same per-stage functions as the IG pipeline: transposed convs, mask, block-diagonal fork sum, with alpha = the spatial sum of d logit / d add2.
- Result: r 0.984 (min 0.901, 95% CI [0.982, 0.985]) against float32 autograd, on 200 images.
- That is lower than 0.997 at the linear head, as expected, since a real backward pass now sits in between.

**Cycles: not run.**
- No generated schedule exists for "forward + backward through block 3 only". The ResNet-8 XAI program runs the full backward to the input and its phase timers do not split per block.
- Measuring it needs a truncated variant of `resnet8_xai_sched.build` (stop after the add2 fork sum, then one Grad-CAM launch) and a Verilator run.
- No number is estimated here.
""", ["item8.txt"])

item("9. Sample-count sensitivity",
     f"{PY} scripts/review_experiments.py r8sweep --images 50 --workers 2   # ResNet-8 per-image data\n"
     f"{PY} scripts/review_experiments.py 9 --images 50\n"
     "# Verilator: PROG=toyadmos_ig_n8 / toyadmos_shap_n8 (new 8-sample programs), R8_METHOD=ig R8X_N=4 R8_SAMPLE=1",
     "ResNet-8 first 50 images, ToyAdmos item-2 windows; IG alpha = 1/N..1; SHAP baselines and alpha rng(100 + k) "
     "(ResNet-8) / rng(11 + clip) (ToyAdmos); N = 128 float32 references rng(500 + k / clip)",
     """**Cycles (Verilator, every output bit-exact):**

| | N=2 | N=4 | N=8 | N=16 |
|---|---:|---:|---:|---:|
| ToyAdmos IG / SHAP | — | — | **60,689** | 91,517 |
| ResNet-8 IG | 755,460 | **1,507,854** | — | 6,027,558 |

- **ToyAdmos:** 8 samples fill one mesh tile, but the cost is 66% of N=16, not 50%. Weight streaming, configuration and the per-layer launch count do not shrink with N; only the row-proportional work does.
- **ResNet-8:** linear in N (one step table per sample): 376,197 cycles per sample from N=2→4, 376,642 from N=4→16, intercept about 2–3k. Cost(N) ≈ 2.5k + 376.5k·N.
- **Caveat on the N=4 ResNet-8 run:** it takes the first 4 alphas of the 16-step table (16, 15, 14, 13). That is a valid cost measurement, since cycles do not depend on data (item 7), but not a 4-step IG.

**Faithfulness (emulator):** see the raw table; the cost-vs-faithfulness summary is in the final table.

- **ResNet-8 IG:** int8-vs-float32 r rises with N (0.908, 0.929, 0.946, 0.956). Float32 IG itself is converged at N=16 (r 0.997 to N=128).
- **ResNet-8 SHAP:** int8 tracks float32 at the same N (0.89–0.93). But SHAP at N=16 is far from its N=128 expectation even in float32 (r 0.59), so the sampling error of SHAP dominates the int8 error.
- **ToyAdmos:** int8-vs-float32 does not improve with N (0.67–0.78 for IG). Float32 IG at N=16 is only r 0.77 to N=128.
""", ["item9.txt", "item9_sim.txt"])

item("10. Baseline sensitivity",
     f"{PY} scripts/review_experiments.py 10 --images 50",
     "ResNet-8 first 50 images (dataset mean = mean of the 512 CIFAR-10 training images of resnet8_xai.cifar_train(); "
     "blur = Gaussian sigma 4 px per channel); ToyAdmos item-2 windows (zeros = real 0 quantised; mean = mean normal "
     "training spectrum)",
     """**IG faithfulness, int8 vs float32 with the same baseline:**

| Baseline | ResNet-8 r [95% CI] | ToyAdmos r [95% CI] |
|---|---|---|
| black | 0.946 [0.940, 0.951] | — |
| dataset mean | 0.950 [0.945, 0.954] | — |
| blurred input | 0.957 [0.953, 0.960] | — |
| mean normal spectrum | — | 0.732 [0.668, 0.792] |
| zeros | — | 0.621 [0.598, 0.643] |

**Cycles:** confirmed unchanged. For each baseline the generator emits byte-identical launch configuration code, step table and DMA table (ResNet-8: black / mean / blur; ToyAdmos: mean / zeros). Since the loops are data-independent (item 7), the cycle count is the same.
""", ["item10.txt"])

item("11. Cheap baseline methods",
     f"{PY} scripts/review_experiments.py 11 --images 50",
     "as item 9",
     """**ResNet-8, gradient × input with one sample at x:**
- Agreement: int8 r 0.678 [0.661, 0.694] against float32 IG (N=16).
- Most of the gap is the method itself: float32 gradient × input vs float32 IG is 0.717. The int8 effect alone, int8 vs float32 gradient × input, is 0.897.
- Cost: one sample of the IG program. Measured sample 0 was 378,443 cycles inside the 16-sample run; the linear fit in item 9 gives about 379k for an N=1 program. That is 7.1× inference, against 112× for IG with N=16.

**ToyAdmos, per-feature reconstruction error (x − y)² from the int8 inference:**
- Agreement: r 0.614 against float32 IG (0.665 against |IG|), versus 0.732 for int8 IG.
- Cost: the inference (9,227 cycles) plus 640 squared differences on a core, run once per window. This was not simulated, so no cycle number is given.
""", ["item11.txt"])

item("12. Higher-precision gradients (ToyAdmos, emulator)",
     f"{PY} scripts/review_experiments.py 12\n{PY} scripts/review_ta_diag.py",
     "item-2 windows and seeds",
     """**Result:** int16 gradients do not help, and float gradients make it worse.

| Gradient precision | IG r | SHAP r |
|---|---:|---:|
| int8 everywhere (as built) | 0.7315 | 0.7072 |
| int16 all stages | 0.7352 | 0.7079 |
| int16 on 3 stages, either end | 0.732–0.733 | ≈ 0.708 |
| float all stages | 0.584 | 0.586 |
| float on the 3 input-side stages | 0.602 | 0.600 |

**Diagnosis** (`review_ta_diag.py`). The backward pass was run in float, with pieces of the int8 forward swapped in one at a time:

| Swapped in | IG r |
|---|---:|
| nothing | 1.000 |
| int8 ReLU masks | 0.633 |
| int8 reconstruction y | 0.722 |
| int8-rounded interpolated inputs x̃ | 0.643 |
| all three | 0.584 |

The last row equals "float all stages", so the two experiments agree.

**Conclusion:**
- The ToyAdmos gap comes from the int8 *forward* pass of the 10-layer autoencoder, not from gradient precision.
- The int8 backward's saturation at the 99.5th percentile clips the outliers this causes, which is why int8 gradients beat float ones here.
- An int16 gradient path is not worth building for this model.

**Launch estimate** (stated as an estimate, not measured). The built schedule has 18 backward GeMM launches per explanation. An int16 × int8 product on the int8 array, with the gradient split into two int8 halves g = 256·g_h + g_l, needs:
- two GeMM launches per current launch, each with an int32 D32 output (the SIMD int8 path cannot produce int16);
- a combine 256·acc_h + acc_l, either on the cores or as an extra block-diagonal launch (the factor 256 exceeds the int8 range of the B pair, so two ×16 steps);
- int16 requantisation and masking on the cores.

That is at least 36 GeMM launches plus 18 combine passes, against 18 today.
""", ["item12.txt", "item12_diag.txt"])

item("13. Perturbation metrics (deletion / insertion)",
     f"{PY} scripts/review_experiments.py 13 --images 200 --workers 2",
     "maps from item 2 (200 images); random maps rng(900 + k)",
     """**Setup:**
- Scored with the float32 dequantised model: softmax probability of the predicted class.
- Pixels are ranked by the channel-summed attribution and changed in 32 steps of 32 pixels.
- Deletion fills black; insertion starts from a blurred copy (sigma 4).
- Lower deletion AUC and higher insertion AUC are better.

**Results (mean AUC [95% CI]):**

| Map | Deletion (lower is better) | Insertion (higher is better) |
|---|---|---|
| IG int8 | 0.142 [0.110, 0.177] | 0.305 [0.271, 0.340] |
| IG float32 | 0.141 [0.110, 0.175] | 0.296 [0.261, 0.331] |
| SHAP int8 | 0.163 [0.128, 0.200] | 0.343 [0.306, 0.383] |
| SHAP float32 | 0.160 [0.126, 0.198] | 0.340 [0.303, 0.378] |
| random | 0.157 [0.122, 0.195] | 0.280 [0.248, 0.314] |

**int8 is as useful as float32.** The paired int8 − float32 differences are not significant, except IG insertion, where int8 is slightly *better* (+0.009 [+0.003, +0.016]).

**Caveat about the methods themselves on this model:**
- On deletion, IG beats random only modestly, and SHAP does not beat random.
- On insertion, both beat random, SHAP by more.
- On 32×32 CIFAR images with a 0.87-accuracy ResNet-8, pixel-level IG / SHAP rankings are weak perturbation explanations whatever the precision. The review's question (is int8 as useful as float32?) is answered yes. Whether these methods are useful at all on this model is a separate, weaker result.
""", ["item13.txt"])

item("14. L3 traffic",
     f"{PY} scripts/review_experiments.py 14",
     "ResNet-8 image 1, IG (rng(101)); ToyAdmos select_input, IG",
     """**ResNet-8 IG, one explanation (16 samples):** 7.19 MB moved between L3 and TCDM.

| Category | Bytes | Share |
|---|---:|---:|
| weights | 2.84 MB | 40% |
| activation spills | 2.33 MB | 32% |
| mask reads | 1.18 MB | 16% |
| sample inputs + attribution inputs | 0.84 MB | 12% |

Also inside the TCDM: polyphase interleave 0.49 MB and border fills 5.35 MB.

**ToyAdmos IG:** 0.58 MB of weight tiles streamed; everything else stays in the TCDM.

**DMA busy / stall fraction: not measured.**
- The simulated cluster gives software no DMA performance counters.
- The runs kept no per-cycle traces (linked to /dev/null to fit the disk).
- The phase timers lump DMA waits together with the core work they overlap. For example, the ResNet-8 mask phase (48%) includes the 2-D activation fetches and the core masking, separated by barriers.
""", ["item14.txt"])

item("15. Software 1-bit masks",
     f"{PY} scripts/review_experiments.py 15",
     "ResNet-8 image 1, IG",
     """**Bytes (computed from the generated schedule):**
- Masks need 73,728 elements per sample, 9,216 B as bits.
- That removes 64,512 B of L3 mask reads per sample (1.03 MB per explanation).
- It replaces 145,920 B of activation spills with 9,216 B of bit-mask writes.
- Together: −201 kB of L3 traffic per sample, −3.2 MB of the 7.2 MB per explanation.

**TCDM space:** the schedule already uses 120,448 B of its 124,928 B limit, so 9 kB of masks per sample do not fit next to it. The bits would have to go to L3 (1/8 of today's bytes) or the buffers would have to be re-planned.

**Cycles: not measured.**
- The change needs new core code: packing bits after each forward ReLU layer (core work the forward pass does not do today) and a bit-expanding mask kernel.
- The mask phase (48% of ResNet-8 XAI) mixes DMA wait and core SWAR time, which the timers do not separate (item 14).
- So no cycle change is given; it needs an implementation and a Verilator run.
""", ["item15.txt"])


SUMMARY = """## Summary of new numbers

| # | Item | Result |
|---|---|---|
| 1 | ResNet-8 top-1, 10,000 test images | GeMMX 0.8704 [0.8637, 0.8768]; TFLite 0.8702 [0.8635, 0.8766]; 187 predictions differ (77 vs 75 correct) |
| 2 | Grad-CAM r (200 images) | 0.9974 [0.9971, 0.9977], min 0.989 |
| 2 | ResNet-8 IG / SHAP map r (200 images) | 0.944 [0.941, 0.948] (min 0.852) / 0.913 [0.909, 0.916] (min 0.835) |
| 2 | ToyAdmos IG / SHAP r (100 windows) | 0.732 [0.668, 0.792] (min −0.264) / 0.707 [0.643, 0.770] (min −0.342) |
| 3 | Calibration, held-out training windows (IG / SHAP r) | exact@99.5: 0.743 / 0.703 (best); pow2@99.5: 0.740 / 0.701 (n.s.); exact@99.9: 0.710 / 0.663; pow2@99.9: 0.641 / 0.602 |
| 4 | IG completeness, median / p90 rel. error | ResNet-8 float32 0.28 / 2.51, int8 0.41 / 2.60 (small Δf in the tail); ToyAdmos float32 0.09 / 0.70, int8 0.04 / 0.12 |
| 5 | Activations at +127 at masked positions | none in ToyAdmos or ResNet-8 IG; 3 of 200 ResNet-8 SHAP images (<= 0.0008% of add1); masking them changes r by < 1e-5 |
| 6 | MACs launched | ResNet-8 inference + Grad-CAM 13.77M (not 14.82M); useful utilisation confirmed: 5.6% / 18.0% / 42.5% / 13.0%; KWS 9.7%, VWW 7.7% |
| 7 | Data-dependent branches in mask / MAC / interleave | none (verified in the binary) |
| 8 | Grad-CAM at the block-2 output, r | 0.984 [0.982, 0.985] (cycles not run) |
| 9 | Cycles vs N | ToyAdmos N=8 60,689 (66% of N=16); ResNet-8 ≈ 2.5k + 376.5k·N (N = 2, 4, 16 measured) |
| 9 | ResNet-8 IG r, N = 4/8/16/32 | 0.908 / 0.929 / 0.946 / 0.956 |
| 9 | ResNet-8 SHAP r, same N / vs float N=128 | 0.888–0.934 / 0.35–0.70 (float32 SHAP itself 0.36–0.71) |
| 9 | ToyAdmos IG r, N = 4/8/16/32 | 0.777 / 0.706 / 0.731 / 0.671 (no trend) |
| 10 | IG r by baseline | ResNet-8 black 0.946, mean 0.950, blur 0.957; ToyAdmos mean 0.732, zeros 0.621; schedule identical |
| 11 | Gradient × input (ResNet-8, 1 sample) | r 0.678 vs float32 IG at ~379k cycles (vs 6.03M for IG) |
| 11 | (x − y)² (ToyAdmos) | r 0.614 vs float32 IG (0.665 vs \\|IG\\|) |
| 12 | ToyAdmos int16 / float gradients | IG 0.735 / 0.584 (int8 0.732): error is in the int8 forward pass |
| 13 | Deletion / insertion AUC (int8 vs float32 vs random) | IG del 0.142 / 0.141 / 0.157, ins 0.305 / 0.296 / 0.280; SHAP del 0.163 / 0.160, ins 0.343 / 0.340; int8 = float32 (no significant difference except IG insertion +0.009 for int8) |
| 14 | L3 traffic per ResNet-8 IG explanation | 7.19 MB (weights 40%, spills 32%, mask reads 16%); ToyAdmos 0.58 MB |
| 15 | 1-bit masks | −3.2 MB of 7.2 MB L3 traffic per explanation; does not fit in the TCDM as built; cycles not measured |

Also new in this round (not requested by the review): MLPerf Tiny keyword spotting and visual wake words on GeMMX, inference + Grad-CAM, bit-exact. KWS 48,024 + 5,787 cycles (+12.0%), VWW 188,250 + 1,538 (+0.8%); see `docs/gemmx/MOBILE_GEMMX.md`.
"""


def main() -> None:
    out = ["# Review Experiments", "",
           "Results for the reviewer's experiment list. The paper is unchanged. Every number comes from a run in this "
           "repository: the bit-exact Python models of the GeMMX pipelines, or (items 6, 9 and the KWS / VWW runs) the "
           "Verilator simulation of `snax_KUL_cluster`. Items that could not be run say why; no number is estimated "
           "unless it is marked as an estimate.", "",
           "**Environment:**",
           "- Python 3.10 venv (numpy 1.26.4, scipy 1.15.3, torch 2.11 CPU, tflite-runtime).",
           "- Run as `env -u PYTHONHOME -u PYTHONPATH .venv/bin/python ...` from the repository root.",
           "- Harness: `scripts/review_experiments.py` (subcommand = item). It reuses the pipeline functions the "
           "simulator was checked against. Variants are local copies, first asserted equal to the original "
           "(`_ta_check_copy`, and the calibration assert in item 3).",
           "- Per-input data is cached in `external/review/` (git-ignored). Raw outputs are in `docs/review_raw/`.",
           "",
           "**Statistics:** CIs are 95% percentile bootstraps of the mean over inputs (10,000 resamples, rng(0)). "
           "Accuracy CIs are Wilson intervals.", ""]
    out.append(SUMMARY)
    for title, cmd, seeds, body, files in ITEMS:
        out += [f"## {title}", "", "```bash", cmd, "```", "", f"**Seeds:** {seeds}", "", body.rstrip(), ""]
        for f in files:
            out += [f"Raw output (`docs/review_raw/{f}`):", "", block(f)]
    open(os.path.join(ROOT, "docs/REVIEW_EXPERIMENTS.md"), "w").write("\n".join(out))
    print("wrote docs/REVIEW_EXPERIMENTS.md")


if __name__ == "__main__":
    main()
