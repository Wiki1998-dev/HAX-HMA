# Review Experiments

Results for the reviewer's experiment list. The paper is unchanged. Every number comes from a run in this repository: the bit-exact Python models of the GeMMX pipelines, or (items 6, 9 and the KWS / VWW runs) the Verilator simulation of `snax_KUL_cluster`. Items that could not be run say why; no number is estimated unless it is marked as an estimate.

**Environment:**
- Python 3.10 venv (numpy 1.26.4, scipy 1.15.3, torch 2.11 CPU, tflite-runtime).
- Run as `env -u PYTHONHOME -u PYTHONPATH .venv/bin/python ...` from the repository root.
- Harness: `scripts/review_experiments.py` (subcommand = item). It reuses the pipeline functions the simulator was checked against. Variants are local copies, first asserted equal to the original (`_ta_check_copy`, and the calibration assert in item 3).
- Per-input data is cached in `external/review/` (git-ignored). Raw outputs are in `docs/review_raw/`.

**Statistics:** CIs are 95% percentile bootstraps of the mean over inputs (10,000 resamples, rng(0)). Accuracy CIs are Wilson intervals.

## Summary of new numbers

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
| 11 | (x − y)² (ToyAdmos) | r 0.614 vs float32 IG (0.665 vs \|IG\|) |
| 12 | ToyAdmos int16 / float gradients | IG 0.735 / 0.584 (int8 0.732): error is in the int8 forward pass |
| 13 | Deletion / insertion AUC (int8 vs float32 vs random) | IG del 0.142 / 0.141 / 0.157, ins 0.305 / 0.296 / 0.280; SHAP del 0.163 / 0.160, ins 0.343 / 0.340; int8 = float32 (no significant difference except IG insertion +0.009 for int8) |
| 14 | L3 traffic per ResNet-8 IG explanation | 7.19 MB (weights 40%, spills 32%, mask reads 16%); ToyAdmos 0.58 MB |
| 15 | 1-bit masks | −3.2 MB of 7.2 MB L3 traffic per explanation; does not fit in the TCDM as built; cycles not measured |

Also new in this round (not requested by the review): MLPerf Tiny keyword spotting and visual wake words on GeMMX, inference + Grad-CAM, bit-exact. KWS 48,024 + 5,787 cycles (+12.0%), VWW 188,250 + 1,538 (+0.8%); see `docs/gemmx/MOBILE_GEMMX.md`.

## 1. ResNet-8 accuracy on the full CIFAR-10 test set

```bash
python scripts/review_experiments.py 1 --workers 2
```

**Seeds:** none (all 10,000 test images, deterministic)

- **Top-1:** GeMMX integer semantics 0.8704, 95% Wilson CI [0.8637, 0.8768]. TFLite int8 0.8702, CI [0.8635, 0.8766].
- **Differences:** 187 of 10,000 predictions differ. Of these, GeMMX is right on 77 and TFLite on 75.
- **Logits vs softmax:** TFLite's argmax over the FC logits and over the softmax output never disagree, so int8 softmax ties play no role.

Raw output (`docs/review_raw/item1.txt`):

```
ResNet-8 pretrainedResnet_quant, CIFAR-10 test set, 10000 images (no sampling, no seed)
  TFLite int8 (argmax of the FC logits): top-1 0.8702 (8702/10000), 95% Wilson CI [0.8635, 0.8766]
  TFLite int8 (argmax of the softmax output): top-1 0.8702 (8702/10000), 95% Wilson CI [0.8635, 0.8766]
  GeMMX integer semantics (resnet8_int): top-1 0.8704 (8704/10000), 95% Wilson CI [0.8637, 0.8768]
  predictions differing GeMMX vs TFLite logits: 187 / 10000; of these GeMMX right 77, TFLite right 75
  predictions differing TFLite logits vs TFLite softmax argmax (int8 ties): 0

real	4m10,026s
user	7m22,554s
sys	0m1,853s
```

## 2. Faithfulness with confidence intervals

```bash
python scripts/review_experiments.py r8 --images 200 --workers 2   # per-image data, cached
python scripts/review_experiments.py 2 --images 200
python scripts/resnet8_gradcam_fidelity.py
python scripts/toyadmos_xai_fidelity.py --clips 50
python scripts/resnet8_xai_fidelity.py --images 200
```

**Seeds:** ResNet-8 baselines rng(100 + k) for image k; ToyAdmos clips rng(3), baselines rng(11 + clip), calibration rng(7) (the scripts' own seeds); bootstrap rng(0), 10,000 resamples over inputs

The harness recomputes each script's loop with the scripts' functions and seeds, keeping per-input values for the bootstrap. The scripts themselves were also rerun unchanged; their output is at the end of this item.

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

Raw output (`docs/review_raw/item2.txt`):

```
ResNet-8 IG / SHAP int8 vs float32, 200 MLPerf samples (baselines seeded rng(100 + k), calibration pct 99.5, bootstrap seed 0)
  ig map Pearson r: mean 0.9444  min 0.8516  95% CI [0.9413, 0.9475]  (n=200)
  ig map Spearman rho: mean 0.9362  min 0.7912  95% CI [0.9323, 0.9400]  (n=200)
  ig top-64 overlap: mean 0.7620  min 0.5156  95% CI [0.7523, 0.7715]  (n=200)
  ig full 32x32x3 Pearson r: mean 0.9467  min 0.8678  95% CI [0.9441, 0.9493]  (n=200)
  shap map Pearson r: mean 0.9126  min 0.8350  95% CI [0.9093, 0.9158]  (n=200)
  shap map Spearman rho: mean 0.8949  min 0.7971  95% CI [0.8915, 0.8983]  (n=200)
  shap top-64 overlap: mean 0.6955  min 0.4844  95% CI [0.6858, 0.7052]  (n=200)
  shap full 32x32x3 Pearson r: mean 0.9149  min 0.8534  95% CI [0.9120, 0.9177]  (n=200)
ResNet-8 Grad-CAM int8 vs float32, 200 MLPerf samples (200 non-constant maps)
  Pearson r: mean 0.9974  min 0.9890  95% CI [0.9971, 0.9977]  (n=200)
  top-1 cell: mean 0.9450  min 0.0000  95% CI [0.9100, 0.9750]  (n=200)
  top-8 overlap: mean 0.9519  min 0.7500  95% CI [0.9425, 0.9606]  (n=200)
ToyAdmos IG / SHAP int8 vs float32, 100 windows (clips rng(3), baselines rng(11 + clip), calibration rng(7))
  ig pearson: mean 0.7315  min -0.2635  95% CI [0.6684, 0.7919]  (n=100)
  ig spearman: mean 0.6780  min -0.3335  95% CI [0.6121, 0.7409]  (n=100)
  ig topk: mean 0.6750  min 0.0938  95% CI [0.6244, 0.7241]  (n=100)
  ig sign: mean 0.9087  min 0.1250  95% CI [0.8750, 0.9387]  (n=100)
  shap pearson: mean 0.7072  min -0.3419  95% CI [0.6428, 0.7695]  (n=100)
  shap spearman: mean 0.6688  min -0.3420  95% CI [0.6075, 0.7281]  (n=100)
  shap topk: mean 0.6072  min 0.0625  95% CI [0.5544, 0.6594]  (n=100)
  shap sign: mean 0.8719  min 0.2188  95% CI [0.8316, 0.9097]  (n=100)
```

Raw output (`docs/review_raw/item2_unchanged_scripts.txt`):

```
$ python scripts/resnet8_gradcam_fidelity.py
ResNet-8, 200 MLPerf CIFAR-10 samples: top-1 GeMMX-int 0.870, dequantised float 0.870
Grad-CAM int8 GeMMX vs float32 (predicted class, 200 maps): Pearson r 0.997 (min 0.989), top-1 cell 0.945, top-8 overlap 0.952
$ python scripts/toyadmos_xai_fidelity.py --clips 50
ToyAdmos int8 GeMMX attribution vs float32 (dequantised model), 100 windows (50 anomalous + 50 normal clips)
  ig  : pearson 0.731 (min -0.264)  spearman 0.678 (min -0.333)  topk 0.675 (min 0.094)  sign 0.909 (min 0.125)
  shap: pearson 0.707 (min -0.342)  spearman 0.669 (min -0.342)  topk 0.607 (min 0.062)  sign 0.872 (min 0.219)
$ python scripts/resnet8_xai_fidelity.py --images 200
ResNet-8 int8 GeMMX-semantics attribution vs float32, 200 images (calibration percentile 99.5)
  ig  : map pearson 0.944 (min 0.852)  spearman 0.936  top-64 overlap 0.762  full pearson 0.947
  shap: map pearson 0.913 (min 0.835)  spearman 0.895  top-64 overlap 0.696  full pearson 0.915
```

## 3. ToyAdmos calibration ablation on held-out normal training windows

```bash
python scripts/review_experiments.py 3
```

**Seeds:** calibration rng(7) (as the repo); held-out windows rng(21) from the 19,056 training windows the calibration does not touch; SHAP baselines rng(1000 + j) from the remaining unused windows; bootstrap rng(0)

**Setup:**
- The four cells re-run the repo's calibration code with the percentile and the multiplier form as parameters. The (exact, 99.5) cell is asserted equal to the repo's calibration.
- All 100 evaluation windows are normal *training* windows that the calibration did not use, so no test data enters the choice.

**Result:** the configuration the repo uses (exact multiplier, 99.5th percentile) is still the best cell for both IG (r 0.743) and SHAP (r 0.703).
- **Power-of-two shift at 99.5:** statistically indistinguishable (paired difference IG −0.003 [−0.011, +0.004], SHAP −0.002 [−0.013, +0.009]).
- **Exact at 99.9:** significantly worse (IG −0.033 [−0.059, −0.010], SHAP −0.041).
- **Power-of-two at 99.9:** significantly worse still (−0.10).

The choice made on test windows therefore holds on held-out training data.

Raw output (`docs/review_raw/item3.txt`):

```
ToyAdmos calibration ablation on 100 held-out NORMAL TRAINING windows (not used by the calibration; picked with rng(21) from 19056 unused windows; SHAP baselines rng(1000 + j) from the remaining unused windows; calibration rng(7))
    multiplier    pct | IG r mean [95% CI] (min) | SHAP r mean [95% CI] (min)
         exact   99.5 | 0.7433 [0.6785, 0.8044] (-0.122) | 0.7031 [0.6443, 0.7590] (-0.039)
         exact   99.9 | 0.7100 [0.6381, 0.7763] (-0.468) | 0.6625 [0.5961, 0.7241] (-0.402)
  power of two   99.5 | 0.7401 [0.6740, 0.8020] (-0.189) | 0.7009 [0.6412, 0.7569] (-0.082)
  power of two   99.9 | 0.6410 [0.5617, 0.7135] (-0.630) | 0.6016 [0.5302, 0.6685] (-0.656)
best cell: ig: exact @ 99.5, shap: exact @ 99.5
  ig: (exact, 99.9) - (exact, 99.5): mean -0.0333, 95% CI [-0.0588, -0.0098]
  shap: (exact, 99.9) - (exact, 99.5): mean -0.0406, 95% CI [-0.0715, -0.0099]
  ig: (pow2, 99.5) - (exact, 99.5): mean -0.0031, 95% CI [-0.0107, +0.0041]
  shap: (pow2, 99.5) - (exact, 99.5): mean -0.0022, 95% CI [-0.0125, +0.0085]
  ig: (pow2, 99.9) - (exact, 99.5): mean -0.1022, 95% CI [-0.1432, -0.0637]
  shap: (pow2, 99.9) - (exact, 99.5): mean -0.1015, 95% CI [-0.1491, -0.0545]

real	1m20,576s
user	1m39,079s
sys	0m39,723s
```

## 4. IG completeness

```bash
python scripts/review_experiments.py 4 --images 200
```

**Seeds:** same inputs and seeds as item 2 (ResNet-8: 200 images, black baseline; ToyAdmos: the 100 windows, mean normal spectrum baseline)

Relative error |Σ attr − (f(x) − f(b))| / |f(x) − f(b)|:

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

Raw output (`docs/review_raw/item4.txt`):

```
ResNet-8 IG completeness, 200 images, black baseline, 16-step Riemann sum (right endpoints alpha = 1/16..1), f = logit of the predicted class
  float32: |sum(attr) - (f(x)-f(b))| / |f(x)-f(b)|: median 0.2759, 90th pct 2.5094
  int8 pipeline vs int8 model's own logits: median 0.4149, 90th pct 2.5982
  int8 pipeline vs float32 f(x)-f(b): median 0.4411, 90th pct 2.5772
ToyAdmos IG completeness, 100 windows (item 2 windows), baseline = mean normal spectrum, f = reconstruction MSE; int8 attribution in real units via K = 2 s_x (2/640) S_gy / (m1 N) = 1.054e-08
  check of K: sum(int8 attr in real units) / sum(float32 attr), median 0.944 (1 = same total)
  float32: median 0.0911, 90th pct 0.6955
  int8 pipeline vs int8 model's own f: median 0.0437, 90th pct 0.1164
  int8 pipeline vs float32 f: median 0.0461, 90th pct 0.1261
```

Raw output (`docs/review_raw/item4_diag.txt`):

```
ResNet-8 IG completeness diagnostics (float32, 200 images, black baseline)
  |f(x)-f(b)|: median 3.566, 10th pct 0.579, min 0.0038
  f(x)-f(b) < 0 (logit of the predicted class lower on x than on black): 42 images
  images with relative error > 1: 38; their median |f(x)-f(b)| 0.566 vs 4.394 for the rest
  absolute error |sum - delta f|: median 0.996, 90th pct 2.498 (logit units)
  sum(attr) / (f(x)-f(b)): median 1.070
```

## 5. Upper-clamp saturation

```bash
python scripts/review_experiments.py 5 --images 200
```

**Seeds:** same inputs as item 2

**Finding:** saturation at +127 at masked positions is essentially absent, but not zero.
- **ToyAdmos** (100 windows x 16 samples x 9 ReLU layers): never reaches +127.
- **ResNet-8 IG** (200 images x 16 samples x 7 masked tensors): never reaches +127.
- **ResNet-8 SHAP:** reaches +127 in 3 of 200 images, only in the first residual add, at most 0.0008% of its activations. That is why the per-tensor means print as 0.000%.

**Effect of also masking +127:** negligible. The r values are identical to 4 decimals. ToyAdmos: 0 windows change. ResNet-8 IG: 0 images change. ResNet-8 SHAP: the 3 images above change by less than 1e-5 in r.

For contrast, the zero-point fraction (gradient masked) is 25–73% per tensor.

Raw output (`docs/review_raw/item5.txt`):

```
ResNet-8: fraction of forward activations at +127 / at -128 (zero point), per masked tensor, mean over 200 images x 16 interpolated samples
  ig: conv1 0.000%/26.8%  b1c1 0.000%/38.0%  add1 0.000%/25.4%  b2c1 0.000%/48.9%  add2 0.000%/43.8%  b3c1 0.000%/73.4%  add3 0.000%/73.2%
  shap: conv1 0.000%/36.8%  b1c1 0.000%/43.7%  add1 0.000%/30.2%  b2c1 0.000%/49.1%  add2 0.000%/46.6%  b3c1 0.000%/72.2%  add3 0.000%/69.8%
  ig: map r with +127 also masked 0.9444 vs 0.9444 now; paired diff +0.00000 95% CI [+0.00000, +0.00000]; images changed 0
  shap: map r with +127 also masked 0.9126 vs 0.9126 now; paired diff +0.00000 95% CI [-0.00000, +0.00000]; images changed 3
ToyAdmos: fraction of hidden-layer outputs (ReLU layers 1-9) at +127, 100 windows x 16 samples
  ig: L1 0.000%  L2 0.000%  L3 0.000%  L4 0.000%  L5 0.000%  L6 0.000%  L7 0.000%  L8 0.000%  L9 0.000%
  shap: L1 0.000%  L2 0.000%  L3 0.000%  L4 0.000%  L5 0.000%  L6 0.000%  L7 0.000%  L8 0.000%  L9 0.000%
  ig: r with +127 masked 0.7315 vs 0.7315; paired diff +0.00000 95% CI [+0.00000, +0.00000]; windows changed 0
  shap: r with +127 masked 0.7072 vs 0.7072; paired diff +0.00000 95% CI [+0.00000, +0.00000]; windows changed 0
Follow-up: per-image +127 fractions (ResNet-8, masked tensors conv1 b1c1 add1 b2c1 add2 b3c1 add3)
  ig: images with any +127: 0 / 200; max fraction per tensor (%): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  shap: images with any +127: 3 / 200; max fraction per tensor (%): [0.0, 0.0, 0.00076, 0.0, 0.0, 0.0, 0.0]
```

## 6. Utilisation check (MACs launched)

```bash
python scripts/review_experiments.py 6
```

**Seeds:** none (schedule generators; ToyAdmos input = toyadmos.select_input, ResNet-8 image 0 (inference) / 1 (XAI), KWS / VWW sample 0)

**MACs launched** (Σ M·N·K blocks × 512, over every launch the generators emit):

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

Raw output (`docs/review_raw/item6.txt`):

```
MACs from the schedule generators (sum over launches of M x N x K blocks x 512); cycles = measured Verilator totals
program                            launches  launched MACs  useful MACs useful/launched     cycles launched util useful util
ToyAdmos inference                       18      2,113,536      264,192           0.125      9,227         0.447       0.056
ToyAdmos IG/SHAP (N=16)                  41      9,273,344    8,454,144           0.912     91,517         0.198       0.180
ResNet-8 inference + Grad-CAM            48     13,770,752   12,505,728           0.908     57,461         0.468       0.425
ResNet-8 IG/SHAP (N=16)                1584    444,727,296  400,052,224           0.900  6,027,558         0.144       0.130
KWS inference + Grad-CAM                 75      9,060,352    2,664,768           0.294     53,811         0.329       0.097
VWW inference + Grad-CAM                346     18,325,504    7,491,968           0.409    189,788         0.189       0.077
useful MACs: ToyAdmos sum(in x out) = 264192 per row; ResNet-8 conv + FC (+ Grad-CAM 64x64) = 12501632; XAI = 16 x (forward + backward) GeMM MACs
```

## 7. Data independence of the core-side loops

```bash
inspection of snax_cluster/sw/xai/*/src/*.c and of the compiled binary (target/.../resnet8_gemmx/build/resnet8_shap_n16_s1.dump)
```

**Seeds:** n/a

**No data-dependent branches** in the ReLU mask (SWAR), the attribution MAC or the interleave (static 2-D DMA descriptors). In the disassembly, the only branches in these loops compare a counter or pointer against a constant bound.

**Data-dependent steps that do exist**, once per explanation:
- the argmax that picks the class;
- the Grad-CAM finish (ReLU, running max, one division).

This is consistent with IG and SHAP measuring identical cycle counts on different data.

Raw output (`docs/review_raw/item7.txt`):

```
Item 7: data-dependent branches in the core-side loops
Sources: snax_cluster/sw/xai/resnet8_gemmx/src/resnet8_xai_run.c, snax_cluster/sw/xai/toyadmos_gemmx/src/toyadmos_run.c,
snax_cluster/sw/xai/resnet8_gemmx/src/resnet8_run.c, snax_cluster/sw/xai/mobile_gemmx/src/mobile_run.c
Binary checked: target/snitch_cluster/sw/apps/xai/resnet8_gemmx/build/resnet8_shap_n16_s1.dump (clang -O3, rv32ima; functions inlined into main)

ReLU mask, relu_mask() (both apps): SWAR, 2 x 32-bit words per iteration (ResNet-8) / 1 word (ToyAdmos):
  t = (((v & 0x7F7F7F7F) + 0x7F7F7F7F) | v) & 0x80808080 with v = a ^ 0x80808080; g &= (t << 1) - (t >> 7)
  Loop bound n is a constant from the step table (row width), not data. Disassembly of the inlined loop
  (0x80003b54-0x80003bc0): lw/not/and/add/or/slli/srli/sub/sw only; the single branch is the loop back-edge
  "bltu a5, s2" (counter vs bound). Outer row/chunk loops: "bne a1, a0" (counter vs bound).
Attribution MAC, attr_mac() (ResNet-8): fixed 32 x 32 x 3 loop, acc += (x - b) * g. Disassembly (0x80003c98-0x80003d34):
  lb/lw/sub/mul/add/sw, branches "bne t5, t1" (pointer vs end) and "bne a3, t1" with t1 = 16 (row counter): none data-dependent.
  mul on this core is fixed-latency; there is no division in these loops.
ToyAdmos diag_gather() and combine(): fixed 80 x 8 and 640 iterations, no conditionals (source).
Polyphase interleave / scatter: not a core loop. The DMA core issues 2-D DMA descriptors from the static table prog_dmas
  (op->a .. op->b range fixed at generation time); sizes and addresses do not depend on data.
Step dispatch: branches on the step table (op type, launch index, prefetch index): static per program, identical for
  every input and every sample.

Data-dependent branches that do exist (outside the loops above):
  - argmax over the logits to pick the explained class (ResNet-8 gpool op, ToyAdmos none, KWS/VWW op_alpha): 10 / 12 / 2
    comparisons once per explanation; the chosen class changes which FC row is copied, not the amount of work.
  - Grad-CAM finish (resnet8_run.c, mobile_run.c op_cam): ReLU "v < 0 ? 0 : v", running max "if (v > mx)", and one
    32-bit division (1 << 31) / mx, whose latency on the iterative divider may depend on the operand; once per map
    (64 / 125 / 9 pixels).
  - BIST result comparison after the timed region (not timed).
Conclusion: the ReLU mask, attribution MAC and interleave have no data-dependent branches; cycle counts of IG / SHAP
are input-independent except for the argmax (a few compares). This matches the measurements: IG and SHAP give
identical totals (6,027,558 and 6,027,558; 91,517 and 91,517; N=8: 60,689 and 60,689) on different data.
```

## 8. (Optional) Grad-CAM at an intermediate layer

```bash
python scripts/review_experiments.py 8
```

**Seeds:** none (first 200 MLPerf samples)

**Faithfulness: measured.** Grad-CAM on the block-2 output (16×16×32):
- The int8 backward pass goes through block 3, using the same per-stage functions as the IG pipeline: transposed convs, mask, block-diagonal fork sum, with alpha = the spatial sum of d logit / d add2.
- Result: r 0.984 (min 0.901, 95% CI [0.982, 0.985]) against float32 autograd, on 200 images.
- That is lower than 0.997 at the linear head, as expected, since a real backward pass now sits in between.

**Cycles: not run.**
- No generated schedule exists for "forward + backward through block 3 only". The ResNet-8 XAI program runs the full backward to the input and its phase timers do not split per block.
- Measuring it needs a truncated variant of `resnet8_xai_sched.build` (stop after the add2 fork sum, then one Grad-CAM launch) and a Verilator run.
- No number is estimated here.

Raw output (`docs/review_raw/item8.txt`):

```
ResNet-8 Grad-CAM on the block-2 output (16x16x32), int8 backward through block 3 vs float32, 200 images
  Pearson r: mean 0.9838  min 0.9013  95% CI [0.9821, 0.9853]  (n=200)
```

## 9. Sample-count sensitivity

```bash
python scripts/review_experiments.py r8sweep --images 50 --workers 2   # ResNet-8 per-image data
python scripts/review_experiments.py 9 --images 50
# Verilator: PROG=toyadmos_ig_n8 / toyadmos_shap_n8 (new 8-sample programs), R8_METHOD=ig R8X_N=4 R8_SAMPLE=1
```

**Seeds:** ResNet-8 first 50 images, ToyAdmos item-2 windows; IG alpha = 1/N..1; SHAP baselines and alpha rng(100 + k) (ResNet-8) / rng(11 + clip) (ToyAdmos); N = 128 float32 references rng(500 + k / clip)

**Cycles (Verilator, every output bit-exact):**

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

Raw output (`docs/review_raw/item9.txt`):

```
Sample-count sweep (emulator). ResNet-8: 50 images; ToyAdmos: 100 windows.
  IG: alpha = 1/N..1 (N = 32 uses a 1/32 grid); SHAP: N baselines + integer alpha 0..16 (/16), seeds rng(100 + k) (ResNet-8) / rng(11 + clip) (ToyAdmos); float32 N = 128 references: IG 128 steps, SHAP 128 baselines with uniform alpha, seed rng(500 + k / clip)
  ResNet-8:
    method   N | int8 vs float32 same N | int8 vs float32 N=128 | float32 N vs float32 N=128
    ig       4 |   0.908 [0.899, 0.916] |  0.898 [0.889, 0.906] | 0.966 [0.963, 0.968]
    ig       8 |   0.929 [0.921, 0.936] |  0.929 [0.921, 0.936] | 0.989 [0.988, 0.990]
    ig      16 |   0.946 [0.940, 0.951] |  0.946 [0.941, 0.952] | 0.997 [0.997, 0.997]
    ig      32 |   0.956 [0.951, 0.960] |  0.956 [0.951, 0.960] | 0.999 [0.999, 0.999]
    shap     4 |   0.888 [0.879, 0.896] |  0.354 [0.302, 0.407] | 0.363 [0.307, 0.419]
    shap     8 |   0.897 [0.890, 0.903] |  0.476 [0.432, 0.520] | 0.485 [0.439, 0.531]
    shap    16 |   0.916 [0.910, 0.922] |  0.580 [0.546, 0.612] | 0.590 [0.555, 0.623]
    shap    32 |   0.934 [0.929, 0.939] |  0.695 [0.672, 0.719] | 0.712 [0.689, 0.735]
  ToyAdmos:
    method   N | int8 vs float32 same N | int8 vs float32 N=128 | float32 N vs float32 N=128
    ig       4 |   0.777 [0.710, 0.839] |  0.746 [0.701, 0.789] | 0.661 [0.590, 0.726]
    ig       8 |   0.706 [0.637, 0.769] |  0.776 [0.730, 0.817] | 0.684 [0.625, 0.741]
    ig      16 |   0.731 [0.668, 0.792] |  0.777 [0.731, 0.820] | 0.770 [0.723, 0.814]
    ig      32 |   0.671 [0.605, 0.734] |  0.782 [0.736, 0.824] | 0.768 [0.712, 0.816]
    shap     4 |   0.821 [0.754, 0.879] |  0.573 [0.529, 0.614] | 0.527 [0.472, 0.580]
    shap     8 |   0.743 [0.679, 0.802] |  0.616 [0.570, 0.660] | 0.517 [0.463, 0.568]
    shap    16 |   0.707 [0.643, 0.770] |  0.662 [0.617, 0.704] | 0.521 [0.461, 0.579]
    shap    32 |   0.673 [0.615, 0.730] |  0.690 [0.642, 0.734] | 0.508 [0.444, 0.569]
```

Raw output (`docs/review_raw/item9_sim.txt`):

```
# Verilator runs for item 9 (simulator: snitch_cluster.vlt, snax_KUL_cluster; per-hart traces linked to /dev/null)
## ToyAdmos IG, N=8 (PROG=toyadmos_ig_n8)
[DMA] x placement + first tile: 536 cycles
[TOTAL] 60689 cycles, 42 steps: config=4954 run=12423 mask=20255 gather+combine=18737
  interp    902
  forward   7114
  loss      782
  backward  27452
  attr      20119
check a1      OK (0/1024 bytes differ)
check a2      OK (0/1024 bytes differ)
check a3      OK (0/1024 bytes differ)
check a4      OK (0/1024 bytes differ)
check a5      OK (0/64 bytes differ)
check a6      OK (0/1024 bytes differ)
check a7      OK (0/1024 bytes differ)
check a8      OK (0/1024 bytes differ)
check a9      OK (0/1024 bytes differ)
check y_inf   OK (0/640 bytes differ)
check D       OK (0/5120 bytes differ)
check attr_gy OK (0/2560 bytes differ)
check g0      OK (0/5120 bytes differ)
check attr    OK (0/2560 bytes differ)
rc=0
## ToyAdmos SHAP, N=8 (PROG=toyadmos_shap_n8)
[DMA] x placement + first tile: 536 cycles
[TOTAL] 60689 cycles, 42 steps: config=4954 run=12423 mask=20255 gather+combine=18737
  interp    902
  forward   7114
  loss      782
  backward  27452
  attr      20119
check a1      OK (0/1024 bytes differ)
check a2      OK (0/1024 bytes differ)
check a3      OK (0/1024 bytes differ)
check a4      OK (0/1024 bytes differ)
check a5      OK (0/64 bytes differ)
check a6      OK (0/1024 bytes differ)
check a7      OK (0/1024 bytes differ)
check a8      OK (0/1024 bytes differ)
check a9      OK (0/1024 bytes differ)
check y_inf   OK (0/640 bytes differ)
check D       OK (0/5120 bytes differ)
check attr_gy OK (0/2560 bytes differ)
check g0      OK (0/5120 bytes differ)
check attr    OK (0/2560 bytes differ)
rc=0
## ResNet-8 IG, N=4 (R8_METHOD=ig R8X_N=4 R8_SAMPLE=1)
[TOTAL] 1507854 cycles, 4 samples x 115 steps: config=46747 run=318290 mask=716460 dma2d=131510 attr=150870, explained class 1
  interp    7028
  forward   190250
  fill      51843
  backward  1184819
  sample 0: 378334  sample 1: 376624 cycles
check attr   OK (0/12288 bytes differ)
rc=0
## KWS / VWW inference + Grad-CAM (MB_TASK=kws|vww MB_SAMPLE=0)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:[TOTAL] 53811 cycles, 75 steps: config=9933 run=24364 dma-ops=4011 core-ops=5379, predicted class 0
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  input    1232
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  conv     9280
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  dw       13917
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  pw       12420
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  fill     2894
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  head     651
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:  gradcam  5787
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:check logits  OK (0/128 bytes differ)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:check fmap    OK (0/21504 bytes differ)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_kws/out.txt:check cam     OK (0/500 bytes differ)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:[TOTAL] 189788 cycles, 346 steps: config=38834 run=65307 dma-ops=40923 core-ops=1237, predicted class 1
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  input    19128
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  conv     3758
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  dw       50810
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  pw       58692
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  fill     21873
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  head     716
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:  gradcam  1538
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:check logits  OK (0/64 bytes differ)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:check fmap    OK (0/30720 bytes differ)
/tmp/claude-1000/-home-wiki-xai-heterogeneous/4875fbcb-ad7d-4a27-aadf-aefdb1508a68/scratchpad/sim_vww/out.txt:check cam     OK (0/36 bytes differ)
```

## 10. Baseline sensitivity

```bash
python scripts/review_experiments.py 10 --images 50
```

**Seeds:** ResNet-8 first 50 images (dataset mean = mean of the 512 CIFAR-10 training images of resnet8_xai.cifar_train(); blur = Gaussian sigma 4 px per channel); ToyAdmos item-2 windows (zeros = real 0 quantised; mean = mean normal training spectrum)

**IG faithfulness, int8 vs float32 with the same baseline:**

| Baseline | ResNet-8 r [95% CI] | ToyAdmos r [95% CI] |
|---|---|---|
| black | 0.946 [0.940, 0.951] | — |
| dataset mean | 0.950 [0.945, 0.954] | — |
| blurred input | 0.957 [0.953, 0.960] | — |
| mean normal spectrum | — | 0.732 [0.668, 0.792] |
| zeros | — | 0.621 [0.598, 0.643] |

**Cycles:** confirmed unchanged. For each baseline the generator emits byte-identical launch configuration code, step table and DMA table (ResNet-8: black / mean / blur; ToyAdmos: mean / zeros). Since the loops are data-independent (item 7), the cycle count is the same.

Raw output (`docs/review_raw/item10.txt`):

```
IG faithfulness (int8 vs float32, same baseline, 16 steps). ResNet-8 50 images, ToyAdmos 100 windows
  ResNet-8 baseline black: mean 0.9459  min 0.8744  95% CI [0.9400, 0.9513]  (n=50)
  ResNet-8 baseline mean: mean 0.9500  min 0.8892  95% CI [0.9453, 0.9544]  (n=50)
  ResNet-8 baseline blur: mean 0.9566  min 0.9192  95% CI [0.9528, 0.9602]  (n=50)
  ToyAdmos baseline mean: mean 0.7315  min -0.2635  95% CI [0.6684, 0.7919]  (n=100)
  ToyAdmos baseline zeros: mean 0.6210  min 0.2246  95% CI [0.5981, 0.6428]  (n=100)
  ResNet-8 schedule identical for black / mean / blur baselines (launch CSR code, step table, DMA table): True
  ToyAdmos schedule identical for mean / zeros baselines: True
```

## 11. Cheap baseline methods

```bash
python scripts/review_experiments.py 11 --images 50
```

**Seeds:** as item 9

**ResNet-8, gradient × input with one sample at x:**
- Agreement: int8 r 0.678 [0.661, 0.694] against float32 IG (N=16).
- Most of the gap is the method itself: float32 gradient × input vs float32 IG is 0.717. The int8 effect alone, int8 vs float32 gradient × input, is 0.897.
- Cost: one sample of the IG program. Measured sample 0 was 378,443 cycles inside the 16-sample run; the linear fit in item 9 gives about 379k for an N=1 program. That is 7.1× inference, against 112× for IG with N=16.

**ToyAdmos, per-feature reconstruction error (x − y)² from the int8 inference:**
- Agreement: r 0.614 against float32 IG (0.665 against |IG|), versus 0.732 for int8 IG.
- Cost: the inference (9,227 cycles) plus 640 squared differences on a core, run once per window. This was not simulated, so no cycle number is given.

Raw output (`docs/review_raw/item11.txt`):

```
Cheap methods vs float32 IG. ResNet-8 50 images (gradient x input = one sample at x, black baseline)
  int8 grad x input vs float32 IG N=16: mean 0.6775  min 0.5344  95% CI [0.6613, 0.6936]  (n=50)
  int8 grad x input vs float32 IG N=128: mean 0.6651  min 0.5143  95% CI [0.6480, 0.6818]  (n=50)
  float32 grad x input vs float32 IG N=16 (method gap alone): mean 0.7165  min 0.5602  95% CI [0.7005, 0.7318]  (n=50)
  int8 vs float32 grad x input (int8 gap alone): mean 0.8966  min 0.8234  95% CI [0.8881, 0.9047]  (n=50)
ToyAdmos, 100 windows: per-feature reconstruction error (x - y)^2 of the int8 inference
  (x - y)^2 vs float32 IG N=16 (signed): mean 0.6139  min -0.0301  95% CI [0.5554, 0.6703]  (n=100)
  (x - y)^2 vs |float32 IG N=16|: mean 0.6647  min 0.0459  95% CI [0.6184, 0.7089]  (n=100)
  for comparison: int8 IG N=16 vs float32 IG N=16: mean 0.7315  min -0.2635  95% CI [0.6684, 0.7919]  (n=100)
```

## 12. Higher-precision gradients (ToyAdmos, emulator)

```bash
python scripts/review_experiments.py 12
python scripts/review_ta_diag.py
```

**Seeds:** item-2 windows and seeds

**Result:** int16 gradients do not help, and float gradients make it worse.

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

Raw output (`docs/review_raw/item12.txt`):

```
ToyAdmos gradient precision (emulator), 100 windows (item 2), r vs float32; masks and forward int8 as built; int16 = the int8 scale with 8 extra fractional bits, saturating at +-32767
  int8 everywhere (as built)                : ig 0.7315 [0.6684, 0.7919] (min -0.264)  shap 0.7072 [0.6428, 0.7695] (min -0.342)
  int16 all stages                          : ig 0.7352 [0.6723, 0.7950] (min -0.264)  shap 0.7079 [0.6435, 0.7701] (min -0.340)
  float all stages                          : ig 0.5844 [0.4931, 0.6711] (min -0.801)  shap 0.5861 [0.5073, 0.6614] (min -0.735)
  int16 output-side 3 (loss, bwd10, bwd9)   : ig 0.7319 [0.6687, 0.7925] (min -0.261)  shap 0.7075 [0.6430, 0.7697] (min -0.341)
  int16 input-side 3 (bwd3, bwd2, bwd1)     : ig 0.7326 [0.6694, 0.7932] (min -0.268)  shap 0.7075 [0.6431, 0.7699] (min -0.339)
  float output-side 3 (loss, bwd10, bwd9)   : ig 0.7359 [0.6727, 0.7970] (min -0.246)  shap 0.7129 [0.6488, 0.7753] (min -0.333)
  float input-side 3 (bwd3, bwd2, bwd1)     : ig 0.6020 [0.5139, 0.6841] (min -0.809)  shap 0.6000 [0.5271, 0.6709] (min -0.457)
  backward GeMM launches in the built schedule: 18 (per explanation)

real	1m12,672s
user	1m15,347s
sys	0m34,191s
```

Raw output (`docs/review_raw/item12_diag.txt`):

```
ToyAdmos IG, 100 windows: float32 backward with pieces of the int8 forward swapped in, r vs float32 IG
  float: mean 1.0000  min 1.0000  95% CI [1.0000, 1.0000]  (n=100)
  int8 masks: mean 0.6333  min -0.9055  95% CI [0.5468, 0.7127]  (n=100)
  int8 y: mean 0.7222  min -0.9616  95% CI [0.6261, 0.8076]  (n=100)
  int8 x~: mean 0.6428  min -0.4736  95% CI [0.5624, 0.7173]  (n=100)
  int8 masks + y + x~: mean 0.5842  min -0.8051  95% CI [0.4929, 0.6709]  (n=100)
  int8 pipeline: mean 0.7315  min -0.2635  95% CI [0.6684, 0.7919]  (n=100)
```

## 13. Perturbation metrics (deletion / insertion)

```bash
python scripts/review_experiments.py 13 --images 200 --workers 2
```

**Seeds:** maps from item 2 (200 images); random maps rng(900 + k)

**Setup:**
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

Raw output (`docs/review_raw/item13.txt`):

```
ResNet-8 deletion / insertion AUC, 200 images, float32 dequantised model, softmax probability of the predicted class; pixels ranked by the channel-summed attribution (descending), 32 steps of 32 pixels; deletion fills black, insertion starts from a Gaussian blur (sigma 4); random map seed rng(900 + k). Lower deletion / higher insertion is better.
  ig_int   : deletion 0.1419 [0.1104, 0.1774]   insertion 0.3048 [0.2705, 0.3396]
  ig_flt   : deletion 0.1405 [0.1100, 0.1748]   insertion 0.2955 [0.2613, 0.3305]
  shap_int : deletion 0.1628 [0.1284, 0.2002]   insertion 0.3434 [0.3060, 0.3825]
  shap_flt : deletion 0.1604 [0.1264, 0.1976]   insertion 0.3401 [0.3027, 0.3784]
  random   : deletion 0.1573 [0.1224, 0.1954]   insertion 0.2799 [0.2480, 0.3136]
  ig int8 - float32 (del): +0.0014 95% CI [-0.0037, +0.0070]
  ig int8 - float32 (ins): +0.0093 95% CI [+0.0032, +0.0155]
  shap int8 - float32 (del): +0.0024 95% CI [-0.0053, +0.0113]
  shap int8 - float32 (ins): +0.0032 95% CI [-0.0051, +0.0114]
```

## 14. L3 traffic

```bash
python scripts/review_experiments.py 14
```

**Seeds:** ResNet-8 image 1, IG (rng(101)); ToyAdmos select_input, IG

**ResNet-8 IG, one explanation (16 samples):** 7.19 MB moved between L3 and TCDM.

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

Raw output (`docs/review_raw/item14.txt`):

```
ResNet-8 IG, one explanation (16 samples), bytes moved per category (schedule generator; each sample runs the same step table)
  weights          L3<->TCDM   per sample    177,664 B   per explanation    2,842,624 B
  spill_write      L3<->TCDM   per sample    145,920 B   per explanation    2,334,720 B
  mask_read        L3<->TCDM   per sample     73,728 B   per explanation    1,179,648 B
  sample_in        L3<->TCDM   per sample     26,240 B   per explanation      419,840 B
  attr_in          L3<->TCDM   per sample     26,112 B   per explanation      417,792 B
  interleave_tcdm  TCDM->TCDM  per sample     30,720 B   per explanation      491,520 B
  fill_tcdm        TCDM->TCDM  per sample    334,080 B   per explanation    5,345,280 B
  total L3<->TCDM per explanation: 7,194,624 B (+ one-time TCDM image 120,448 B)
ToyAdmos IG, one explanation (16 samples stacked in the mesh rows): weight / bias tiles streamed L3->TCDM 581,888 B; inputs x, b_1..b_16 and constants arrive with the TCDM image (109,696 B); masks, interleaves and gradients stay in the TCDM (no spills).
DMA busy / stall counters: the simulated cluster exposes no DMA performance counters to software and the runs record no per-cycle trace (traces were linked to /dev/null for disk space), so the fraction of time the cores wait on DMA is not measured. The phase timers count DMA-bound phases together with the core work that overlaps them (see item 14 text).
```

## 15. Software 1-bit masks

```bash
python scripts/review_experiments.py 15
```

**Seeds:** ResNet-8 image 1, IG

**Bytes (computed from the generated schedule):**
- Masks need 73,728 elements per sample, 9,216 B as bits.
- That removes 64,512 B of L3 mask reads per sample (1.03 MB per explanation).
- It replaces 145,920 B of activation spills with 9,216 B of bit-mask writes.
- Together: −201 kB of L3 traffic per sample, −3.2 MB of the 7.2 MB per explanation.

**TCDM space:** the schedule already uses 120,448 B of its 124,928 B limit, so 9 kB of masks per sample do not fit next to it. The bits would have to go to L3 (1/8 of today's bytes) or the buffers would have to be re-planned.

**Cycles: not measured.**
- The change needs new core code: packing bits after each forward ReLU layer (core work the forward pass does not do today) and a bit-expanding mask kernel.
- The mask phase (48% of ResNet-8 XAI) mixes DMA wait and core SWAR time, which the timers do not separate (item 14).
- So no cycle change is given; it needs an implementation and a Verilator run.

Raw output (`docs/review_raw/item15.txt`):

```
ResNet-8 IG: 1-bit ReLU masks instead of re-reading int8 activations (per sample)
  masked gradient elements per sample: 73,728; int8 activation bytes read from L3 now: 73,728 B; activation bytes spilled to L3 now: 145,920 B
  as 1-bit masks: 9,216 B per sample (saves 64,512 B of mask reads and replaces 145,920 B of spills with 9,216 B)
  TCDM in use by the schedule: 120,448 B image of 124,928 B limit
```
