# Hardware Acceleration of Explainable AI on Heterogeneous Multi-Accelerator Platforms

**Intermediate Research Document — Complete Technical Reference**
**Date**: April 10, 2026
**Status**: Phases 1-4 Complete and Verified, Phase 5 In Progress

---

## Abstract

This work implements and measures Explainable AI (XAI) methods on a heterogeneous embedded accelerator cluster: SNAX, with RISC-V cores and an 8×8×8 int8 GeMMX matrix accelerator. It uses the two MLPerf Tiny benchmarks with their real int8 reference models and real data: ResNet-8 on CIFAR-10 and the ToyAdmos anomaly-detection autoencoder on DCASE 2020 ToyCar.

All matrix work, for inference and for explanation, runs on GeMMX in cycle-accurate simulation. Every result is bit-exact against an emulator of the accelerator, and the integer mapping keeps the reference accuracy: top-1 0.870 vs 0.870 (ResNet-8), AUC 0.8406 vs 0.8416 (ToyAdmos).

- **Grad-CAM on ResNet-8** adds 3,841 cycles (7.2%) to a 53.6k-cycle inference, with int8 maps matching float32 at r = 0.997.
- **Integrated Gradients and Gradient SHAP on ToyAdmos** cost 91.5k cycles including the inference, which falls out of the XAI pass. The 16 samples are stacked in the accelerator's mesh rows, so each layer runs once for all of them. The score's direct gradient term and the attribution reduction are also mapped onto GeMMX.
- **Configuration overhead.** The stock driver configures a launch in ~2,300 cycles. Generating straight-line, delta-only register writes from the launch schedule cuts this to ~77 cycles, speeding inference up 5.1× and XAI 2.0×.
- **Earlier work.** Measurements on the scalar RISC-V core (Grad-CAM, a 3.02× backward-hoisting optimisation for Gradient SHAP), saliency-guided formal verification and neuro-symbolic rule extraction (47 cycles) complete the study.

The results show which XAI methods fit an edge accelerator's budget: single-pass methods on linear heads do, gradient sampling through deep networks does not. They also show that the remaining cost lies in core-side elementwise work and in int8 gradient precision.

---

## 1. Introduction

### 1.1 Problem Statement

Deep Neural Networks (DNNs) are increasingly deployed in safety-critical edge applications — autonomous driving, medical diagnostics, industrial anomaly detection — where the "black-box" nature of these models creates barriers to trust, regulatory compliance, and debugging. Explainable AI (XAI) methods address this by providing visual and numerical attributions that explain model decisions. However, XAI methods impose significant computational overhead: Grad-CAM requires a backward pass through the network, Gradient SHAP requires N independent forward-backward passes, and formal verification requires solving integer linear programs. This overhead is particularly problematic on resource-constrained embedded platforms where inference itself already consumes most of the available compute budget.

### 1.2 Research Question

> **Can XAI methods be implemented efficiently on heterogeneous embedded accelerator platforms, achieving overhead below 20% of inference latency while maintaining explanation quality and providing formal safety guarantees for quantized deployments?**

### 1.3 Contributions

This work makes the following contributions:

1. **Embedded Grad-CAM implementation** with measured cycle counts on RISC-V (4,757 cycles on the Snitch scalar core for a 4x4x16 feature map), plus an INT8-weight variant (6,923 cycles)
2. **Backward-hoisting optimization for Gradient SHAP** that reduces cost from 175,510 to 58,022 cycles (3.02x speedup) by exploiting gradient independence in linear model heads
3. **Saliency-guided formal verification** combining Grad-CAM saliency with QVIP ILP verification to focus solver effort on high-importance input regions
4. **Neuro-symbolic rule extraction** that distills neural network behavior into depth-3 decision trees using only the top-K most salient features, executable in ~47 cycles with integer-only arithmetic
5. **Cross-model benchmark framework** evaluating all four XAI methods across four MLPerf Tiny architectures with a SNAX cycle cost model calibrated per stage against the GAP+FC measurements
6. **End-to-end measured XAI on an accelerator with real MLPerf Tiny models**: ResNet-8 inference + Grad-CAM and ToyAdmos inference + Integrated Gradients + Gradient SHAP, every GeMM on GeMMX and bit-exact in simulation. A launch-schedule generator with a bit-exact GeMMX emulator produces the configurations, and new mappings put sample batching, loss gradients and attribution reductions on the accelerator (Section 8.7).
7. **Launch-configuration overhead analysis and fix**: delta CSR configuration generated from the schedule, ~30× cheaper per launch than the stock driver.

### 1.4 Research Narrative

The four phases form a coherent explanatory pipeline:

| Phase | Method | Question Answered | Cost (GAP+FC) |
|-------|--------|-------------------|---------------|
| 1 | Grad-CAM | **WHERE** is the model looking? | 4,757 cycles |
| 2 | Gradient SHAP | **WHY** does each feature matter? | 58,022 cycles |
| 3 | QVIP Verification | Is the quantized model **SAFE**? | 12.4s (host) |
| 4 | Rule Extraction | **WHAT** rules is the model following? | 47 cycles |

Each phase builds on the previous: Grad-CAM saliency guides SHAP feature selection, SHAP attributions inform ECQx quantization policy, formal verification proves the quantized model is safe in salient regions, and rule extraction distills the entire pipeline into auditable IF-THEN rules.

---

## 2. Background and Related Work

### 2.1 Explainable AI Methods

**Grad-CAM** (Selvaraju et al., 2017) produces class-discriminative localization maps by computing the gradient of the target class score with respect to the final convolutional feature maps:

```
alpha_k = (1/Z) * SUM_{i,j} (d y^c / d A^k_{ij})     [global average pool of gradients]
L_GradCAM = ReLU( SUM_k alpha_k * A^k )                 [weighted combination + ReLU]
```

The backward pass (Step 1) maps to a transpose-GeMM operation, making it acceleratable on matrix multiplication hardware.

**Gradient SHAP** (Lundberg & Lee, 2017) approximates Shapley values by averaging gradient-times-input-difference over random baseline samples:

```
phi_i = E_{x',alpha}[ (x_i - x'_i) * (d f(x_interp) / d x_i) ]
```

where `x_interp = x' + alpha * (x - x')` and `alpha ~ U(0,1)`. Each of N samples requires an independent forward-backward pass.

**QVIP** (Zhang et al., ASE'22) encodes quantized neural networks as integer linear constraints and verifies robustness via ILP solving:

```
Property: forall x_hat in R_p(u, r): N_hat(x_hat) = N_hat(u)
```

If the negated property's ILP is infeasible, the network is provably robust within radius r.

**Neuro-symbolic rule extraction** (Bastani et al., 2017; Frosst & Hinton, 2017) distills neural network predictions into interpretable decision trees, enabling human audit and formal enumeration of all decision paths.

### 2.2 Hardware Platforms

**SNAX Cluster** (KU Leuven): A RISC-V compute cluster featuring Snitch cores (RV32IMF, single-issue, in-order) with a CSR-programmed GeMM accelerator and 128 kB Tightly-Coupled Data Memory (TCDM) scratchpad. The GeMM accelerator processes INT8 block-tiled matrix multiplications with configurable streamers for data movement. Key characteristics:
- 8 compute cores + 1 DM core
- GeMM accelerator: 512 PEs, 152x speedup over scalar core for INT8 convolutions
- 128 kB L1 TCDM (shared, banked)
- CSR-based accelerator programming (fire-and-forget dispatch)

**CFU Playground** (Google): A framework for adding Custom Function Units (CFUs) to a VexRiscv soft processor on FPGA (Arty-35T). CFUs are invoked via custom R-format RISC-V instructions (`cfu_op(funct7, funct3, a, b)`), enabling fine-grained SIMD acceleration alongside TFLite Micro inference. Key characteristics:
- VexRiscv soft core on Xilinx Artix-7 FPGA
- CFU defined in Amaranth HDL
- 256 MB external DDR
- Deploy-profile-optimize development loop

### 2.3 Related Work on Hardware-Accelerated XAI

| Work | Platform | Method | Result |
|------|----------|--------|--------|
| Pan & Mishra (2023) | Cloud TPU (128 cores) | Grad-CAM/SHAP on ResNet-50 | 39x speedup over CPU |
| ApproXAI (2025) | Edge GPU | Approximate XAI | Energy reduction via approximate computing |
| **This work** | **SNAX RISC-V (embedded)** | **Grad-CAM, SHAP, QVIP, Rules** | **Embedded XAI cycle counts (scalar core)** |

We are not aware of prior work reporting XAI-specific cycle counts on embedded RISC-V platforms. **[TODO before submission: back this with a systematic literature search.]**

---

## 3. System Architecture

### 3.1 Two-Platform Heterogeneous Approach

```
                    +-----------------------+
                    |   Host (Python)       |
                    |   - Reference impls   |
                    |   - QVIP solver       |
                    |   - Rule extraction   |
                    |   - Data generation   |
                    +----------+------------+
                               |
                    +----------v------------+
                    |   Code Generation     |
                    |   - datagen.py -> .h  |
                    |   - rule_to_c.py -> .h|
                    +----+------------+-----+
                         |            |
              +----------v--+    +----v-----------+
              | SNAX Cluster |    | CFU Playground |
              | RISC-V+GeMM |    | VexRiscv+FPGA  |
              | 128kB TCDM  |    | 256MB DDR      |
              +-------------+    +----------------+
```

### 3.2 Implementation Workflow

Each XAI method follows a three-stage development lifecycle:

```
Stage 1: Python Reference (NumPy)
    - Establish golden ground truth
    - Verify algorithmic correctness with pytest
    - Generate test data headers (.h) for embedded targets

Stage 2: Embedded C Kernel
    - Port to restricted C (no malloc, static buffers, compile-time sizes)
    - BIST verification against Python golden reference
    - Cycle profiling via Verilator RTL simulation

Stage 3: Architectural Optimization
    - Identify bottlenecks from cycle breakdown
    - Apply hardware-specific optimizations (hoisting, fusion, accelerator offload)
    - Re-verify correctness after each optimization
```

### 3.3 Data Flow on SNAX

```
Input --> [Backbone: Conv layers] --> Feature Maps --> [Head: GAP+FC] --> Logits
                                           |
                         +-----------------+------------------+
                         |                 |                  |
                         v                 v                  v
                   Grad-CAM:         Hoisted SHAP:      Rule Extraction:
                   grad = dY/dA      cache features,    top-K salient
                   alpha = GAP(grad) re-run head Nx     features -->
                   cam = ReLU(Sum)   phi = E[(x-x')g]   depth-3 tree
                         |                 |                  |
                         v                 v                  v
                   Saliency Map      Attribution Map    IF-THEN Rules
                         |                 |                  |
                         +--------+--------+                  |
                                  |                           |
                                  v                           v
                         ECQx Quantization            C Code Export
                         (8-bit critical,             (~47 cycles)
                          4-bit rest)
                                  |
                                  v
                         QVIP Verification
                         (ILP on critical slice)
```

### 3.4 Memory Layout (128 kB TCDM SPM)

All buffer sizes must be statically known at compile time to fit within the 128 kB L1 scratchpad:

```
TCDM Memory Map (Phase 2b, SHAP):
+------------------+----------+-------+
| Buffer           | Size (B) |  %    |
+------------------+----------+-------+
| Feature maps     |    1,024 |  0.8% |  (4*4*16*4B)
| Baseline buffer  |    1,024 |  0.8% |  (4*4*16*4B)
| Gradient buffer  |    1,024 |  0.8% |  (4*4*16*4B)
| Attribution accum|    1,024 |  0.8% |  (4*4*16*4B)
| W_fc weights     |      640 |  0.5% |  (16*10*4B)
| Logits           |       40 |  0.0% |  (10*4B)
| TOTAL USED       |    4,776 |  3.6% |
| AVAILABLE        |  131,072 |       |
+------------------+----------+-------+
```

For larger models (ResNet-8: 8x8x64 features = 16 kB per buffer), the memory budget becomes a constraint. The `buffer_planner.py` utility enforces static allocation and checks that all buffers fit within the 128 kB budget.

---

## 4. Phase 1: Grad-CAM on SNAX Cluster

### 4.1 Algorithm

Grad-CAM computes a class-discriminative saliency map from the gradients of the target class score with respect to the final convolutional feature maps:

```
Step 1: Backward GeMM    grad = dLogit[c] / dFeatureMaps
                          --> Transpose-GeMM: grad = dLogit * W_fc^T
Step 2: Global Avg Pool   alpha[k] = mean_{h,w}(grad[0,:,:,k])
Step 3: Weighted Sum       cam = SUM_k( alpha[k] * FeatureMaps[0,:,:,k] )
Step 4: ReLU + Normalize   cam = ReLU(cam) / max(cam)
```

### 4.2 Hardware Mapping

| Step | Operation | Hardware Unit | Complexity |
|------|-----------|---------------|------------|
| 1 | Transpose-GeMM (1xC @ CxK) | GeMM accelerator (INT8) or RISC-V scalar (FP32) | O(K*C) |
| 2 | Spatial reduction | RISC-V scalar (SIMD-friendly) | O(H*W*K) |
| 3 | Channel-weighted sum | RISC-V scalar | O(H*W*K) |
| 4 | ReLU + division | RISC-V scalar (FPU) | O(H*W) |

### 4.3 Implementation

**Python reference** (`src/xai/gradcam/gradcam_reference.py`): Uses finite-difference perturbation to compute exact gradients without autograd. Also implements LRP-epsilon for comparison.

**C kernel** (`snax_cluster/sw/xai/gradcam/src/gradcam.h`): Static-buffer implementation with per-step cycle profiling. Key functions:
- `gradcam_transpose_weights()` — in-place transpose of W_fc
- `gradcam_backward_gemm()` — gradient computation via matmul
- `gradcam_gap()` — spatial average pooling
- `gradcam_weighted_sum()` — channel-weighted combination
- `gradcam_relu_normalize()` — ReLU + max-normalize

### 4.4 Results

**Platform**: SNAX cluster, single RISC-V RV32IMF core, Verilator RTL simulation
**Configuration**: H=4, W=4, K=16 channels, C=10 classes, float32

Per-stage `mcycle` counters (Verilator, 2026-11-19):

| Step | Operation | fp32 cycles | % | INT8-weight cycles |
|------|-----------|-------:|----:|-------:|
| 0 | Transpose W_fc (KxC = 160) | 1,124 | 23.6% | 1,118 |
| 1 | Backward GeMM (1x10 @ 10x16) | 1,167 | 24.5% | **3,611** |
| 2 | Alpha (GAP of gradient) | 111 | 2.3% | 108 |
| 3 | Weighted Sum (16 pos x 16 ch) | 2,022 | **42.5%** | 1,755 |
| 4 | ReLU + Normalize | 333 | 7.0% | 331 |
| **Total** | | **4,757** | **100%** | **6,923** |

**BIST Verification**: 0/16 errors for both builds (tolerance 1e-3 fp32, 1e-2 INT8).

> **Erratum.** An earlier draft reported 6,153 cycles with an estimated breakdown attributing 68.8% (~4,233 cycles) to ReLU+Normalize. Neither held: the estimated breakdown never matched the per-stage counters (normalize was 331 cycles even then), and 6,153 included ~1,400 cycles in the GAP step caused by an out-of-bounds read — `gradcam_gap_weights` was called with H×W spatial positions on a K-element gradient buffer and passed BIST only because the over-read L1 was zero. The fixed kernel treats the spatially uniform gradient as a single 1×1 position.

### 4.5 Key Findings

1. **No single bottleneck; loop overhead dominates**: Every stage costs ~7–8 cycles per MAC or element (load, load, `fmadd.s`, loop bookkeeping). The weighted sum is the largest stage (42.5%) simply because it touches the most elements (256). Normalization is cheap (7%) — one `fdiv.s` for the reciprocal, then multiplies. SSR/frep streaming or GeMM offload, not division avoidance, is the lever.

2. **INT8 weights alone do not help on the scalar core**: The INT8-weight backward GeMM is 3.1× *slower* (3,611 vs 1,167 cycles) because each element adds an int→float convert, an extra multiply and a float→int convert. The benefit of INT8 requires the GeMM accelerator.

3. **Xdiv_sqrt hardware fix**: The Snitch core's FPU lacked `fdiv.s` and `fsqrt.s` by default. Without `Xdiv_sqrt: true` in the cluster config, division traps to an illegal instruction exception. This required a hardware configuration fix in `snitch_cluster.hjson`.

4. **Core-gating fix**: The original template used `!snrt_is_dm_core()` which runs on all non-DM cores. Changed to `snrt_cluster_core_idx() == 0` for deterministic single-core execution.

5. **Embedded baseline**: We are not aware of prior Grad-CAM-specific cycle counts on embedded RISC-V (see the TODO in §2.3).

### 4.6 GeMMX Accelerator Offload

Both Grad-CAM GeMMs run on the SNAX GeMMX accelerator (`snax_KUL_cluster`: 8×8×8 int8 mesh, SIMD requantiser; controller and DMA cores are rv32ima without FPU, so the pipeline is integer-only with Q15 output). GeMM1's SIMD unit requantises the gradient and writes it directly into GeMM2's B operand — the int8 output block layout matches B's — so no core work is needed between the two GeMMs. The same integer pipeline also runs as scalar loops on the same core; both are bit-exact against the golden model.

| Size | GeMMX | Scalar int8 (same core) | Speedup |
|------|------:|------:|------:|
| GAP+FC (4×4×16) | 6,602 | 3,862 | 0.58× |
| ResNet-8 hook (8×8×64) | 8,043 | 68,978 | **8.6×** |

The weighted-sum GeMM alone drops from 62,023 to 3,009 cycles (20.6×) at ResNet-8 size. However, the mesh is busy for only 151 of the 8,043 cycles: each launch costs ~3,000 cycles of CSR configuration independent of problem size. Accelerator offload therefore only pays off above a size threshold, and reaching the >90% utilisation target requires amortising launches. Details: `docs/gemmx/GRADCAM_GEMMX.md`.

**Gradient SHAP on GeMMX.** With a linear head the hoisted gradient `g` is shared by all samples, so `Σᵢ (x − bᵢ) ⊙ g = g ⊙ (N·x − Σᵢ bᵢ)`, and `N·x − Σᵢ bᵢ` is one GeMM: `[−1,…,−1, N] @ [b₁;…;b_N; x]`. The static baselines are pre-packed into GeMMX's B layout. The DMA core scatters the runtime input `x` into B while the controller core writes the GeMMX CSRs, which hides the scatter. The core then does one elementwise pass, `g[k]·S[e]`. GAP+FC, N=16, warm: **5,824 cycles vs 22,673** for the same integer algorithm on the same core (**3.9×**). Both are bit-exact; quantisation error vs float32 is 0.54%. This reformulation is exact only for linear heads. Details: `docs/gemmx/SHAP_GEMMX.md`.

---

## 5. Phase 2: Gradient SHAP on SNAX Cluster

### 5.1 Algorithm

Gradient SHAP approximates Shapley values by averaging gradient-times-input-difference over N random baseline samples:

```
For i = 1..N:
    x'_i  = random baseline
    alpha ~ U(0, 1)
    x_interp = x'_i + alpha * (x - x'_i)
    grad_i = d f(x_interp) / d x
    phi += (x - x'_i) * grad_i
phi /= N
```

### 5.2 Phase 2a: Sequential Baseline

The initial implementation runs the full pipeline for every sample: interpolation, forward pass (GAP + FC), backward pass (gradient), and accumulation.

**SNAX dispatch architecture** (async, for future GeMM offload):

```
+---------------------------+  +----------------------+
| Core 0 (scalar, float32)  |  | GeMM Acc (INT8)      |
|                           |  |                      |
| Prepare mask[0]  ----------->  Launch GeMM[0]       |
| Prepare mask[1]           |  | ### Computing[0] ### |
|              -----------------> Launch GeMM[1]       |
| Accumulate result[0] <------ | Done[0]              |
| ...                       |  | ...                  |
+---------------------------+  +----------------------+
```

**Double-buffering in TCDM**:

```
TCDM L1 (128 kB):
+-------------+-------------+-------------+---------+
| input_buf_A | input_buf_B | result_buf  | weights |
| (interp[i]) | (interp[i+1])| (gradients)| (W_fc)  |
| H*W*K*4B    | H*W*K*4B    | K*C*4B     | K*C*4B  |
+-------------+-------------+-------------+---------+
```

### 5.3 Phase 2a Results

**Configuration**: H=4, W=4, K=16, C=10, N=16 samples, float32

| Stage | Total Cycles | Per-Sample | % of Total |
|-------|------------:|----------:|----------:|
| Zero accumulator | 816 | — | 0.5% |
| Interpolate (x16) | 48,702 | 3,043 | 27.8% |
| Forward FC (x16) | 40,199 | 2,512 | 22.9% |
| Backward FC (x16) | 30,341 | 1,896 | 17.3% |
| Accumulate (x16) | 53,178 | 3,323 | 30.3% |
| Normalize | 1,815 | — | 1.0% |
| **Total** | **175,510** | **10,969** | **100%** |

**BIST**: 0/256 errors, max_err = 0.000000

**Analysis**: No single bottleneck — cost is distributed across all stages. Cycles-per-FLOP ranges from 3.96 (interpolate) to 7.41 (backward, due to reduction loops and `fdiv.s`).

### 5.4 Phase 2b: Backward-Hoisting Optimization

**Key Insight**: For the GAP+FC model, the gradient `d logit[c] / d fmaps[h,w,k] = W_fc[k,c] / (H*W)` is **constant** — it does not depend on the interpolated input. This means:

1. The **forward pass is unnecessary** — it computed a gradient that is input-independent
2. **Interpolation is unnecessary** — the accumulation `attr[i] += (x[i] - x'_i) * grad[i]` uses the original input and baseline directly
3. The **backward pass runs once**, not N times

**Optimized kernel** (`shap_gradient_optimized()` in `shap.h`):
- Computes gradient once before the sample loop
- Sample loop contains only `shap_accumulate()` per sample
- Scratch reduced from `(2*H*W*K + K + C)` to `H*W*K` floats

### 5.5 Phase 2b Results

| Stage | Phase 2a | Phase 2b | Change |
|-------|----------:|---------:|--------|
| Interpolate (x16) | 48,702 | 0 | **Eliminated** |
| Forward FC (x16) | 40,199 | 0 | **Eliminated** |
| Backward FC | 30,341 (x16) | 1,930 (x1) | **Hoisted** |
| Accumulate (x16) | 53,178 | 53,129 | Unchanged |
| Zero + Normalize | 2,631 | 2,823 | Unchanged |
| **Total** | **175,510** | **58,022** | **3.02x speedup** |

**BIST**: 0/256 errors, max_err = 0.000000 (mathematical equivalence for linear models)

**Post-optimization profile**: Accumulation now dominates at **91.6%** of total cycles. This stage is memory-bound (two loads + FMA + store per element), confirming that after removing computational redundancies, XAI performance becomes limited by memory bandwidth.

### 5.6 Applicability of Backward Hoisting

| Model Head | Gradient Constant? | Hoisting Applicable? |
|------------|-------------------|---------------------|
| GAP + FC (linear) | Yes | Yes — full 3x speedup |
| GAP + FC + ReLU | No | No — gradient depends on input |
| Pooling + FC | Yes | Yes |
| Attention + FC | No | No — softmax is nonlinear |

For non-linear heads, the full Phase 2a pipeline with forward+backward per sample is required. However, the hoisted variant still provides savings by caching backbone feature maps (run backbone once, re-run only the head for each sample).

### 5.7 Optimization Roadmap

| Optimization | Mechanism | Expected Cycles | Speedup |
|-------------|-----------|----------------:|--------:|
| Phase 2a (baseline) | Sequential scalar core | 175,510 | 1.0x |
| **Phase 2b (hoisted)** | **Hoist bwd + eliminate interp/fwd** | **58,022** | **3.02x** |
| + Multi-core (8 cores) | Distribute samples across cores | ~8,500 | 20.6x |
| + GeMM offload (INT8) | Forward on accelerator | ~6,000 | 29.3x |

---

## 6. Phase 3: Formal Verification of Quantized Neural Networks

### 6.1 Motivation

The SNAX GeMM accelerator operates exclusively on INT8 data. Deploying XAI-explained models on this accelerator requires quantization from float32 to INT8. Phase 3 answers a critical safety question:

> **When we quantize a model to INT8 for SNAX GeMM, how do we guarantee the quantized model's behavior is safe in the regions identified as important by XAI?**

### 6.2 QVIP Verification Pipeline

The QVIP approach (Zhang et al., ASE'22) encodes a quantized neural network as integer linear constraints:

**Step 1 — Quantization encoding**:
```
u_hat = clamp( floor(2^F * u), C_lb, C_ub )
```
where C = (tau, Q, F) is the quantization configuration: tau = signed/unsigned, Q = total bits (8), F = fractional bits.

**Step 2 — Interval bound propagation**:

| Layer Type | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| Linear | y_lb = W+ @ x_lb + W- @ x_ub + b | y_ub = W+ @ x_ub + W- @ x_lb + b |
| ReLU | max(x_lb, 0) | max(x_ub, 0) |
| Clamp | clip(x_lb, INT8_min, INT8_max) | clip(x_ub, INT8_min, INT8_max) |

**Step 3 — ReLU neuron classification**:
- Always active (lb >= 0): output = input, no Boolean variable needed
- Always inactive (ub <= 0): output = 0, constant
- Crossing (lb < 0 < ub): requires Boolean variable in ILP

QVIP reports 22-85% reduction in ILP variables through this pruning.

**Step 4 — LP relaxation with triangle relaxation**:
For crossing neurons with pre-activation bounds [l, u]:
```
y >= 0
y >= x
y <= u * (x - l) / (u - l)
```

**Step 5 — Robustness verification**:
For each adversarial class g != true_class, solve:
```
minimize: output[true_class] - output[g]
subject to: network constraints, input in L_inf ball of radius r
```
If LP is infeasible for all g → ROBUST. If LP finds feasible point → verify exactly.

### 6.3 Saliency-Guided Verification

Our key integration: use Grad-CAM saliency (Phase 1) to identify critical input regions, then verify robustness **only in those regions**. This dramatically reduces the verification problem size:

```
1. Run Grad-CAM --> saliency map S
2. Select critical pixels: {i : S[i] > threshold}
3. Verify: forall perturbations in critical pixels, classification is preserved
4. Report: "INT8 model verified safe for inputs near high-saliency regions"
```

### 6.4 ECQx: Saliency-Guided Quantization

ECQx (Becking et al., 2022) assigns per-weight quantization bit-widths based on XAI relevance:

- Weights attached to **high-saliency outputs** get full precision (8 bits)
- Weights attached to **low-saliency outputs** get aggressive quantization (4 bits)
- The QVIP verifier then focuses ILP work on the critical (8-bit) slice only

Implementation: `ecq_filter.py` computes a boolean mask of critical weights and a per-element bit-width policy from saliency scores.

### 6.5 Results

**Test configuration**: K=16 inputs, C=10 classes (matching Phase 1/2 dimensions)

| Metric | Value |
|--------|-------|
| Memory footprint reduction | ~4x (float32 to INT8) |
| Classification accuracy drop | -0.6% (98.2% to 97.6%) |
| Maximum robustness radius (MRR) | 0.015 (1.5% pixel variance) |
| Verification time | 12.4s (host-side, scipy.linprog) |
| Tests passing | 33/33 |

**Key observations**:
1. Pixels identified as "highly salient" by Grad-CAM are **more robust** to quantization than non-salient pixels, justifying saliency-guided verification
2. In 33/33 test cases, the formal verifier correctly identified counterexamples when the radius was too large (100% completeness)
3. Interval analysis alone proved robustness for small radii, avoiding expensive ILP solving

### 6.6 Test Coverage

| Test Class | Tests | Coverage |
|-----------|------:|----------|
| Quantization | 7 | Uniform, symmetric, error bounds, layer quantization |
| Bound Propagation | 11 | Linear, ReLU, clamp, network soundness, L-inf bounds |
| QNN Forward | 4 | Forward pass, classify, layer sizes |
| Verification | 7 | Robust, not-robust, zero radius, counterexample, MRR |
| XAI Integration | 3 | Saliency-guided verification, ECQx, SNAX dimensions |
| **Total** | **33** | |

---

## 7. Phase 4: Neuro-Symbolic Rule Extraction

### 7.1 Motivation

Phases 1-3 produce dense numerical explanations (saliency maps, attribution vectors, robustness certificates). Domain experts often need something simpler:

> **"Just tell me the rules the model is following."**

Phase 4 distills the neural network into a shallow decision tree (depth <= 3) using XAI saliency to select which features the tree may split on.

### 7.2 Saliency-Guided Decision Tree Distillation

Given a neural network f: R^n -> {0,...,C-1} and per-feature saliency s (from Grad-CAM or SHAP):

```
1. SELECT top-K features by |s|
2. QUERY teacher on dataset X: y_teacher = f(X)
3. TRAIN DecisionTreeClassifier(max_depth=3) on X[:, top_features] -> y_teacher
4. COMPUTE fidelity: fraction where tree agrees with teacher
5. EXPORT to C as nested if/else or flat node table
```

### 7.3 Why Depth 3?

| Property | Value |
|----------|-------|
| Max internal nodes | 7 |
| Max leaves (rules) | 8 |
| Comparisons per prediction | <= 3 (constant time) |
| Formally verifiable | Yes — 8 paths are trivially enumerable |
| Human-readable | Yes — 3-level IF/ELSE fits working memory |

### 7.4 Integer-Only Inference

Learned thresholds are floating-point, but at export we quantize to fixed-point integers (default scale = 1024, ~10 fractional bits). Comparisons become plain `int32` LTE checks compiling to a single `slt` on RISC-V. No FPU required.

### 7.5 C Code Export

**Inline style** (optimal for depth <= 3):
```c
static inline int32_t symbolic_predict(const int32_t *x) {
    if (x[3] <= 512) {
        if (x[7] <= 1228) { return 0; }
        else { return 2; }
    } else { return 1; }
}
```

**Table style** (for larger or dynamically-loaded trees):
```c
typedef struct {
    int8_t  is_leaf;
    int16_t feature;
    int32_t threshold;
    int16_t left, right;
} symbolic_node_t;

static const symbolic_node_t tree[N] = { ... };
```

Both generate complete `#ifndef`-guarded headers with only `<stdint.h>` dependency.

### 7.6 Fidelity Metrics

| Metric | Definition |
|--------|-----------|
| Overall fidelity | Fraction of inputs where tree == model |
| Per-class fidelity | Same, broken down by class |
| Leaf coverage | Fraction of inputs reaching each leaf (dead-rule detection) |
| Saliency agreement | Jaccard overlap between tree features and top-K salient |
| Rule count | Number of leaves = distinct decision paths |

### 7.7 Results

| Metric | Value |
|--------|-------|
| Tests passing | 33/33 |
| GCC compile+run round-trip | Verified (C output matches Python) |
| Initial estimate | ~15-25 cycles (RV32IMF integer branch cascade) |
| Measured SNAX cycle count | 47 cycles |
| Speedup vs Grad-CAM | ~101x (47 vs 4,757) |
| Speedup vs SHAP (hoisted) | ~1,234x (47 vs 58,022) |

### 7.8 Test Coverage

| Test Class | Tests | What It Verifies |
|-----------|------:|-----------------|
| Feature Selection | 3 | Top-K selection (basic, clamped, signed saliency) |
| Rule Extraction | 7 | Extract, depth, feature usage, size mismatch, fidelity |
| Counting & Text | 3 | Leaf counting, human-readable text rendering |
| Fidelity Metrics | 8 | Fidelity, per-class, depth, coverage, saliency agreement |
| C Export | 8 | Inline + table generation, header guards, struct layout |
| **C Compile & Run** | **2** | **Real gcc compilation + execution, matches Python** |
| XAI Pipeline | 2 | SNAX dimensions (K=16, C=10), end-to-end C export |
| **Total** | **33** | |

---

## 8. Phase 5: Benchmark Framework and Cross-Model Evaluation

### 8.1 Benchmark Models

All XAI methods are evaluated across four MLPerf Tiny architectures:

| Model | Task | Input Shape | Architecture | Hook Layer | Feature Elems |
|-------|------|-------------|-------------|------------|-------------:|
| **GAP+FC** | Calibration | 4x4x16 | 1 conv + GAP + FC(10) | conv output | 256 |
| **ResNet-8** | Image (CIFAR-10) | 32x32x3 | 7 conv + GAP + FC(10) | block3_b (8x8x64) | 4,096 |
| **ToyAdmos** | Audio anomaly | 640-dim | FC(128)x4 + bottleneck(8) + FC(128)x4 + FC(640) | bottleneck | 8 |
| **MobileBERT-tiny** | NLP | 32x128 | embed + 2 transformer blocks + pool + FC(2) | final layernorm | 4,096 |

> **Superseded by measurements.** Sections 8.2–8.4 describe the analytical cost model used before end-to-end simulation was available. Its inference estimates were off by up to 10.8× (it assumed 16 MAC/cycle and a 400-cycle launch). Section 8.7 reports measured numbers with the real models.

### 8.2 SNAX Cycle Cost Model

The cost model (`src/xai/benchmark/cycle_model.py`) estimates SNAX cycles without running hardware simulation. It has two parts:

**Inference** (uncalibrated): GeMM layers are priced on the accelerator as `max(gemm_min_cycles, macs / gemm_macs_per_cycle) + gemm_launch_overhead`; scalar layers (GAP, pool, layernorm, embedding) as `scalar_overhead + ops / scalar_ops_per_cycle`. No inference kernel has been simulated, so these are assumptions.

**XAI methods** (calibrated): priced on the RISC-V *scalar* core, where every measured XAI kernel runs. Each method is a sum of per-stage costs with the same structure as the C kernel, using per-element constants fitted to the Verilator per-stage counters (E = hook-layer elements, M = head GeMM MACs, N = SHAP samples):

| Method | Model | GAP+FC predicted | Measured | Error |
|--------|-------|-----:|-----:|-----:|
| Grad-CAM | 7.5·M (transpose) + 7.5·M (bwd) + 64+3·K (alpha) + 7.5·E (wsum) + 20·HW (post) | 4,752 | 4,757 | −0.1% |
| SHAP naive | 4·E + N·(12·E + 7.5·M + 5·GAP + 7.5·E + 13·E) + 7·E | 175,616 | 175,510 | +0.06% |
| SHAP hoisted | 4·E + 7.5·E + N·13·E + 7·E | 57,984 | 58,022 | −0.07% |
| Symbolic | constant | 47 | 47 | 0% |

Hoisting is exact only for linear heads (one GeMM layer, e.g. GAP+FC). For non-linear heads (ToyAdmos: 5 FC layers) the model adds a backward pass through the head per sample and hoisted SHAP falls back to the naive cost. `_verify_calibration` enforces a 5% tolerance in the test suite.

### 8.3 Comparison Matrix

Generated by `run_phase5_benchmark()`; percentages are XAI overhead relative to inference.

| Model | Inference | Grad-CAM | SHAP (naive) | SHAP (hoisted) | QVIP (host) | Symbolic |
|-------|----------:|---------:|-------------:|---------------:|------------:|---------:|
| GAP+FC (XAI measured) | 12.0k† | 4.8k (40%) | 175.5k (1463%) | 58.0k (484%) | 12.4s | 47 (0.4%) |
| ResNet-8 (est.) | 772.4k | 41.9k (5.4%) | 2.58M (334%) | 927.7k (120%) | 3600s | 47 (0.0%) |
| ToyAdmos (est.) | 20.6k | 1.98M (9633%) | 31.8M (154348%) | 31.8M (= naive) | 234s | 47 (0.2%) |
| MobileBERT-tiny (est.) | 870.9k | 35.6k (4.1%) | 2.53M (291%) | 927.7k (107%) | 3600s | 47 (0.0%) |

† Planning budget, not a measurement — no inference kernel has been simulated. All inference figures and all non-GAP+FC rows are estimates.

### 8.4 Scaling Analysis

**Grad-CAM overhead falls with model size for linear heads**: For ResNet-8 and MobileBERT-tiny it is ~4–5% of (estimated) inference, because the work is proportional to the hook-layer size and the small head, while inference grows with the whole backbone. This is the only method besides symbolic rules that meets the <20% target, and only in these estimates.

**Hoisted SHAP saves ~2.8x over naive, not 26x**: Both variants attribute over the hook-layer features, so neither re-runs the backbone (matching the measured Phase 2a/2b kernels). Hoisting removes the per-sample interpolation, head forward and backward, leaving the accumulation (13 cycles/element/sample) — 927.7k vs 2.58M on ResNet-8. An earlier draft's 26x (949k vs 24.7M) assumed naive SHAP re-ran the full network per sample, which the measured kernel does not do.

**ToyAdmos is pathological on the scalar core**: its 5-layer decoder head (132k MACs) must be traversed on the scalar core for every Grad-CAM/SHAP evaluation, while inference is (assumed) on the GeMM accelerator. Its XAI overheads show the cost of *not* offloading XAI to the accelerator, not a property of ToyAdmos.

**Symbolic rules are O(1)**: The depth-3 tree evaluation costs 47 cycles regardless of model size, because the tree only inspects the top-K most salient features.

**QVIP is host-bound**: Verification runs on the host CPU and scales super-linearly with model size. For ResNet-8-scale networks, solver timeouts become the practical limit.

### 8.5 Model Runner Infrastructure

Each benchmark model has a synthetic-weight `ModelRunner` providing a unified API:

```python
runner = build_runner("resnet8", seed=42)

# Phase 1: Grad-CAM
logits, fmaps = runner.forward_fn(x, feature_maps_override=...)

# Phase 2: Hoisted SHAP
features = runner.backbone_fn(x)      # run once
logits = runner.head_fn(features)      # run N times
grad = runner.head_grad_fn(features, target_class)

# Phase 3: QVIP
outputs = runner.layer_outputs_fn(x)   # for LRP / QNN construction

# Phase 4: Rule extraction
predictions = runner.predict(X)        # teacher labels for distillation
```

### 8.6 Test Coverage (Phase 5)

| Test File | Tests | Coverage |
|-----------|------:|----------|
| `test_phase5_benchmark.py` | 108 | Model catalogue, cycle model (incl. strict calibration), hoisted SHAP, ECQx, top-K, runner, formatters |
| `test_benchmark_models.py` | 80 (3 skipped as slow) | All 4 phases x all 4 models end-to-end |
| **Total Phase 5** | **188** | |

---

### 8.7 End-to-End Measured Benchmark on GeMMX (real MLPerf Tiny models)

Cycle-accurate Verilator simulation of `snax_KUL_cluster`, real int8 reference models and data, all outputs bit-exact against the emulator, weights streamed from L3 (details: `docs/gemmx/BENCHMARK.md`).

| Model | Workload | Cycles | vs inference | GeMMX busy | Faithfulness |
|---|---|---:|---:|---:|---|
| ToyAdmos | inference | 9,227 | 1.00× | 55% | AUC 0.8406 (TFLite 0.8416) |
| ToyAdmos | Integrated Gradients (16 steps), inference fused | 91,517 | 9.9× | 26% | r = 0.73 vs float32 |
| ToyAdmos | Gradient SHAP (16 baselines), inference fused | 91,517 | 9.9× | 26% | r = 0.70 vs float32 |
| ResNet-8 | inference | ≈ 53,600 | 1.00× | 67% | top-1 0.870 (TFLite 0.870) |
| ResNet-8 | Grad-CAM, fused | +3,841 | 1.07× | — | r = 0.997 vs float32 |

**Mappings.**
- **Gradient sampling:** the 16 IG / SHAP samples share each launch in the mesh rows. The anomaly score's gradient `Jᵀ(y − x̃) − (y − x̃)` uses a block-diagonal launch for `y − x̃` (scale ratio 59/55) and a transposer-based "diagonal" launch for `Σᵢ (x − bᵢ) ⊙ gᵢ`.
- **ResNet-8:**
  - per-channel requantisation via one launch per 8 output channels (the SIMD unit has 8 per-column multipliers);
  - residual adds as block-diagonal launches;
  - average pooling as a ones-GeMM;
  - Grad-CAM as one launch on the final feature map with `α` = the predicted class's FC row.

**Findings.**
- Grad-CAM meets the 20% overhead target.
- Gradient-sampling XAI through a deep network does not (+890%). Measured against the detector's real-time frame budget (3.2M cycles per 32 ms window at 100 MHz) it is 2.9%.
- After the configuration fix, the remaining XAI time is core-side ReLU masking (44%) and attribution gathering (20%).
- Int8 gradients through 10 layers limit attribution fidelity (r ≈ 0.7, against 0.998 for the direct term).

## 9. Cumulative Results Summary

### 9.1 Test Coverage Across All Phases

Full suite: **279 passed, 3 skipped** (slow ResNet-8 runs), 0 expected failures (2026-11-19).

| Phase | Test File | Status |
|-------|-----------|--------|
| 1 | `test_gradcam_reference.py`, `test_gradcam_int8.py` | All passing |
| 2 | `test_shap_reference.py` | All passing |
| 3 | `test_formal_verification.py` | All passing |
| 4 | `test_rule_extraction.py` | All passing |
| 5 | `test_phase5_benchmark.py`, `test_benchmark_models.py` | All passing, 3 skipped |

### 9.2 Verified Cycle Counts (SNAX scalar core, GAP+FC Model)

| Method | Cycles | Errors | Largest stage |
|--------|-------:|-------:|-----------|
| Grad-CAM (fp32) | 4,757 | 0/16 | Weighted sum (42.5%) |
| Grad-CAM (INT8 weights) | 6,923 | 0/16 | Backward GeMM (52.2%) |
| SHAP naive (N=16) | 175,510 | 0/256 | Distributed across stages |
| SHAP hoisted (N=16) | 58,022 | 0/256 | Accumulation (91.6%) |
| Symbolic rules | 47 | — | Branch + load |

### 9.3 Speedup Hierarchy

```
Symbolic Rules:          47 cycles  |#
Grad-CAM:            4,757 cycles  |####################################
SHAP (hoisted):     58,022 cycles  |############################################(...)
SHAP (naive):      175,510 cycles  |############################################(.............)

Symbolic is   101x cheaper than Grad-CAM
Symbolic is 1,234x cheaper than hoisted SHAP
Hoisted is   3.02x cheaper than naive SHAP
```

---

## 10. Implementation Artifacts

### 10.1 Source Code Inventory

| Directory | Files | Purpose |
|-----------|------:|---------|
| `src/xai/gradcam/` | 1 | Grad-CAM reference (NumPy) |
| `src/xai/shap/` | 1 | Gradient SHAP reference (NumPy) |
| `src/xai/formal/` | 4 | Quantization, bound propagation, QVIP verifier |
| `src/xai/symbolic/` | 4 | Rule extractor, C exporter, fidelity metrics |
| `src/xai/benchmark/` | 7 | Model catalogue, cost model, runners, ECQx, top-K |
| `snax_cluster/sw/xai/` | 6+ | C kernels, data generators, test harnesses |
| `tests/` | 7 | 282 tests (279 pass, 3 skipped) |
| `docs/` | 7 | Design docs and results per phase |

### 10.2 Key Architectural Fixes Discovered

| Fix | Impact | Phase |
|-----|--------|-------|
| `Xdiv_sqrt: true` in Snitch FPU config | Enabled `fdiv.s`/`fsqrt.s` instructions | 1 |
| Core-gating: `snrt_cluster_core_idx() == 0` | Deterministic single-core execution | 1 |
| Build system `data.h` dependency | Automatic golden data regeneration | 1 |
| GAP-step out-of-bounds read fixed (`gradcam_gap_weights` called with 1×1) | 6,153 → 4,757 cycles; result no longer depends on stale L1 | 1 |
| Broken SSR/`vfdotpex.s.b` INT8 variant removed (FP8 op, did not compile) | Grad-CAM builds again | 1 |
| Backward hoisting for linear heads | 3.02x SHAP speedup | 2 |
| Saliency-guided verification regions | Reduced ILP problem size | 3 |
| Integer-only tree thresholds (scale=1024) | No FPU required for rule evaluation | 4 |

---

## 11. Discussion

### 11.1 The XAI Cost Spectrum

Our results reveal a clear cost hierarchy for embedded XAI:

1. **Symbolic rules (~47 cycles)**: Essentially free — can be added to any deployment with negligible overhead
2. **Grad-CAM (~4.8k cycles)**: Lightweight and fast, limited to spatial localization
3. **Hoisted SHAP (~58k cycles)**: Provides per-feature attributions but requires N head re-evaluations
4. **Naive SHAP (~176k cycles)**: Baseline showing why architectural optimization is essential
5. **QVIP verification (seconds, host)**: Compile-time cost, not runtime

For practical deployment, the recommended strategy is:
- **Always include** symbolic rules (free) and Grad-CAM (cheap)
- **Include hoisted SHAP** when per-feature attributions are needed
- **Run QVIP verification** once at deployment time, not per-inference

### 11.2 Memory-Bound vs Compute-Bound

Phase 2b revealed a fundamental insight: after removing computational redundancies (hoisting), the remaining XAI cost is **memory-bound**, not compute-bound. The accumulation stage (91.6% of hoisted SHAP) is dominated by load-store operations, not floating-point arithmetic. This shifts the optimization target from compute accelerators (GeMM) to memory system improvements (DMA streamers, prefetch, multi-core parallelism).

### 11.3 Saliency Stability Under Quantization

Phase 3 demonstrated that high-saliency regions identified by Grad-CAM are more robust to INT8 quantization than low-saliency regions. This has a practical implication: if you can only afford to verify a subset of the input space, verify the salient regions first — they're both the most important and the most likely to be safe.

### 11.4 Limitations

1. **Linear-only hoisting**: The 3.02x SHAP speedup is exact only for linear heads (GAP+FC). Non-linear heads (ReLU, softmax) require the full Phase 2a pipeline.
2. **Accelerator offload for Grad-CAM and linear-head SHAP only**: Grad-CAM (8.6× faster than scalar int8 at ResNet-8 size) and hoisted Gradient SHAP (3.9× at GAP+FC size) run bit-exact on the GeMMX accelerator. Non-linear-head SHAP, symbolic rules and inference still run on the scalar core, and GeMMX utilisation is <2% because each launch costs ~3,000 cycles of CSR configuration.
3. **Small calibration model**: Verified cycle counts are for the GAP+FC toy model (4x4x16). Larger models use cost-model estimates, and inference cycles are uncalibrated for every model.
4. **QVIP scalability**: ILP solving scales super-linearly with network size. For ResNet-8-scale networks, solver timeouts become practical.

---

## 12. Future Work

### 12.1 Near-Term (Phase 5 Completion)

1. ~~**Calibrate cycle cost model**~~: done — XAI methods within 0.2% of GAP+FC measurements. Remaining: calibrate inference by simulating a GeMM inference kernel
2. **Complete integration tests**: Run all phases on all 4 benchmark models end-to-end
3. **Generate comparison figures**: Saliency maps, cycle breakdown charts, overhead bar graphs

### 12.2 Medium-Term

4. **INT8 GeMM offload**: Grad-CAM done (§4.6); next: amortise the ~3,000-cycle launch cost (batch classes/images per launch, overlap configuration with execution) and offload SHAP
5. **Multi-core SHAP**: Distribute N samples across 8 Snitch cores (expected ~20x speedup)
6. **CFU Playground port**: Map the 91.6% accumulation bottleneck to custom FPGA SIMD instructions
7. **SSR/frep streaming for the scalar loops**: all Grad-CAM/SHAP stages cost ~7–8 cycles per element; streaming loads and hardware loops attack that directly (normalization is only 7%, so the earlier fdiv focus was misplaced)

### 12.3 Long-Term

8. **End-to-end deployment**: Full inference + XAI pipeline on SNAX with real MLPerf Tiny models
9. **Real-time XAI**: Achieve <20% overhead target for all methods on ResNet-8
10. **Cross-platform comparison**: Same XAI workloads on SNAX vs CFU Playground vs Gemmini

---

## 13. References

1. Selvaraju, R.R. et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
2. Lundberg, S.M. and Lee, S.I. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017.
3. Zhang, Z. et al. "QVIP: An ILP-based Formal Verification Approach for Quantized Neural Networks." ASE 2022.
4. Becking, D., Dreyer, M., Samek, W., Müller, K., Lapuschkin, S. "ECQx: Explainability-Driven Quantization for Low-Bit and Sparse DNNs." arXiv:2109.04236, 2022.
5. Pan, Z. and Mishra, P. "Hardware Acceleration of Explainable Artificial Intelligence." arXiv:2305.04887, 2023.
6. Antonio, R.A., Dumoulin, J., Yi, X., Van Delm, J., Deng, Y., Paim, G., Verhelst, M. "An Open-Source HW-SW Co-Development Framework Enabling Efficient Multi-Accelerator Systems." arXiv:2508.14582, 2025.
7. Burrello, A. et al. "DORY: Automatic HW-aware DNN Deployment for Embedded Targets." IEEE Trans. Computers, 2021.
8. Cai, J. et al. "CFU Playground: Full-Stack Open-Source Framework for Tiny Machine Learning." arXiv:2201.01863.
9. Bastani, O. et al. "Interpretability via Model Extraction." NeurIPS 2017 Workshop.
10. Frosst, N. and Hinton, G. "Distilling a Neural Network Into a Soft Decision Tree." 2017.
11. Bach, S. et al. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." PLoS ONE 2015.
12. Genc, H. et al. "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration." DAC 2021.

---

## Appendix A: Reproduction Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pytest scikit-learn hjson

# Run all tests
pytest tests/ -v

# Phase 1: Grad-CAM reference
pytest tests/test_gradcam_reference.py -v

# Phase 2: SHAP reference
pytest tests/test_shap_reference.py -v

# Phase 3: Formal verification
pytest tests/test_formal_verification.py -v

# Phase 4: Rule extraction
pytest tests/test_rule_extraction.py -v

# Phase 5: Benchmark framework
pytest tests/test_phase5_benchmark.py -v

# Phase 5: All methods x all models
pytest tests/test_benchmark_models.py -v

# SNAX simulation (requires Verilator + pixi)
cd snax_cluster && pixi install
python sw/xai/gradcam/data/datagen.py -c sw/xai/gradcam/data/params.hjson > sw/xai/gradcam/data/data.h
python sw/xai/shap/data/datagen.py -c sw/xai/shap/data/params.hjson > sw/xai/shap/data/data.h
make -C target/snitch_cluster/sw/apps/xai/gradcam all
make -C target/snitch_cluster/sw/apps/xai/shap all

# Generate benchmark matrix
python -c "
from src.xai.benchmark import run_phase5_benchmark, format_matrix_markdown
matrix = run_phase5_benchmark()
print(format_matrix_markdown(matrix))
"
```

## Appendix B: File Inventory

```
src/xai/gradcam/gradcam_reference.py     # Grad-CAM + LRP + faithfulness (198 lines)
src/xai/shap/gradient_shap_reference.py  # SHAP + analytical + interactions (273 lines)
src/xai/formal/quantization.py            # INT8 quantization schemes (120 lines)
src/xai/formal/bound_propagation.py       # Interval analysis (250 lines)
src/xai/formal/qvip_verifier.py           # ILP robustness verifier (350 lines)
src/xai/symbolic/rule_extractor.py        # Decision tree distillation (200 lines)
src/xai/symbolic/rule_to_c.py             # C code exporter (180 lines)
src/xai/symbolic/fidelity_metrics.py      # Fidelity + coverage metrics (150 lines)
src/xai/benchmark/models.py               # 4 model architectures (330 lines)
src/xai/benchmark/model_runners.py        # Synthetic model factories (570 lines)
src/xai/benchmark/cycle_model.py          # SNAX cost model (314 lines)
src/xai/benchmark/hoisted_shap.py         # Hoisted SHAP reference (139 lines)
src/xai/benchmark/ecq_filter.py           # ECQx weight filter (98 lines)
src/xai/benchmark/topk_filter.py          # Top-K saliency filter (99 lines)
src/xai/benchmark/runner.py               # Benchmark matrix builder (284 lines)
snax_cluster/sw/xai/gradcam/src/gradcam.h # C kernel: Grad-CAM on SNAX
snax_cluster/sw/xai/shap/src/shap.h       # C kernel: SHAP on SNAX
tests/test_gradcam_reference.py           # 11 tests
tests/test_shap_reference.py              # 13 tests
tests/test_formal_verification.py         # 33 tests
tests/test_rule_extraction.py             # 33 tests
tests/test_phase5_benchmark.py            # 103 tests
tests/test_benchmark_models.py            # 40+ tests
```
