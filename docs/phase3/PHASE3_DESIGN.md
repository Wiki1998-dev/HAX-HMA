# Phase 3 Design: Formal Verification of Quantized Neural Networks

**Status**: COMPLETE — Python reference implementation, 33/33 tests passing
**Date**: 2026-04-07
**Platform**: Host-side Python (scipy.optimize.linprog for ILP/LP solving)

## Motivation

The SNAX GeMM accelerator is INT8-only, but our XAI methods (Grad-CAM, SHAP)
run in float32 on the scalar RISC-V core. Any model deployed on the GeMM
accelerator must be quantized from float32 to INT8. Phase 3 answers:

> **When we quantize a model to INT8 for SNAX GeMM, how do we guarantee
> the quantized model's behavior is safe?**

This directly connects to the research narrative:
- Phase 1 (Grad-CAM) tells us **WHERE** the model is looking
- Phase 3 (Formal verification) proves the quantized model is **SAFE** in those regions
- Phase 2 (SHAP) explains **WHY** individual features matter

## Technical Approach

### QVIP: ILP-based Verification (Zhang et al., ASE'22)

The core algorithm encodes a quantized neural network (QNN) as integer linear
constraints, then checks robustness properties using an ILP solver.

**Quantization encoding** (Section 2.2):
```
û = clamp(⌊2^F · u⌋, C^lb, C^ub)
```
where `C = (τ, Q, F)` is the quantization configuration:
- `τ ∈ {±, +}`: signed or unsigned
- `Q`: total bits (8 for SNAX GeMM)
- `F`: fractional bits

**Network encoding** (Section 3.2): Each layer's computation is expressed as
integer linear constraints. ReLU activations become piecewise constant
functions, encoded with Boolean indicator variables.

**Interval analysis** (Section 3.5): Before building the full ILP, propagate
interval bounds through the network. Neurons that are always-active (lb ≥ 0)
or always-inactive (ub ≤ 0) don't need Boolean variables, dramatically
reducing the constraint count. QVIP reports 22-85% reduction.

**Robustness property** (Section 3.4): For class c and attack radius r:
```
∀ x̂ ∈ R̂_p(û, r): N̂(x̂) = N̂(û)
```
Negated for ILP: find x̂ in the region where output class changes.
If ILP is infeasible → network is provably robust.

**Maximum robustness radius** (Section 4): Binary search over r,
calling the verification procedure at each step.

### ECQx: Explainability-Driven Quantization (Becking et al., 2022)

ECQx uses LRP (Layer-wise Relevance Propagation) to guide quantization:

**Key insight**: Weight magnitude ≠ weight relevance. Small weights can be
highly relevant (especially in early layers close to input). Standard
magnitude-based quantization may destroy relevant small weights while
preserving irrelevant large ones.

**ECQx assignment function** (Equation 11):
```
A_x^(l) = argmin_c {
    ρ R_W · (d(W, w_{c=0}) - λ log₂(P_{c=0}))  if c = 0  (zero cluster)
    d(W, w_c) - λ log₂(P_c)                      if c ≠ 0  (non-zero cluster)
}
```
The relevance term `ρ R_W` increases the cost of assigning relevant weights
to the zero cluster, preventing their removal during quantization.

### Our Integration

We combine both approaches:
1. **Quantize** the float32 model to INT8 (matching SNAX GeMM scheme)
2. **Use Grad-CAM saliency** (Phase 1) to identify critical input regions
3. **Formally verify** that the quantized model preserves classification
   in those critical regions
4. **Report**: "This INT8 model is verified safe for inputs near the
   high-saliency regions identified by Grad-CAM"

## Implementation

### Module Structure

```
src/xai/formal/
├── __init__.py
├── quantization.py        # INT8 quantization (symmetric + QVIP-style)
├── bound_propagation.py   # Interval arithmetic bounds through network
└── qvip_verifier.py       # ILP verification engine
```

### quantization.py

Two quantization schemes:

| Scheme | Use Case | Formula |
|--------|----------|---------|
| **Symmetric** | SNAX GeMM (INT8) | `x_q = round(x / scale)`, `scale = max(\|x\|) / 127` |
| **QVIP uniform** | Formal verification | `û = clamp(⌊2^F · u⌋, C^lb, C^ub)` |

Both schemes are interconvertible. The symmetric scheme is what SNAX hardware
uses; the QVIP scheme is what the formal verifier needs.

### bound_propagation.py

Interval arithmetic for three layer types:

| Layer | Propagation Rule |
|-------|-----------------|
| **Linear** | `y_lb = W⁺ @ x_lb + W⁻ @ x_ub + b` |
| **ReLU** | `y_lb = max(x_lb, 0)`, `y_ub = max(x_ub, 0)` |
| **Clamp** | `y_lb = clip(x_lb, lb, ub)` |

**Neuron classification** for ReLU pruning:
- Always active (`lb ≥ 0`): no Boolean variable needed
- Always inactive (`ub ≤ 0`): output is constant 0
- Crossing (`lb < 0 < ub`): requires Boolean variable in ILP

### qvip_verifier.py

Verification pipeline:
1. Compute input bounds for attack radius r
2. Propagate bounds through network (interval analysis)
3. If output bounds alone prove robustness → return ROBUST
4. Otherwise, solve LP relaxation for each adversarial class
5. For 2-layer networks: triangle relaxation for crossing ReLU neurons
6. If LP says no adversarial input exists → ROBUST
7. If LP finds candidate → verify exactly on QNN → NOT_ROBUST or ROBUST

**LP relaxation** (triangle relaxation for ReLU):
```
For crossing neuron with pre-activation bounds [l, u]:
    y ≥ 0
    y ≥ x
    y ≤ u·(x - l) / (u - l)
```

**Enumeration fallback**: For small input spaces (≤100K points), exact
enumeration is faster and complete.

## Test Coverage

33 tests across 5 test classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| TestQuantization | 7 | Uniform, symmetric, error bounds, layer quantization |
| TestBoundPropagation | 11 | Linear, ReLU, clamp, network soundness, L-inf bounds |
| TestQNN | 4 | Forward pass, classify, layer sizes |
| TestVerification | 7 | Robust, not-robust, zero radius, counterexample, MRR |
| TestXAIIntegration | 3 | Saliency-guided verification, ECQx insight, SNAX dims |

### Key test properties verified:
- **Soundness**: Propagated bounds always contain actual network outputs (100 random samples)
- **Completeness**: Counterexamples are valid (produce different classification)
- **Error bounds**: Actual quantization error ≤ theoretical bound for all elements
- **SNAX dimensions**: Verification works at K=16, C=10 (matching Phase 1/2)

## Connection to SNAX Hardware

| Phase | Method | Platform | Purpose |
|-------|--------|----------|---------|
| Phase 1 | Grad-CAM | SNAX scalar (float32) | Identify salient regions |
| Phase 2 | SHAP | SNAX scalar (float32) | Explain feature importance |
| Phase 3 | QVIP verify | Host Python (scipy) | Prove INT8 safety |
| Future | INT8 inference | SNAX GeMM accelerator | Quantized deployment |

The verification runs on the host before deployment. Once verified, the
INT8 model can be deployed on SNAX GeMM with formal safety guarantees
for the verified input regions.

## Benchmarks from QVIP Paper

| QNN Architecture | Dataset | Accuracy (Q=8) | MRR (avg) |
|-----------------|---------|----------------|-----------|
| P1 (784:64:10) | MNIST | 97.10% | ~20 |
| P2 (784:100:10) | MNIST | 97.45% | ~9 |
| P3 (784:64:32:10) | MNIST | 97.01% | ~5 |
| P4 (784:64:10) | F-MNIST | — | — |

Our SNAX test network (K=16 → C=10) is comparable in scale to the output
layer of P1, making QVIP verification tractable.
