# src/xai — XAI Implementation

## Purpose
This directory contains all our XAI method implementations.
It bridges the paper formulations (../papers/) with the hardware targets
(../snax_cluster/ and ../CFU-Playground/).

## Structure
```
src/xai/
├── gradcam/
│   ├── gradcam_pass.py        ← MLIR compiler pass (SNAX)
│   ├── gradcam_kernel.c       ← C runtime kernel (embedded)
│   └── gradcam_reference.py  ← NumPy reference for correctness checking
├── shap/
│   ├── shap_dispatch.py       ← Async multi-pass SHAP dispatcher (SNAX)
│   ├── shap_kernel.c          ← Embedded C for masked forward passes
│   └── shap_reference.py     ← Reference using shap library (host-side)
├── formal/
│   ├── qvip_pass.py           ← Compile-time ILP verification pass
│   ├── property_spec.py       ← Input-output property specification DSL
│   └── README.md              ← Formal verification methodology
├── symbolic/
│   ├── rule_extractor.c       ← RISC-V decision tree (depth ≤ 3)
│   ├── rule_extractor.py      ← Host-side training of rules from saliency
│   └── rules.h                ← Generated rule table (C header)
└── utils/
    ├── buffer_planner.py      ← Static SPM buffer layout calculator
    ├── zigzag_wrapper.py      ← ZigZag cost estimation for XAI ops
    └── metrics.py             ← Faithfulness, localization accuracy metrics
```

## Implementation Order
1. `gradcam/gradcam_reference.py` — verify algorithm on host first
2. `gradcam/gradcam_kernel.c` — port to embedded C, test in simulation
3. `gradcam/gradcam_pass.py` — MLIR automation pass
4. `shap/shap_reference.py` → `shap_kernel.c` → `shap_dispatch.py`
5. `formal/` — QVIP-style verification on generated MLIR
6. `symbolic/` — rule extraction on RISC-V

## Key Formulas

### Grad-CAM (from paper: "Hardware Acceleration of Explainable AI")
```
alpha_c^k = (1/Z) * sum_{i,j} (d y^c / d A^k_{ij})   # global avg pool of grads
L^c_GradCAM = ReLU( sum_k alpha_c^k * A^k )            # weighted sum of feature maps
```
Where A^k is the k-th feature map, y^c is the score for class c.
Step 1 (gradient) = transpose-GeMM on SNAX GeMM accelerator.
Step 2 (weighted sum) = element-wise SIMD on RISC-V.

### Gradient SHAP
```
phi_i = E_x'[ (x_i - x'_i) * (d f(x') / d x'_i) ]   # expectation over baselines
```
Approximate with N=8-16 random baseline samples.
Each sample = one async SNAX forward pass.

### Formal property (QVIP-style)
```
Property: forall x in [lb, ub]: f(x)[true_class] > f(x)[j]  for all j != true_class
```
Encode as ILP over quantized weight bounds, solve with scipy.optimize.linprog.

## Testing Protocol
```bash
# 1. Run reference (Python, host)
pytest tests/test_gradcam_reference.py -v

# 2. Run embedded simulation (Verilator)
pytest tests/test_gradcam_snax.py -v --snax-sim

# 3. Compare: embedded vs reference output
pytest tests/test_gradcam_match.py -v --tolerance=1e-3

# 4. Measure overhead
pytest tests/test_xai_overhead.py -v  # reports XAI/inference cycle ratio
```

## XAI Quality Metrics (implement in utils/metrics.py)
- **Faithfulness**: delete top-K salient features → check prediction drop
- **Localization**: does saliency peak fall within ground-truth region?
- **Stability**: Lipschitz estimate — does small input change → small explanation change?
- **Overhead**: XAI_cycles / inference_cycles (target: < 0.2)

## Buffer Planner Usage
```python
from src.xai.utils.buffer_planner import SPMPlanner

planner = SPMPlanner(total_bytes=128*1024)
planner.reserve("model_activations", 40*1024)
planner.reserve("weight_tile", 32*1024)
planner.reserve("gradient_buffer", size="same_as:model_activations")
planner.reserve("shap_baselines", 16*1024)
planner.check_fits()   # raises if over budget
planner.print_layout() # prints memory map
```

## ZigZag Cost Estimation
Before implementing any XAI op, estimate its cost:
```python
from src.xai.utils.zigzag_wrapper import estimate_xai_cost

cost = estimate_xai_cost(
    op="transpose_gemm",
    M=16, N=16, K=64,        # gradient dimensions
    accelerator="snax_gemm"  # ../zigzag/inputs/hardware/snax_gemm.yaml
)
print(f"Estimated cycles: {cost['total_cycles']}")
print(f"Data movement:    {cost['dram_accesses_bytes']} bytes")
```
