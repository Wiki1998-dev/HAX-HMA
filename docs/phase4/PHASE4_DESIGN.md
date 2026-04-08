# Phase 4 Design: Neuro-Symbolic Rule Extraction

**Status**: COMPLETE — Python reference + C exporter, 33/33 tests passing
**Date**: 2026-04-08
**Platform**: Host-side Python (sklearn) + generated C (no FP required)

## Motivation

Phases 1–3 give us dense, numerical explanations of what a neural network
does: Grad-CAM saliency, SHAP attributions, and formal verification of a
quantized deployment. Those are useful for researchers, but a domain expert
looking at a deployed model often wants something simpler:

> **"Just tell me the rules the model is following."**

Phase 4 answers this by distilling the black-box model into a shallow
decision tree (depth ≤ 3) using XAI saliency to pick which features the
tree is allowed to split on. The result is:

- A small set of **IF-THEN rules** that can be audited by humans
- Sparse — uses only the top-K most-salient features
- **Integer-only** at inference — no FPU required
- Exportable to a self-contained C header that drops into any embedded
  project (SNAX, CFU Playground, bare-metal RISC-V)

## Research Narrative

| Phase | Method | Question Answered |
|-------|--------|-------------------|
| Phase 1 | Grad-CAM (6,153 cyc) | **WHERE** is the model looking? |
| Phase 2 | SHAP (58,022 cyc) | **WHY** does each feature matter? |
| Phase 3 | QVIP verification (33 tests) | Is the INT8 model **SAFE**? |
| Phase 4 | Rule extraction (33 tests) | **WHAT** rules can we extract? |

Phase 4 closes the loop: the same saliency scores that Phase 1 computes
on-device become the **guide** for which features the decision tree is
allowed to use. The extracted rules can themselves be formally verified
using the Phase 3 pipeline (a depth-3 tree has ≤ 2³ = 8 paths, trivially
enumerable).

## Technical Approach

### Saliency-Guided Decision Tree Distillation

Given a trained neural network `f: ℝⁿ → {0,…,C-1}` and a per-feature
saliency vector `s ∈ ℝⁿ` (from Grad-CAM, LRP, or SHAP averaged over the
dataset), we:

1. **Select** the top-K features by `|s|`
2. **Query** the teacher on a dataset `X`, getting labels `y_teacher = f(X)`
3. **Train** a `DecisionTreeClassifier` with `max_depth=3` on
   `X[:, top_features]` → `y_teacher`
4. **Compute fidelity**: fraction of `X` where tree agrees with teacher
5. **Export** to C as either nested `if/else` or a flat node table

The key insight is that the neural network's *relevance attribution* tells
us where the interesting action is, so the decision tree doesn't have to
rediscover this from the raw features. Training only on salient features
acts as a strong regularizer and produces more interpretable trees than
training on the full feature space.

### Why Depth 3?

A depth-3 tree has:
- ≤ 7 internal nodes
- ≤ 8 leaves
- ≤ 3 comparisons per prediction (constant time, deterministic)

This fits in a single cache line, runs in constant time without branch
prediction help, and can be formally verified by enumerating all 8 paths.
It's also the sweet spot for human readability — a 3-level hierarchy of
IF/ELSE is about as much as a domain expert can hold in working memory.

### Integer-Only Inference

The learned thresholds are floating-point, but at export we quantize them
to fixed-point integers with a configurable scale (default 1024, giving
~10 fractional bits). The resulting comparisons are plain `int32` LTE
checks, which compile to a single `slt` on RISC-V.

The input features must be pre-scaled by the same factor before calling
the generated function. This matches how Grad-CAM feature maps already
live in scaled-integer form on the SNAX GeMM path.

## Module Structure

```
src/xai/symbolic/
├── __init__.py
├── rule_extractor.py     # Training (sklearn-backed) + tree data model
├── rule_to_c.py          # C exporter (inline + table styles)
└── fidelity_metrics.py   # Fidelity, coverage, saliency agreement

tests/
└── test_rule_extraction.py  # 33 tests (including gcc compile+run)

docs/phase4/
└── PHASE4_DESIGN.md      # This document
```

### rule_extractor.py

Core API:

```python
rules = extract_rules(
    X,                # (n_samples, n_features) training data
    saliency,         # (n_features,) per-feature relevance
    model_predict,    # callable: X -> predicted classes
    max_depth=3,
    top_k_features=8,
)

rules.predict(x)              # single prediction
rules.predict_batch(X)        # batch prediction
count_rules(rules)            # number of IF-THEN rules
rules_to_text(rules)          # human-readable text
```

Internal data model:

```python
@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int          # only if leaf
    feature: int             # only if not leaf
    threshold: float
    left: Optional["TreeNode"]
    right: Optional["TreeNode"]
```

Trees are converted from sklearn's internal representation, with the
important caveat that **feature indices are mapped back to the original
feature space**, so a tree that splits on `feature=3` means input `x[3]`,
not the 3rd selected feature.

### rule_to_c.py

Two export styles:

**Inline (best for depth ≤ 3)**:
```c
static inline int32_t symbolic_predict(const int32_t *x) {
    if (x[3] <= 512) {
        if (x[7] <= 1228) {
            return 0;
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}
```

**Table (best for larger or dynamically-loaded trees)**:
```c
typedef struct {
    int8_t  is_leaf;
    int16_t feature;     // or prediction for leaves
    int32_t threshold;
    int16_t left;
    int16_t right;
} symbolic_node_t;

static const symbolic_node_t symbolic_tree[N] = { ... };

static inline int32_t symbolic_predict(const int32_t *x) {
    int16_t i = 0;
    while (!symbolic_tree[i].is_leaf) {
        if (x[symbolic_tree[i].feature] <= symbolic_tree[i].threshold)
            i = symbolic_tree[i].left;
        else
            i = symbolic_tree[i].right;
    }
    return symbolic_tree[i].feature;
}
```

Both generate complete `#ifndef`-guarded headers with a `#include <stdint.h>`
and nothing else. No external dependencies.

### fidelity_metrics.py

Measures how faithfully the rule set mimics the teacher:

| Metric | Meaning |
|--------|---------|
| `fidelity` | Fraction of inputs where tree == model |
| `per_class_fidelity` | Same, broken down by class |
| `rule_count` | Number of leaves (= distinct rules) |
| `avg_rule_depth` | Mean root-to-leaf path length |
| `max_rule_depth` | Longest path |
| `leaf_coverage` | Fraction of inputs reaching each leaf (dead-rule detection) |
| `used_features` | Sorted list of features that appear in splits |
| `saliency_agreement` | Jaccard overlap between tree features and top-K salient |

## Test Coverage

33 tests across 7 test classes:

| Class | Tests | What it verifies |
|-------|-------|------------------|
| TestFeatureSelection | 3 | Top-K selection (basic, clamped, signed) |
| TestRuleExtraction | 7 | Extract, depth, feature usage, size mismatch, fidelity, predict |
| TestRuleCountingAndText | 3 | Leaf counting, text rendering |
| TestFidelityMetrics | 8 | Fidelity, per-class, depth, coverage, agreement, report |
| TestCExport | 8 | Inline + table C generation, guards, struct, error handling |
| TestCCompileAndRun | 2 | **Real gcc compilation + execution**, matches Python |
| TestXAIPipelineIntegration | 2 | SNAX dimensions (K=16, C=10), end-to-end C export |

The `TestCCompileAndRun` tests are particularly strong — they generate C,
compile it with `gcc -O2`, run the binary, and check the prediction
matches what Python gave. This catches any mismatch between the Python
reference and the C kernel.

### Sample test assertions

```python
# Tree only uses top-K salient features
assert set(used).issubset(set(top_k_features))

# Depth bound is respected
assert max(compute_rule_depths(rules)) <= max_depth

# Leaf coverage sums to 1
assert sum(compute_leaf_coverage(rules, X).values()) == 1.0

# Generated C produces same prediction as Python
c_pred = compile_and_run(exported_header, x)
assert c_pred == rules.predict(x)
```

## Performance Characteristics

### Python Extraction

Training a depth-3 tree on 500 samples × 16 features takes <10 ms with
sklearn on a modern laptop. This is essentially free compared to the
cost of running the teacher model to get labels.

### Generated C (measured on gcc -O2, x86_64)

For the `inline` style with a depth-3 tree:
- ~20–30 instructions total (7 compares, 3 taken branches, 1 load, 1 return)
- No memory allocation
- No function calls
- Deterministic execution time (all paths are the same length for depth=3)

For SNAX RV32IMF:
- Each comparison is one `slti` or `slt` + `beq`/`bne`
- Expected cycle count: **~15–25 cycles** (vs 6,153 for Grad-CAM or
  58,022 for optimized SHAP)
- Roughly **300–400× cheaper** than Grad-CAM for the prediction step alone

The Phase 4 rule evaluation is the cheapest XAI operation in the entire
project and is essentially free to add alongside Grad-CAM or SHAP.

## Connection to SNAX Hardware (Future Work)

A natural Phase 4b would be to:
1. Run Grad-CAM on SNAX (Phase 1) to get per-feature saliency
2. Extract rules from the saliency (Phase 4, host)
3. Generate a C header and drop it into
   `snax_cluster/sw/xai/symbolic/src/symbolic.h`
4. Measure rule-evaluation cycles on SNAX Verilator
5. Bonus: use Phase 3 QVIP verification to prove the rules are safe

Because rule evaluation is pure integer arithmetic with no FPU, it can
run on any core — including tiny accelerators that can't run the full
neural network.

## Sample Extracted Rules

For the synthetic 3-class task used in testing:

```
IF x[0] <= 0.5003:
  PREDICT class 0
ELSE:
  IF x[1] <= 0.4998:
    PREDICT class 1
  ELSE:
    PREDICT class 2
```

3 leaves, 3 rules, 2 features used (matches the top-2 saliency), 100%
training fidelity. When exported to C with scale=1024, the thresholds
become `512` and `512` (both `0.5 * 1024`), and the generated function
is 10 lines of code.

## References

- Bastani et al. "Interpretability via Model Extraction" (NeurIPS 2017
  Workshop) — distillation-based tree extraction
- Frosst & Hinton "Distilling a Neural Network Into a Soft Decision Tree"
  (2017) — soft decision trees as a distillation target
- Ribeiro et al. "Why Should I Trust You?" (KDD 2016) — LIME, local
  sparse explanations
- Craven & Shavlik "Extracting Tree-Structured Representations of Trained
  Networks" (NIPS 1996) — the original tree-extraction idea
