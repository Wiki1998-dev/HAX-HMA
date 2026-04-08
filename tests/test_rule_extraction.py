"""Tests for Phase 4 neuro-symbolic rule extraction.

Covers:
  - Rule extraction from synthetic saliency + teacher model
  - Tree structure (depth, leaf count, feature usage)
  - Fidelity metrics (overall, per-class, coverage)
  - Saliency agreement
  - C code export (inline and table styles)
  - End-to-end: extract rules from a Grad-CAM-style saliency map,
    export to C, verify the generated code compiles and runs correctly.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.xai.symbolic.rule_extractor import (
    ExtractedRules,
    TreeNode,
    count_rules,
    extract_rules,
    rules_to_text,
    select_salient_features,
)
from src.xai.symbolic.rule_to_c import (
    export_to_c_header,
    export_to_c_inline,
    export_to_c_table,
)
from src.xai.symbolic.fidelity_metrics import (
    compute_fidelity,
    compute_leaf_coverage,
    compute_per_class_fidelity,
    compute_rule_depths,
    compute_used_features,
    fidelity_report,
    saliency_agreement,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_teacher():
    """A simple 3-class teacher model based on thresholds.

    Class assignment:
        x[0] > 0.5 and x[1] > 0.5 → class 2
        x[0] > 0.5 and x[1] <= 0.5 → class 1
        x[0] <= 0.5 → class 0

    Features 0 and 1 are fully informative; features 2-9 are noise.
    """
    def predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        y = np.zeros(X.shape[0], dtype=np.int32)
        class1_mask = (X[:, 0] > 0.5) & (X[:, 1] <= 0.5)
        class2_mask = (X[:, 0] > 0.5) & (X[:, 1] > 0.5)
        y[class1_mask] = 1
        y[class2_mask] = 2
        return y
    return predict


@pytest.fixture
def simple_data():
    """Synthetic data: 200 samples, 10 features."""
    rng = np.random.RandomState(42)
    X = rng.rand(200, 10).astype(np.float64)
    return X


@pytest.fixture
def simple_saliency():
    """Saliency concentrated on features 0 and 1 (the informative ones)."""
    sal = np.zeros(10)
    sal[0] = 0.8
    sal[1] = 0.6
    sal[2:] = np.array([0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
    return sal


@pytest.fixture
def extracted(simple_data, simple_saliency, simple_teacher):
    """Pre-extracted rules for reuse across tests."""
    return extract_rules(
        simple_data, simple_saliency, simple_teacher,
        max_depth=3, top_k_features=4,
    )


# ============================================================================
# Test feature selection
# ============================================================================

class TestFeatureSelection:
    """Tests for saliency-based feature selection."""

    def test_top_k_basic(self):
        """Top-k selects the highest-saliency features."""
        sal = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        top = select_salient_features(sal, top_k=3)
        assert list(top) == [3, 1, 2]

    def test_top_k_clamped(self):
        """Top-k is clamped to the number of features."""
        sal = np.array([1.0, 2.0, 3.0])
        top = select_salient_features(sal, top_k=10)
        assert len(top) == 3

    def test_top_k_uses_absolute_value(self):
        """Negative saliencies are ranked by magnitude."""
        sal = np.array([-0.9, 0.2, 0.3, -0.5])
        top = select_salient_features(sal, top_k=2)
        assert set(top.tolist()) == {0, 3}


# ============================================================================
# Test rule extraction
# ============================================================================

class TestRuleExtraction:
    """Tests for the main extract_rules() function."""

    def test_returns_extracted_rules(self, extracted):
        """extract_rules returns an ExtractedRules object."""
        assert isinstance(extracted, ExtractedRules)
        assert isinstance(extracted.root, TreeNode)

    def test_respects_max_depth(self, simple_data, simple_saliency, simple_teacher):
        """Tree depth does not exceed max_depth."""
        for depth in [1, 2, 3, 5]:
            rules = extract_rules(
                simple_data, simple_saliency, simple_teacher,
                max_depth=depth, top_k_features=4,
            )
            depths = compute_rule_depths(rules)
            assert max(depths) <= depth

    def test_uses_top_features(self, simple_data, simple_saliency, simple_teacher):
        """Extracted tree only uses features from the top-k salient set."""
        rules = extract_rules(
            simple_data, simple_saliency, simple_teacher,
            max_depth=3, top_k_features=4,
        )
        allowed = set(select_salient_features(simple_saliency, 4).tolist())
        used = set(compute_used_features(rules))
        assert used.issubset(allowed)

    def test_feature_size_mismatch_raises(
        self, simple_data, simple_teacher,
    ):
        """Mismatched saliency size raises ValueError."""
        bad_saliency = np.array([0.5, 0.5])  # wrong size
        with pytest.raises(ValueError):
            extract_rules(
                simple_data, bad_saliency, simple_teacher,
                max_depth=3, top_k_features=2,
            )

    def test_high_fidelity_on_separable_task(
        self, simple_data, simple_saliency, simple_teacher,
    ):
        """For a linearly-separable task, depth-3 tree should have high fidelity."""
        rules = extract_rules(
            simple_data, simple_saliency, simple_teacher,
            max_depth=3, top_k_features=4,
        )
        # With the correct top features available, fidelity should be perfect
        assert rules.train_fidelity >= 0.99

    def test_predict_single(self, extracted, simple_data):
        """ExtractedRules.predict works on a single input."""
        pred = extracted.predict(simple_data[0])
        assert isinstance(pred, int)
        assert 0 <= pred <= 2

    def test_predict_batch(self, extracted, simple_data):
        """ExtractedRules.predict_batch has correct shape."""
        preds = extracted.predict_batch(simple_data)
        assert preds.shape == (simple_data.shape[0],)
        assert preds.dtype == np.int32


# ============================================================================
# Test rule counting and text
# ============================================================================

class TestRuleCountingAndText:
    """Tests for rule counting and text rendering."""

    def test_count_matches_leaves(self, extracted):
        """count_rules returns the number of leaves."""
        assert count_rules(extracted) == extracted.n_leaves

    def test_text_is_multiline(self, extracted):
        """rules_to_text returns a non-empty multi-line string."""
        text = rules_to_text(extracted)
        assert len(text) > 0
        assert "\n" in text

    def test_text_mentions_predict(self, extracted):
        """Text representation contains 'PREDICT' lines (one per leaf)."""
        text = rules_to_text(extracted)
        predict_count = text.count("PREDICT")
        assert predict_count == extracted.n_leaves


# ============================================================================
# Test fidelity metrics
# ============================================================================

class TestFidelityMetrics:
    """Tests for fidelity_metrics module."""

    def test_compute_fidelity_range(
        self, extracted, simple_data, simple_teacher,
    ):
        """Fidelity is in [0, 1]."""
        f = compute_fidelity(extracted, simple_data, simple_teacher)
        assert 0.0 <= f <= 1.0

    def test_compute_fidelity_identical_on_training(
        self, extracted, simple_data, simple_teacher,
    ):
        """On training data, fidelity matches extracted.train_fidelity."""
        f = compute_fidelity(extracted, simple_data, simple_teacher)
        assert abs(f - extracted.train_fidelity) < 1e-10

    def test_per_class_fidelity_keys(
        self, extracted, simple_data, simple_teacher,
    ):
        """per_class_fidelity has one entry per class present in predictions."""
        per_class = compute_per_class_fidelity(
            extracted, simple_data, simple_teacher,
        )
        y = simple_teacher(simple_data)
        for c in np.unique(y):
            assert int(c) in per_class
            assert 0.0 <= per_class[int(c)] <= 1.0

    def test_rule_depths_bounded(self, extracted):
        """All leaf depths are within max_depth."""
        depths = compute_rule_depths(extracted)
        assert max(depths) <= extracted.max_depth
        assert min(depths) >= 1

    def test_leaf_coverage_sums_to_one(self, extracted, simple_data):
        """Leaf coverage sums to 1.0 over all reachable leaves."""
        coverage = compute_leaf_coverage(extracted, simple_data)
        total = sum(coverage.values())
        assert abs(total - 1.0) < 1e-10

    def test_used_features_subset(self, extracted):
        """used_features is a subset of the selected top-k features."""
        used = compute_used_features(extracted)
        allowed = set(extracted.feature_indices.tolist())
        assert set(used).issubset(allowed)

    def test_saliency_agreement_perfect(
        self, simple_data, simple_saliency, simple_teacher,
    ):
        """For a task where tree uses the top features, agreement is high."""
        rules = extract_rules(
            simple_data, simple_saliency, simple_teacher,
            max_depth=3, top_k_features=2,
        )
        agreement = saliency_agreement(rules, simple_saliency, top_k=2)
        # Tree should use both top-2 features
        assert agreement >= 0.5

    def test_fidelity_report_fields(
        self, extracted, simple_data, simple_teacher,
    ):
        """fidelity_report returns all expected fields."""
        report = fidelity_report(extracted, simple_data, simple_teacher)
        assert 0.0 <= report.fidelity <= 1.0
        assert report.rule_count == extracted.n_leaves
        assert report.avg_rule_depth > 0
        assert report.max_rule_depth <= extracted.max_depth
        assert isinstance(report.used_features, list)
        assert report.n_used_features == len(report.used_features)


# ============================================================================
# Test C code export
# ============================================================================

class TestCExport:
    """Tests for C code generation."""

    def test_inline_export_produces_function(self, extracted):
        """Inline export contains a C function definition."""
        code = export_to_c_inline(extracted)
        assert "static inline int32_t symbolic_predict" in code
        assert "return" in code
        # Must have one 'return N;' per leaf
        returns = re.findall(r"return\s+\d+\s*;", code)
        assert len(returns) == extracted.n_leaves

    def test_inline_export_has_comparisons(self, extracted):
        """Inline export contains feature comparisons."""
        code = export_to_c_inline(extracted)
        # Must have comparisons like `if (x[N] <= M)`
        assert re.search(r"if\s*\(x\[\d+\]\s*<=\s*-?\d+\)", code)

    def test_table_export_node_count(self, extracted):
        """Table export has the right number of nodes in the array."""
        code = export_to_c_table(extracted)
        # Count entries in the table by looking for lines starting with "    { "
        entries = re.findall(r"^\s*\{ [01],", code, re.MULTILINE)
        assert len(entries) == extracted.n_nodes

    def test_table_export_has_struct(self, extracted):
        """Table export defines the struct once."""
        code = export_to_c_table(extracted)
        assert "typedef struct" in code
        assert "symbolic_node_t" in code

    def test_header_has_guard(self, extracted):
        """Exported header has include guards."""
        code = export_to_c_header(extracted, guard_name="TEST_GUARD_H")
        assert "#ifndef TEST_GUARD_H" in code
        assert "#define TEST_GUARD_H" in code
        assert "#endif" in code
        assert "#include <stdint.h>" in code

    def test_header_inline_style(self, extracted):
        """Header with style='inline' uses nested if/else."""
        code = export_to_c_header(extracted, style="inline")
        assert "static inline int32_t" in code
        # No struct definition for inline style
        assert "symbolic_node_t" not in code

    def test_header_table_style(self, extracted):
        """Header with style='table' uses flat node table."""
        code = export_to_c_header(extracted, style="table")
        assert "symbolic_node_t" in code
        assert "static const symbolic_node_t" in code

    def test_header_unknown_style_raises(self, extracted):
        """Unknown style raises ValueError."""
        with pytest.raises(ValueError):
            export_to_c_header(extracted, style="martian")


# ============================================================================
# Integration: end-to-end C compile & run
# ============================================================================

@pytest.mark.integration
class TestCCompileAndRun:
    """Compile the exported C code with gcc and check the output matches Python."""

    def _compile_and_run(self, header_code: str, test_input: np.ndarray) -> int:
        """Compile header with a driver and run on test_input.

        Returns the integer prediction.
        """
        n = len(test_input)
        scale = 1024
        scaled_input = [int(round(float(v) * scale)) for v in test_input]
        values = ", ".join(str(v) for v in scaled_input)

        driver = f"""
#include <stdio.h>
#include "symbolic_rules.h"

int main(void) {{
    int32_t x[{n}] = {{ {values} }};
    int32_t y = symbolic_predict(x);
    printf("%d\\n", y);
    return 0;
}}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "symbolic_rules.h").write_text(header_code)
            (tmp / "driver.c").write_text(driver)
            binary = tmp / "driver"
            compile_res = subprocess.run(
                ["gcc", "-O2", "-o", str(binary),
                 str(tmp / "driver.c"), f"-I{tmp}"],
                capture_output=True, text=True,
            )
            assert compile_res.returncode == 0, (
                f"gcc failed:\nSTDERR: {compile_res.stderr}\nCODE:\n{header_code}"
            )
            run_res = subprocess.run(
                [str(binary)], capture_output=True, text=True,
            )
            assert run_res.returncode == 0, f"Runtime error: {run_res.stderr}"
            return int(run_res.stdout.strip())

    def test_inline_compiles_and_runs(
        self, extracted, simple_data,
    ):
        """Inline-style C code compiles and matches Python prediction."""
        if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
            pytest.skip("gcc not available")

        header = export_to_c_header(extracted, style="inline", scale=1024)
        for i in range(10):
            py_pred = extracted.predict(simple_data[i])
            c_pred = self._compile_and_run(header, simple_data[i])
            assert c_pred == py_pred, (
                f"Sample {i}: python={py_pred} c={c_pred}"
            )

    def test_table_compiles_and_runs(
        self, extracted, simple_data,
    ):
        """Table-style C code compiles and matches Python prediction."""
        if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
            pytest.skip("gcc not available")

        header = export_to_c_header(extracted, style="table", scale=1024)
        for i in range(10):
            py_pred = extracted.predict(simple_data[i])
            c_pred = self._compile_and_run(header, simple_data[i])
            assert c_pred == py_pred


# ============================================================================
# Integration with XAI pipeline
# ============================================================================

class TestXAIPipelineIntegration:
    """Connect Phase 1 (saliency) → Phase 4 (rules) on SNAX dimensions."""

    def test_gradcam_style_extraction(self):
        """Extract rules using Grad-CAM-style saliency on K=16, C=10 model."""
        rng = np.random.RandomState(42)
        K, C = 16, 10
        n_samples = 500

        # Synthetic "fc layer" teacher
        W = rng.randn(C, K).astype(np.float64)
        b = rng.randn(C).astype(np.float64)

        def teacher(X):
            logits = X @ W.T + b
            return np.argmax(logits, axis=1).astype(np.int32)

        # Grad-CAM analogue: average absolute weight per input feature
        saliency = np.abs(W).mean(axis=0)

        # Training data
        X = rng.randn(n_samples, K).astype(np.float64)

        rules = extract_rules(
            X, saliency, teacher,
            max_depth=3, top_k_features=8,
        )

        # Sanity: tree uses only top-8 salient features
        top_8 = set(select_salient_features(saliency, 8).tolist())
        used = set(compute_used_features(rules))
        assert used.issubset(top_8)

        # Sanity: tree has reasonable fidelity on training data
        report = fidelity_report(rules, X, teacher)
        assert report.fidelity > 0.2  # Shallow tree on 10-class problem
        assert report.rule_count <= 2 ** rules.max_depth
        assert report.n_used_features <= 8

    def test_end_to_end_c_export(self):
        """Full pipeline: saliency → rules → C header with SNAX dimensions."""
        rng = np.random.RandomState(0)
        K = 16
        X = rng.randn(300, K).astype(np.float64)

        # Simple 3-class teacher
        def teacher(X):
            return np.argmax(np.stack([
                X[:, 0] + X[:, 1],
                X[:, 2] - X[:, 3],
                X[:, 4] * 2,
            ], axis=1), axis=1).astype(np.int32)

        saliency = np.zeros(K)
        saliency[:5] = [1.0, 0.9, 0.8, 0.7, 0.6]

        rules = extract_rules(X, saliency, teacher, max_depth=3, top_k_features=5)
        header = export_to_c_header(rules, style="inline")

        assert "symbolic_predict" in header
        assert "#ifndef" in header
        # Tree should use some subset of the top-5 salient features
        used = compute_used_features(rules)
        assert all(f < 5 for f in used)
