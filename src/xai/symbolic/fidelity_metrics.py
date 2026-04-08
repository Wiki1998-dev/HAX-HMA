"""Fidelity metrics for extracted symbolic rules.

Measures how well an extracted decision tree mimics the original neural
network, as well as rule-set complexity and coverage.

Key metrics:
  - **Fidelity**: fraction of inputs where tree == model
  - **Per-class fidelity**: fidelity broken down by class
  - **Average rule length**: mean depth of leaves (shorter = simpler)
  - **Coverage**: fraction of inputs that reach each leaf
  - **Saliency agreement**: Jaccard overlap between tree's used features
    and top-K salient features

References:
    - Bastani et al. "Interpretability via Model Extraction"
    - Frosst & Hinton "Distilling a Neural Network Into a Soft Decision Tree"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List

from .rule_extractor import ExtractedRules, TreeNode


@dataclass
class FidelityReport:
    """Summary of how well a rule set matches the original model.

    Attributes:
        fidelity: overall agreement fraction
        per_class_fidelity: dict mapping class → agreement fraction
        rule_count: number of rules (= leaves)
        avg_rule_depth: mean path length from root to leaf
        max_rule_depth: maximum path length
        leaf_coverage: dict mapping leaf_id → fraction of samples reaching it
        used_features: sorted list of feature indices appearing in splits
        n_used_features: number of distinct features used in splits
    """
    fidelity: float
    per_class_fidelity: Dict[int, float]
    rule_count: int
    avg_rule_depth: float
    max_rule_depth: int
    leaf_coverage: Dict[int, float]
    used_features: List[int]
    n_used_features: int


def compute_fidelity(
    rules: ExtractedRules,
    X: np.ndarray,
    model_predict: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Compute overall fidelity: fraction of inputs where rules agree with model.

    Args:
        rules: extracted rule set
        X: (n_samples, n_features) test inputs
        model_predict: callable that returns model predictions for X

    Returns:
        Fidelity in [0, 1]
    """
    y_model = np.asarray(model_predict(X)).astype(np.int32).flatten()
    y_rules = rules.predict_batch(X)
    return float(np.mean(y_rules == y_model))


def compute_per_class_fidelity(
    rules: ExtractedRules,
    X: np.ndarray,
    model_predict: Callable[[np.ndarray], np.ndarray],
) -> Dict[int, float]:
    """Compute fidelity broken down by the model's predicted class.

    Useful for spotting classes where the tree under-performs.

    Args:
        rules: extracted rule set
        X: test inputs
        model_predict: teacher model predictor

    Returns:
        Dict mapping class index → fidelity on samples of that class
    """
    y_model = np.asarray(model_predict(X)).astype(np.int32).flatten()
    y_rules = rules.predict_batch(X)

    per_class: Dict[int, float] = {}
    for c in np.unique(y_model):
        mask = y_model == c
        if mask.any():
            per_class[int(c)] = float(np.mean(y_rules[mask] == y_model[mask]))
    return per_class


def compute_rule_depths(rules: ExtractedRules) -> List[int]:
    """Return the depth of every leaf in the tree.

    Args:
        rules: extracted rule set

    Returns:
        List of leaf depths (root is depth 0)
    """
    depths: List[int] = []

    def _walk(node: TreeNode, d: int) -> None:
        if node.is_leaf:
            depths.append(d)
        else:
            _walk(node.left, d + 1)
            _walk(node.right, d + 1)

    _walk(rules.root, 0)
    return depths


def compute_leaf_coverage(
    rules: ExtractedRules,
    X: np.ndarray,
) -> Dict[int, float]:
    """Compute the fraction of inputs that land in each leaf.

    A rule with near-zero coverage is dead code — it doesn't apply to
    any real inputs and can usually be pruned.

    Args:
        rules: extracted rule set
        X: (n_samples, n_features) inputs

    Returns:
        Dict mapping leaf node_id → coverage fraction
    """
    counts: Dict[int, int] = {}
    total = len(X)

    for x in X:
        node = rules.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        counts[node.node_id] = counts.get(node.node_id, 0) + 1

    return {leaf_id: count / total for leaf_id, count in counts.items()}


def compute_used_features(rules: ExtractedRules) -> List[int]:
    """Return the sorted list of feature indices that appear in splits.

    Args:
        rules: extracted rule set

    Returns:
        Sorted list of feature indices
    """
    used: set = set()

    def _walk(node: TreeNode) -> None:
        if not node.is_leaf:
            used.add(node.feature)
            _walk(node.left)
            _walk(node.right)

    _walk(rules.root)
    return sorted(used)


def saliency_agreement(
    rules: ExtractedRules,
    saliency: np.ndarray,
    top_k: int,
) -> float:
    """Compute Jaccard overlap between tree features and top-k salient features.

    This measures whether the extracted tree is using the same features
    that the XAI saliency map highlighted. High agreement means the
    symbolic rules faithfully reflect the model's learned feature
    importance.

    Args:
        rules: extracted rule set
        saliency: (n_features,) saliency scores
        top_k: number of top-saliency features to consider

    Returns:
        Jaccard similarity in [0, 1]
    """
    from .rule_extractor import select_salient_features

    top_salient = set(select_salient_features(saliency, top_k).tolist())
    used_features = set(compute_used_features(rules))

    if not top_salient and not used_features:
        return 1.0
    if not top_salient or not used_features:
        return 0.0

    intersection = top_salient & used_features
    union = top_salient | used_features
    return len(intersection) / len(union)


def fidelity_report(
    rules: ExtractedRules,
    X: np.ndarray,
    model_predict: Callable[[np.ndarray], np.ndarray],
) -> FidelityReport:
    """Generate a complete fidelity report for an extracted rule set.

    Args:
        rules: extracted rule set
        X: test inputs
        model_predict: teacher model predictor

    Returns:
        FidelityReport with all metrics populated
    """
    depths = compute_rule_depths(rules)
    used_features = compute_used_features(rules)

    return FidelityReport(
        fidelity=compute_fidelity(rules, X, model_predict),
        per_class_fidelity=compute_per_class_fidelity(rules, X, model_predict),
        rule_count=rules.n_leaves,
        avg_rule_depth=float(np.mean(depths)) if depths else 0.0,
        max_rule_depth=int(max(depths)) if depths else 0,
        leaf_coverage=compute_leaf_coverage(rules, X),
        used_features=used_features,
        n_used_features=len(used_features),
    )
