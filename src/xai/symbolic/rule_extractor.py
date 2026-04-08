"""Saliency-guided decision tree extraction for neuro-symbolic XAI.

Extracts a shallow decision tree (depth ≤ 3) from a black-box neural network
using XAI saliency maps to guide feature selection. The result is a small,
human-readable rule set that can run on embedded hardware (RISC-V) with
integer comparisons only.

The approach combines two ideas:
  1. **Saliency-weighted feature selection**: Use Grad-CAM/LRP relevance
     scores to identify which input features the model relies on.
  2. **Distillation via decision tree**: Train a depth-bounded decision
     tree to mimic the neural network's predictions on a sample dataset.

The resulting tree:
  - Uses only the top-K most-salient features (sparse)
  - Has bounded depth (auditable, fast to evaluate)
  - Can be exported to C for embedded inference (no FP required)

References:
    - Ribeiro et al. "Why Should I Trust You?" (LIME): local explanations
    - Frosst & Hinton "Distilling a Neural Network Into a Soft Decision Tree"
    - Bastani et al. "Interpretability via Model Extraction"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from sklearn.tree import DecisionTreeClassifier


@dataclass
class TreeNode:
    """A single decision tree node (used for our exported representation).

    A node is either a leaf (predicted class) or an internal split
    (feature index + threshold + left/right children).

    Attributes:
        is_leaf: True if this is a leaf node
        prediction: predicted class (only valid if is_leaf)
        feature: input feature index for the split (only if not leaf)
        threshold: split threshold; goes left if x[feature] <= threshold
        left: left child (taken if x[feature] <= threshold)
        right: right child (taken if x[feature] > threshold)
        node_id: unique integer id for serialization
    """
    is_leaf: bool
    prediction: int = -1
    feature: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    node_id: int = -1


@dataclass
class ExtractedRules:
    """A complete extracted rule set.

    Attributes:
        root: root TreeNode of the decision tree
        feature_indices: original feature indices used (after saliency masking)
        n_features_orig: total number of input features (before masking)
        max_depth: depth bound used during training
        n_nodes: total number of nodes in the tree
        n_leaves: number of leaf nodes (= number of distinct rules)
        train_fidelity: fraction of training samples where tree agrees with model
    """
    root: TreeNode
    feature_indices: np.ndarray
    n_features_orig: int
    max_depth: int
    n_nodes: int
    n_leaves: int
    train_fidelity: float = 0.0

    def predict(self, x: np.ndarray) -> int:
        """Predict class for a single input by walking the tree.

        Args:
            x: input feature vector (full original feature space)

        Returns:
            Predicted class index
        """
        node = self.root
        while not node.is_leaf:
            # The split feature is in the *original* feature space
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for a batch of inputs.

        Args:
            X: (n_samples, n_features) input matrix

        Returns:
            (n_samples,) array of predictions
        """
        return np.array([self.predict(x) for x in X], dtype=np.int32)


def select_salient_features(
    saliency: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Select the top-k most salient features.

    Args:
        saliency: (n_features,) saliency scores (Grad-CAM, LRP, etc.)
        top_k: number of features to keep

    Returns:
        Array of feature indices, sorted by descending saliency
    """
    saliency = np.asarray(saliency).flatten()
    abs_sal = np.abs(saliency)
    top_k = min(top_k, saliency.size)
    return np.argsort(-abs_sal)[:top_k]


def _convert_sklearn_tree(
    sklearn_tree: DecisionTreeClassifier,
    feature_indices: np.ndarray,
) -> Tuple[TreeNode, int, int]:
    """Convert a fitted sklearn DecisionTreeClassifier to our TreeNode format.

    Maps the (compact) sklearn feature indices back to the original
    feature space using `feature_indices`.

    Args:
        sklearn_tree: fitted sklearn classifier
        feature_indices: mapping from compact (training) features to
                         original feature indices

    Returns:
        Tuple of (root TreeNode, total nodes, leaf count)
    """
    tree = sklearn_tree.tree_
    n_nodes = tree.node_count
    classes = sklearn_tree.classes_

    # Build node objects
    nodes: List[TreeNode] = []
    for i in range(n_nodes):
        is_leaf = tree.children_left[i] == -1
        if is_leaf:
            class_idx = int(np.argmax(tree.value[i]))
            prediction = int(classes[class_idx])
            nodes.append(TreeNode(
                is_leaf=True,
                prediction=prediction,
                node_id=i,
            ))
        else:
            compact_feature = int(tree.feature[i])
            original_feature = int(feature_indices[compact_feature])
            nodes.append(TreeNode(
                is_leaf=False,
                feature=original_feature,
                threshold=float(tree.threshold[i]),
                node_id=i,
            ))

    # Wire up children
    for i in range(n_nodes):
        if not nodes[i].is_leaf:
            nodes[i].left = nodes[tree.children_left[i]]
            nodes[i].right = nodes[tree.children_right[i]]

    n_leaves = sum(1 for n in nodes if n.is_leaf)
    return nodes[0], n_nodes, n_leaves


def extract_rules(
    X: np.ndarray,
    saliency: np.ndarray,
    model_predict: Callable[[np.ndarray], np.ndarray],
    max_depth: int = 3,
    top_k_features: int = 8,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> ExtractedRules:
    """Extract a shallow decision tree from a neural network using saliency.

    The tree is trained to mimic the model's predictions, but only using
    the top-k most-salient features. This produces a sparse, auditable
    rule set that captures the model's decision logic.

    Args:
        X: (n_samples, n_features) sample input data
        saliency: (n_features,) per-feature saliency scores (e.g., from
                  Grad-CAM averaged over the dataset)
        model_predict: callable that takes X and returns predicted class
                       indices (n_samples,)
        max_depth: maximum decision tree depth (default 3 for embedded)
        top_k_features: number of salient features to use
        min_samples_leaf: sklearn min_samples_leaf parameter
        random_state: random seed for reproducibility

    Returns:
        ExtractedRules containing the trained tree and metadata
    """
    X = np.asarray(X, dtype=np.float64)
    saliency = np.asarray(saliency, dtype=np.float64).flatten()
    n_samples, n_features = X.shape

    if saliency.size != n_features:
        raise ValueError(
            f"saliency size ({saliency.size}) must match "
            f"X.shape[1] ({n_features})"
        )

    # Select salient features
    feature_indices = select_salient_features(saliency, top_k_features)
    X_selected = X[:, feature_indices]

    # Get teacher (model) predictions
    y_teacher = np.asarray(model_predict(X)).astype(np.int32).flatten()

    # Train shallow decision tree to mimic the model
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X_selected, y_teacher)

    # Convert to our format
    root, n_nodes, n_leaves = _convert_sklearn_tree(tree, feature_indices)

    # Compute training fidelity
    y_tree = tree.predict(X_selected)
    train_fidelity = float(np.mean(y_tree == y_teacher))

    return ExtractedRules(
        root=root,
        feature_indices=feature_indices,
        n_features_orig=n_features,
        max_depth=max_depth,
        n_nodes=n_nodes,
        n_leaves=n_leaves,
        train_fidelity=train_fidelity,
    )


def rules_to_text(rules: ExtractedRules) -> str:
    """Convert a decision tree to human-readable IF-THEN text.

    Produces output like:
        IF x[3] <= 0.5:
          IF x[7] <= 1.2:
            PREDICT class 0
          ELSE:
            PREDICT class 2
        ELSE:
          PREDICT class 1

    Args:
        rules: ExtractedRules from extract_rules()

    Returns:
        Multi-line text representation
    """
    lines: List[str] = []

    def _walk(node: TreeNode, depth: int) -> None:
        indent = "  " * depth
        if node.is_leaf:
            lines.append(f"{indent}PREDICT class {node.prediction}")
        else:
            lines.append(f"{indent}IF x[{node.feature}] <= {node.threshold:.4f}:")
            _walk(node.left, depth + 1)
            lines.append(f"{indent}ELSE:")
            _walk(node.right, depth + 1)

    _walk(rules.root, 0)
    return "\n".join(lines)


def count_rules(rules: ExtractedRules) -> int:
    """Count the number of distinct IF-THEN rules (= number of leaves).

    Each leaf corresponds to one root-to-leaf path, which is one rule.

    Args:
        rules: ExtractedRules

    Returns:
        Number of leaf paths
    """
    return rules.n_leaves
