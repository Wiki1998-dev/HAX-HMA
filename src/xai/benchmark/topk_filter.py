"""Top-K saliency filter for neuro-symbolic rule distillation (Phase 5 §4).

Training a decision tree over the full input space of large models
(e.g. MobileBERT's ``seq_len × hidden`` = 32×128 = 4096 features)
produces trees that are either unauditably deep or that pick
non-obvious splits.  The Phase 4 rule extractor already supports a
``top_k_features`` parameter; this module provides a thin wrapper
that selects features from SHAP / Grad-CAM saliency and prepares
them for :func:`src.xai.symbolic.rule_extractor.extract_rules`.

Keeping the tree to ``k ≤ 16`` salient features preserves the
``≈47 cycles`` RV32IMF run-cost guaranteed by Phase 4, regardless of
how many features the parent model has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..symbolic.rule_extractor import (
    ExtractedRules,
    extract_rules,
    select_salient_features,
)


@dataclass
class TopKFilterResult:
    """Output of :func:`topk_saliency_filter`.

    Attributes:
        rules:             extracted :class:`ExtractedRules` (depth ≤
                           ``max_depth``, using only the top-K features).
        selected_features: indices of the top-K salient features used.
        dropped_fraction:  fraction of features discarded by the filter.
    """

    rules: ExtractedRules
    selected_features: np.ndarray
    dropped_fraction: float


def topk_saliency_filter(
    X: np.ndarray,
    saliency: np.ndarray,
    model_predict: Callable[[np.ndarray], np.ndarray],
    top_k: int = 16,
    max_depth: int = 3,
    random_state: int = 42,
) -> TopKFilterResult:
    """Distill a neural model into a rule set using Top-K salient features.

    The distillation uses :func:`extract_rules` under the hood but
    enforces the Phase 5 policy: only the top-``k`` most salient
    features are ever shown to the decision tree.  This guarantees
    the extracted tree's embedded run-cost stays bounded regardless
    of the parent model's input dimensionality.

    Args:
        X:             sample dataset, shape ``(n_samples, n_features)``
        saliency:      per-feature saliency scores, shape
                       ``(n_features,)`` — typically averaged SHAP
                       values from :func:`hoisted_gradient_shap`
        model_predict: teacher model predictor (returns class indices)
        top_k:         number of features to retain (default 16)
        max_depth:     decision tree depth bound (default 3)
        random_state:  RNG seed for reproducibility

    Returns:
        :class:`TopKFilterResult` with the extracted rules and filter
        statistics.

    Raises:
        ValueError: if ``top_k`` is not in ``[1, n_features]``.
    """
    saliency = np.asarray(saliency).flatten()
    n_features = saliency.size
    if not 1 <= top_k <= n_features:
        raise ValueError(
            f"top_k ({top_k}) must be in [1, n_features={n_features}]"
        )

    rules = extract_rules(
        X=X,
        saliency=saliency,
        model_predict=model_predict,
        max_depth=max_depth,
        top_k_features=top_k,
        random_state=random_state,
    )
    dropped = 1.0 - top_k / n_features
    return TopKFilterResult(
        rules=rules,
        selected_features=select_salient_features(saliency, top_k),
        dropped_fraction=float(dropped),
    )
