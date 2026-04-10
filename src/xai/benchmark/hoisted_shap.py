"""Backward-hoisted Gradient-SHAP (Phase 2b scaling strategy).

Naive Gradient SHAP runs ``N`` full forward + backward passes through
the entire network, giving overhead of ``N × inference_cost``.  For
larger models this quickly dominates runtime.

Phase 2b observed that for architectures ending in a **GAP + FC head**
(which covers ResNet-8, ToyAdmos, and pooled transformers), the
expensive backbone only needs to run **once** per input: the feature
maps are cached, and only the lightweight head is re-evaluated for
each of the ``N`` SHAP samples.  This module implements that
strategy as a host-side reference, using the same interface as
:mod:`src.xai.shap.gradient_shap_reference` but exposing the
backbone/head split explicitly.

The resulting attributions live in the *feature-map* space (not the
pixel space), matching what the SNAX kernel produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class HoistedShapResult:
    """Output of :func:`hoisted_gradient_shap`.

    Attributes:
        attributions:     SHAP attribution map over feature-map space,
                          shape matching the cached feature maps
                          (e.g. (H, W, K) for vision, (S, D) for NLP).
        backbone_calls:   number of times the backbone was executed
                          (always 1 for the hoisted strategy).
        head_calls:       number of head evaluations performed (= N).
        n_samples:        number of SHAP samples used.
        feature_map_shape: shape of the cached backbone feature maps.
    """

    attributions: np.ndarray
    backbone_calls: int
    head_calls: int
    n_samples: int
    feature_map_shape: Tuple[int, ...]


def hoisted_gradient_shap(
    backbone_fn: Callable[[np.ndarray], np.ndarray],
    head_fn: Callable[[np.ndarray], np.ndarray],
    head_grad_fn: Callable[[np.ndarray, int], np.ndarray],
    input_tensor: np.ndarray,
    target_class: int,
    n_samples: int = 16,
    baselines: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> HoistedShapResult:
    """Compute Gradient SHAP with backbone hoisted out of the sample loop.

    Args:
        backbone_fn:   maps the input ``(1, *input_shape)`` to feature
                       maps of shape matching the Grad-CAM hook layer.
                       Called **exactly once** by this routine.
        head_fn:       maps feature maps to logits.  Called ``n_samples``
                       times (once per SHAP sample).  Not strictly
                       required by the math, but calling it lets tests
                       verify that the head is what's being re-run.
        head_grad_fn:  maps (feature_maps, target_class) to the gradient
                       of the target logit w.r.t. the feature maps.
                       Called ``n_samples`` times.
        input_tensor:  input to the backbone, shape ``(1, *input_shape)``
        target_class:  class index to explain
        n_samples:     number of SHAP samples (default 16 to match the
                       Phase 2b embedded kernel)
        baselines:     optional array of feature-map baselines, shape
                       ``(n_samples, *feature_map_shape)``.  If omitted,
                       Gaussian baselines are drawn.
        seed:          RNG seed for reproducibility

    Returns:
        A :class:`HoistedShapResult` with the attribution map.
    """
    rng = np.random.default_rng(seed)

    # One expensive backbone pass.
    cached_features = backbone_fn(input_tensor)
    if cached_features.ndim == 0:
        raise ValueError("backbone_fn must return a tensor, got a scalar")
    feat_shape = cached_features.shape
    # Strip the batch dim for attribution accounting.
    if feat_shape[0] == 1:
        feat_shape_noB = feat_shape[1:]
    else:
        feat_shape_noB = feat_shape

    if baselines is None:
        baselines = rng.standard_normal((n_samples,) + feat_shape_noB).astype(
            np.float32
        ) * 0.1
    elif baselines.shape[0] != n_samples:
        n_samples = baselines.shape[0]

    attributions = np.zeros(feat_shape_noB, dtype=np.float32)
    head_calls = 0

    features_flat = cached_features[0] if cached_features.shape[0] == 1 else cached_features

    for i in range(n_samples):
        baseline = baselines[i]
        alpha = float(rng.uniform(0.0, 1.0))

        interp = baseline + alpha * (features_flat - baseline)

        # Run the head forward (kept for test observability)
        _ = head_fn(interp[np.newaxis, ...] if interp.ndim == len(feat_shape) - 1
                    else interp)
        head_calls += 1

        grad = head_grad_fn(interp, target_class)
        if grad.shape != features_flat.shape:
            raise ValueError(
                f"head_grad_fn returned shape {grad.shape}, "
                f"expected {features_flat.shape}"
            )

        diff = features_flat - baseline
        attributions += (diff * grad).astype(np.float32)

    attributions /= n_samples
    return HoistedShapResult(
        attributions=attributions,
        backbone_calls=1,
        head_calls=head_calls,
        n_samples=n_samples,
        feature_map_shape=feat_shape_noB,
    )
