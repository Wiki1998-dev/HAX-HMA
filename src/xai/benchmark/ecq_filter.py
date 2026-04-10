"""ECQx (Explainability-Driven Quantization) weight filtering.

Phase 5, Method 3 uses ECQx — the idea that per-weight quantization
bit-width should be informed by the *explanation* of what a weight
actually contributes to the decision.  Weights that Grad-CAM/SHAP
identify as important to the model's output get *more* bits; the
rest can be aggressively quantized (or clustered to fewer levels).

This module implements the host-side pre-processing step: given a
weight matrix and a saliency map over its outputs, compute a boolean
mask of "critical" weights and a per-element bit-width policy.  The
QVIP verifier then focuses ILP work on the critical slice only.

References:
    - ECQ×: Guillard et al., "Explanation-guided Quantization",
      arXiv:2109.04236
    - QVIP (Zhang et al., ASE'22): tooling we plug the mask into
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def ecqx_weight_mask(
    weights: np.ndarray,
    output_saliency: np.ndarray,
    critical_fraction: float = 0.25,
) -> np.ndarray:
    """Select the *critical* slice of a weight matrix from saliency.

    Args:
        weights:          weight matrix, shape ``(n_out, n_in)``.
        output_saliency:  per-output saliency scores, shape ``(n_out,)``.
                          Typically Grad-CAM or SHAP averaged over a
                          held-out set.
        critical_fraction: fraction of outputs to mark as critical
                          (default 0.25 = top 25 %).

    Returns:
        Boolean mask with the same shape as ``weights``.  Entries are
        ``True`` for weights attached to the top-K most-salient output
        neurons — these must stay at full precision.

    Raises:
        ValueError: if shapes don't match or ``critical_fraction`` is
            out of range.
    """
    weights = np.asarray(weights)
    saliency = np.asarray(output_saliency).flatten()

    if not 0.0 < critical_fraction <= 1.0:
        raise ValueError(
            f"critical_fraction must be in (0, 1], got {critical_fraction}"
        )
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape {weights.shape}")
    if saliency.size != weights.shape[0]:
        raise ValueError(
            f"saliency size ({saliency.size}) must match "
            f"weights.shape[0] ({weights.shape[0]})"
        )

    k = max(1, int(round(critical_fraction * saliency.size)))
    top_outputs = np.argsort(-np.abs(saliency))[:k]

    mask = np.zeros_like(weights, dtype=bool)
    mask[top_outputs, :] = True
    return mask


def ecqx_bitwidth_policy(
    weights: np.ndarray,
    output_saliency: np.ndarray,
    critical_bits: int = 8,
    noncritical_bits: int = 4,
    critical_fraction: float = 0.25,
) -> np.ndarray:
    """Produce a per-weight quantization bit-width policy.

    Args:
        weights:          weight matrix, shape ``(n_out, n_in)``.
        output_saliency:  per-output saliency scores.
        critical_bits:    bit-width to assign to critical weights.
        noncritical_bits: bit-width to assign to the rest.
        critical_fraction: fraction of outputs flagged as critical.

    Returns:
        Integer array with the same shape as ``weights``, containing
        the bit-width to use for each element.
    """
    mask = ecqx_weight_mask(weights, output_saliency, critical_fraction)
    policy = np.where(mask, critical_bits, noncritical_bits).astype(np.int32)
    return policy
