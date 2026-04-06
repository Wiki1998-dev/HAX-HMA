"""
Gradient SHAP reference implementation (host-side, NumPy).

Ground truth for verifying the embedded C implementation on SNAX.
Based on: Lundberg & Lee "A Unified Approach to Interpreting Model Predictions"
and hardware mapping from: Pan & Mishra arXiv 2305.04887

Gradient SHAP approximates Shapley values by averaging gradient × input
differences over random baseline samples:

    phi_i = E_{x', alpha}[ (x_i - x'_i) * (d f(x_interp) / d x_i) ]

where x_interp = x' + alpha * (x - x'), alpha ~ U(0, 1).

Usage:
    python -m src.xai.shap.gradient_shap_reference
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def gradient_shap(
    forward_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    input_tensor: np.ndarray,
    target_class: int,
    n_samples: int = 16,
    baselines: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Compute Gradient SHAP attribution map.

    For each of n_samples random baselines, interpolates between baseline
    and input, computes the gradient at the interpolated point, and
    multiplies by (input - baseline). Averages over all samples.

    Args:
        forward_fn:     Function taking input (1, H, W, C) returning
                        (logits (1, num_classes), feature_maps).
        input_tensor:   Input, shape (1, H, W, C), float32.
        target_class:   Class index to explain.
        n_samples:      Number of baseline samples (default 16).
        baselines:      Optional (n_samples, H, W, C) baselines. If None,
                        random Gaussian baselines are generated.
        seed:           Random seed for reproducibility.

    Returns:
        attributions:   SHAP values, shape (H, W, C), float32.
                        Positive = supports target class.

    Hardware mapping note:
        Each masked forward pass is an independent inference → can be
        dispatched to SNAX GeMM accelerator in fire-and-forget mode.
        Pattern: for i in 0..N: launch_gemm(masked_input[i]); barrier();
        Scalar core prepares mask[i+1] while GeMM processes mask[i].
    """
    rng = np.random.default_rng(seed)
    _, H, W, C = input_tensor.shape

    if baselines is None:
        baselines = rng.standard_normal(
            (n_samples, H, W, C)
        ).astype(np.float32) * 0.1
    else:
        n_samples = baselines.shape[0]

    attributions = np.zeros((H, W, C), dtype=np.float32)

    for i in range(n_samples):
        baseline = baselines[i]
        alpha = rng.uniform(0.0, 1.0)

        # Interpolated input
        x_interp = baseline + alpha * (input_tensor[0] - baseline)
        x_interp_batch = x_interp[np.newaxis, ...]  # (1, H, W, C)

        # Compute gradient via finite differences
        grad = _compute_gradient(forward_fn, x_interp_batch, target_class)

        # Attribution for this sample: (x - x') * grad
        diff = input_tensor[0] - baseline  # (H, W, C)
        attributions += diff * grad

    attributions /= n_samples
    return attributions


def _compute_gradient(
    forward_fn: Callable,
    input_tensor: np.ndarray,
    target_class: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Compute gradient of target class score w.r.t. input via finite differences.

    Args:
        forward_fn:     Model forward function.
        input_tensor:   Shape (1, H, W, C).
        target_class:   Class to differentiate.
        eps:            Perturbation size.

    Returns:
        grad:   Shape (H, W, C), gradient of logit[target_class] w.r.t. input.
    """
    logits_base, _ = forward_fn(input_tensor)
    base_score = logits_base[0, target_class]

    _, H, W, C = input_tensor.shape
    grad = np.zeros((H, W, C), dtype=np.float32)

    for h in range(H):
        for w in range(W):
            for c in range(C):
                perturbed = input_tensor.copy()
                perturbed[0, h, w, c] += eps
                logits_p, _ = forward_fn(perturbed)
                grad[h, w, c] = (logits_p[0, target_class] - base_score) / eps

    return grad


def gradient_shap_analytical(
    input_tensor: np.ndarray,
    w_fc: np.ndarray,
    target_class: int,
    baselines: np.ndarray,
) -> np.ndarray:
    """
    Analytical Gradient SHAP for the simple GAP+FC model.

    For a linear model (GAP → FC), the gradient w.r.t. input feature maps
    is constant: grad[h,w,k] = w_fc[k, target_class] / (H * W).
    So SHAP values simplify to:
        phi[h,w,k] = mean_over_baselines[ (x[h,w,k] - x'[h,w,k]) ] * w_fc[k,c] / (H*W)

    This serves as a fast golden reference for testing.

    Args:
        input_tensor:   Shape (H, W, K), float32.
        w_fc:           Shape (K, C), float32.
        target_class:   Class index.
        baselines:      Shape (N, H, W, K), float32.

    Returns:
        attributions:   Shape (H, W, K), float32.
    """
    H, W, K = input_tensor.shape
    N = baselines.shape[0]

    # For linear model, gradient is constant
    grad = w_fc[:, target_class] / (H * W)  # shape (K,)

    # Mean difference over baselines
    mean_diff = input_tensor - baselines.mean(axis=0)  # (H, W, K)

    # SHAP attribution
    attributions = mean_diff * grad[np.newaxis, np.newaxis, :]

    return attributions.astype(np.float32)


def shap_interaction_values(
    forward_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    input_tensor: np.ndarray,
    target_class: int,
    n_samples: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """
    Compute pairwise SHAP interaction values (simplified).

    Measures how pairs of features interact in their contribution to
    the prediction. For feature pair (i, j):
        Phi_{ij} = E[ (grad_i(x+) - grad_i(x-)) * (x_j - x'_j) ]

    where x+ includes feature j, x- excludes it.

    Args:
        forward_fn:     Model forward function.
        input_tensor:   Shape (1, H, W, C).
        target_class:   Class to explain.
        n_samples:      Number of samples.
        seed:           Random seed.

    Returns:
        interactions:   Shape (H*W*C, H*W*C), float32.
                        interactions[i,j] = interaction between flat feature i and j.
    """
    rng = np.random.default_rng(seed)
    _, H, W, C = input_tensor.shape
    n_features = H * W * C
    flat_input = input_tensor[0].flatten()

    interactions = np.zeros((n_features, n_features), dtype=np.float32)

    for _ in range(n_samples):
        baseline = rng.standard_normal(n_features).astype(np.float32) * 0.1

        for j in range(n_features):
            # x+ = baseline with feature j set to input value
            x_plus = baseline.copy()
            x_plus[j] = flat_input[j]

            # x- = baseline without feature j
            x_minus = baseline.copy()

            # Gradients at both points
            grad_plus = _compute_gradient(
                forward_fn,
                x_plus.reshape(1, H, W, C),
                target_class,
            ).flatten()
            grad_minus = _compute_gradient(
                forward_fn,
                x_minus.reshape(1, H, W, C),
                target_class,
            ).flatten()

            diff_j = flat_input[j] - baseline[j]
            interactions[:, j] += (grad_plus - grad_minus) * diff_j

    interactions /= n_samples
    return interactions


def expected_gradients(
    forward_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    input_tensor: np.ndarray,
    target_class: int,
    baselines: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    """
    Expected Gradients: integrate gradient along paths from baselines to input.

    A variant of Integrated Gradients averaged over multiple baselines,
    which satisfies the SHAP completeness axiom.

    Args:
        forward_fn:     Model forward function.
        input_tensor:   Shape (1, H, W, C).
        target_class:   Class to explain.
        baselines:      Shape (N, H, W, C).
        n_steps:        Integration steps per baseline.

    Returns:
        attributions:   Shape (H, W, C), float32.
    """
    _, H, W, C = input_tensor.shape
    n_baselines = baselines.shape[0]
    attributions = np.zeros((H, W, C), dtype=np.float32)

    for i in range(n_baselines):
        baseline = baselines[i]
        diff = input_tensor[0] - baseline  # (H, W, C)

        path_attr = np.zeros((H, W, C), dtype=np.float32)
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            x_interp = baseline + alpha * diff
            x_interp_batch = x_interp[np.newaxis, ...]
            grad = _compute_gradient(forward_fn, x_interp_batch, target_class)
            path_attr += grad

        path_attr = diff * path_attr / n_steps
        attributions += path_attr

    attributions /= n_baselines
    return attributions
