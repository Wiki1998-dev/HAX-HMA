"""
Grad-CAM reference implementation (host-side, NumPy).

This is the ground truth we verify the embedded C implementation against.
Based on: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks"
and the hardware mapping from: Pan & Mishra arXiv 2305.04887

Usage:
    python -m src.xai.gradcam.gradcam_reference --model resnet8 --input test_img.npy
"""

from __future__ import annotations
import numpy as np
from typing import Callable


def gradcam(
    forward_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    input_tensor: np.ndarray,
    target_class: int,
) -> np.ndarray:
    """
    Compute Grad-CAM saliency map.

    Args:
        forward_fn:     Function that takes input (1, H, W, C) and returns
                        (logits, feature_maps) where feature_maps is the last
                        conv layer output (1, h, w, num_filters).
        input_tensor:   Input image, shape (1, H, W, C), float32.
        target_class:   Class index to explain.

    Returns:
        saliency:   Grad-CAM heatmap, shape (h, w), float32, values in [0, 1].

    Hardware mapping note (Pan & Mishra 2305.04887, Section III):
        Step 1 — gradient w.r.t. feature maps:
            grad[b, h, w, k] = d logits[target_class] / d feature_maps[b, h, w, k]
            → This is a backward GeMM: grad = (d_logit * W_fc^T)
            → Maps to SNAX GeMM accelerator with transposed streamer config
        Step 2 — global average pool over spatial dims:
            alpha[k] = mean_{h,w}(grad[0, :, :, k])
            → Simple SIMD reduction on RISC-V (no accelerator needed)
        Step 3 — weighted sum of feature maps:
            cam = sum_k(alpha[k] * feature_maps[0, :, :, k])
            → Element-wise multiply + accumulate → SNAX SIMD accelerator
        Step 4 — ReLU + normalize:
            cam = ReLU(cam) / max(cam)
            → RISC-V (cheap, single pass)
    """
    # ── Step 1: compute gradient of target class score w.r.t. feature maps ──
    # We use finite differences as an approximation (no autograd at runtime)
    eps = 1e-4

    logits_base, feature_maps = forward_fn(input_tensor)  # (1, num_classes), (1, h, w, K)

    h, w, K = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
    grads = np.zeros_like(feature_maps)  # (1, h, w, K)

    # Full gradient via perturbation (reference only — expensive, but exact)
    for k in range(K):
        perturbed = feature_maps.copy()
        perturbed[0, :, :, k] += eps
        # Recompute logits from perturbed features (forward_fn must support this)
        logits_perturbed, _ = forward_fn(input_tensor, feature_maps_override=perturbed)
        grads[0, :, :, k] = (logits_perturbed[0, target_class] - logits_base[0, target_class]) / eps

    # ── Step 2: global average pool over spatial dims (alpha_c^k in paper) ──
    alpha = np.mean(grads[0], axis=(0, 1))  # shape (K,)

    # ── Step 3: weighted combination of forward activation maps ──
    cam = np.einsum("hwk,k->hw", feature_maps[0], alpha)  # shape (h, w)

    # ── Step 4: ReLU + normalize to [0, 1] ──
    cam = np.maximum(cam, 0.0)
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    return cam.astype(np.float32)


def lrp_epsilon(
    layer_outputs: list[np.ndarray],
    layer_weights: list[np.ndarray],
    target_class: int,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """
    Layer-wise Relevance Propagation (LRP-epsilon rule).

    Propagates relevance scores backward through the network.
    Reference: Bach et al. 2015, "On Pixel-Wise Explanations for Non-Linear
               Classifier Decisions by Layer-Wise Relevance Propagation"

    Args:
        layer_outputs:  List of layer activations from forward pass,
                        each shape (1, N_neurons).
        layer_weights:  List of weight matrices, each shape (N_in, N_out).
        target_class:   Class to explain.
        epsilon:        Stabilizer to avoid division by zero.

    Returns:
        relevances: Input relevance scores, same shape as layer_outputs[0].

    Hardware mapping note:
        Each LRP backward step = (R / z) @ W^T  where z = a @ W + b
        → Transpose-GeMM, same as Grad-CAM backward → SNAX GeMM
    """
    num_layers = len(layer_weights)

    # Initialize relevance at output layer
    R = np.zeros_like(layer_outputs[-1])
    R[0, target_class] = layer_outputs[-1][0, target_class]

    # Backpropagate relevance layer by layer
    for l in range(num_layers - 1, -1, -1):
        W = layer_weights[l]             # (N_in, N_out)
        a = layer_outputs[l]             # (1, N_in)

        z = a @ W                        # (1, N_out) — pre-activation
        z_stable = z + epsilon * np.sign(z)  # stabilize
        z_stable[z_stable == 0] = epsilon

        s = R / z_stable                 # (1, N_out) — relevance message
        c = s @ W.T                      # (1, N_in)  — backpropagated
        R = a * c                        # (1, N_in)  — element-wise relevance

    return R.astype(np.float32)


def upsample_saliency(cam: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Bilinear upsample saliency map to input image resolution.

    Args:
        cam:            Saliency map, shape (h, w).
        target_shape:   Target (H, W) matching input image.

    Returns:
        upsampled:  Shape (H, W), values in [0, 1].
    """
    from scipy.ndimage import zoom
    scale_h = target_shape[0] / cam.shape[0]
    scale_w = target_shape[1] / cam.shape[1]
    return zoom(cam, (scale_h, scale_w), order=1).astype(np.float32)


def faithfulness_score(
    forward_fn: Callable,
    input_tensor: np.ndarray,
    saliency: np.ndarray,
    target_class: int,
    top_k_fractions: list[float] | None = None,
) -> dict[str, float]:
    """
    Measure faithfulness by progressively deleting top-K salient features.

    A faithful explanation should cause a large prediction drop when its
    most important features are removed (replaced with baseline = 0).

    Args:
        forward_fn:         Model forward function.
        input_tensor:       Original input, shape (1, H, W, C).
        saliency:           Saliency map, shape (H, W).
        target_class:       Class to measure.
        top_k_fractions:    List of fractions of features to delete (default [.1,.2,.5]).

    Returns:
        Dict mapping fraction → prediction score after deletion.
        Perfect faithfulness: score drops to ~0 after removing top 10-20%.
    """
    if top_k_fractions is None:
        top_k_fractions = [0.1, 0.2, 0.5]

    H, W = input_tensor.shape[1], input_tensor.shape[2]
    flat_saliency = saliency.flatten()
    total_pixels = len(flat_saliency)

    logits_original, _ = forward_fn(input_tensor)
    original_score = logits_original[0, target_class]

    scores = {"baseline": float(original_score)}

    for frac in top_k_fractions:
        k = int(frac * total_pixels)
        top_indices = np.argsort(flat_saliency)[-k:]  # top-K most salient

        masked = input_tensor.copy()
        mask_2d = np.zeros(H * W, dtype=bool)
        mask_2d[top_indices] = True
        mask_2d = mask_2d.reshape(H, W)
        masked[0, mask_2d, :] = 0.0  # zero out most salient pixels

        logits_masked, _ = forward_fn(masked)
        scores[f"top_{int(frac*100)}pct_deleted"] = float(logits_masked[0, target_class])

    return scores
