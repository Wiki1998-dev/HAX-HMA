"""Layer-by-layer interval arithmetic bound propagation for QNNs.

Computes output bounds given input bounds by propagating intervals through
each layer of a quantized neural network. Used by the ILP verifier to:
  1. Prune infeasible ReLU intervals (QVIP Section 3.5)
  2. Determine which neurons are always active/inactive
  3. Provide initial bounds for ILP variables

Supports: Linear (FC), ReLU, and quantized clamp layers.

References:
    - QVIP (Zhang et al., ASE'22): Section 3.5, interval analysis
    - Standard interval arithmetic for neural networks
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .quantization import QuantConfig, quantize_uniform


@dataclass
class Bounds:
    """Interval bounds [lb, ub] for a set of neurons.

    Attributes:
        lb: lower bounds array, shape (n,)
        ub: upper bounds array, shape (n,)
    """
    lb: np.ndarray
    ub: np.ndarray

    def __post_init__(self) -> None:
        self.lb = np.asarray(self.lb, dtype=np.float64)
        self.ub = np.asarray(self.ub, dtype=np.float64)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lb.shape

    @property
    def size(self) -> int:
        return self.lb.size

    def width(self) -> np.ndarray:
        """Per-element interval width."""
        return self.ub - self.lb

    def contains(self, x: np.ndarray) -> bool:
        """Check if x is within bounds (element-wise)."""
        return bool(np.all(x >= self.lb - 1e-10) and np.all(x <= self.ub + 1e-10))


@dataclass
class LayerSpec:
    """Specification of a single network layer.

    Attributes:
        layer_type: one of 'linear', 'relu', 'clamp'
        weights: weight matrix for linear layers (out, in)
        bias: bias vector for linear layers (out,)
        clamp_lb: lower clamp bound (for quantized clamp layers)
        clamp_ub: upper clamp bound (for quantized clamp layers)
    """
    layer_type: str
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    clamp_lb: Optional[float] = None
    clamp_ub: Optional[float] = None


def propagate_linear(bounds: Bounds, W: np.ndarray, b: np.ndarray) -> Bounds:
    """Propagate intervals through a linear layer: y = Wx + b.

    Uses standard interval arithmetic:
        y_lb = W+ @ x_lb + W- @ x_ub + b
        y_ub = W+ @ x_ub + W- @ x_lb + b

    where W+ = max(W, 0) and W- = min(W, 0).

    Args:
        bounds: input bounds
        W: weight matrix (out_features, in_features)
        b: bias vector (out_features,)

    Returns:
        Output bounds after linear transformation
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)

    lb = W_pos @ bounds.lb + W_neg @ bounds.ub + b
    ub = W_pos @ bounds.ub + W_neg @ bounds.lb + b

    return Bounds(lb=lb, ub=ub)


def propagate_relu(bounds: Bounds) -> Bounds:
    """Propagate intervals through ReLU: y = max(x, 0).

    Three cases per neuron:
        - Always active (lb >= 0): bounds unchanged
        - Always inactive (ub <= 0): bounds = [0, 0]
        - Crossing (lb < 0 < ub): lb=0, ub unchanged

    Args:
        bounds: input bounds

    Returns:
        Output bounds after ReLU
    """
    lb = np.maximum(bounds.lb, 0)
    ub = np.maximum(bounds.ub, 0)
    return Bounds(lb=lb, ub=ub)


def propagate_clamp(bounds: Bounds, clamp_lb: float, clamp_ub: float) -> Bounds:
    """Propagate intervals through clamp: y = clamp(x, lb, ub).

    Args:
        bounds: input bounds
        clamp_lb: lower clamp value
        clamp_ub: upper clamp value

    Returns:
        Output bounds after clamping
    """
    lb = np.clip(bounds.lb, clamp_lb, clamp_ub)
    ub = np.clip(bounds.ub, clamp_lb, clamp_ub)
    return Bounds(lb=lb, ub=ub)


def propagate_quantize(bounds: Bounds, config: QuantConfig) -> Bounds:
    """Propagate intervals through quantization: û = clamp(floor(2^F * u), C^lb, C^ub).

    Args:
        bounds: input bounds (float)
        config: quantization configuration

    Returns:
        Output bounds (integer) after quantization
    """
    scale = 2.0 ** config.frac_bits
    lb = np.clip(np.floor(bounds.lb * scale), config.clamp_lb, config.clamp_ub)
    ub = np.clip(np.floor(bounds.ub * scale), config.clamp_lb, config.clamp_ub)
    return Bounds(lb=lb, ub=ub)


def propagate_network(
    input_bounds: Bounds,
    layers: List[LayerSpec],
) -> List[Bounds]:
    """Propagate bounds through an entire network.

    Args:
        input_bounds: bounds on network input
        layers: list of layer specifications

    Returns:
        List of bounds at each layer output (including input bounds at index 0)
    """
    all_bounds = [input_bounds]
    current = input_bounds

    for layer in layers:
        if layer.layer_type == 'linear':
            current = propagate_linear(current, layer.weights, layer.bias)
        elif layer.layer_type == 'relu':
            current = propagate_relu(current)
        elif layer.layer_type == 'clamp':
            current = propagate_clamp(current, layer.clamp_lb, layer.clamp_ub)
        else:
            raise ValueError(f"Unknown layer type: {layer.layer_type}")
        all_bounds.append(current)

    return all_bounds


def classify_relu_neurons(pre_relu_bounds: Bounds) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Classify ReLU neurons as always-active, always-inactive, or crossing.

    Used by QVIP to eliminate Boolean variables for non-crossing neurons.

    Args:
        pre_relu_bounds: bounds on pre-activation values

    Returns:
        Tuple of (active_mask, inactive_mask, crossing_mask) boolean arrays
    """
    active = pre_relu_bounds.lb >= 0      # always active: lb >= 0
    inactive = pre_relu_bounds.ub <= 0    # always inactive: ub <= 0
    crossing = ~active & ~inactive        # crossing: lb < 0 < ub
    return active, inactive, crossing


def compute_input_bounds_linf(
    x: np.ndarray,
    radius: int,
    config: QuantConfig,
) -> Bounds:
    """Compute input bounds for L-infinity perturbation in quantized space.

    Per QVIP Section 3.3: R̂_p(û, r) for L_inf norm.
    x̂_i^lb = clamp(û_i - r, C_in^lb, C_in^ub)
    x̂_i^ub = clamp(û_i + r, C_in^lb, C_in^ub)

    Args:
        x: quantized input (integer values)
        radius: L-infinity attack radius in quantized space
        config: input quantization configuration

    Returns:
        Bounds on perturbed input region
    """
    x = np.asarray(x, dtype=np.float64)
    lb = np.clip(x - radius, config.clamp_lb, config.clamp_ub)
    ub = np.clip(x + radius, config.clamp_lb, config.clamp_ub)
    return Bounds(lb=lb, ub=ub)


def compute_input_bounds_l1(
    x: np.ndarray,
    radius: int,
    config: QuantConfig,
) -> Bounds:
    """Compute element-wise bounds for L1 perturbation (over-approximation).

    Over-approximates L1 ball with L-infinity box of same radius.
    This is sound but not tight.

    Args:
        x: quantized input
        radius: L1 attack radius
        config: input quantization configuration

    Returns:
        Over-approximate bounds
    """
    return compute_input_bounds_linf(x, radius, config)


def output_robustness_check(output_bounds: Bounds, true_class: int) -> bool:
    """Quick check: can we prove robustness from output bounds alone?

    If the lower bound of the true class exceeds the upper bound of all
    other classes, the network is provably robust.

    Args:
        output_bounds: bounds on network output logits
        true_class: correct class index

    Returns:
        True if provably robust, False if inconclusive
    """
    true_lb = output_bounds.lb[true_class]
    for i in range(output_bounds.size):
        if i != true_class and output_bounds.ub[i] >= true_lb:
            return False
    return True
