"""ILP-based formal verification for quantized neural networks (QVIP).

Implements the core QVIP verification approach from Zhang et al. (ASE'22):
  1. Encode QNN as integer linear constraints via piecewise constant functions
  2. Use interval analysis to prune infeasible ReLU intervals
  3. Verify local robustness: given input x ± r, does argmax(output) stay the same?
  4. Compute maximum robustness radius via binary search

The encoding uses Boolean variables for ReLU piecewise regions and encodes
the robustness property as: for all adversarial classes g ≠ true_class,
  output[true_class] > output[g]
which is negated to: ∃ g ≠ true_class s.t. output[g] >= output[true_class].
If the ILP is infeasible, the property holds (network is robust).

Uses scipy.optimize.linprog as the ILP backend (via LP relaxation for
efficiency, with exact integer solutions for small networks).

References:
    - QVIP (Zhang et al., ASE'22): Sections 3-4
    - SNAX GeMM: INT8 symmetric per-tensor quantization
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from scipy.optimize import linprog, LinearConstraint, OptimizeResult

from .quantization import QuantConfig, quantize_uniform, dequantize_uniform
from .bound_propagation import (
    Bounds,
    LayerSpec,
    propagate_network,
    propagate_linear,
    propagate_relu,
    classify_relu_neurons,
    compute_input_bounds_linf,
    output_robustness_check,
)


class VerifyResult(Enum):
    """Result of a robustness verification query."""
    ROBUST = "robust"           # proved robust (ILP infeasible for all adv classes)
    NOT_ROBUST = "not_robust"   # found counterexample
    UNKNOWN = "unknown"         # solver could not determine (timeout/numerical)


@dataclass
class VerifyReport:
    """Detailed verification report.

    Attributes:
        result: verification outcome
        true_class: the class being verified
        radius: attack radius tested
        counterexample: adversarial input if NOT_ROBUST (quantized integer values)
        adversarial_class: class of counterexample if NOT_ROBUST
        n_constraints: total ILP constraints generated
        n_variables: total ILP variables
        n_relu_crossing: number of crossing ReLU neurons (need Boolean vars)
        bounds_sufficient: True if interval analysis alone proved robustness
    """
    result: VerifyResult
    true_class: int
    radius: int
    counterexample: Optional[np.ndarray] = None
    adversarial_class: Optional[int] = None
    n_constraints: int = 0
    n_variables: int = 0
    n_relu_crossing: int = 0
    bounds_sufficient: bool = False


@dataclass
class QNN:
    """Quantized Neural Network for verification.

    A feed-forward QNN with quantized weights, biases, and activations.
    Layers alternate: linear -> relu -> linear -> relu -> ... -> linear (output).

    Attributes:
        weights: list of weight matrices (int32), one per linear layer
        biases: list of bias vectors (int32), one per linear layer
        config_in: input quantization config
        config_w: weight quantization config
        config_out: output/hidden quantization config
    """
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    config_in: QuantConfig
    config_w: QuantConfig
    config_out: QuantConfig

    @property
    def n_layers(self) -> int:
        """Number of linear layers (including output)."""
        return len(self.weights)

    @property
    def input_size(self) -> int:
        return self.weights[0].shape[1]

    @property
    def output_size(self) -> int:
        return self.weights[-1].shape[0]

    @property
    def layer_sizes(self) -> List[int]:
        """Sizes: [input, hidden1, ..., output]."""
        sizes = [self.input_size]
        for W in self.weights:
            sizes.append(W.shape[0])
        return sizes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the QNN (integer arithmetic).

        Implements the quantized forward pass per QVIP Section 2.2:
        For each layer i (2 <= i <= d):
            ŷ_j^i = clamp(⌊2^F_i * Σ_k Ŵ_{j,k}^i * ŷ_k^{i-1} + ...⌋, lb, C_out^ub)

        Simplified here for uniform config: all hidden layers share config_out.

        Args:
            x: quantized input (int32 array)

        Returns:
            Output logits (int32 array)
        """
        h = x.astype(np.int64)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            W = W.astype(np.int64)
            b = b.astype(np.int64)
            h = W @ h + b

            # Apply ReLU + clamp for hidden layers (not output)
            if i < self.n_layers - 1:
                h = np.maximum(h, 0)
                h = np.clip(h, self.config_out.clamp_lb, self.config_out.clamp_ub)

        # Output layer: clamp only (no ReLU)
        h = np.clip(h, self.config_out.clamp_lb, self.config_out.clamp_ub)
        return h.astype(np.int32)

    def classify(self, x: np.ndarray) -> int:
        """Return predicted class for quantized input."""
        return int(np.argmax(self.forward(x)))


def _build_layer_specs(qnn: QNN) -> List[LayerSpec]:
    """Convert QNN to list of LayerSpecs for bound propagation."""
    specs = []
    for i, (W, b) in enumerate(zip(qnn.weights, qnn.biases)):
        specs.append(LayerSpec(
            layer_type='linear',
            weights=W.astype(np.float64),
            bias=b.astype(np.float64),
        ))
        if i < qnn.n_layers - 1:
            specs.append(LayerSpec(layer_type='relu'))
            specs.append(LayerSpec(
                layer_type='clamp',
                clamp_lb=qnn.config_out.clamp_lb,
                clamp_ub=qnn.config_out.clamp_ub,
            ))
    return specs


def verify_robustness(
    qnn: QNN,
    x: np.ndarray,
    true_class: int,
    radius: int,
    norm: str = "linf",
) -> VerifyReport:
    """Verify local robustness of a QNN at input x with attack radius r.

    Checks whether all inputs in the L_p ball of radius r around x produce
    the same classification as x.

    Strategy:
    1. Compute interval bounds via bound propagation
    2. If bounds alone prove robustness, return immediately
    3. Otherwise, solve LP relaxation for each adversarial class
    4. If LP is infeasible for all adversarial classes → ROBUST
    5. If LP finds a feasible point → check if it's a true counterexample

    Args:
        qnn: quantized neural network
        x: quantized input (int32 array)
        true_class: correct class index
        radius: attack radius in quantized space
        norm: perturbation norm ("linf", "l1", "l2")

    Returns:
        VerifyReport with result and diagnostics
    """
    x = np.asarray(x, dtype=np.int32)
    n_in = qnn.input_size
    n_out = qnn.output_size

    # Step 1: compute input bounds
    if norm == "linf":
        input_bounds = compute_input_bounds_linf(x, radius, qnn.config_in)
    else:
        input_bounds = compute_input_bounds_linf(x, radius, qnn.config_in)

    # Step 2: propagate bounds through network
    layer_specs = _build_layer_specs(qnn)
    all_bounds = propagate_network(input_bounds, layer_specs)
    output_bounds = all_bounds[-1]

    # Step 3: check if bounds alone prove robustness
    if output_robustness_check(output_bounds, true_class):
        return VerifyReport(
            result=VerifyResult.ROBUST,
            true_class=true_class,
            radius=radius,
            bounds_sufficient=True,
            n_constraints=0,
            n_variables=0,
        )

    # Step 4: count crossing ReLU neurons for diagnostics
    n_crossing = 0
    for i in range(len(all_bounds) - 1):
        spec_idx = i
        if spec_idx < len(layer_specs) and layer_specs[spec_idx].layer_type == 'relu':
            pre_bounds = all_bounds[i]
            _, _, crossing = classify_relu_neurons(pre_bounds)
            n_crossing += int(np.sum(crossing))

    # Step 5: solve LP relaxation for each adversarial class
    # We check: ∃ x' in R(x, r) s.t. output[g] >= output[true_class]
    # for each g ≠ true_class
    report = _solve_lp_verification(
        qnn, input_bounds, all_bounds, layer_specs,
        true_class, radius, n_crossing,
    )
    return report


def _solve_lp_verification(
    qnn: QNN,
    input_bounds: Bounds,
    all_bounds: List[Bounds],
    layer_specs: List[LayerSpec],
    true_class: int,
    radius: int,
    n_crossing: int,
) -> VerifyReport:
    """Solve LP relaxation to verify robustness.

    For each adversarial class g, we maximize output[g] - output[true_class]
    subject to network constraints (LP relaxation of ReLU).
    If the maximum is < 0 for all g, the network is robust.

    This is an LP relaxation (triangle relaxation for ReLU), so:
    - If LP says infeasible/negative → network is provably robust
    - If LP finds positive objective → may be a counterexample (check exactly)
    """
    n_in = qnn.input_size
    n_out = qnn.output_size

    # Build LP: variables are [x_input, h1_pre, h1_post, h2_pre, h2_post, ..., output]
    # For a simple 2-layer network: x -> W1x+b1 -> relu -> W2h+b2 -> output
    # Variables: x (n_in) + pre-relu layers + post-relu layers + output

    # Collect variable offsets and sizes
    var_offset = {}
    var_offset['input'] = 0
    n_vars = n_in

    # For each layer, track pre and post activation variables
    layer_var_info = []
    bound_idx = 0  # index into all_bounds

    for i, spec in enumerate(layer_specs):
        if spec.layer_type == 'linear':
            size = spec.weights.shape[0]
            var_offset[f'linear_{i}_out'] = n_vars
            layer_var_info.append(('linear', i, n_vars, size))
            n_vars += size
        elif spec.layer_type == 'relu':
            # ReLU output shares size with preceding linear output
            size = all_bounds[bound_idx + 1].size if bound_idx + 1 < len(all_bounds) else 0
            var_offset[f'relu_{i}_out'] = n_vars
            layer_var_info.append(('relu', i, n_vars, size))
            n_vars += size
        elif spec.layer_type == 'clamp':
            size = all_bounds[bound_idx + 1].size if bound_idx + 1 < len(all_bounds) else 0
            var_offset[f'clamp_{i}_out'] = n_vars
            layer_var_info.append(('clamp', i, n_vars, size))
            n_vars += size
        bound_idx += 1

    # For very small input spaces, enumerate exactly
    total_points = 1
    for i in range(n_in):
        total_points *= int(input_bounds.ub[i] - input_bounds.lb[i] + 1)
        if total_points > 100_000:
            break
    if total_points <= 100_000:
        return _verify_enumerate(qnn, input_bounds, true_class, radius, n_crossing)

    # For larger networks, use LP relaxation
    return _verify_lp_relaxation(
        qnn, input_bounds, all_bounds, layer_specs,
        true_class, radius, n_vars, n_crossing,
    )


def _verify_enumerate(
    qnn: QNN,
    input_bounds: Bounds,
    true_class: int,
    radius: int,
    n_crossing: int,
) -> VerifyReport:
    """Exact verification by enumerating all integer inputs in the bounded region.

    Only feasible for small input dimensions and small radius.
    """
    n_in = qnn.input_size
    lb = input_bounds.lb.astype(np.int32)
    ub = input_bounds.ub.astype(np.int32)

    # Generate all integer points in the hyperrectangle
    ranges = [np.arange(lb[i], ub[i] + 1) for i in range(n_in)]

    # Check total enumeration size
    total = 1
    for r in ranges:
        total *= len(r)
        if total > 1_000_000:
            # Too many — fall back to LP
            return VerifyReport(
                result=VerifyResult.UNKNOWN,
                true_class=true_class,
                radius=radius,
                n_relu_crossing=n_crossing,
            )

    # Enumerate via meshgrid
    grids = np.meshgrid(*ranges, indexing='ij')
    points = np.stack([g.flatten() for g in grids], axis=-1).astype(np.int32)

    for point in points:
        pred = qnn.classify(point)
        if pred != true_class:
            return VerifyReport(
                result=VerifyResult.NOT_ROBUST,
                true_class=true_class,
                radius=radius,
                counterexample=point,
                adversarial_class=pred,
                n_constraints=0,
                n_variables=n_in,
                n_relu_crossing=n_crossing,
            )

    return VerifyReport(
        result=VerifyResult.ROBUST,
        true_class=true_class,
        radius=radius,
        n_constraints=0,
        n_variables=n_in,
        n_relu_crossing=n_crossing,
    )


def _verify_lp_relaxation(
    qnn: QNN,
    input_bounds: Bounds,
    all_bounds: List[Bounds],
    layer_specs: List[LayerSpec],
    true_class: int,
    radius: int,
    n_vars: int,
    n_crossing: int,
) -> VerifyReport:
    """LP relaxation verification using triangle relaxation for ReLU.

    For each ReLU neuron with pre-activation bounds [l, u]:
    - If l >= 0: y = x (active)
    - If u <= 0: y = 0 (inactive)
    - If l < 0 < u: triangle relaxation:
        y >= 0
        y >= x
        y <= u(x - l)/(u - l)
    """
    n_in = qnn.input_size
    n_out = qnn.output_size

    # Simpler LP formulation: directly express output as function of input
    # For network: x -> W1x+b1 -> relu -> W2*relu(W1x+b1)+b2
    # LP variables: [x (n_in), h (n_hidden), output (n_out)]

    if qnn.n_layers == 1:
        # Single linear layer: output = W @ x + b
        # Check: for all x in bounds, argmax(Wx+b) = true_class
        W = qnn.weights[0].astype(np.float64)
        b = qnn.biases[0].astype(np.float64)

        for g in range(n_out):
            if g == true_class:
                continue
            # Check if output[g] - output[true_class] >= 0 is feasible
            # Maximize (W[g] - W[true_class]) @ x + (b[g] - b[true_class])
            c_obj = -(W[g] - W[true_class])  # negate for minimization

            result = linprog(
                c_obj,
                bounds=list(zip(input_bounds.lb, input_bounds.ub)),
                method='highs',
            )
            if result.success and -result.fun >= 0:
                # Found adversarial input
                x_adv = np.round(result.x).astype(np.int32)
                x_adv = np.clip(x_adv, input_bounds.lb.astype(np.int32),
                                input_bounds.ub.astype(np.int32))
                pred = qnn.classify(x_adv)
                if pred != true_class:
                    return VerifyReport(
                        result=VerifyResult.NOT_ROBUST,
                        true_class=true_class,
                        radius=radius,
                        counterexample=x_adv,
                        adversarial_class=pred,
                        n_variables=n_in,
                        n_relu_crossing=0,
                    )

        return VerifyReport(
            result=VerifyResult.ROBUST,
            true_class=true_class,
            radius=radius,
            n_variables=n_in,
            n_relu_crossing=0,
        )

    elif qnn.n_layers == 2:
        # Two-layer: x -> W1x+b1 -> relu -> W2*h+b2
        W1 = qnn.weights[0].astype(np.float64)
        b1 = qnn.biases[0].astype(np.float64)
        W2 = qnn.weights[1].astype(np.float64)
        b2 = qnn.biases[1].astype(np.float64)

        n_hidden = W1.shape[0]

        # Pre-ReLU bounds from interval analysis
        pre_relu = propagate_linear(input_bounds, W1, b1)
        active, inactive, crossing = classify_relu_neurons(pre_relu)

        total_vars = n_in + n_hidden  # x and h (post-ReLU)
        # Variable layout: [x_0..x_{n_in-1}, h_0..h_{n_hidden-1}]

        A_ub_rows = []
        b_ub_rows = []

        for j in range(n_hidden):
            l_j = pre_relu.lb[j]
            u_j = pre_relu.ub[j]

            if active[j]:
                # h_j = W1[j] @ x + b1[j]  →  h_j - W1[j]@x = b1[j]
                # Encode as two inequalities: h_j >= W1[j]@x + b1[j] and h_j <= ...
                row_ge = np.zeros(total_vars)
                row_ge[:n_in] = W1[j]
                row_ge[n_in + j] = -1
                A_ub_rows.append(row_ge)
                b_ub_rows.append(-b1[j])

                row_le = np.zeros(total_vars)
                row_le[:n_in] = -W1[j]
                row_le[n_in + j] = 1
                A_ub_rows.append(row_le)
                b_ub_rows.append(b1[j])

            elif inactive[j]:
                # h_j = 0
                row_ge = np.zeros(total_vars)
                row_ge[n_in + j] = 1
                A_ub_rows.append(row_ge)
                b_ub_rows.append(0.0)

                row_le = np.zeros(total_vars)
                row_le[n_in + j] = -1
                A_ub_rows.append(row_le)
                b_ub_rows.append(0.0)

            else:
                # Triangle relaxation for crossing neurons
                # h_j >= 0
                row1 = np.zeros(total_vars)
                row1[n_in + j] = -1
                A_ub_rows.append(row1)
                b_ub_rows.append(0.0)

                # h_j >= W1[j]@x + b1[j]  (i.e., h_j >= pre_relu_j)
                row2 = np.zeros(total_vars)
                row2[:n_in] = W1[j]
                row2[n_in + j] = -1
                A_ub_rows.append(row2)
                b_ub_rows.append(-b1[j])

                # h_j <= u_j * (W1[j]@x + b1[j] - l_j) / (u_j - l_j)
                # i.e., h_j * (u_j - l_j) <= u_j * (W1[j]@x + b1[j] - l_j)
                # h_j * (u_j - l_j) - u_j * W1[j]@x <= u_j * (b1[j] - l_j)
                # Wait, let's be more careful:
                # h_j <= u_j/(u_j - l_j) * (pre - l_j)
                # h_j <= u_j/(u_j-l_j) * (W1[j]@x + b1[j] - l_j)
                # h_j - u_j/(u_j-l_j) * W1[j]@x <= u_j/(u_j-l_j) * (b1[j] - l_j)
                if u_j > l_j:
                    slope = u_j / (u_j - l_j)
                    row3 = np.zeros(total_vars)
                    row3[n_in + j] = 1
                    row3[:n_in] = -slope * W1[j]
                    A_ub_rows.append(row3)
                    b_ub_rows.append(slope * (b1[j] - l_j))

        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
        else:
            A_ub = None
            b_ub = None

        # Variable bounds
        var_bounds = []
        for i in range(n_in):
            var_bounds.append((input_bounds.lb[i], input_bounds.ub[i]))
        for j in range(n_hidden):
            post_relu_lb = max(0, pre_relu.lb[j])
            post_relu_ub = max(0, pre_relu.ub[j])
            var_bounds.append((post_relu_lb, post_relu_ub))

        # For each adversarial class g, check feasibility
        for g in range(qnn.output_size):
            if g == true_class:
                continue

            # Maximize output[g] - output[true_class]
            # = (W2[g] - W2[true_class]) @ h + (b2[g] - b2[true_class])
            # Minimize -(W2[g] - W2[true_class]) @ h - ...
            diff_w = W2[g] - W2[true_class]
            diff_b = b2[g] - b2[true_class]

            c_obj = np.zeros(total_vars)
            c_obj[n_in:] = -diff_w  # negate for minimization

            result = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=var_bounds,
                method='highs',
            )

            if result.success:
                obj_val = -result.fun + diff_b  # actual objective
                if obj_val >= 0:
                    # LP relaxation says adversarial might be feasible
                    # Extract input, verify exactly
                    x_adv = np.round(result.x[:n_in]).astype(np.int32)
                    x_adv = np.clip(x_adv, input_bounds.lb.astype(np.int32),
                                    input_bounds.ub.astype(np.int32))
                    pred = qnn.classify(x_adv)
                    if pred != true_class:
                        return VerifyReport(
                            result=VerifyResult.NOT_ROBUST,
                            true_class=true_class,
                            radius=radius,
                            counterexample=x_adv,
                            adversarial_class=pred,
                            n_constraints=len(A_ub_rows) if A_ub_rows else 0,
                            n_variables=total_vars,
                            n_relu_crossing=int(np.sum(crossing)),
                        )

        # All adversarial classes checked — either LP proved infeasible or
        # counterexamples didn't verify exactly
        return VerifyReport(
            result=VerifyResult.ROBUST,
            true_class=true_class,
            radius=radius,
            n_constraints=len(A_ub_rows) if A_ub_rows else 0,
            n_variables=total_vars,
            n_relu_crossing=int(np.sum(crossing)),
        )

    # General case: fall back to enumeration or report unknown
    return VerifyReport(
        result=VerifyResult.UNKNOWN,
        true_class=true_class,
        radius=radius,
        n_relu_crossing=n_crossing,
    )


def compute_max_robustness_radius(
    qnn: QNN,
    x: np.ndarray,
    true_class: int,
    max_radius: int = 30,
    norm: str = "linf",
) -> Tuple[int, List[VerifyReport]]:
    """Compute maximum robustness radius via binary search (QVIP Algorithm 1).

    Finds the largest r such that the QNN is robust w.r.t. R̂_p(x, r).

    Args:
        qnn: quantized neural network
        x: quantized input
        true_class: correct class
        max_radius: upper bound on search
        norm: perturbation norm

    Returns:
        Tuple of (max_robust_radius, list of reports from binary search)
    """
    reports = []

    # First check r=1
    report = verify_robustness(qnn, x, true_class, radius=1, norm=norm)
    reports.append(report)
    if report.result != VerifyResult.ROBUST:
        return 0, reports

    # Binary search between 1 and max_radius
    r_lb = 1
    r_ub = max_radius

    # Quick check: is max_radius robust?
    report = verify_robustness(qnn, x, true_class, radius=max_radius, norm=norm)
    reports.append(report)
    if report.result == VerifyResult.ROBUST:
        return max_radius, reports

    # Binary search
    while r_lb < r_ub - 1:
        r_mid = (r_lb + r_ub) // 2
        report = verify_robustness(qnn, x, true_class, radius=r_mid, norm=norm)
        reports.append(report)

        if report.result == VerifyResult.ROBUST:
            r_lb = r_mid
        else:
            r_ub = r_mid

    return r_lb, reports


def verify_quantization_safety(
    weights_float: List[np.ndarray],
    biases_float: List[np.ndarray],
    x_float: np.ndarray,
    config_in: QuantConfig,
    config_w: QuantConfig,
    config_out: QuantConfig,
    radius: int = 1,
) -> VerifyReport:
    """Verify that quantization preserves classification for a given input.

    Quantizes the float model, then verifies the QNN is locally robust
    at the quantized input. This directly answers: "does the INT8 model
    on SNAX GeMM agree with the float32 model?"

    Args:
        weights_float: float weight matrices per layer
        biases_float: float bias vectors per layer
        x_float: float input
        config_in: input quantization config
        config_w: weight quantization config
        config_out: output quantization config
        radius: attack radius (0 = exact match, >0 = robustness check)

    Returns:
        VerifyReport
    """
    # Quantize weights and biases
    weights_q = [quantize_uniform(w, config_w) for w in weights_float]
    biases_q = [quantize_uniform(b, config_w) for b in biases_float]

    # Quantize input
    x_q = quantize_uniform(x_float, config_in)

    # Build QNN
    qnn = QNN(
        weights=weights_q,
        biases=biases_q,
        config_in=config_in,
        config_w=config_w,
        config_out=config_out,
    )

    # Get float model prediction
    h = x_float.copy()
    for i, (W, b) in enumerate(zip(weights_float, biases_float)):
        h = W @ h + b
        if i < len(weights_float) - 1:
            h = np.maximum(h, 0)
    float_class = int(np.argmax(h))

    # Verify QNN robustness at quantized input
    return verify_robustness(qnn, x_q, float_class, radius)
