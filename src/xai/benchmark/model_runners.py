"""Synthetic model runners for Phase 5 benchmarking.

Each factory builds a NumPy-only forward function with random weights
matching the architecture described in :mod:`.models`.  These are NOT
trained models — weights are drawn from a scaled normal distribution so
that activations stay in a reasonable range and XAI methods produce
non-degenerate outputs.

Every factory returns a :class:`ModelRunner` that unifies the interface
across all four phases:

* **Grad-CAM / LRP**: ``runner.forward_fn(x, feature_maps_override=...)``
  returns ``(logits, feature_maps)`` with the hook at the backbone end.
* **SHAP**: ``runner.backbone_fn`` / ``runner.head_fn`` /
  ``runner.head_grad_fn`` for hoisted Gradient SHAP.
* **Formal (QVIP)**: ``runner.qnn`` returns a :class:`QNN` built from
  the quantized weights.
* **Rule extraction**: ``runner.predict(X)`` for teacher predictions on
  a flat feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import MODEL_CATALOG, ModelSpec, get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _he_init(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """He-normal initialization (fan_in scaling)."""
    fan_in = shape[-1] if len(shape) == 2 else np.prod(shape[:-1])
    std = np.sqrt(2.0 / max(fan_in, 1))
    return (rng.standard_normal(shape) * std).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _conv2d_valid(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Minimal valid-padding conv2d.  x: (H, W, C_in), W: (k, k, C_in, C_out)."""
    H, Wi, C_in = x.shape
    k, _, _, C_out = W.shape
    oH, oW = H - k + 1, Wi - k + 1
    out = np.zeros((oH, oW, C_out), dtype=np.float32)
    for co in range(C_out):
        for i in range(oH):
            for j in range(oW):
                out[i, j, co] = np.sum(x[i:i + k, j:j + k, :] * W[:, :, :, co])
    return out


def _conv2d_same(x: np.ndarray, W: np.ndarray, stride: int = 1) -> np.ndarray:
    """Same-padding conv2d with stride support.
    x: (H, W, C_in), W: (k, k, C_in, C_out)."""
    H, Wi, C_in = x.shape
    k, _, _, C_out = W.shape
    pad = k // 2
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    oH, oW = H // stride, Wi // stride
    out = np.zeros((oH, oW, C_out), dtype=np.float32)
    for co in range(C_out):
        for i in range(oH):
            for j in range(oW):
                si, sj = i * stride, j * stride
                out[i, j, co] = np.sum(
                    padded[si:si + k, sj:sj + k, :] * W[:, :, :, co]
                )
    return out


def _gap(x: np.ndarray) -> np.ndarray:
    """Global average pooling.  x: (H, W, C) -> (C,)."""
    return x.mean(axis=(0, 1))


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------


@dataclass
class ModelRunner:
    """Unified interface for running all XAI phases on a benchmark model.

    Attributes:
        spec:           the underlying :class:`ModelSpec`
        forward_fn:     Grad-CAM compatible forward function:
                        ``(x, feature_maps_override=None) -> (logits, feature_maps)``
        backbone_fn:    maps input to feature maps (for hoisted SHAP)
        head_fn:        maps feature maps to logits (for hoisted SHAP)
        head_grad_fn:   gradient of target logit w.r.t. feature maps
        weights:        list of weight matrices per layer (for LRP, QVIP)
        biases:         list of bias vectors per layer
        layer_outputs_fn: given an input, returns list of per-layer activations
                          (for LRP backward pass)
        input_shape:    expected input tensor shape (without batch dim)
        feature_shape:  shape of feature maps at the Grad-CAM hook point
    """

    spec: ModelSpec
    forward_fn: Callable
    backbone_fn: Callable
    head_fn: Callable
    head_grad_fn: Callable
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    layer_outputs_fn: Callable
    input_shape: Tuple[int, ...]
    feature_shape: Tuple[int, ...]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Teacher predictions for rule extraction.

        Args:
            X: (n_samples, n_features) flat feature matrix.

        Returns:
            (n_samples,) predicted class indices.
        """
        preds = []
        for i in range(X.shape[0]):
            x = X[i]
            # Reshape flat features to input_shape and run forward
            x_shaped = x.reshape((1,) + self.input_shape)
            logits, _ = self.forward_fn(x_shaped)
            preds.append(int(np.argmax(logits[0])))
        return np.array(preds, dtype=np.int32)

    def sample_input(self, seed: int = 0) -> np.ndarray:
        """Generate a random input tensor (with batch dim)."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1,) + self.input_shape).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# GAP+FC factory (calibration model)
# ---------------------------------------------------------------------------


def build_gap_fc(seed: int = 42) -> ModelRunner:
    """Build the Phase 1/2 calibration model: conv features -> GAP -> FC.

    Architecture: 4x4x16 feature maps (treated as one conv layer output),
    then GAP to (16,), then FC(16, 10).
    """
    rng = np.random.default_rng(seed)
    spec = get_model("gap_fc")

    H, W, K = 4, 4, 16
    n_classes = spec.n_classes  # 10

    W_conv = _he_init((3, 3, K, K), rng)  # identity-ish conv
    b_conv = np.zeros(K, dtype=np.float32)
    W_fc = _he_init((K, n_classes), rng)
    b_fc = np.zeros(n_classes, dtype=np.float32)

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        if feature_maps_override is not None:
            fmaps = feature_maps_override
        else:
            # x: (1, H, W, K) — treat as already being feature maps
            # Apply a lightweight conv to make gradients non-trivial
            fmaps_raw = _conv2d_same(x[0], W_conv)
            fmaps = _relu(fmaps_raw + b_conv)[np.newaxis, ...]
        pooled = _gap(fmaps[0])  # (K,)
        logits = (pooled @ W_fc + b_fc).reshape(1, -1)
        return logits, fmaps

    def backbone_fn(x):
        _, fmaps = forward_fn(x)
        return fmaps

    def head_fn(features):
        f = features.reshape(-1) if features.ndim > 1 else features
        if f.size == H * W * K:
            pooled = f.reshape(H, W, K).mean(axis=(0, 1))
        else:
            pooled = f
        return (pooled @ W_fc + b_fc).reshape(1, -1)

    def head_grad_fn(features, target_class):
        # d(logit[c]) / d(feature_maps) = W_fc[:, c] / (H * W) broadcast
        grad_pooled = W_fc[:, target_class]  # (K,)
        grad = np.broadcast_to(
            grad_pooled / (H * W), features.shape
        ).astype(np.float32)
        return grad

    def layer_outputs_fn(x):
        _, fmaps = forward_fn(x)
        pooled = _gap(fmaps[0]).reshape(1, -1)
        logits = pooled @ W_fc + b_fc
        return [x.reshape(1, -1), pooled, logits.reshape(1, -1)]

    return ModelRunner(
        spec=spec,
        forward_fn=forward_fn,
        backbone_fn=backbone_fn,
        head_fn=head_fn,
        head_grad_fn=head_grad_fn,
        weights=[W_fc],
        biases=[b_fc],
        layer_outputs_fn=layer_outputs_fn,
        input_shape=(H, W, K),
        feature_shape=(H, W, K),
    )


# ---------------------------------------------------------------------------
# ResNet-8 factory
# ---------------------------------------------------------------------------


def build_resnet8(seed: int = 42) -> ModelRunner:
    """Build a synthetic ResNet-8 (MLPerf Tiny, CIFAR-10).

    Simplified: conv1(3->16) + 3 blocks of 2 convs each + GAP + FC(64,10).
    Skip connections omitted for simplicity (they don't affect XAI API shape).
    """
    rng = np.random.default_rng(seed)
    spec = get_model("resnet8")

    # Layer weights: (k, k, c_in, c_out)
    convs = [
        ("conv1",    _he_init((3, 3, 3, 16), rng)),
        ("block1_a", _he_init((3, 3, 16, 16), rng)),
        ("block1_b", _he_init((3, 3, 16, 16), rng)),
        ("block2_a", _he_init((3, 3, 16, 32), rng)),
        ("block2_b", _he_init((3, 3, 32, 32), rng)),
        ("block3_a", _he_init((3, 3, 32, 64), rng)),
        ("block3_b", _he_init((3, 3, 64, 64), rng)),
    ]
    strides = [1, 1, 1, 2, 1, 2, 1]
    W_fc = _he_init((64, 10), rng)
    b_fc = np.zeros(10, dtype=np.float32)

    def _run_backbone(x_2d: np.ndarray) -> np.ndarray:
        """x_2d: (H, W, C) -> feature maps (h, w, 64)."""
        h = x_2d
        for (_, W), s in zip(convs, strides):
            h = _relu(_conv2d_same(h, W, stride=s))
        return h

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        if feature_maps_override is not None:
            fmaps = feature_maps_override
            pooled = _gap(fmaps[0])
        else:
            fm = _run_backbone(x[0])
            fmaps = fm[np.newaxis, ...]
            pooled = _gap(fm)
        logits = (pooled @ W_fc + b_fc).reshape(1, -1)
        return logits, fmaps

    def backbone_fn(x):
        fm = _run_backbone(x[0])
        return fm[np.newaxis, ...]

    def head_fn(features):
        f = features[0] if features.ndim == 4 else features
        pooled = _gap(f) if f.ndim == 3 else f
        return (pooled @ W_fc + b_fc).reshape(1, -1)

    def head_grad_fn(features, target_class):
        # GAP + FC: gradient = W_fc[:, c] / (H * W)
        f = features[0] if features.ndim == 4 else features
        spatial = f.shape[0] * f.shape[1] if f.ndim == 3 else 1
        grad_per_elem = W_fc[:, target_class] / max(spatial, 1)
        return np.broadcast_to(grad_per_elem, features.shape).astype(np.float32)

    def layer_outputs_fn(x):
        h = x[0]
        outputs = [x.reshape(1, -1)]
        for (_, W), s in zip(convs, strides):
            h = _relu(_conv2d_same(h, W, stride=s))
            outputs.append(h.reshape(1, -1))
        pooled = _gap(h).reshape(1, -1)
        outputs.append(pooled)
        logits = (pooled @ W_fc + b_fc).reshape(1, -1)
        outputs.append(logits)
        return outputs

    return ModelRunner(
        spec=spec,
        forward_fn=forward_fn,
        backbone_fn=backbone_fn,
        head_fn=head_fn,
        head_grad_fn=head_grad_fn,
        weights=[W_fc],  # only FC for LRP (conv weights stored internally)
        biases=[b_fc],
        layer_outputs_fn=layer_outputs_fn,
        input_shape=(32, 32, 3),
        feature_shape=(8, 8, 64),
    )


# ---------------------------------------------------------------------------
# ToyAdmos factory
# ---------------------------------------------------------------------------


def build_toyadmos(seed: int = 42) -> ModelRunner:
    """Build a synthetic ToyAdmos autoencoder (MLPerf Tiny, anomaly detection).

    FC(640->128) x4 -> bottleneck FC(128->8) -> FC(8->128) x4 -> FC(128->640).
    XAI hooks into the bottleneck (layer index 4).
    """
    rng = np.random.default_rng(seed)
    spec = get_model("toyadmos")

    # Encoder: 640->128, 128->128, 128->128, 128->128, 128->8
    # Decoder: 8->128, 128->128, 128->128, 128->128, 128->640
    layer_dims = [
        (640, 128), (128, 128), (128, 128), (128, 128), (128, 8),
        (8, 128), (128, 128), (128, 128), (128, 128), (128, 640),
    ]
    fc_weights = []
    fc_biases = []
    for d_in, d_out in layer_dims:
        fc_weights.append(_he_init((d_in, d_out), rng))
        fc_biases.append(np.zeros(d_out, dtype=np.float32))

    bottleneck_idx = 4  # output of 5th layer is the bottleneck

    def _run_all(x_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Returns (output, bottleneck_features, layer_outputs)."""
        h = x_flat.astype(np.float32)
        outputs = [h.reshape(1, -1)]
        for i, (W, b) in enumerate(zip(fc_weights, fc_biases)):
            h = h @ W + b
            if i < len(fc_weights) - 1:
                h = _relu(h)
            outputs.append(h.reshape(1, -1))
        bottleneck = outputs[bottleneck_idx + 1]  # +1 because outputs[0] is input
        return h, bottleneck, outputs

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        if feature_maps_override is not None:
            # Run decoder from bottleneck
            fmaps = feature_maps_override
            h = fmaps[0].flatten() if fmaps.ndim > 1 else fmaps.flatten()
            for i in range(bottleneck_idx, len(fc_weights)):
                W, b = fc_weights[i], fc_biases[i]
                h = h @ W + b
                if i < len(fc_weights) - 1:
                    h = _relu(h)
            # For anomaly detection: "logits" are reconstruction + anomaly score
            recon = h
            anomaly_score = np.sum((x[0].flatten()[:640] - recon[:640]) ** 2)
            logits = np.array([[0.0, anomaly_score]], dtype=np.float32)
            return logits, fmaps

        x_flat = x[0].flatten()
        output, bottleneck, _ = _run_all(x_flat)
        fmaps = bottleneck  # shape (1, 8)

        recon = output
        anomaly_score = np.sum((x_flat[:640] - recon[:640]) ** 2)
        logits = np.array([[0.0, anomaly_score]], dtype=np.float32)
        return logits, fmaps

    def backbone_fn(x):
        h = x[0].flatten().astype(np.float32)
        for i in range(bottleneck_idx + 1):
            h = h @ fc_weights[i] + fc_biases[i]
            if i < bottleneck_idx:
                h = _relu(h)
        return h.reshape(1, -1)  # (1, 8)

    def head_fn(features):
        h = features.flatten().astype(np.float32)
        for i in range(bottleneck_idx + 1, len(fc_weights)):
            h = h @ fc_weights[i] + fc_biases[i]
            if i < len(fc_weights) - 1:
                h = _relu(h)
        return h.reshape(1, -1)

    def head_grad_fn(features, target_class):
        """Numerical gradient of head output w.r.t. bottleneck features."""
        eps = 1e-4
        f = features.flatten().astype(np.float32)
        base_out = head_fn(f).flatten()
        grad = np.zeros_like(f)
        for j in range(f.size):
            f_p = f.copy()
            f_p[j] += eps
            out_p = head_fn(f_p).flatten()
            # Use reconstruction output (index depends on target_class)
            grad[j] = (out_p.sum() - base_out.sum()) / eps
        return grad.reshape(features.shape).astype(np.float32)

    def layer_outputs_fn(x):
        _, _, outputs = _run_all(x[0].flatten())
        return outputs

    return ModelRunner(
        spec=spec,
        forward_fn=forward_fn,
        backbone_fn=backbone_fn,
        head_fn=head_fn,
        head_grad_fn=head_grad_fn,
        weights=fc_weights,
        biases=fc_biases,
        layer_outputs_fn=layer_outputs_fn,
        input_shape=(640,),
        feature_shape=(8,),
    )


# ---------------------------------------------------------------------------
# MobileBERT-tiny factory
# ---------------------------------------------------------------------------


def build_mobilebert_tiny(seed: int = 42) -> ModelRunner:
    """Build a synthetic MobileBERT-tiny encoder (embedded NLP).

    Simplified: embedding + 2 transformer blocks (linear attention + FFN)
    + mean pool + FC(128, 2).  Self-attention is approximated as a linear
    projection for benchmarking (real QK^TV is too expensive in NumPy for
    large seq_len, and XAI cares about the shapes, not accuracy).
    """
    rng = np.random.default_rng(seed)
    spec = get_model("mobilebert_tiny")

    seq_len = 32
    hidden = 128
    n_classes = 2

    # Embedding: (vocab=256, hidden) — we just project random input
    W_embed = _he_init((hidden, hidden), rng)

    # Two transformer blocks, each: attn_proj + ffn_up + ffn_down
    blocks = []
    for _ in range(2):
        block = {
            "attn_proj": _he_init((hidden, hidden), rng),  # combined QKV+O
            "ffn_up": _he_init((hidden, 4 * hidden), rng),
            "ffn_down": _he_init((4 * hidden, hidden), rng),
        }
        blocks.append(block)

    W_cls = _he_init((hidden, n_classes), rng)
    b_cls = np.zeros(n_classes, dtype=np.float32)

    def _layer_norm(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return ((x - mean) / np.sqrt(var + 1e-5)).astype(np.float32)

    def _run_backbone(x_seq: np.ndarray) -> np.ndarray:
        """x_seq: (seq_len, hidden) -> (seq_len, hidden)."""
        h = x_seq @ W_embed
        h = _layer_norm(h)
        for block in blocks:
            # Simplified attention: linear proj (approximation)
            attn_out = _relu(h @ block["attn_proj"])
            h = _layer_norm(h + attn_out)
            # FFN
            ffn_mid = _relu(h @ block["ffn_up"])
            ffn_out = ffn_mid @ block["ffn_down"]
            h = _layer_norm(h + ffn_out)
        return h

    def forward_fn(x: np.ndarray, feature_maps_override=None):
        if feature_maps_override is not None:
            fmaps = feature_maps_override
            h_seq = fmaps[0] if fmaps.ndim == 3 else fmaps
        else:
            x_seq = x[0] if x.ndim == 3 else x.reshape(seq_len, hidden)
            h_seq = _run_backbone(x_seq)
            fmaps = h_seq[np.newaxis, ...]  # (1, seq_len, hidden)

        pooled = h_seq.mean(axis=0)  # (hidden,)
        logits = (pooled @ W_cls + b_cls).reshape(1, -1)
        return logits, fmaps

    def backbone_fn(x):
        x_seq = x[0] if x.ndim == 3 else x.reshape(seq_len, hidden)
        h = _run_backbone(x_seq)
        return h[np.newaxis, ...]

    def head_fn(features):
        f = features[0] if features.ndim == 3 else features
        if f.ndim == 2:
            pooled = f.mean(axis=0)
        else:
            pooled = f
        return (pooled @ W_cls + b_cls).reshape(1, -1)

    def head_grad_fn(features, target_class):
        # Mean pool + FC: gradient = W_cls[:, c] / seq_len
        grad_per_token = W_cls[:, target_class] / seq_len
        return np.broadcast_to(grad_per_token, features.shape).astype(np.float32)

    def layer_outputs_fn(x):
        x_seq = x[0] if x.ndim == 3 else x.reshape(seq_len, hidden)
        outputs = [x_seq.reshape(1, -1)]
        h = x_seq @ W_embed
        h = _layer_norm(h)
        outputs.append(h.reshape(1, -1))
        for block in blocks:
            attn_out = _relu(h @ block["attn_proj"])
            h = _layer_norm(h + attn_out)
            outputs.append(h.reshape(1, -1))
            ffn_mid = _relu(h @ block["ffn_up"])
            ffn_out = ffn_mid @ block["ffn_down"]
            h = _layer_norm(h + ffn_out)
            outputs.append(h.reshape(1, -1))
        pooled = h.mean(axis=0).reshape(1, -1)
        outputs.append(pooled)
        logits = (pooled @ W_cls + b_cls).reshape(1, -1)
        outputs.append(logits)
        return outputs

    return ModelRunner(
        spec=spec,
        forward_fn=forward_fn,
        backbone_fn=backbone_fn,
        head_fn=head_fn,
        head_grad_fn=head_grad_fn,
        weights=[W_cls],
        biases=[b_cls],
        layer_outputs_fn=layer_outputs_fn,
        input_shape=(seq_len, hidden),
        feature_shape=(seq_len, hidden),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


RUNNER_FACTORIES: Dict[str, Callable[..., ModelRunner]] = {
    "gap_fc": build_gap_fc,
    "resnet8": build_resnet8,
    "toyadmos": build_toyadmos,
    "mobilebert_tiny": build_mobilebert_tiny,
}


def build_runner(name: str, seed: int = 42) -> ModelRunner:
    """Build a :class:`ModelRunner` by model name.

    Args:
        name: one of ``gap_fc``, ``resnet8``, ``toyadmos``,
              ``mobilebert_tiny``.
        seed: RNG seed for weight initialization.

    Returns:
        A ready-to-use :class:`ModelRunner`.

    Raises:
        KeyError: if ``name`` is not a known model.
    """
    if name not in RUNNER_FACTORIES:
        raise KeyError(
            f"Unknown model {name!r}. Known: {sorted(RUNNER_FACTORIES)}"
        )
    return RUNNER_FACTORIES[name](seed=seed)
