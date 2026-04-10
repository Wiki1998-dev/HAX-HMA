"""Model catalogue used by the Phase 5 benchmark.

We represent each benchmark model as a flat list of :class:`LayerSpec`
entries.  This is deliberately **architectural-only** — no weights are
allocated — because the benchmark estimates SNAX cycles from the layer
shapes (MACs, spatial extents, parallelism), not from running the model.

Four models are supplied:

* ``gap_fc``  – the Phase 1/2 GAP+FC toy used for calibration
  (H=4, W=4, K=16, C=10).
* ``resnet8`` – MLPerf Tiny ResNet-8 for CIFAR-10.
* ``toyadmos`` – MLPerf Tiny ToyAdmos fully-connected autoencoder
  for anomaly detection on audio.
* ``mobilebert_tiny`` – a shrunk MobileBERT encoder stack suitable for
  embedded NLP (2 transformer blocks, hidden 128, 4 heads).

Each model exposes a ``backbone`` / ``head`` split: the backbone is
everything up to and including the layer Grad-CAM hooks into (the
"final feature layer"); the head is everything after.  Hoisted SHAP
re-runs only the head while reusing cached backbone activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

LayerKind = Literal[
    "conv",
    "depthwise_conv",
    "pool",
    "gap",
    "fc",
    "attention",
    "ffn",
    "layernorm",
    "embedding",
]


@dataclass(frozen=True)
class LayerSpec:
    """Architectural descriptor for a single layer.

    The ``h_out`` / ``w_out`` / ``seq_len`` fields describe the *output*
    tensor shape.  ``h_in`` / ``w_in`` describe the *input* spatial
    extent — for pool / gap layers this is the reduction extent and is
    larger than ``h_out`` / ``w_out``.

    Attributes:
        name:    human-readable layer name
        kind:    layer kind (drives cost-model dispatch)
        h_out:   output spatial height (1 for non-spatial layers)
        w_out:   output spatial width (1 for non-spatial layers)
        h_in:    input spatial height; defaults to ``h_out`` for
                 stride-1 operations
        w_in:    input spatial width; defaults to ``w_out``
        c_in:    input channels / hidden features
        c_out:   output channels / hidden features
        k:       kernel size (1 for non-conv layers)
        seq_len: sequence length for transformer layers (1 otherwise)
        heads:   number of attention heads (0 for non-attention layers)
        is_head: True if this layer belongs to the classification head
    """

    name: str
    kind: LayerKind
    h_out: int = 1
    w_out: int = 1
    h_in: int = 0
    w_in: int = 0
    c_in: int = 1
    c_out: int = 1
    k: int = 1
    seq_len: int = 1
    heads: int = 0
    is_head: bool = False

    @property
    def h_input(self) -> int:
        return self.h_in if self.h_in > 0 else self.h_out

    @property
    def w_input(self) -> int:
        return self.w_in if self.w_in > 0 else self.w_out

    @property
    def macs(self) -> int:
        """Multiply-accumulate operations executed by this layer."""
        h, w = self.h_out, self.w_out
        hi, wi = self.h_input, self.w_input
        if self.kind == "conv":
            return h * w * self.c_in * self.c_out * self.k * self.k
        if self.kind == "depthwise_conv":
            return h * w * self.c_in * self.k * self.k
        if self.kind in ("pool", "gap"):
            # Reduction workload scales with the INPUT extent.
            return hi * wi * self.c_in * max(self.seq_len, 1)
        if self.kind == "fc":
            return self.c_in * self.c_out
        if self.kind == "attention":
            s = self.seq_len
            d = self.c_out
            proj = 4 * s * d * d               # Q, K, V, O projections
            qk = s * s * d                     # scaled dot product
            av = s * s * d                     # attention * V
            return proj + qk + av
        if self.kind == "ffn":
            # Two FC layers with 4x expansion in the middle.
            return 2 * self.seq_len * self.c_in * self.c_out
        if self.kind == "layernorm":
            return 2 * self.seq_len * self.c_in
        if self.kind == "embedding":
            return self.seq_len * self.c_in
        return 0

    @property
    def feature_map_elems(self) -> int:
        """Number of elements in the layer's *input* feature map.

        Grad-CAM and SHAP operate on this many activations when hooking
        into the layer — e.g. a 4×4×16 conv output has 256 elements.
        """
        h = self.h_input
        w = self.w_input
        return h * w * self.c_in * max(self.seq_len, 1)

    @property
    def activation_elems(self) -> int:
        """Number of elements in this layer's output tensor."""
        return self.h_out * self.w_out * self.c_out * max(self.seq_len, 1)


@dataclass(frozen=True)
class ModelSpec:
    """Complete benchmark model descriptor.

    Attributes:
        name:           model identifier
        display_name:   human-readable name for tables
        input_shape:    input tensor shape (H, W, C) or (seq_len, hidden)
        n_classes:      number of output classes
        layers:         ordered list of layers
        backbone_end:   index of the *final feature* layer — the last
                        layer Grad-CAM hooks into.  Layers at indices
                        ``> backbone_end`` form the classification head.
        head_is_linear: True when the head is a single linear
                        transformation of the feature maps (GAP + FC
                        or pooling + FC).  Used to pick between the
                        fast Phase 2b hoisting variant (linear head,
                        backward computed once) and the slower general
                        variant (nonlinear head, backward re-run per
                        SHAP sample).
    """

    name: str
    display_name: str
    input_shape: Tuple[int, ...]
    n_classes: int
    layers: List[LayerSpec]
    backbone_end: int
    head_is_linear: bool = True

    @property
    def backbone(self) -> List[LayerSpec]:
        """Layers up to and including the Grad-CAM hook layer."""
        return self.layers[: self.backbone_end + 1]

    @property
    def head(self) -> List[LayerSpec]:
        """Layers after the Grad-CAM hook layer (the re-run head)."""
        return self.layers[self.backbone_end + 1 :]

    @property
    def final_feature_layer(self) -> LayerSpec:
        """The last layer whose activations are the XAI hook point."""
        return self.layers[self.backbone_end]

    @property
    def total_macs(self) -> int:
        return sum(layer.macs for layer in self.layers)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def _gap_fc_model() -> ModelSpec:
    """Phase 1 calibration model: 4×4×16 feature map → GAP → FC(10)."""
    layers = [
        LayerSpec("feature_maps", "conv", h_out=4, w_out=4, c_in=16, c_out=16, k=3),
        LayerSpec("gap", "gap", h_out=1, w_out=1, h_in=4, w_in=4, c_in=16, c_out=16),
        LayerSpec("fc", "fc", c_in=16, c_out=10, is_head=True),
    ]
    # Hook into the feature_maps conv (index 0); head = gap + fc.
    return ModelSpec(
        name="gap_fc",
        display_name="GAP+FC (Test)",
        input_shape=(4, 4, 16),
        n_classes=10,
        layers=layers,
        backbone_end=0,
    )


def _resnet8_model() -> ModelSpec:
    """MLPerf Tiny ResNet-8 for CIFAR-10.

    Architecture (simplified — matches parameter count of the MLPerf
    reference within a few percent):

        conv1  : 3x3, 3→16,  32x32
        block1 : 3x3, 16→16, 32x32 (two conv layers)
        block2 : 3x3, 16→32, 16x16 (stride 2) + 3x3, 32→32
        block3 : 3x3, 32→64, 8x8  (stride 2) + 3x3, 64→64
        gap    : 8x8 → 1x1
        fc     : 64 → 10
    """
    layers = [
        LayerSpec("conv1",    "conv", 32, 32, c_in= 3, c_out=16, k=3),
        LayerSpec("block1_a", "conv", 32, 32, c_in=16, c_out=16, k=3),
        LayerSpec("block1_b", "conv", 32, 32, c_in=16, c_out=16, k=3),
        LayerSpec("block2_a", "conv", 16, 16, c_in=16, c_out=32, k=3),
        LayerSpec("block2_b", "conv", 16, 16, c_in=32, c_out=32, k=3),
        LayerSpec("block3_a", "conv",  8,  8, c_in=32, c_out=64, k=3),
        LayerSpec("block3_b", "conv",  8,  8, c_in=64, c_out=64, k=3),
        LayerSpec("gap",      "gap",   1,  1, h_in=8, w_in=8, c_in=64, c_out=64),
        LayerSpec("fc",       "fc", c_in=64, c_out=10, is_head=True),
    ]
    return ModelSpec(
        name="resnet8",
        display_name="ResNet-8 (Image)",
        input_shape=(32, 32, 3),
        n_classes=10,
        layers=layers,
        backbone_end=6,  # final conv before GAP
    )


def _toyadmos_model() -> ModelSpec:
    """MLPerf Tiny ToyAdmos dense autoencoder (anomaly detection).

    Input: 640-dim log-mel spectrogram frame.
    Architecture: FC(128)×4 → FC(8) (bottleneck) → FC(128)×4 → FC(640).
    Saliency hooks on the bottleneck — XAI explains *why* a frame was
    flagged anomalous by inspecting the bottleneck activation pattern.
    """
    layers = [
        LayerSpec("enc1",        "fc", c_in=640, c_out=128),
        LayerSpec("enc2",        "fc", c_in=128, c_out=128),
        LayerSpec("enc3",        "fc", c_in=128, c_out=128),
        LayerSpec("enc4",        "fc", c_in=128, c_out=128),
        LayerSpec("bottleneck",  "fc", c_in=128, c_out=8),
        LayerSpec("dec1",        "fc", c_in=8,   c_out=128),
        LayerSpec("dec2",        "fc", c_in=128, c_out=128),
        LayerSpec("dec3",        "fc", c_in=128, c_out=128),
        LayerSpec("dec4",        "fc", c_in=128, c_out=128),
        LayerSpec("recon",       "fc", c_in=128, c_out=640, is_head=True),
    ]
    return ModelSpec(
        name="toyadmos",
        display_name="ToyAdmos (Audio)",
        input_shape=(640,),
        n_classes=2,   # normal / anomaly
        layers=layers,
        backbone_end=4,  # bottleneck is the hook layer
    )


def _mobilebert_tiny_model() -> ModelSpec:
    """Shrunk MobileBERT encoder for embedded NLP (seq_len=32, hidden=128).

    Two transformer blocks (attention + FFN + layernorm) + pooled FC.
    Intended to represent the MLPerf Tiny "NLP" workload class.
    """
    seq = 32
    hidden = 128
    layers = [
        LayerSpec("embed", "embedding", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("ln0",   "layernorm", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("attn1", "attention", c_in=hidden, c_out=hidden, seq_len=seq, heads=4),
        LayerSpec("ln1",   "layernorm", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("ffn1",  "ffn",       c_in=hidden, c_out=4 * hidden, seq_len=seq),
        LayerSpec("ln2",   "layernorm", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("attn2", "attention", c_in=hidden, c_out=hidden, seq_len=seq, heads=4),
        LayerSpec("ln3",   "layernorm", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("ffn2",  "ffn",       c_in=hidden, c_out=4 * hidden, seq_len=seq),
        LayerSpec("ln4",   "layernorm", c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("pool",  "gap",       c_in=hidden, c_out=hidden, seq_len=seq),
        LayerSpec("cls",   "fc",        c_in=hidden, c_out=2, is_head=True),
    ]
    # Hook the final hidden state (ln4) as the "feature map".
    return ModelSpec(
        name="mobilebert_tiny",
        display_name="MobileBERT (NLP)",
        input_shape=(seq, hidden),
        n_classes=2,
        layers=layers,
        backbone_end=9,
    )


MODEL_CATALOG: Dict[str, ModelSpec] = {
    "gap_fc": _gap_fc_model(),
    "resnet8": _resnet8_model(),
    "toyadmos": _toyadmos_model(),
    "mobilebert_tiny": _mobilebert_tiny_model(),
}


def get_model(name: str) -> ModelSpec:
    """Fetch a model from the catalogue by name.

    Args:
        name: model identifier (one of :data:`MODEL_CATALOG` keys)

    Returns:
        The corresponding :class:`ModelSpec`.

    Raises:
        KeyError: if ``name`` is not a known model.
    """
    if name not in MODEL_CATALOG:
        raise KeyError(
            f"Unknown model {name!r}. Known models: {sorted(MODEL_CATALOG)}"
        )
    return MODEL_CATALOG[name]
