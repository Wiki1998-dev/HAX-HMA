#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Generate Grad-CAM test data from real MLPerf Tiny ResNet-8 model.
#
# Extracts real weights and feature maps from the quantized tflite,
# dequantizes to float32, and computes golden Grad-CAM output.
# The SNAX kernel receives real data — not synthetic random values.

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import (emit_license, format_scalar_definition,  # noqa: E402
                        format_vector_definition, format_ifdef_wrapper)

BURST_ALIGNMENT = 4096

# Paths relative to repo root (snax_cluster/..)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../../.."))
TFLITE_PATH = os.path.join(
    REPO_ROOT,
    "CFU-Playground/third_party/mlcommons/tiny/v0.1/"
    "training/image_classification/trained_models/pretrainedResnet_quant.tflite",
)
INPUT_PATH = os.path.join(REPO_ROOT, "data/benchmark_inputs/resnet8_input.npy")

# TFLite tensor indices for ResNet-8 (pretrainedResnet_quant.tflite)
FMAP_TENSOR_IDX = 33   # model/activation_6/Relu;model/add_2/add  [1,8,8,64]
FC_WEIGHT_TENSOR_IDX = 7   # model/dense/MatMul  [10,64]


def load_real_model_data():
    """Run real inference on the tflite model and extract hook-point data.

    Returns:
        feature_maps: (H, W, K) float32 — dequantized last-conv activations
        w_fc:         (K, C) float32 — dequantized FC weight matrix
        target_class: int — predicted class from real inference
    """
    import tflite_runtime.interpreter as tflite

    interp = tflite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()

    inp_detail = interp.get_input_details()[0]

    # Load and quantize real input
    real_input = np.load(INPUT_PATH)  # (32, 32, 3) float32
    scale_in, zp_in = inp_detail["quantization"]
    input_int8 = np.clip(
        np.round(real_input / scale_in + zp_in), -128, 127
    ).astype(np.int8)
    input_batch = input_int8.reshape(1, 32, 32, 3)

    # Run real inference
    interp.set_tensor(inp_detail["index"], input_batch)
    interp.invoke()

    # Predicted class
    output = interp.get_tensor(interp.get_output_details()[0]["index"])
    target_class = int(np.argmax(output))

    # Extract real feature maps at Grad-CAM hook point
    fmap_detail = interp.get_tensor_details()[FMAP_TENSOR_IDX]
    fmap_int8 = interp.get_tensor(FMAP_TENSOR_IDX)  # [1, 8, 8, 64] int8
    fmap_scale = fmap_detail["quantization_parameters"]["scales"][0]
    fmap_zp = fmap_detail["quantization_parameters"]["zero_points"][0]
    feature_maps = (fmap_int8[0].astype(np.float32) - fmap_zp) * fmap_scale

    # Extract real FC weights
    fc_detail = interp.get_tensor_details()[FC_WEIGHT_TENSOR_IDX]
    fc_int8 = interp.get_tensor(FC_WEIGHT_TENSOR_IDX)  # [10, 64] int8
    fc_scale = fc_detail["quantization_parameters"]["scales"][0]
    fc_zp = fc_detail["quantization_parameters"]["zero_points"][0]
    fc_f32 = (fc_int8.astype(np.float32) - fc_zp) * fc_scale
    # tflite stores FC as (C, K), kernel expects (K, C)
    w_fc = fc_f32.T

    return feature_maps, w_fc, target_class


def gradcam_golden(feature_maps, w_fc, target_class):
    """Compute golden Grad-CAM matching the SNAX kernel logic exactly.

    Args:
        feature_maps: (H, W, K) float32
        w_fc:         (K, C) float32
        target_class: int

    Returns:
        cam: (H, W) float32, values in [0, 1]
    """
    H, W, K = feature_maps.shape
    C = w_fc.shape[1]

    # One-hot d_logit
    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Backward GeMM: grad = d_logit @ W_fc_T
    w_fc_t = w_fc.T  # (C, K)
    grad = d_logit @ w_fc_t  # (K,)

    # GAP (grad is spatially uniform, so alpha = grad)
    alpha = grad

    # Weighted sum
    cam = np.einsum("hwk,k->hw", feature_maps, alpha)

    # ReLU + normalize
    cam = np.maximum(cam, 0.0)
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    return cam.astype(np.float32)


def emit_header(**kwargs):
    section = kwargs.get("section")

    feature_maps, w_fc, target_class = load_real_model_data()
    H, W, K = feature_maps.shape
    C = w_fc.shape[1]

    # One-hot d_logit
    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Golden output
    cam_golden = gradcam_golden(feature_maps, w_fc, target_class)

    # Emit C header
    data_str = [emit_license()]
    data_str += [f"// Real data from MLPerf Tiny ResNet-8 (pretrainedResnet_quant.tflite)"]
    data_str += [f"// Input: data/benchmark_inputs/resnet8_input.npy"]
    data_str += [f"// Predicted class: {target_class}"]
    data_str += [format_scalar_definition("uint32_t", "H", H)]
    data_str += [format_scalar_definition("uint32_t", "W", W)]
    data_str += [format_scalar_definition("uint32_t", "K_CH", K)]
    data_str += [format_scalar_definition("uint32_t", "N_CLASSES", C)]
    data_str += [format_scalar_definition("uint32_t", "TARGET_CLASS", target_class)]
    data_str += [
        format_vector_definition(
            "float", "feature_maps", feature_maps.flatten(),
            alignment=BURST_ALIGNMENT, section=section,
        )
    ]
    data_str += [
        format_vector_definition(
            "float", "w_fc", w_fc.flatten(),
            alignment=BURST_ALIGNMENT, section=section,
        )
    ]
    data_str += [
        format_vector_definition(
            "float", "d_logit", d_logit.flatten(),
            alignment=BURST_ALIGNMENT, section=section,
        )
    ]
    data_str += [
        format_vector_definition(
            "float", "cam_out", np.zeros(H * W, dtype=np.float32),
            alignment=BURST_ALIGNMENT, section=section,
        )
    ]
    golden_def = format_vector_definition("float", "cam_golden", cam_golden.flatten())
    data_str += [format_ifdef_wrapper("BIST", golden_def)]

    return "\n\n".join(data_str)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM data from real ResNet-8 tflite model")
    parser.add_argument("-c", "--cfg", type=pathlib.Path,
                        help="Optional param config file (hjson)")
    parser.add_argument("--section", type=str,
                        help="Section to store arrays in")
    args = parser.parse_args()

    kwargs = {}
    if args.cfg:
        with args.cfg.open() as f:
            kwargs = hjson.loads(f.read())
    kwargs["section"] = args.section

    print(emit_header(**kwargs))


if __name__ == "__main__":
    main()