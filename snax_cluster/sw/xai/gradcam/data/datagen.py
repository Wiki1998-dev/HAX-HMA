#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Generate test data + golden Grad-CAM output for SNAX simulation.

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../util/sim/"))
from data_utils import (emit_license, format_scalar_definition,  # noqa: E402
                        format_vector_definition, format_ifdef_wrapper)

np.random.seed(42)

BURST_ALIGNMENT = 4096


def gradcam_golden(feature_maps: np.ndarray, w_fc: np.ndarray,
                   target_class: int) -> np.ndarray:
    """Compute golden Grad-CAM output matching gradcam_reference.py logic.

    Args:
        feature_maps: (h, w, K) float32
        w_fc:         (K, C) float32 — FC weight matrix
        target_class: class index to explain

    Returns:
        cam: (h, w) float32, values in [0, 1]
    """
    h, w, K = feature_maps.shape
    # Gradient of logit[target_class] w.r.t. feature_maps via FC weights.
    # After GAP: logit[c] = sum_k (mean_{ij} fmaps[i,j,k]) * w_fc[k,c]
    # So d logit[c] / d fmaps[i,j,k] = w_fc[k, c] / (h*w)
    # => alpha[k] = mean over spatial of grad = w_fc[k, target_class] / (h*w)
    #    then multiply by (h*w) from GAP cancels, leaving alpha[k] = w_fc[k,c]
    # But our kernel computes the full chain, so replicate that:

    # d_logit: one-hot
    C = w_fc.shape[1]
    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Backward GeMM: grad[k] = sum_c d_logit[c] * w_fc_T[c, k]
    #              = w_fc[k, target_class]  (since d_logit is one-hot)
    grad = d_logit @ w_fc.T  # shape (K,) — but this is spatially uniform

    # Broadcast grad to (h, w, K) for the GAP step
    grad_spatial = np.broadcast_to(grad, (h, w, K))

    # GAP over spatial dims
    alpha = np.mean(grad_spatial, axis=(0, 1))  # (K,)

    # Weighted sum
    cam = np.einsum("hwk,k->hw", feature_maps, alpha)

    # ReLU + normalize
    cam = np.maximum(cam, 0.0)
    cam_max = cam.max()
    if cam_max > 0:
        cam = cam / cam_max

    return cam.astype(np.float32)


def emit_header(**kwargs):
    h = kwargs['h']
    w = kwargs['w']
    K = kwargs['K']
    C = kwargs['C']
    target_class = kwargs['target_class']
    section = kwargs.get('section')

    # Generate random inputs
    feature_maps = np.random.rand(h, w, K).astype(np.float32)
    w_fc = np.random.rand(K, C).astype(np.float32)

    # One-hot d_logit
    d_logit = np.zeros(C, dtype=np.float32)
    d_logit[target_class] = 1.0

    # Golden output
    cam_golden = gradcam_golden(feature_maps, w_fc, target_class)

    # Emit C header
    data_str = [emit_license()]
    data_str += [format_scalar_definition('uint32_t', 'H', h)]
    data_str += [format_scalar_definition('uint32_t', 'W', w)]
    data_str += [format_scalar_definition('uint32_t', 'K_CH', K)]
    data_str += [format_scalar_definition('uint32_t', 'N_CLASSES', C)]
    data_str += [format_scalar_definition('uint32_t', 'TARGET_CLASS',
                                          target_class)]
    data_str += [format_vector_definition('float', 'feature_maps',
                                          feature_maps.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    data_str += [format_vector_definition('float', 'w_fc',
                                          w_fc.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    data_str += [format_vector_definition('float', 'd_logit',
                                          d_logit.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    # cam_out: destination buffer in DRAM for DMA write-back
    cam_out_def = format_vector_definition('float', 'cam_out',
                                           np.zeros(h * w, dtype=np.float32),
                                           alignment=BURST_ALIGNMENT,
                                           section=section)
    data_str += [cam_out_def]
    # Golden reference under BIST guard
    golden_def = format_vector_definition('float', 'cam_golden',
                                          cam_golden.flatten())
    data_str += [format_ifdef_wrapper('BIST', golden_def)]

    return '\n\n'.join(data_str)


def main():
    parser = argparse.ArgumentParser(
        description='Generate data for Grad-CAM kernel')
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True,
                        help='Param config file (hjson)')
    parser.add_argument("--section", type=str,
                        help='Section to store arrays in')
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())
    param['section'] = args.section

    print(emit_header(**param))


if __name__ == '__main__':
    main()
