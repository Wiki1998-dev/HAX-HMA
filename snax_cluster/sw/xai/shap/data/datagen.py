#!/usr/bin/env python3
# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Generate test data + golden Gradient SHAP output for SNAX simulation.

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


def shap_golden(feature_maps: np.ndarray, w_fc: np.ndarray,
                target_class: int, baselines: np.ndarray,
                alphas: np.ndarray) -> np.ndarray:
    """Compute golden Gradient SHAP output matching the C kernel logic.

    For the GAP+FC model, gradient is constant:
        grad[h,w,k] = w_fc[k, target_class] / (H*W)

    SHAP attribution:
        attr[h,w,k] = (1/N) * sum_i (fmaps[h,w,k] - baselines[i,h,w,k]) * grad[k]

    Args:
        feature_maps: (h, w, K) float32
        w_fc:         (K, C) float32
        target_class: class index
        baselines:    (N, h, w, K) float32
        alphas:       (N,) float32 — interpolation factors (used in C kernel
                      but gradient is input-independent for linear model)

    Returns:
        attr: (h, w, K) float32
    """
    h, w, K = feature_maps.shape
    N = baselines.shape[0]

    # Gradient is constant for linear GAP+FC model
    grad = w_fc[:, target_class] / (h * w)  # (K,)

    # Accumulate over samples
    attr = np.zeros((h, w, K), dtype=np.float32)
    for i in range(N):
        diff = feature_maps - baselines[i]  # (h, w, K)
        attr += diff * grad[np.newaxis, np.newaxis, :]

    attr /= N
    return attr.astype(np.float32)


def emit_header(**kwargs):
    """Generate C header with test data and golden output."""
    h = kwargs['h']
    w = kwargs['w']
    K = kwargs['K']
    C = kwargs['C']
    target_class = kwargs['target_class']
    n_samples = kwargs['n_samples']
    tolerance = kwargs.get('tolerance', 1e-2)
    section = kwargs.get('section')

    # Generate random inputs
    feature_maps = np.random.rand(h, w, K).astype(np.float32)
    w_fc = np.random.rand(K, C).astype(np.float32) * 0.2 - 0.1  # [-0.1, 0.1]

    # Generate random baselines and interpolation factors
    baselines = np.random.randn(n_samples, h, w, K).astype(np.float32) * 0.1
    alphas = np.random.rand(n_samples).astype(np.float32)

    # Golden output
    attr_golden = shap_golden(feature_maps, w_fc, target_class, baselines, alphas)

    # Emit C header
    data_str = [emit_license()]
    data_str += [format_scalar_definition('uint32_t', 'H', h)]
    data_str += [format_scalar_definition('uint32_t', 'W', w)]
    data_str += [format_scalar_definition('uint32_t', 'K_CH', K)]
    data_str += [format_scalar_definition('uint32_t', 'N_CLASSES', C)]
    data_str += [format_scalar_definition('uint32_t', 'TARGET_CLASS',
                                          target_class)]
    data_str += [format_scalar_definition('uint32_t', 'N_SAMPLES', n_samples)]
    data_str += [format_scalar_definition('float', 'TOLERANCE', tolerance)]

    data_str += [format_vector_definition('float', 'feature_maps',
                                          feature_maps.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    data_str += [format_vector_definition('float', 'w_fc',
                                          w_fc.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    data_str += [format_vector_definition('float', 'baselines',
                                          baselines.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]
    data_str += [format_vector_definition('float', 'alphas',
                                          alphas.flatten(),
                                          alignment=BURST_ALIGNMENT,
                                          section=section)]

    # Output buffer
    attr_out_def = format_vector_definition(
        'float', 'attr_out',
        np.zeros(h * w * K, dtype=np.float32),
        alignment=BURST_ALIGNMENT, section=section)
    data_str += [attr_out_def]

    # Golden reference under BIST guard
    golden_def = format_vector_definition('float', 'attr_golden',
                                          attr_golden.flatten())
    data_str += [format_ifdef_wrapper('BIST', golden_def)]

    return '\n\n'.join(data_str)


def main():
    parser = argparse.ArgumentParser(
        description='Generate data for Gradient SHAP kernel')
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
