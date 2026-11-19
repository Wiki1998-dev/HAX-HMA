# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# MLPerf Tiny ResNet-8 inference + fused Grad-CAM on GeMMX (real model, real
# CIFAR-10 image). Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/resnet8_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson [R8_SAMPLE=k] all
# R8_SAMPLE: index into the 200 MLPerf performance samples (default 0).

MK_DIR    := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR  := $(realpath $(MK_DIR)/data)
SRC_DIR   := $(realpath $(MK_DIR)/src)
R8_SAMPLE ?= 0

APP      := resnet8_gemmx_s$(R8_SAMPLE)
DATA_H   := $(DATA_DIR)/s$(R8_SAMPLE)/data.h
SRCS     := $(realpath $(SRC_DIR)/resnet8_run.c)
INCDIRS  := $(dir $(DATA_H)) $(SRC_DIR)
# Project venv (numpy, tflite-runtime); pixi may define PYTHON as empty.
R8_PYTHON := $(abspath $(MK_DIR)/../../../../.venv/bin/python)
DEPS     := $(DATA_DIR)/resnet8.py $(DATA_DIR)/resnet8_int.py \
            $(realpath $(MK_DIR)/../toyadmos_gemmx/data/gemmx_sched.py)

$(DATA_H): $(DEPS)
	mkdir -p $(dir $@)
	cd $(DATA_DIR) && env -u PYTHONHOME -u PYTHONPATH R8_SAMPLE=$(R8_SAMPLE) $(R8_PYTHON) resnet8.py > $@

.DELETE_ON_ERROR:
.PHONY: clean-data
clean-data:
	rm -rf $(DATA_DIR)/s*/data.h
