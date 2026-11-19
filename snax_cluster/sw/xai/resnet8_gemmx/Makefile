# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# MLPerf Tiny ResNet-8 inference + fused Grad-CAM on GeMMX (real model, real
# CIFAR-10 image). Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/resnet8_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson [R8_SAMPLE=k] all
# R8_SAMPLE: index into the 200 MLPerf performance samples (default 0).
# Input-level XAI instead of inference + Grad-CAM: R8_METHOD=ig|shap [R8X_N=16]

MK_DIR    := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR  := $(realpath $(MK_DIR)/data)
SRC_DIR   := $(realpath $(MK_DIR)/src)
R8_SAMPLE ?= 0

ifeq ($(R8_METHOD),)
APP      := resnet8_gemmx_s$(R8_SAMPLE)
DATA_H   := $(DATA_DIR)/s$(R8_SAMPLE)/data.h
SRCS     := $(realpath $(SRC_DIR)/resnet8_run.c)
GEN      := resnet8.py
else
R8X_N    ?= 16
APP      := resnet8_$(R8_METHOD)_n$(R8X_N)_s$(R8_SAMPLE)
DATA_H   := $(DATA_DIR)/$(R8_METHOD)_n$(R8X_N)_s$(R8_SAMPLE)/data.h
SRCS     := $(realpath $(SRC_DIR)/resnet8_xai_run.c)
GEN      := resnet8_xai_sched.py
endif
INCDIRS  := $(dir $(DATA_H)) $(SRC_DIR)
# Project venv (numpy, tflite-runtime); pixi may define PYTHON as empty.
R8_PYTHON := $(abspath $(MK_DIR)/../../../../.venv/bin/python)
DEPS     := $(DATA_DIR)/resnet8.py $(DATA_DIR)/resnet8_int.py $(DATA_DIR)/resnet8_xai.py \
            $(DATA_DIR)/resnet8_xai_sched.py \
            $(realpath $(MK_DIR)/../toyadmos_gemmx/data/gemmx_sched.py)

$(DATA_H): $(DEPS)
	mkdir -p $(dir $@)
	cd $(DATA_DIR) && env -u PYTHONHOME -u PYTHONPATH R8_SAMPLE=$(R8_SAMPLE) R8X_METHOD=$(R8_METHOD) \
	  R8X_N=$(R8X_N) $(R8_PYTHON) $(GEN) > $@

.DELETE_ON_ERROR:
.PHONY: clean-data
clean-data:
	rm -rf $(DATA_DIR)/s*/data.h $(DATA_DIR)/ig_*/data.h $(DATA_DIR)/shap_*/data.h
