# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# ToyAdmos (MLPerf Tiny AD, ad01_int8) inference + XAI on GeMMX, driven by
# launch tables generated and emulated bit-exactly in data/.
# Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/toyadmos_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson PROG=<prog> all
# PROG: test_patterns (launch-pattern hardware check),
#       toyadmos_inference, toyadmos_ig, toyadmos_shap (real ad01_int8 + ToyCar data;
#       needs scripts/toyadmos_data.py run once for the cached features)

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)
PROG     ?= test_patterns

APP      := toyadmos_gemmx_$(PROG)
DATA_H   := $(DATA_DIR)/$(PROG)/data.h
ifneq ($(filter toyadmos_%,$(PROG)),)
SRCS     := $(realpath $(SRC_DIR)/toyadmos_run.c)
EXTRA_DEPS := $(DATA_DIR)/toyadmos.py $(DATA_DIR)/ad01_int.py
else
SRCS     := $(realpath $(SRC_DIR)/$(PROG).c)
endif
INCDIRS  := $(dir $(DATA_H)) $(SRC_DIR)
DATAGEN_PY := $(DATA_DIR)/$(PROG).py
# Project venv (numpy, tflite-runtime); pixi may define PYTHON as empty.
TA_PYTHON := $(abspath $(MK_DIR)/../../../../.venv/bin/python)

$(DATA_H): $(DATAGEN_PY) $(DATA_DIR)/gemmx_sched.py $(DATA_DIR)/emit.py $(EXTRA_DEPS)
	mkdir -p $(dir $@)
	cd $(DATA_DIR) && env -u PYTHONHOME -u PYTHONPATH $(TA_PYTHON) $(DATAGEN_PY) > $@

.DELETE_ON_ERROR:
.PHONY: clean-data
clean-data:
	rm -rf $(DATA_DIR)/*/data.h
