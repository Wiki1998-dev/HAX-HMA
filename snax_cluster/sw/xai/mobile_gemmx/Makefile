# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# MLPerf Tiny keyword spotting (DS-CNN) / visual wake words (MobileNetV1)
# inference + fused Grad-CAM on GeMMX (real models, real inputs). Needs the
# snax_KUL_cluster simulator and the prepared inputs (scripts/kws_vww_data.py).
#   make -C target/snitch_cluster/sw/apps/xai/mobile_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson MB_TASK=kws|vww [MB_SAMPLE=k] all

MK_DIR    := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR  := $(realpath $(MK_DIR)/data)
SRC_DIR   := $(realpath $(MK_DIR)/src)
MB_TASK   ?= kws
MB_SAMPLE ?= 0

APP      := mobile_$(MB_TASK)_s$(MB_SAMPLE)
DATA_H   := $(DATA_DIR)/$(MB_TASK)_s$(MB_SAMPLE)/data.h
SRCS     := $(realpath $(SRC_DIR)/mobile_run.c)
INCDIRS  := $(dir $(DATA_H)) $(SRC_DIR)
# Project venv (numpy, tflite-runtime); pixi may define PYTHON as empty.
MB_PYTHON := $(abspath $(MK_DIR)/../../../../.venv/bin/python)
DEPS     := $(DATA_DIR)/mobile.py $(DATA_DIR)/mobile_int.py \
            $(realpath $(MK_DIR)/../toyadmos_gemmx/data/gemmx_sched.py)

$(DATA_H): $(DEPS)
	mkdir -p $(dir $@)
	cd $(DATA_DIR) && env -u PYTHONHOME -u PYTHONPATH MB_TASK=$(MB_TASK) MB_SAMPLE=$(MB_SAMPLE) \
	  $(MB_PYTHON) mobile.py > $@

.DELETE_ON_ERROR:
.PHONY: clean-data
clean-data:
	rm -rf $(DATA_DIR)/kws_s*/data.h $(DATA_DIR)/vww_s*/data.h
