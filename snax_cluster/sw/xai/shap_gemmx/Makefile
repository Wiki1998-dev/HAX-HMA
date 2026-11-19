# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Gradient SHAP on the SNAX GeMMX accelerator. Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/shap_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson all

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)

APP      ?= shap_gemmx
DATA_CFG ?= $(DATA_DIR)/params.hjson
DATA_H   ?= $(DATA_DIR)/data.h

HW_CFG   := $(realpath $(MK_DIR)/../../../target/snitch_cluster/cfg/snax_KUL_cluster.hjson)
SRCS     ?= $(realpath $(SRC_DIR)/main.c)
INCDIRS  ?= $(dir $(DATA_H)) $(SRC_DIR)
DATAGEN_PY = $(DATA_DIR)/datagen.py

$(DATA_H): $(DATAGEN_PY) $(DATA_CFG) $(MK_DIR)/../gradcam_gemmx/data/datagen.py
	mkdir -p $(dir $@)
	$< --swcfg $(DATA_CFG) --hwcfg $(HW_CFG) > $@

.DELETE_ON_ERROR:
.PHONY: clean-data clean
clean-data:
	rm -f $(DATA_H)
clean: clean-data
