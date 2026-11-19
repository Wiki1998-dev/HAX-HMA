# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Grad-CAM on the SNAX GeMMX accelerator. Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/gradcam_gemmx CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson all
# ResNet-8 hook-layer size (8x8x64): add SIZE=resnet8

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)

ifeq ($(SIZE),resnet8)
APP      ?= gradcam_gemmx_resnet8
DATA_CFG ?= $(DATA_DIR)/params_resnet8.hjson
DATA_H   ?= $(DATA_DIR)/resnet8/data.h
else
APP      ?= gradcam_gemmx
DATA_CFG ?= $(DATA_DIR)/params.hjson
DATA_H   ?= $(DATA_DIR)/data.h
endif

HW_CFG   := $(realpath $(MK_DIR)/../../../target/snitch_cluster/cfg/snax_KUL_cluster.hjson)
SRCS     ?= $(realpath $(SRC_DIR)/main.c)
INCDIRS  ?= $(dir $(DATA_H)) $(SRC_DIR)
DATAGEN_PY = $(DATA_DIR)/datagen.py

$(DATA_H): $(DATAGEN_PY) $(DATA_CFG)
	mkdir -p $(dir $@)
	$< --swcfg $(DATA_CFG) --hwcfg $(HW_CFG) > $@

.DELETE_ON_ERROR:
.PHONY: clean-data clean
clean-data:
	rm -f $(DATA_H)
clean: clean-data
