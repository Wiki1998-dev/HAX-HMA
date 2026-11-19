# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# One ResNet-8 inference layer on GeMMX. Needs the snax_KUL_cluster simulator.
#   make -C target/snitch_cluster/sw/apps/xai/inference_gemmx \
#        CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson LAYER=block3_b all
# Layers: params/*.hjson. Data comes from the stock snax-gemmx-conv generator.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SRC_DIR  := $(realpath $(MK_DIR)/src)
LAYER    ?= block3_b

APP      := inference_gemmx_$(LAYER)
DATA_CFG := $(realpath $(MK_DIR)/params/$(LAYER).hjson)
DATA_H   := $(MK_DIR)data/$(LAYER)/data.h
HW_CFG   := $(realpath $(MK_DIR)/../../../target/snitch_cluster/cfg/snax_KUL_cluster.hjson)
DATAGEN_PY := $(realpath $(MK_DIR)/../../../target/snitch_cluster/sw/apps/snax-gemmx-conv/data/datagen.py)
SRCS     := $(realpath $(SRC_DIR)/main.c)
INCDIRS  := $(dir $(DATA_H)) $(SRC_DIR)

RISCV_CFLAGS += $(EXTRA_CFLAGS)
ifeq ($(LAYER),fc)
RISCV_CFLAGS += -DMEASURE_GAP
endif

$(DATA_H): $(DATAGEN_PY) $(DATA_CFG)
	mkdir -p $(dir $@)
	$< --swcfg $(DATA_CFG) --hwcfg $(HW_CFG) > $@

.DELETE_ON_ERROR:
.PHONY: clean-data
clean-data:
	rm -rf $(MK_DIR)data
