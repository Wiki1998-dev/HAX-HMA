# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Grad-CAM kernel for SNAX cluster.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)

DATA_CFG ?= $(DATA_DIR)/params.hjson
SECTION  ?=

APP     ?= gradcam
SRCS    ?= $(realpath $(SRC_DIR)/main.c)
INCDIRS ?= $(DATA_DIR) $(SRC_DIR)

DATAGEN_PY = $(DATA_DIR)/datagen.py
DATA_H     = $(DATA_DIR)/data.h

$(DATA_H): $(DATAGEN_PY) $(DATA_CFG)
	$< -c $(DATA_CFG) --section="$(SECTION)" > $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data

# INT8-weight backward GeMM variant (builds build/gradcam_i8.elf):
#   make -C target/snitch_cluster/sw/apps/xai/gradcam GEMM_I8=1 all
ifeq ($(GEMM_I8),1)
APP := gradcam_i8
RISCV_CFLAGS += -DGEMM_I8
endif
