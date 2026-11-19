# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Grad-CAM kernel on real MLPerf Tiny ResNet-8 data.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)

# Reuse shared gradcam.h kernel header from the Phase 1 gradcam directory
GRADCAM_INC_DIR := $(realpath $(MK_DIR)/../gradcam/src)

APP     ?= gradcam_resnet8
SRCS    ?= $(realpath $(SRC_DIR)/main.c)
INCDIRS ?= $(DATA_DIR) $(SRC_DIR) $(GRADCAM_INC_DIR)

DATAGEN_PY = $(DATA_DIR)/datagen.py
DATA_H     = $(DATA_DIR)/data.h

$(DATA_H): $(DATAGEN_PY)
	$< > $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data