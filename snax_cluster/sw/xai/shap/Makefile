# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Gradient SHAP kernel for SNAX cluster.

MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DATA_DIR := $(realpath $(MK_DIR)/data)
SRC_DIR  := $(realpath $(MK_DIR)/src)

DATA_CFG ?= $(DATA_DIR)/params.hjson
SECTION  ?=

APP     ?= shap
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
