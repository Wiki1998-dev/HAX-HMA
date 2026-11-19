# Copyright 2024 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

include ../../../../../../sw/xai/toyadmos_gemmx/Makefile
INCDIRS += ../../../snax/gemmx/include
RISCV_LDFLAGS += $(abspath ../../../snax/gemmx/build/snax-gemmx-lib.o)
include ../../common.mk
$(DEP): $(DATA_H)
