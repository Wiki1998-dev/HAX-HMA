#!/usr/bin/env bash
# Build + simulate every ResNet-8 layer of sw/xai/inference_gemmx on the GeMMX
# cluster and print one result line per layer. Needs the GeMMX mode active
# (scripts/snax_cluster_mode.sh gemmx).
#
# Usage: scripts/run_inference_gemmx.sh [layer ...]
#        MODEL=toyadmos scripts/run_inference_gemmx.sh   (default MODEL=resnet8)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAX="$ROOT/snax_cluster"
APP="target/snitch_cluster/sw/apps/xai/inference_gemmx"
PARAMS="$SNAX/sw/xai/inference_gemmx/params"
SIM_TIMEOUT="${SIM_TIMEOUT:-600}"

grep -q "NR_CORES 2" "$SNAX/target/snitch_cluster/sw/runtime/common/snitch_cluster_cfg.h" ||
  { echo "GeMMX mode not active: run scripts/snax_cluster_mode.sh gemmx" >&2; exit 1; }

layers=("$@")
if [ ${#layers[@]} -eq 0 ]; then
  case "${MODEL:-resnet8}" in
    resnet8)  layers=(conv1 block1_a block1_b block2_a block2_b block3_a block3_b fc) ;;
    toyadmos) layers=(ta_enc1 ta_h128 ta_bottleneck ta_dec1 ta_recon) ;;
    *) echo "unknown MODEL=$MODEL" >&2; exit 2 ;;
  esac
fi

cd "$SNAX"
for L in "${layers[@]}"; do
  launches=$(grep -oE 'launches_per_layer: [0-9]+' "$PARAMS/$L.hjson" | grep -oE '[0-9]+$')
  rm -rf "$APP/build"
  if ! pixi run make -s -C "$APP" CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson LAYER="$L" all >/dev/null 2>"${TMPDIR:-/tmp}/inference_gemmx_$L.build.log"; then
    echo "$L BUILD-FAILED (see ${TMPDIR:-/tmp}/inference_gemmx_$L.build.log)"; continue
  fi
  out=$(cd target/snitch_cluster && timeout "$SIM_TIMEOUT" ./bin/snitch_cluster.vlt \
        "$SNAX/$APP/build/inference_gemmx_$L.elf" 2>&1)
  rc=$?
  rm -f target/snitch_cluster/logs/trace_chip_*.dasm
  count=$(grep -oE 'layers_with_this_shape: [0-9]+' "$PARAMS/$L.hjson" | grep -oE '[0-9]+$')
  echo "== $L (launches/layer: ${launches:-1}, layers with this shape: ${count:-1}, sim rc=$rc)"
  echo "$out" | grep -E '^\[(LAYER|GAP) warm\]'
done
