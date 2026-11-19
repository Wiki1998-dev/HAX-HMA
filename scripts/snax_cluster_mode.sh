#!/usr/bin/env bash
# Switch the SNAX checkout between the two simulated clusters.
#
#   plain  - snitch_cluster.hjson: 8 compute cores with FPU + DMA core.
#            Phase 1/2/4 fp32 kernels (gradcam, shap, symbolic).
#   gemmx  - snax_KUL_cluster.hjson: 1 GeMMX-controller core + DMA core,
#            both rv32ima (no FPU). gradcam_gemmx app.
#
# Each mode needs (a) its own Verilator binary, (b) runtime headers generated
# from its config and (c) a runtime library rebuilt against those headers
# (the library does not track the generated headers, so it is rebuilt with -B).
# Both binaries are kept in target/snitch_cluster/bin/ (git-ignored):
#   snitch_cluster.plain.vlt, snitch_cluster.kul.vlt
# and the active one is copied to bin/snitch_cluster.vlt.
#
# Usage: scripts/snax_cluster_mode.sh plain|gemmx|status
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(dirname "$(dirname "$SELF")")"
SNAX="$ROOT/snax_cluster"
TGT="$SNAX/target/snitch_cluster"
BIN="$TGT/bin"
HDRS="sw/runtime/common/snitch_cluster_cfg.h sw/runtime/common/snitch_cluster_addrmap.h"

case "${1:-}" in
  plain) cfg="cfg/snitch_cluster.hjson"; vlt="snitch_cluster.plain.vlt" ;;
  gemmx) cfg="cfg/snax_KUL_cluster.hjson"; vlt="snitch_cluster.kul.vlt" ;;
  status)
    echo "lru config : $(readlink "$TGT/cfg/lru.hjson")"
    grep -h "CFG_CLUSTER_NR_CORES" "$TGT/sw/runtime/common/snitch_cluster_cfg.h"
    for f in "$BIN"/snitch_cluster*.vlt; do echo "$(md5sum "$f" | cut -c1-8)  $(basename "$f")"; done
    exit 0 ;;
  *) echo "usage: $0 plain|gemmx|status" >&2; exit 2 ;;
esac

[ -f "$BIN/$vlt" ] || { echo "missing $BIN/$vlt — build it with:
  pixi run make CFG_OVERRIDE=$cfg -C target/snitch_cluster bin/snitch_cluster.vlt" >&2; exit 1; }

cd "$SNAX"
cp -p "$BIN/$vlt" "$BIN/snitch_cluster.vlt"
# Regenerate runtime headers for this config (also repoints cfg/lru.hjson).
pixi run make -s -C target/snitch_cluster CFG_OVERRIDE="$cfg" $HDRS >/dev/null
pixi run make -s -B -C target/snitch_cluster/sw/runtime/rtl-generic CFG_OVERRIDE="$cfg" >/dev/null
echo "Switched to '$1' ($cfg):"
"$SELF" status
echo "Rebuild apps for this mode before simulating (their ELFs link the runtime)."
