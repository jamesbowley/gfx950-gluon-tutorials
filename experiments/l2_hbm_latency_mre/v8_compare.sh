#!/usr/bin/env bash
# ATT of the a16w16 v8 kernel (fp16, 4096 x 4096 x 8192, MFMA scheduler alone) on one GPU, hot and cold:
#   cold: rotating inputs of 2048 MB, so each launch's 128 MB of A and B was last read 1.8 GB earlier
#   hot:  one 160 MB copy reused by every launch, so A and B stay in the 256 MB memory-side cache
# Traces dispatch 15 of the bench.py --rocprof loop and reduces it with scripts/process_json.py (loop
# MFMA efficiency) and experiments/tcp_latency/att_latency.py --loop (round trips, load stalls, clock).
#
#   GPU=0 bash v8_compare.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
A16=$REPO/kernels/gemm/intra_wave/a16w16
export HIP_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH=$REPO/scripts/triton_deepbind_shim
export ROCPROF_ATT_LIBRARY_PATH=${ROCPROF_ATT_LIBRARY_PATH:-/opt/rocm/lib/}
export TRITON_CACHE_DIR=$HERE/tmp/v8_cache
unset TRITON_AMDGCNAS_PLUGIN LLVM_PASS_PLUGIN_PATH
OUT=$HERE/results/v8
mkdir -p "$OUT"

for cache in cold hot; do
  mb=$([ $cache = cold ] && echo 2048 || echo 160)
  d=$HERE/tmp/v8_$cache
  rm -rf "$d" && mkdir -p "$d"
  cat > "$d/att.json" <<EOF
{"jobs": [{"kernel_include_regex": "v8_sliceMN", "kernel_exclude_regex": "", "kernel_iteration_range": "[15]",
           "advanced_thread_trace": true, "att_target_cu": 0, "att_shader_engine_mask": "0xF",
           "att_simd_select": "0xF", "att_buffer_size": "0x60000000"}]}
EOF
  rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks > "$OUT/${cache}_smi_before.txt" 2>&1
  (cd "$A16" && rocprofv3 --att -i "$d/att.json" -d "$d" -- \
      python bench.py --K 8192 --dtype fp16 --version 8 --rocprof --rotating-buffer-size "$mb") > "$d/rocprof.log" 2>&1
  rocm-smi -d "$HIP_VISIBLE_DEVICES" --showclocks > "$OUT/${cache}_smi_after.txt" 2>&1
  ui=$(find "$d" -maxdepth 3 -type d -name 'ui_*' | head -1)
  echo "== v8 $cache: $(grep -E 'match|differ' "$d/rocprof.log" | head -1)"
  echo "   $(grep 'rotating tensors' "$d/rocprof.log")"
  echo "   trace: $ui"
  python "$REPO/scripts/process_json.py" "$ui" > "$OUT/${cache}_mfma.json" 2>/dev/null || true
  grep -i "efficiency" "$OUT/${cache}_mfma.json" | sed 's/^/   /'
  python "$REPO/experiments/tcp_latency/att_latency.py" "$ui" --name "v8_$cache" --loop --plot --exact-stall 64 \
      --out "$OUT" | sed 's/^/   /'
done
