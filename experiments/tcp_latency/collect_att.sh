#!/usr/bin/env bash
# ATT traces of the a16w16 v7 / v8 hot loops at K = 8192 and 16384, decoded into tmp/att/<v>_K<K>/ui_*,
# then reduced by att_latency.py into results/att_<v>_K<K>.json (+ _timeline.png, _dk.png).
#
#   GPU=2 SHIM=<dir with the deepbind sitecustomize.py> bash collect_att.sh
#
# Calls rocprofv3 directly rather than scripts/run_perf_table.py, which deletes the whole
# kernels/gemm/intra_wave/a16w16/tmp/ before each run. Config: the LLIR scheduler plugin alone
# (llir+amdgcnas miscompiles v8 on the installed Triton: "Triton and Torch differ").
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
A16=$REPO/kernels/gemm/intra_wave/a16w16
export HIP_VISIBLE_DEVICES=${GPU:-2}
export PYTHONPATH=${SHIM:-$REPO/scripts/triton_deepbind_shim}${PYTHONPATH:+:$PYTHONPATH}
export ROCPROF_ATT_LIBRARY_PATH=${ROCPROF_ATT_LIBRARY_PATH:-/opt/rocm/lib/}
export LLVM_PASS_PLUGIN_PATH=$REPO/plugins/llir_scheduler/libLlirSched.so
export TRITON_CACHE_DIR=$HERE/tmp/cache
unset TRITON_AMDGCNAS_PLUGIN

declare -A DIR=([7]=v7_sliceN [8]=v8_sliceMN)
for v in 7 8; do
  for K in 8192 16384; do
    out=$HERE/tmp/att/v${v}_K${K}
    rm -rf "$out" && mkdir -p "$out"
    cat > "$out/att.json" <<EOF
{"jobs": [{"kernel_include_regex": "${DIR[$v]}", "kernel_exclude_regex": "", "kernel_iteration_range": "[15]",
           "advanced_thread_trace": true, "att_target_cu": 0, "att_shader_engine_mask": "0xF",
           "att_simd_select": "0xF", "att_buffer_size": "0x60000000"}]}
EOF
    (cd "$A16" && rocprofv3 --att -i "$out/att.json" -d "$out" -- python bench.py --K "$K" --dtype fp16 --version "$v") \
      > "$out/rocprof.log" 2>&1
    ui=$(find "$out" -maxdepth 3 -type d -name 'ui_*' | head -1)
    echo "v$v K=$K: $(grep -E 'match|differ' "$out/rocprof.log" | head -1)  ui=$ui"
    python "$HERE/att_latency.py" "$ui" --name "v${v}_K${K}" --plot
  done
done
