#!/usr/bin/env bash
# ATT traces of att_stream_kernel (bw_kernel's read loop), one workgroup per CU on `grid` CUs. Every
# s_waitcnt in its loop waits for a round trip, and with more than the TCP's worth of loads in flight its
# buffer_loads wait for slots, so one trace has both the exact L and the stretched issues att_latency.py
# needs to count the slots (D_k). xcd0 runs on the 32 CUs of XCD 0 (the traced ones among them; the memory
# system is lightly loaded), all on every CU (HBM-bound).
#
#   GPU=2 SHIM=<dir with the deepbind sitecustomize.py> bash collect_att_stream.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export HIP_VISIBLE_DEVICES=${GPU:-2}
export PYTHONPATH=${SHIM:-$REPO/scripts/triton_deepbind_shim}${PYTHONPATH:+:$PYTHONPATH}
export ROCPROF_ATT_LIBRARY_PATH=${ROCPROF_ATT_LIBRARY_PATH:-/opt/rocm/lib/}
export TRITON_CACHE_DIR=$HERE/tmp/cache

# num_warps,unroll,tier,scope: nominal KB in flight per CU = 2 x num_warps x unroll
for point in 4,2,HBM,xcd0 4,4,HBM,xcd0 8,4,HBM,xcd0 16,4,HBM,xcd0 16,4,MALL,xcd0 16,4,HBM,all; do
  tag=stream_$(echo "$point" | tr ',' '_')
  out=$HERE/tmp/att/$tag
  rm -rf "$out" && mkdir -p "$out"
  cat > "$out/att.json" <<EOF
{"jobs": [{"kernel_include_regex": "att_stream_kernel", "kernel_exclude_regex": "", "kernel_iteration_range": "[10]",
           "advanced_thread_trace": true, "att_target_cu": 0, "att_shader_engine_mask": "0xF",
           "att_simd_select": "0xF", "att_buffer_size": "0x60000000"}]}
EOF
  # 18 att_stream_kernel dispatches (the HBM flush is a torch kernel).
  (cd "$HERE" && rocprofv3 --att -i "$out/att.json" -d "$out" -- python bench.py --att-point "$point" --reps 3) \
    > "$out/rocprof.log" 2>&1
  ui=$(find "$out" -maxdepth 3 -type d -name 'ui_*' | head -1)
  echo "$tag: $(grep 'att point' "$out/rocprof.log")  ui=$ui"
  python "$HERE/att_latency.py" "$ui" --name "$tag" --plot
done
