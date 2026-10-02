#!/usr/bin/env bash
# Published-config perf matrix (run_perf_table.py --rocprof) for one checkout, on GPU 3.
#
#   run_matrix.sh <repo root> <out dir> [--bias]
#
# a16w16 fp16 K=8192: every REPORTED_COMBINATIONS pair (base v0-v9, llir v5-v13,
# llir+amdgcnas v7-v13); a16w16 bf16 K=8192: v9-v13 under llir and llir+amdgcnas;
# a8w8 K=16384 and a4w4 v0/v1 K=32768 under all configs. With --bias, only the a16w16
# v9-v13 rows, with bias. One log per table in <out dir>.
set -euo pipefail

REPO="$(cd "$1" && pwd)"
OUT="$2"
BIAS="${3:-}"
mkdir -p "$OUT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM="$HERE/../../scripts/triton_deepbind_shim"
TRITON="${TRITON_PYTHON:-/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python}"

export PYTHONPATH="$SHIM:$TRITON"
export HIP_VISIBLE_DEVICES=3
unset ROCR_VISIBLE_DEVICES

run() {
    local name="$1"
    shift
    echo "== $name"
    python "$REPO/scripts/run_perf_table.py" "$@" --rocprof >"$OUT/$name.log" 2>&1 || true
    sed -n '/RESULTS SUMMARY/,$p' "$OUT/$name.log"
}

if [[ "$BIAS" == "--bias" ]]; then
    run a16w16_fp16_bias --kernel a16w16 --versions 9 10 11 12 13 \
        --configs llir llir+amdgcnas --K 8192 --dtype fp16 --bias
    run a16w16_bf16_bias --kernel a16w16 --versions 9 10 11 12 13 \
        --configs llir llir+amdgcnas --K 8192 --dtype bf16 --bias
    exit 0
fi

run a16w16_fp16 --kernel a16w16 --versions 0 1 2 3 4 5 6 7 8 9 10 11 12 13 \
    --configs base llir llir+amdgcnas --K 8192 --dtype fp16
run a16w16_bf16 --kernel a16w16 --versions 9 10 11 12 13 \
    --configs llir llir+amdgcnas --K 8192 --dtype bf16
run a8w8 --kernel a8w8 --configs base llir llir+amdgcnas --K 16384
run a4w4 --kernel a4w4 --versions 0 1 --configs base llir llir+amdgcnas --K 32768
