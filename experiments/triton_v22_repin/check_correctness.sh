#!/usr/bin/env bash
# Correctness sweep of every kernel family on one checkout, on GPU 3.
#
#   check_correctness.sh <repo root> <out dir>
#
# a16w16 intra v0-v13 in fp16 and bf16 over bench.py's default K sweep (M=N=4096), each
# under its published config (base v0-v4, llir v5-v6, llir+amdgcnas v7-v13); v10-v13 also at
# 8192x8192 (4 tiles per program) and 4608x4096 (uneven tiles per program). --bias for
# v9-v13 under base, llir and llir+amdgcnas at the same shapes. a8w8 / a4w4 intra under
# llir+amdgcnas, the inter_wave kernels (no plugins), and the attention kernels under llir.
# Prints every mismatch and a pass/fail count.
set -uo pipefail

REPO="$(cd "$1" && pwd)"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE/../../scripts/triton_deepbind_shim:${TRITON_PYTHON:-/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python}"
export HIP_VISIBLE_DEVICES=3
unset ROCR_VISIBLE_DEVICES LLVM_PASS_PLUGIN_PATH TRITON_AMDGCNAS_PLUGIN
PLUGIN="$REPO/plugins/llir_scheduler/libLlirSched.so"

n=0
run() {  # name config dir cmd...
    local name="$1" config="$2" dir="$3"
    shift 3
    local env=()
    [[ "$config" != base ]] && env+=(LLVM_PASS_PLUGIN_PATH="$PLUGIN")
    [[ "$config" == llir+amdgcnas ]] && env+=(TRITON_AMDGCNAS_PLUGIN=1)
    n=$((n + 1))
    (cd "$dir" && env "${env[@]}" "$@" >"$OUT/$name.$config.log" 2>&1)
    local rc=$?
    local ok bad
    ## a4w4 intra prints "Triton and Torch match/differ" without the check marks.
    ok=$(grep -cE "✅|Torch match" "$OUT/$name.$config.log")
    bad=$(grep -cE "❌|Torch differ|MISMATCH" "$OUT/$name.$config.log")
    if [[ $rc -ne 0 || $bad -ne 0 || $ok -eq 0 ]]; then
        echo "FAIL $name $config (exit $rc, $ok ok, $bad mismatches)"
        grep "❌" "$OUT/$name.$config.log" | head -5
    else
        echo "ok   $name $config ($ok checks)"
    fi
}

A16="$REPO/kernels/gemm/intra_wave/a16w16"
for v in $(seq 0 13); do
    if ((v <= 4)); then c=base; elif ((v <= 6)); then c=llir; else c=llir+amdgcnas; fi
    run "a16w16_v$v" "$c" "$A16" python bench.py --version "$v" --dtype fp16 bf16
done
for v in 10 11 12 13; do
    for mn in "8192 8192" "4608 4096"; do
        set -- $mn
        run "a16w16_v${v}_${1}x${2}" llir+amdgcnas "$A16" python bench.py --version "$v" --dtype fp16 bf16 --K 8192 --M "$1" --N "$2"
    done
done
for c in base llir llir+amdgcnas; do
    for v in 9 10 11 12 13; do
        run "a16w16_v${v}_bias" "$c" "$A16" python bench.py --version "$v" --dtype fp16 bf16 --bias
        for mn in "8192 8192" "4608 4096"; do
            set -- $mn
            run "a16w16_v${v}_bias_${1}x${2}" "$c" "$A16" python bench.py --version "$v" --dtype fp16 bf16 --K 8192 --M "$1" --N "$2" --bias
        done
    done
done

run a8w8 llir+amdgcnas "$REPO/kernels/gemm/intra_wave/a8w8" python bench.py
for v in 0 1; do run "a4w4_v$v" llir+amdgcnas "$REPO/kernels/gemm/intra_wave/a4w4" python bench.py --version "$v"; done
run inter_a16w16 base "$REPO/kernels/gemm/inter_wave/a16w16" python bench.py
run inter_a8w8 base "$REPO/kernels/gemm/inter_wave/a8w8" python bench.py
for v in 0 1 2; do run "inter_a4w4_v$v" base "$REPO/kernels/gemm/inter_wave/a4w4" python bench.py --version "$v"; done
for m in fmha_v3 fmha_v4; do
    run "attention_$m" llir "$REPO/kernels/attention" env FA_MODULE="$m" python bench.py --batch 32 --hq 8 --hk 8 --seqlen 8192
done
echo "ran $n checks"
