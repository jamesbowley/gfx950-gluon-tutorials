#!/usr/bin/env bash
# Compile kernels from one checkout through its bench.py and keep each .amdgcn, on GPU 3.
#
#   dump_asm.sh <repo root> <out dir> amdgcnas|nobias
#
# amdgcnas: a16w16 v7-v9 (fp16, K=8192), a8w8 (K=16384), a4w4 v0/v1 (K=32768) under
#           llir and llir+amdgcnas -- the kernels whose published numbers amdgcnas_ext.py
#           can move.
# nobias:   a16w16 v9-v13, fp16 and bf16, K=8192, under base, llir and llir+amdgcnas.
# Compare two dumps with compare_asm.py.
set -euo pipefail

REPO="$(cd "$1" && pwd)"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
SET="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM="$HERE/../../scripts/triton_deepbind_shim"
TRITON="${TRITON_PYTHON:-/home/jbowley/repos/triton_gfx950-tutorial-v2.2/python}"
PLUGIN="$REPO/plugins/llir_scheduler/libLlirSched.so"

export PYTHONPATH="$SHIM:$TRITON"
export HIP_VISIBLE_DEVICES=3
unset ROCR_VISIBLE_DEVICES LLVM_PASS_PLUGIN_PATH TRITON_AMDGCNAS_PLUGIN

dump() {  # name config workdir bench-args...
    local name="$1" config="$2" dir="$3"
    shift 3
    local cache
    cache="$(mktemp -d)"
    local env=(TRITON_CACHE_DIR="$cache" TRITON_ALWAYS_COMPILE=1)
    [[ "$config" != base ]] && env+=(LLVM_PASS_PLUGIN_PATH="$PLUGIN")
    [[ "$config" == llir+amdgcnas ]] && env+=(TRITON_AMDGCNAS_PLUGIN=1)
    (cd "$dir" && env "${env[@]}" python bench.py "$@" >"$OUT/$name.$config.log" 2>&1) || true
    local f
    f="$(find "$cache" -name '*.amdgcn' | head -1)"
    if [[ -n "$f" ]]; then cp "$f" "$OUT/$name.$config.amdgcn"; else echo "NO ASM: $name $config"; fi
    rm -rf "$cache"
}

A16="$REPO/kernels/gemm/intra_wave/a16w16"
case "$SET" in
amdgcnas)
    for config in llir llir+amdgcnas; do
        for v in 7 8 9; do dump "a16w16_v$v" "$config" "$A16" --version "$v" --K 8192 --dtype fp16; done
        dump a8w8 "$config" "$REPO/kernels/gemm/intra_wave/a8w8" --K 16384
        for v in 0 1; do dump "a4w4_v$v" "$config" "$REPO/kernels/gemm/intra_wave/a4w4" --version "$v" --K 32768; done
    done
    ;;
nobias)
    for config in base llir llir+amdgcnas; do
        for v in 9 10 11 12 13; do
            for dt in fp16 bf16; do
                dump "a16w16_v${v}_$dt" "$config" "$A16" --version "$v" --K 8192 --dtype "$dt"
            done
        done
    done
    ;;
*)
    echo "unknown set $SET" >&2
    exit 2
    ;;
esac
echo "dumped $(ls "$OUT"/*.amdgcn | wc -l) kernels to $OUT"
