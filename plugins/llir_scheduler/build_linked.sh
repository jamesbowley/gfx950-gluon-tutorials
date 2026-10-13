#!/usr/bin/env bash
# Build libLlirSched_linked.so -- the stock libLlirSched.so with one change:
# it links against libtriton.so, so it carries a DT_NEEDED for it.
#
# The stock plugin resolves its llvm:: symbols from whatever is in the global
# symbol scope, which assumes libtriton is the only LLVM in the process. That
# holds on a /opt/rocm install, where libamd_comgr hides its LLVM. It does not
# hold on a pip-installed ROCm, where comgr links a *shared*
# libLLVM.so.23.0git that exports llvm:: and gets loaded by HIP and by
# rocprofiler-sdk. The plugin then binds to the wrong LLVM and hangs.
#
# With a DT_NEEDED for libtriton, the plugin can be dlopened RTLD_DEEPBIND to
# make it prefer libtriton's LLVM. scripts/triton_deepbind_shim does that.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM="${LLVM:-$(dirname "$(dirname "$(find "$HOME/.triton/llvm" -name llvm-config | head -1)")")}"
TRITON_C="${TRITON_C:-$(python -c 'import triton._C.libtriton as l, os; print(os.path.dirname(l.__file__))')}"

echo "LLVM     = $LLVM"
echo "libtriton= $TRITON_C"

g++ -shared -fPIC -fvisibility=default \
    $("$LLVM/bin/llvm-config" --cxxflags) \
    -o "$HERE/libLlirSched_linked.so" "$HERE/LlirSchedPlugin.cpp" \
    -L"$TRITON_C" -l:libtriton.so -Wl,-rpath,"$TRITON_C"

echo
echo "Built $HERE/libLlirSched_linked.so"
readelf -d "$HERE/libLlirSched_linked.so" | grep -E 'NEEDED|RUNPATH'
