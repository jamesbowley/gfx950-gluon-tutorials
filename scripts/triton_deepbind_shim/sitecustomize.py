# Make Triton survive sharing a process with ROCm's shared libLLVM.
#
# This environment has two complete LLVMs in play:
#   * libtriton.so statically links LLVM b010a18d (Triton's pin) and, because
#     the tutorial needs TRITON_EXT_ENABLED=1 / default visibility, exports all
#     ~23k of its llvm:: symbols.
#   * A pip-installed ROCm (rocm-sdk-core) ships LLVM as a *shared* library,
#     libLLVM.so.23.0git, which exports llvm:: too. It is pulled into the
#     process by libamd_comgr.so.3, which arrives via HIP (any `import torch`)
#     and via rocprofiler-sdk (anything run under rocprofv3).
#
# Whichever object is earlier in the global symbol scope wins, and the two are
# different builds, so the wrong pairing corrupts one side or the other:
#
#   * rocprofv3 LD_PRELOADs rocprofiler-sdk, so libLLVM lands in the global
#     scope ahead of libtriton and ~17k of libtriton's references to its own
#     LLVM bind to ROCm's copy. libtriton then dies in its static initializers
#     (SIGSEGV in __libc_realloc).
#   * Loading libtriton RTLD_GLOBAL instead (what bench.py does when
#     LLVM_PASS_PLUGIN_PATH is set) inverts it: libLLVM loads later, its
#     cl::opt registrations bind to libtriton's LLVM, and LLVM aborts with
#     "Option 'default' already exists!".
#
# The fix is to keep each LLVM bound to itself. RTLD_DEEPBIND puts an object's
# own dependencies ahead of the global scope, so libtriton prefers its own
# LLVM, and libLLVM (loaded later, with libtriton kept out of the global scope)
# prefers its own.
#
# Enable by putting this directory on PYTHONPATH:
#     export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:$PYTHONPATH
#
# Because this imports triton before the benchmark script runs, bench.py's own
# `sys.setdlopenflags(RTLD_GLOBAL)` becomes a no-op -- triton is already in
# sys.modules -- which is what keeps libtriton out of the global scope.
import ctypes
import os
import sys

_needed = "rocprofiler" in os.environ.get("LD_PRELOAD", "") or bool(
    os.environ.get("LLVM_PASS_PLUGIN_PATH")
)

if _needed:
    _prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    try:
        import triton  # noqa: F401
    finally:
        sys.setdlopenflags(_prev)

    # The LLIR pass plugin carries no LLVM of its own; it resolves llvm::
    # symbols at load time. Left alone it binds them to ROCm's libLLVM (global
    # scope outranks a dlopened object's own dependencies) and then hangs inside
    # the wrong AnalysisManager. Pre-dlopening it with DEEPBIND binds it to
    # libtriton instead; LLVM's PassPlugin::Load dlopen()s the same path later
    # and gets this already-relocated handle back.
    #
    # This requires a plugin that actually lists libtriton as a dependency, so
    # DEEPBIND has something to prefer. The stock libLlirSched.so does not, so
    # transparently redirect to the relinked sibling built by
    # plugins/llir_scheduler/build_linked.sh when it is present. Doing the
    # swap here means every entry point -- run_perf_table.py, the manual
    # command lines in the kernel READMEs -- picks it up without edits.
    _plugin = os.environ.get("LLVM_PASS_PLUGIN_PATH")
    if _plugin:
        _base, _ext = os.path.splitext(_plugin)
        if not _base.endswith("_linked"):
            _linked = _base + "_linked" + _ext
            if os.path.exists(_linked):
                # putenv too, so Triton's C++ side sees the redirect.
                os.environ["LLVM_PASS_PLUGIN_PATH"] = _linked
                _plugin = _linked
        ctypes.CDLL(_plugin, mode=os.RTLD_NOW | os.RTLD_DEEPBIND | os.RTLD_LOCAL)
