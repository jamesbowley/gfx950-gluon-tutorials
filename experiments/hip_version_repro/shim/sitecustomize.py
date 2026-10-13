# DEEPBIND loader shim for the llirSched Triton (TRITON_EXT_ENABLED=1) next to pip/deb ROCm,
# copied from /tmp/llirsched_bench/shim. Keeps libtriton and ROCm's libLLVM (pulled in by
# libamd_comgr via torch) each bound to their own LLVM, and loads AITER's llirSched plugin.
#
# Only acts for `python script.py`, `python -m ...` and `python -c ...`. PYTHONPATH is
# inherited by ROCm's Python tools (rocm_agent_enumerator, the venv's hipconfig/rocminfo
# wrappers) that hipcc runs during AITER's JIT build; importing AITER there waits on the
# build lock the parent holds, and the build hangs.
import ctypes
import os
import sys

_argv = getattr(sys, "orig_argv", [])
if len(_argv) > 1 and (_argv[1].endswith(".py") or _argv[1] in ("-m", "-c")):
    _prev = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_DEEPBIND)
    try:
        import triton  # noqa: F401
    finally:
        sys.setdlopenflags(_prev)

    if os.environ.get("AITER_LLIR_SCHED", "1") != "0":
        from aiter.ops.triton import llir_sched

        _so = llir_sched.plugin_path()
        if _so:
            ctypes.CDLL(_so, mode=os.RTLD_NOW | os.RTLD_DEEPBIND | os.RTLD_LOCAL)
