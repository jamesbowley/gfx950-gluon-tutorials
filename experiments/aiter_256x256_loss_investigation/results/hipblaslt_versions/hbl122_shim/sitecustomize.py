# The llirSched DEEPBIND shim (/tmp/llirsched_bench/shim) for run_aiter_compare_hbl122.sh.
# rocm_sdk's preload must be disabled before anything imports torch, or it loads the 7.14
# hipBLASLt next to the LD_PRELOADed 1.2.2; the shim below imports aiter, and so torch.
import ctypes
import os
import sys

import rocm_sdk

rocm_sdk.preload_libraries = lambda *a, **k: None

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
