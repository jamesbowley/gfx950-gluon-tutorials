# For AITER's tuner on the pip-installed ROCm: put this directory ahead of the
# triton_deepbind_shim on PYTHONPATH (it runs the shim afterwards).
#
# HIP's initialization setenv()s ROCPROFILER_REGISTER_LIBRARY and
# HSA_TOOLS_ROCPROFILER_V1_TOOLS=0 at the C level. AITER's mp_tuner spawns its
# workers after the parent has initialized HIP, the workers inherit both, and
# torch.profiler then records no device events in them: run_perftest returns
# 0 us for every candidate ("no valida data after post process!"). Clearing
# them before torch loads HIP in each process restores the device timeline.
import os
import runpy

for _name in ("ROCPROFILER_REGISTER_LIBRARY", "HSA_TOOLS_ROCPROFILER_V1_TOOLS"):
    os.environ.pop(_name, None)

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts",
                            "triton_deepbind_shim", "sitecustomize.py"))
