#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""benchmark_prepared.py with the same arguments, resolving intra-wave a16w16 versions through
bench_ext.py's VERSION_MAP, for kernels that take an optional bias (bias_ptr after c_ptr, an
ADD_BIAS constexpr). They are timed without it. benchmark_prepared.py itself stays as on main.

It also compiles with the schedule_hint the kernel's matmul() passes to its launch (v5 on:
"mfma-schedule"), which benchmark_prepared.py leaves out; --schedule-hint overrides it ("" off).

The persistent versions (v10 on) also need NUM_PROGRAMS and a CU-sized grid, which
benchmark_prepared.py's launch configuration does not supply, so they do not run here.
"""

import inspect
import pathlib
import sys

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
A16W16_DIR = SCRIPTS_DIR.parent / "kernels" / "gemm" / "intra_wave" / "a16w16"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_prepared as bp  # noqa: E402

_load_kernel = bp.load_kernel
_make_runtime_arguments = bp.make_runtime_arguments
_launch_configuration = bp.launch_configuration
_jit_kernel = None
_module = None
_schedule_hint = None


def _has_bias(jit_kernel):
    return "bias_ptr" in jit_kernel.arg_names


def load_kernel(args):
    global _jit_kernel, _module
    if args.route == "intra" and args.kernel == "a16w16":
        # load_kernel imports "bench" for VERSION_MAP and the hooks; hand it bench_ext instead.
        sys.path.insert(0, str(bp.UTILS_DIR))
        sys.path.insert(0, str(A16W16_DIR))
        import bench_ext

        sys.modules["bench"] = bench_ext
    _module, _jit_kernel, label = _load_kernel(args)
    return _module, _jit_kernel, label


def make_runtime_arguments(args, device):
    runtime_sets, bytes_per_set = _make_runtime_arguments(args, device)
    if _jit_kernel is not None and _has_bias(_jit_kernel):
        runtime_sets = [(*rs[:3], None, *rs[3:]) for rs in runtime_sets]
    return runtime_sets, bytes_per_set


def launch_configuration(args, jit_kernel):
    grid, constexprs, compiler_options = _launch_configuration(args, jit_kernel)
    if _has_bias(jit_kernel):
        constexprs["ADD_BIAS"] = False
    matmul = getattr(_module, "matmul", None)
    param = inspect.signature(matmul).parameters.get("schedule_hint") if matmul else None
    hint = _schedule_hint if _schedule_hint is not None else (param.default if param else None)
    if param and hint:
        compiler_options["schedule_hint"] = hint
    return grid, constexprs, compiler_options


bp.load_kernel = load_kernel
bp.make_runtime_arguments = make_runtime_arguments
bp.launch_configuration = launch_configuration

if __name__ == "__main__":
    if "--schedule-hint" in sys.argv:
        i = sys.argv.index("--schedule-hint")
        _schedule_hint = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    bp.main()
