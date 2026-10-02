#!/usr/bin/env python3
"""Run AITER pytest cases in this environment (no flydsl, pip ROCm with a shared libLLVM).

    PYTHONPATH=<triton gfx950-tutorial-v2.2>/python:<aiter> \
    python run_aiter_tests.py op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py -k "..."

Same bootstrap as check_spills.py's m1 worker: libtriton and AITER's llirSched plugin
deep-bound to their own LLVM, and `aiter` registered without running aiter/__init__.py.
Arguments are passed to pytest, run from the AITER checkout.
"""

import os
import sys

import check_spills

check_spills._bare_aiter_package()
check_spills._deepbind_triton_and_aiter_plugin()

import pytest  # noqa: E402

os.chdir(check_spills.AITER)
sys.path.insert(0, check_spills.AITER)
sys.exit(pytest.main(sys.argv[1:]))
