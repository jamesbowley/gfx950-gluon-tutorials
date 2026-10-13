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
"""Version specs for the v20 measurement tools: "20" or "20:STREAMK_POLICY=spread" (several
variables separated by "+"). The matmul functions read STREAMK_POLICY / STREAMK_FIXUP when
called, so setting them around a version's calls is enough."""

import contextlib
import os


def parse(spec):
    """(version, {name: value}) from "V" or "V:NAME=VALUE+NAME=VALUE"."""
    v, _, env = spec.partition(":")
    pairs = dict(kv.split("=", 1) for kv in env.split("+")) if env else {}
    return int(v), pairs


def label(spec):
    v, env = parse(spec)
    return f"v{v}" + "".join(f" {k.removeprefix('STREAMK_').lower()}={x}" for k, x in env.items())


@contextlib.contextmanager
def environment(env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, x in old.items():
            if x is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = x
