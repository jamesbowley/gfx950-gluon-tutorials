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
"""Fast/slow C placement: back-to-back GEMMs in a CUDA graph with controlled C buffers.

    export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
    export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so TRITON_AMDGCNAS_PLUGIN=1
    HIP_VISIBLE_DEVICES=2 python place.py classify --n 8
    HIP_VISIBLE_DEVICES=2 python place.py e1
    HIP_VISIBLE_DEVICES=2 python place.py e2
    HIP_VISIBLE_DEVICES=2 python place.py e3 --path hip
    HIP_VISIBLE_DEVICES=2 python place.py e7

Every measurement is a graph of --calls GEMMs (A and B fixed, C taken from a pattern of
buffers in turn), replayed --replays times; the result is the median time per GEMM. While the
graph replays, a thread reads the device's hwmon sclk and power about every millisecond.
"""

import argparse
import ctypes
import glob
import importlib
import json
import os
import random
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
A16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")
sys.path.insert(0, A16)

import bench  # noqa: E402  (plugin hooks and the path to common.py)
import torch  # noqa: E402
from v20_streamk_reduce_scatter import version_spec  # noqa: E402

DT = torch.bfloat16
MB = 2**20


def hip():
    for line in open("/proc/self/maps"):
        if "libamdhip64" in line:
            return ctypes.CDLL(line.split()[-1])
    raise RuntimeError("libamdhip64 not loaded")


class RawBuf:
    """A device buffer from hipMalloc or hipExtMallocWithFlags, viewed as a torch tensor."""

    def __init__(self, nbytes, flags=None):
        self.lib = hip()
        p = ctypes.c_void_p()
        if flags is None:
            err = self.lib.hipMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes))
        else:
            err = self.lib.hipExtMallocWithFlags(ctypes.byref(p), ctypes.c_size_t(nbytes), ctypes.c_uint(flags))
        assert err == 0, f"hip alloc failed: {err}"
        self.ptr, self.nbytes = p.value, nbytes
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (self.ptr, False),
                                         "version": 3}

    def tensor(self):
        return torch.as_tensor(self, device="cuda")


class Setup:
    def __init__(self, args):
        self.args = args
        self.M, self.N, self.K = (int(x) for x in args.shape.split("x"))
        v, self.env = version_spec.parse(args.version)
        os.environ.update(self.env)
        self.mod = importlib.import_module(bench.VERSION_MAP[v] + ".matmul_kernel")
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.a = torch.randn(self.M, self.K, device="cuda", dtype=DT)
        self.b = torch.randn(self.N, self.K, device="cuda", dtype=DT).T
        self.rng = random.Random(args.seed)
        self.keep = []
        self.c_bytes = self.M * self.N * 2
        bus = torch.cuda.get_device_properties(0).pci_bus_id
        d = glob.glob(f"/sys/bus/pci/devices/0000:{bus:02x}:00.0/hwmon/hwmon*")[0]
        self.hwmon = os.path.join(d, "freq1_input"), os.path.join(d, "power1_input")
        self.sleep_cycles_per_us = None

    # ---- C buffers -------------------------------------------------------------------------
    def view_c(self, flat_u8, offset):
        n = self.M * self.N
        return flat_u8[offset:offset + 2 * n].view(DT).view(self.M, self.N)

    def alloc_c(self, path):
        """One C buffer from the given allocation path. Earlier buffers stay alive, with a
        random-size pad in between, so each one lands somewhere new."""
        pad = self.rng.randint(1, 64) * MB + self.rng.randint(0, 255) * 4096
        if path == "torch":
            self.keep.append(torch.empty(pad, dtype=torch.uint8, device="cuda"))
            return torch.empty(self.M, self.N, device="cuda", dtype=DT)
        if path in ("hip", "fine", "coarse_ext"):
            flags = {"hip": None, "fine": 1, "coarse_ext": 0}[path]
            self.keep.append(RawBuf(pad, flags))
            buf = RawBuf(self.c_bytes, flags)
            self.keep.append(buf)
            return self.view_c(buf.tensor(), 0)
        if path == "slice":
            if not hasattr(self, "big"):
                self.big = RawBuf(1024 * MB).tensor()
            off = self.rng.randrange(0, 1024 * MB - self.c_bytes, 4096)
            return self.view_c(self.big, off)
        raise ValueError(path)

    # ---- timing ----------------------------------------------------------------------------
    def calibrate_sleep(self):
        if self.sleep_cycles_per_us is None:
            probe = 100000
            us = self._graph_us(lambda i: torch.cuda._sleep(probe), 50, 10)[0]
            self.sleep_cycles_per_us = probe / us
        return self.sleep_cycles_per_us

    def _graph_us(self, body, calls, replays, sample=False):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            for i in range(calls):
                body(i)
        g.replay()
        torch.cuda.synchronize()
        stop, out = threading.Event(), []
        th = None
        if sample:
            th = threading.Thread(target=self._sample, args=(stop, out))
            th.start()
        times = []
        for _ in range(replays):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1e3 / calls)
        if th:
            stop.set()
            th.join()
        del g
        mhz = statistics.median(x for x, _ in out) if out else float("nan")
        w = statistics.median(y for _, y in out) if out else float("nan")
        return statistics.median(times), mhz, w

    def _sample(self, stop, out):
        fq, pw = self.hwmon
        while not stop.is_set():
            with open(fq) as f:
                mhz = int(f.read()) / 1e6
            with open(pw) as f:
                w = int(f.read()) / 1e6
            out.append((mhz, w))
            time.sleep(0.001)

    def time_pattern(self, cs, gap_us=0.0, between=None, sample=False, replays=None):
        """Median us per GEMM over a graph whose call i writes cs[i % len(cs)]. gap_us inserts
        an idle sleep after each GEMM (subtracted again); between(i) is extra work after each
        GEMM (not subtracted). Returns (us, MHz, W)."""
        a, b, mod = self.a, self.b, self.mod
        cycles = int(gap_us * self.calibrate_sleep()) if gap_us > 0 else 0
        for c in cs:
            mod.matmul(a, b, c)

        def body(i):
            mod.matmul(a, b, cs[i % len(cs)])
            if cycles:
                torch.cuda._sleep(cycles)
            if between:
                between(i)

        r = replays or self.args.replays
        us, mhz, w = self._graph_us(body, self.args.calls, r, sample)
        if cycles:
            us -= self._graph_us(lambda i: torch.cuda._sleep(cycles), self.args.calls, r)[0]
        return us, mhz, w

    def classify(self, path, n):
        """n new C buffers with their single-buffer time, fastest first."""
        res = []
        for _ in range(n):
            c = self.alloc_c(path)
            res.append((self.time_pattern([c])[0], c))
        return res


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |", flush=True)


def split(res, gap_min=0.5):
    """Fastest two and slowest two buffers; warns when the gap is too small to call it bimodal."""
    s = sorted(res, key=lambda x: x[0])
    if s[-1][0] - s[0][0] < gap_min:
        print(f"warning: spread {s[-1][0] - s[0][0]:.2f} us, no clear fast/slow split", flush=True)
    return s[0][1], s[1][1], s[-1][1], s[-2][1], s


# ---- experiments ---------------------------------------------------------------------------
def cmd_classify(S):
    res = S.classify(S.args.path, S.args.n)
    table(["buffer", "C address (MB)", "us"], [(i, f"{c.data_ptr() / MB:.0f}", f"{t:.2f}") for i, (t, c) in
                                              enumerate(res)])


def cmd_e1(S):
    f1, f2, s1, s2, ranked = split(S.classify(S.args.path, S.args.n))
    print("classified (us): " + " ".join(f"{t:.2f}" for t, _ in ranked))
    pats = [("F1", [f1]), ("F2", [f2]), ("S1", [s1]), ("S2", [s2]), ("F1,F2", [f1, f2]), ("F1,S1", [f1, s1]),
            ("S1,S2", [s1, s2]), ("F1,F1,F2,F2", [f1, f1, f2, f2]), ("F1x4,F2x4", [f1] * 4 + [f2] * 4),
            ("F1x16,F2x16", [f1] * 16 + [f2] * 16), ("S1,S1,S2,S2", [s1, s1, s2, s2])]
    rows = []
    for _ in range(S.args.repeats):
        for name, cs in pats:
            us, mhz, w = S.time_pattern(cs, sample=True, replays=100)
            rows.append((name, f"{us:.2f}", f"{mhz:.0f}", f"{w:.0f}"))
    table(["pattern", "us", "MHz", "W"], rows)


def cmd_e2(S):
    big = RawBuf(1024 * MB)
    t = big.tensor()
    offs = list(range(0, 512 * 1024 + 1, 4096 * S.args.step4k)) + [x * 2 * MB for x in range(0, 448, S.args.step2m)]
    offs = sorted(set(o for o in offs if o + S.c_bytes <= 1024 * MB))
    rows = []
    for o in offs:
        us = S.time_pattern([S.view_c(t, o)])[0]
        rows.append((o // 4096, f"{o / MB:.3f}", f"{us:.2f}"))
    print(f"buffer base: {big.ptr / MB:.0f} MB")
    table(["offset (4 KB pages)", "offset (MB)", "us"], rows)


def cmd_sweep(S):
    """A, B and C all inside one 1 GB buffer: sweep one of them (--what c or ab) from --start
    to --stop MB in --step MB steps, the others at --fixed MB (A at --fixed, B 8 MB above it
    for --what c; C at --fixed for --what ab)."""
    big = RawBuf(1024 * MB)
    t = big.tensor()
    a_bytes = S.M * S.K * 2
    b_bytes = S.N * S.K * 2

    def ab_at(off):
        a = t[off:off + a_bytes].view(DT).view(S.M, S.K)
        b = t[off + a_bytes:off + a_bytes + b_bytes].view(DT).view(S.N, S.K).T
        a.normal_()
        b.normal_()
        return a, b

    rows = []
    x = S.args.start
    while x <= S.args.stop:
        o, f = int(x * MB), int(S.args.fixed * MB)
        if S.args.what == "c":
            c_off, ab_off = o, f
        else:
            c_off, ab_off = f, o
        lo_c, hi_c = c_off, c_off + S.c_bytes
        lo_ab, hi_ab = ab_off, ab_off + a_bytes + b_bytes
        if hi_c <= 1024 * MB and hi_ab <= 1024 * MB and (hi_c <= lo_ab or hi_ab <= lo_c):
            S.a, S.b = ab_at(ab_off)
            us, mhz, _ = S.time_pattern([S.view_c(t, c_off)], gap_us=S.args.gap, sample=S.args.gap > 0,
                                        replays=40 if S.args.gap > 0 else None)
            rows.append((f"{x:g}", f"{c_off / MB:g}", f"{ab_off / MB:g}", f"{us:.2f}", f"{mhz:.0f}"))
        x += S.args.step
    print(f"buffer base: {big.ptr / MB:.0f} MB, idle gap {S.args.gap} us")
    table(["swept (MB)", "C offset (MB)", "A/B offset (MB)", "us", "MHz"], rows)


def cmd_launch(S):
    """Plain stream launches for counter collection: inside one 1 GB buffer, A/B at --fixed MB
    and C at each of --offsets MB in turn, --n launches each (no graph, no timing)."""
    big = RawBuf(1024 * MB)
    t = big.tensor()
    a_bytes, b_bytes = S.M * S.K * 2, S.N * S.K * 2
    f = int(S.args.fixed * MB)
    S.a = t[f:f + a_bytes].view(DT).view(S.M, S.K)
    S.b = t[f + a_bytes:f + a_bytes + b_bytes].view(DT).view(S.N, S.K).T
    S.a.normal_()
    S.b.normal_()
    for o in S.args.offsets.split(","):
        c = S.view_c(t, int(float(o) * MB))
        for _ in range(S.args.n):
            S.mod.matmul(S.a, S.b, c)
    torch.cuda.synchronize()


def cmd_layout(S):
    """A, B and C inside one 1.5 GB buffer: A at --a-off, then every (B offset, C offset) pair
    from --b-offs x --c-offs (MB); pairs that overlap are skipped."""
    total = 1536 * MB
    big = RawBuf(total)
    t = big.tensor()
    kp = S.K + S.args.b_pad
    a_bytes, b_bytes = S.M * S.K * 2, S.N * kp * 2
    oa = int(S.args.a_off * MB)
    S.a = t[oa:oa + a_bytes].view(DT).view(S.M, S.K)
    S.a.normal_()
    rows = []
    for bo in (float(x) for x in S.args.b_offs.split(",")):
        ob = int(bo * MB)
        S.b = t[ob:ob + b_bytes].view(DT).view(S.N, kp)[:, :S.K].T
        S.b.normal_()
        for co in (float(x) for x in S.args.c_offs.split(",")):
            oc = int(co * MB)
            np_ = S.N + S.args.c_pad
            c_span = S.M * np_ * 2
            spans = sorted([(oa, oa + a_bytes), (ob, ob + b_bytes), (oc, oc + c_span)])
            if any(spans[i][1] > spans[i + 1][0] for i in range(2)) or spans[-1][1] > total:
                continue
            c = t[oc:oc + c_span].view(DT).view(S.M, np_)[:, :S.N]
            if S.args.check:
                ref = (S.a.float() @ S.b.float()).to(DT)
                S.mod.matmul(S.a, S.b, c)
                ok = torch.allclose(c.float(), ref.float(), atol=1e-1, rtol=1e-2)
                print(f"check B {bo:g} C {co:g}: {'match' if ok else 'DIFFER'}", flush=True)
            us = S.time_pattern([c])[0]
            tflops = 2 * S.M * S.N * S.K / us / 1e6
            rows.append((f"{bo:g}", f"{co:g}", f"{(co % 256):g}", f"{us:.1f}", f"{tflops:.0f}"))
    print(f"buffer base: {big.ptr / MB:.0f} MB, A at {S.args.a_off:g} MB")
    table(["B offset (MB)", "C offset (MB)", "C offset mod 256", "us", "TFLOPS"], rows)


def cmd_e3(S):
    res = S.classify(S.args.path, S.args.n)
    ts = sorted(t for t, _ in res)
    thr = S.args.threshold
    fast = sum(t < thr for t in ts)
    print(json.dumps({"path": S.args.path, "n": len(ts), "fast": fast, "threshold": thr,
                      "pairs": [(c.data_ptr() / MB, round(t, 2)) for t, c in res]}), flush=True)


def cmd_e7(S):
    """Inside one 1 GB buffer (A/B at --fixed MB): C at 32 MB (fast half) and 160 MB (slow half).
    1) idle gaps after each GEMM; 2) a write of some size into the fast half (80 MB) or the slow
    half (200 MB) after each GEMM, with C in the fast half. The extra work's own graph time is
    subtracted, so the GEMM column is the GEMM's share."""
    big = RawBuf(1024 * MB)
    t = big.tensor()
    a_bytes, b_bytes = S.M * S.K * 2, S.N * S.K * 2
    f = int(S.args.fixed * MB)
    S.a = t[f:f + a_bytes].view(DT).view(S.M, S.K)
    S.b = t[f + a_bytes:f + a_bytes + b_bytes].view(DT).view(S.N, S.K).T
    S.a.normal_()
    S.b.normal_()
    cf, cs = S.view_c(t, 32 * MB), S.view_c(t, 160 * MB)
    rows = []
    for gap in [0, 2, 5, 10, 20, 50, 100, 200]:
        for name, c in (("fast half", cf), ("slow half", cs)):
            us, mhz, w = S.time_pattern([c], gap_us=gap, sample=True, replays=40)
            rows.append((f"idle {gap} us", name, f"{us:.2f}", f"{mhz:.0f}", f"{w:.0f}"))
    for size_kb in [64, 1024, 4096, 16384, 32768]:
        for where, off in (("fast half (80 MB)", 80), ("slow half (200 MB)", 200)):
            dst = t[off * MB:off * MB + size_kb * 1024]
            fn = lambda i, d=dst: d.fill_(1)
            base = S._graph_us(fn, S.args.calls, 40)[0]
            us, mhz, w = S.time_pattern([cf], between=fn, sample=True, replays=40)
            rows.append((f"write {size_kb} KB to {where}", "fast half", f"{us - base:.2f}", f"{mhz:.0f}", f"{w:.0f}"))
    table(["after each GEMM", "C", "GEMM us", "MHz", "W"], rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["classify", "e1", "e2", "sweep", "launch", "layout", "e3", "e7"])
    p.add_argument("--a-off", type=float, default=0, help="layout: A offset (MB)")
    p.add_argument("--b-offs", default="64", help="layout: B offsets (MB)")
    p.add_argument("--c-offs", default="600,700", help="layout: C offsets (MB)")
    p.add_argument("--b-pad", type=int, default=0, help="layout: extra elements per row of B (N, K)")
    p.add_argument("--c-pad", type=int, default=0, help="layout: extra elements per row of C (M, N)")
    p.add_argument("--check", action="store_true", help="layout: compare each output against torch")
    p.add_argument("--offsets", default="32,160", help="launch: C offsets (MB) in the 1 GB buffer")
    p.add_argument("--what", default="c", choices=["c", "ab"])
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--stop", type=float, default=256)
    p.add_argument("--step", type=float, default=1)
    p.add_argument("--fixed", type=float, default=600)
    p.add_argument("--gap", type=float, default=0, help="sweep: idle us after each GEMM")
    p.add_argument("--version", default="9")
    p.add_argument("--shape", default="4096x4096x1024")
    p.add_argument("--path", default="torch", choices=["torch", "hip", "fine", "coarse_ext", "slice"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--calls", type=int, default=400)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step4k", type=int, default=8, help="E2: 4 KB-page step below 512 KB")
    p.add_argument("--step2m", type=int, default=8, help="E2: 2 MB step above")
    p.add_argument("--threshold", type=float, default=29.0, help="E3: fast if below (us)")
    args = p.parse_args()
    S = Setup(args)
    {"classify": cmd_classify, "e1": cmd_e1, "e2": cmd_e2, "sweep": cmd_sweep, "launch": cmd_launch,
     "layout": cmd_layout, "e3": cmd_e3, "e7": cmd_e7}[args.cmd](S)


if __name__ == "__main__":
    main()
