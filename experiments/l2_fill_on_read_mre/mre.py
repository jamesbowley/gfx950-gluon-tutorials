"""Minimal reproducer: on gfx950 the first read of a line that was written into L2, but was not
valid in L2 before the write, still fetches the line from memory (see README.md).

    HIP_VISIBLE_DEVICES=3 python mre.py                  # kernel timing, sc0 nt (.cg) chase loads
    HIP_VISIBLE_DEVICES=3 python mre.py --load default   # the same with default-policy chase loads
    HIP_VISIBLE_DEVICES=3 python mre.py --load cv        # the same with cache-volatile (sc0 sc1) chase loads
    HIP_VISIBLE_DEVICES=3 python mre.py --pmc            # rocprofv3 TCC counters (takes --load too)

One wave on one CU. A 1 MiB buffer holds a linked list over its lines. Optionally the wave reads
every line first (PRIME), optionally it writes the whole list (WRITE), then it follows the list
CHASES times with dependent loads. A 1 GiB read before every launch evicts L2 and the 256 MB
last-level cache.
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import tempfile

import triton  # before torch: torch's HIP pulls in ROCm's shared libLLVM, which breaks libtriton's
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

N_LINES = 8192  # 1 MiB of 128-byte lines; element 32 l holds the next line
# For N = 2^k, A % 4 == 1 and B odd make next(l) = (A*l + B) % N
# one cycle through every line, returning to START after N steps.
A, B = 4949, 2731
START = 1
REPS = 10  # launches per configuration, after one warm-up
MODES = {  # (PRIME, WRITE)
    "plain miss": (False, False),
    "L2 hit (primed, not written)": (True, False),
    "written into L2": (False, True),
    "primed, then written": (True, True),
}
LOADS = {  # --load: (Triton cache modifier, description)
    "cg": (".cg", "sc0 nt (.cg)"),
    "default": ("", "default"),
    "cv": (".cv", "sc0 sc1 (.cv, cache volatile)"),
}
COUNTERS = {
    "L2 hits": "TCC_HIT_sum",
    "L2 misses": "TCC_MISS_sum",
    "memory reads": "TCC_EA0_RDREQ_sum",
    "of which DRAM": "TCC_EA0_RDREQ_DRAM_sum",
}


@gluon.jit
def mre_kernel(chain_ptr, out_ptr, a, b, start, N: gl.constexpr, PRIME: gl.constexpr, WRITE: gl.constexpr,
               CHASES: gl.constexpr, LOAD: gl.constexpr):
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    if PRIME:
        # one dword of every line; storing the sum keeps the loads
        lines = gl.arange(0, 64, L1)
        s = gl.zeros([64], gl.int32, L1)
        for t in range(0, N * 32, 64 * 32):
            s += gl.amd.cdna4.buffer_load(ptr=chain_ptr + t, offsets=lines * 32)
        gl.store(out_ptr + 2 + lines, s)
    if WRITE:
        # every byte of every line, 8 lines per store
        L4: gl.constexpr = gl.BlockedLayout([4], [64], [1], [0])
        offs = gl.arange(0, 256, L4)
        for t in range(0, N * 32, 256):
            e = t + offs
            nxt = ((e // 32) * a + b) % N
            gl.amd.cdna4.buffer_store(stored_value=gl.where(e % 32 == 0, nxt, 0), ptr=chain_ptr + t, offsets=offs)
    # Each chase step waits with s_waitcnt vmcnt(0); the first one also drains the prime and the write.
    for c in gl.static_range(CHASES):
        p = gl.full([1], start, gl.int32, L1)
        for _ in range(N):
            p = gl.amd.cdna4.buffer_load(ptr=chain_ptr, offsets=p * 32, cache=LOAD)
        gl.store(out_ptr + c + gl.zeros([1], gl.int32, L1), p)


class Buffers:

    def __init__(self):
        self.chain = torch.empty(N_LINES * 32, device="cuda", dtype=torch.int32)
        self.list = torch.zeros_like(self.chain)
        self.list[0::32] = (torch.arange(N_LINES, device="cuda") * A + B) % N_LINES
        self.out = torch.empty(2 + 64, device="cuda", dtype=torch.int32)
        self.flush = torch.ones(2**28, device="cuda")


def run(bufs, prime, write, chases, load):
    """One launch from a known cache state. Returns the kernel time in us."""
    # The write cases start from zeros: a chase that read the old copy would stick at line 0.
    if write:
        bufs.chain.zero_()
    else:
        bufs.chain.copy_(bufs.list)
    bufs.out.fill_(-1)
    bufs.flush.sum()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    mre_kernel[(1, )](bufs.chain, bufs.out, A, B, START, N=N_LINES, PRIME=prime, WRITE=write, CHASES=chases,
                      LOAD=load, num_warps=1)
    e1.record()
    torch.cuda.synchronize()
    assert bufs.out[:chases].tolist() == [START] * chases, "the chase did not return to its start line"
    return e0.elapsed_time(e1) * 1e3


def timing(load_name):
    load, desc = LOADS[load_name]
    bufs = Buffers()
    print(f"Chase loads: {desc}. One wave, {N_LINES} lines (1 MiB), median "
          f"of {REPS} launches after one warm-up, a 1 GiB read before each, CUDA events around the kernel. "
          f"Per load: kernel time with one chase minus with none (first chase), two minus one (second chase), "
          f"divided by {N_LINES}.\n")
    print("| case | kernel us, 0 / 1 / 2 chases | first chase ns per load | second chase ns per load |")
    print("|---|---|---|---|")
    for name, (prime, write) in MODES.items():
        t = [statistics.median([run(bufs, prime, write, c, load) for _ in range(REPS + 1)][1:]) for c in (0, 1, 2)]
        first, second = ((t[c + 1] - t[c]) * 1e3 / N_LINES for c in (0, 1))
        print(f"| {name} | {t[0]:.0f} / {t[1]:.0f} / {t[2]:.0f} | {first:.0f} | {second:.0f} |", flush=True)


PMC_CONFIGS = [(prime, write, chases) for prime, write in MODES.values() for chases in (0, 1)]


def pmc_worker(load_name):
    bufs = Buffers()
    for prime, write, chases in PMC_CONFIGS:
        for _ in range(REPS + 1):
            run(bufs, prime, write, chases, LOADS[load_name][0])


def pmc(load_name):
    d = tempfile.mkdtemp(prefix="l2mre_")
    cmd = ["rocprofv3", "--pmc", *COUNTERS.values(), "--kernel-include-regex", "mre_kernel", "-d", d, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "--worker", "--load", load_name]
    # rocprofv3 refuses to start under a process that has loaded HIP, which sets these.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROCPROFILER_REGISTER")}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode:
        sys.exit(p.stderr[-3000:])
    by = {}  # dispatch id -> counter -> value
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                by.setdefault(int(r["Dispatch_Id"]), {})[r["Counter_Name"]] = float(r["Counter_Value"])
    disp = sorted(by)
    assert len(disp) == len(PMC_CONFIGS) * (REPS + 1), len(disp)
    # per configuration: the median per line over its launches after the warm-up
    res = [{c: statistics.median(by[x][c] for x in disp[i * (REPS + 1) + 1:(i + 1) * (REPS + 1)]) / N_LINES
            for c in COUNTERS.values()} for i in range(len(PMC_CONFIGS))]

    print(f"Chase loads: {LOADS[load_name][1]}. Per 128-byte line of the 1 MiB buffer ({N_LINES} lines), median of "
          f"{REPS} launches after one warm-up, "
          "a 1 GiB read before each. 'Before the chase' is the kernel without the chase (prime and write only); "
          "'chase' is the kernel with one chase minus that.\n")
    print(f"| case | phase | {' | '.join(COUNTERS)} |")
    print("|---|---|" + "---|" * len(COUNTERS))
    for k, name in enumerate(MODES):
        before, after = res[2 * k], res[2 * k + 1]
        print(f"| {name} | before the chase | " + " | ".join(f"{before[c]:.2f}" for c in COUNTERS.values()) + " |")
        print("| | chase | " + " | ".join(f"{after[c] - before[c]:.2f}" for c in COUNTERS.values()) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pmc", action="store_true", help="collect TCC counters with rocprofv3")
    p.add_argument("--load", choices=list(LOADS), default="cg",
                   help="cache policy of the chase loads: cg (sc0 nt), default, or cv (sc0 sc1, cache volatile)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        pmc_worker(args.load)
        return
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}\n")
    if args.pmc:
        pmc(args.load)
    else:
        timing(args.load)


if __name__ == "__main__":
    main()
