#!/usr/bin/env python3
"""Is the shader clock where it was pinned? A compute-only kernel, timed three ways.

    python clock_check.py --gpu 0

A dependent FP32 FMA chain runs for n and 2n iterations. At a fixed clock the 2n run takes twice
the wall time, and cycles per iteration stay constant. The shader clock is the wave's s_memtime
cycles over s_memrealtime (100 MHz) time, and also cycles over host-event time. The SMU's
current_gfxclks (amdsmi gpu_metrics) is sampled during the run, along with the sysfs DPM state.
"""

import argparse
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUS = {0: "0000:05:00.0"}
p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--n", type=int, default=2_000_000)
args = p.parse_args()
os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
sys.path.insert(0, os.path.join(REPO, "scripts", "triton_deepbind_shim"))
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/share/amd_smi")

import amdsmi  # noqa: E402
import torch  # noqa: E402
from triton.experimental import gluon  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402


@gluon.jit
def _stamp(z, ASM: gl.constexpr):
    return gl.max(gl.inline_asm_elementwise(ASM, "=s,v,~{memory}", [z], dtype=gl.int64, is_pure=False, pack=1),
                  axis=0)


@gluon.jit
def spin(out, clk, n_iter):
    lay: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    z = gl.zeros([1], gl.int32, lay)
    c0 = _stamp(z, "s_memtime $0\ns_waitcnt lgkmcnt(0)")
    r0 = _stamp(z, "s_memrealtime $0\ns_waitcnt lgkmcnt(0)")
    x = gl.arange(0, 256, lay).to(gl.float32)
    for _ in range(n_iter):
        x = x * 1.0000001 + 0.5
    c1 = _stamp(z, "s_memtime $0\ns_waitcnt lgkmcnt(0)")
    r1 = _stamp(z, "s_memrealtime $0\ns_waitcnt lgkmcnt(0)")
    pid = gl.program_id(0)
    gl.store(out + pid, gl.sum(x, axis=0))
    k = gl.arange(0, 256, lay).to(gl.int64)
    gl.store(clk + pid * 4 + k, gl.where(k == 0, c0, gl.where(k == 1, r0, gl.where(k == 2, c1, r1))), mask=k < 4)


def sysfs():
    for c in os.listdir("/sys/class/drm"):
        d = f"/sys/class/drm/{c}/device"
        try:
            if f"PCI_SLOT_NAME={BUS[args.gpu]}" in open(f"{d}/uevent").read():
                return {f: open(f"{d}/{f}").read().split() for f in
                        ("power_dpm_force_performance_level", "pp_dpm_sclk", "pp_od_clk_voltage")}
        except OSError:
            pass
    return {}


def main():
    amdsmi.amdsmi_init()
    h = next(x for x in amdsmi.amdsmi_get_processor_handles()
             if amdsmi.amdsmi_get_gpu_device_bdf(x).lower() == BUS[args.gpu])
    s = sysfs()
    print("perf level", s.get("power_dpm_force_performance_level"), "| pp_dpm_sclk", " ".join(s.get("pp_dpm_sclk", [])),
          "| OD_SCLK", " ".join(x for x in s.get("pp_od_clk_voltage", [])[:5]))
    for G in (256, 1024):  # 1 or 4 workgroups (4 or 16 waves) per CU: two power levels
        run(h, G)


def run(h, G):
    out = torch.empty(G, device="cuda")
    clk = torch.zeros(G * 4, dtype=torch.int64, device="cuda")
    n0 = args.n | 1  # odd, like every n below, so Triton's divisibility specialisation never recompiles
    spin[(G,)](out, clk, 1001, num_warps=4)
    torch.cuda.synchronize()
    res = {}
    for n in (n0, 2 * n0 + 1):
        samples, stop = [], threading.Event()

        def sample():
            while not stop.is_set():
                samples.append(statistics.mean(amdsmi.amdsmi_get_gpu_metrics_info(h)["current_gfxclks"]))
                time.sleep(0.002)

        th = threading.Thread(target=sample)
        th.start()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        spin[(G,)](out, clk, n, num_warps=4)
        e1.record()
        torch.cuda.synchronize()
        stop.set()
        th.join()
        ck = clk.view(G, 4).cpu().tolist()
        ghz = statistics.median((x[2] - x[0]) / (x[3] - x[1]) / 10 for x in ck)
        cyc = statistics.median(x[2] - x[0] for x in ck)
        ms = e0.elapsed_time(e1)
        res[n] = ms
        print(f"{G // 256} WG/CU n={n}: {ms:.1f} ms (host events); s_memtime/s_memrealtime {ghz:.3f} GHz; "
              f"{cyc / n:.3f} cycles per iteration; SMU current_gfxclks median {statistics.median(samples):.0f} MHz")
    ns = sorted(res)
    print(f"{G // 256} WG/CU: time ratio {res[ns[1]] / res[ns[0]]:.3f} for {ns[1] / ns[0]:.3f}x the iterations")


main()
