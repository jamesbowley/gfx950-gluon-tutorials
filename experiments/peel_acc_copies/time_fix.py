#!/usr/bin/env python3
"""Cold kernel time and accumulator copies of a16w16 builds, interleaved rounds.

    python time_fix.py --versions 9,13 --K 1024,8192 --rounds 3 --configs inproc,llc,llc:-amdgpu-mfma-vgpr-form=0

Configs: "inproc" is bench.py as is (Triton's in-process codegen); "llc[:FLAGS]" builds the amdgcn
stage with llc_stage.py and FLAGS (space-separated within the config, e.g.
"llc:-amdgpu-mfma-vgpr-form=0"). Each run compiles in a fresh TRITON_CACHE_DIR and is timed as in
rocprof_check.py: rocprofv3 --kernel-trace over bench.py --rocprof (1000 launches over a rotating
512 MB), median dispatch duration. Uses the environment's plugin settings (LLVM_PASS_PLUGIN_PATH,
TRITON_AMDGCNAS_PLUGIN), which should be the standard stack. 4096x4096xK bf16.
"""

import argparse
import csv
import glob
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.abspath(os.path.join(HERE, "..", "..", "kernels", "gemm", "intra_wave", "a16w16"))
sys.path[:0] = [HERE, A16]
from compile_stats import asm_stats  # noqa: E402
from v20_streamk_reduce_scatter.rocprof_check import NAMES  # noqa: E402


def run(version, K, config):
    cache = tempfile.mkdtemp(prefix="pac_tc_")
    out = tempfile.mkdtemp(prefix="pac_rp_")
    env = {**os.environ, "TRITON_CACHE_DIR": cache}
    if config == "inproc":
        script = os.path.join(A16, "bench.py")
    else:
        script = os.path.join(HERE, "bench_llc.py")
        env["PAC_LLC_FLAGS"] = config.partition(":")[2]
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", NAMES[version], "-d", out, "-o", "run",
           "--output-format", "csv", "--", sys.executable, script, "--version", str(version), "--M", "4096",
           "--N", "4096", "--K", str(K), "--dtype", "bf16", "--rocprof"]
    p = subprocess.run(cmd, cwd=A16, env=env, capture_output=True, text=True)
    ok = "differ" not in p.stdout and p.returncode == 0
    t = []
    for f in glob.glob(os.path.join(out, "**", "*kernel_trace.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            if NAMES[version] in r["Kernel_Name"]:
                t.append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
    asm = [f for f in glob.glob(os.path.join(cache, "*", "*.amdgcn"))]
    st = asm_stats(open(max(asm, key=os.path.getsize)).read()) if asm else {}
    shutil.rmtree(cache, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    if not ok:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:])
    return (statistics.median(t) if t else float("nan")), st, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", default="9,13")
    ap.add_argument("--K", default="1024,8192")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--configs", default="inproc,llc,llc:-amdgpu-mfma-vgpr-form=0")
    a = ap.parse_args()
    versions = [int(v) for v in a.versions.split(",")]
    Ks = [int(k) for k in a.K.split(",")]
    configs = a.configs.split(",")
    res = {}
    print("| round | K | version | config | kernel us | accvgpr_mov | s_nop | vgpr | spills | correct |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in range(a.rounds):
        for K in Ks:
            for v in versions:
                for c in configs:
                    us, st, ok = run(v, K, c)
                    res.setdefault((K, v, c), []).append(us)
                    print(f"| {r + 1} | {K} | v{v} | {c} | {us:.2f} | {st.get('accvgpr_mov')} | {st.get('s_nop')} | "
                          f"{st.get('vgpr_count')} | {st.get('vgpr_spill_count')} | {ok} |", flush=True)
    print("\n| K | version | config | median of rounds (us) | min | max |")
    print("|---|---|---|---|---|---|")
    for (K, v, c), ts in res.items():
        print(f"| {K} | v{v} | {c} | {statistics.median(ts):.2f} | {min(ts):.2f} | {max(ts):.2f} |")


if __name__ == "__main__":
    main()
