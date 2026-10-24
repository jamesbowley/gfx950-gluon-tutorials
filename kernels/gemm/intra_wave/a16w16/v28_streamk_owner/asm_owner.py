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
"""The owner ending's compiled code in v28, against v27 and the archived experiment variants.

    source v28_streamk_owner/env.sh
    python v28_streamk_owner/asm_owner.py --v27 --compare bs > v28_streamk_owner/results/asm_owner_v28.md
    python v28_streamk_owner/asm_owner.py --archive "" b bs        # the experiment variants alone

v28's own builds (`v28`) are always made. --archive VARIANTS builds the archived experiment kernel
(experiments/matmul_kernel_variants.py, the code WORKLOG sections 1-9 timed) with STREAMK_OWNER_OPT=VARIANT ('' for
v27's ending). --compare VARIANT also builds that archive variant and checks v28's instructions against it.

Each build compiles without launching (matmul_kernel.WARMUP), in its own process with an empty Triton cache,
on BUILDS: per large tile, one-tile owner splits on a shape without a full wave (S = 64: n = 2 and all) and on one
with a full wave (S = 32: n = 2, 4 and all), two-tile and spread on the latter. Per build it reports:
- VGPRs (.amdhsa_next_free_vgpr, arch + acc), AGPRs and VGPR spills;
- the owner's partial reads: the .cv loads (buffer_load ... sc0 sc1) counted between vmcnt(0) waits, as the
  largest batch and the number of batches;
- the poll: global_atomic_or with return (v27's polled atomic_add 0) and global_load_dword ... sc1 (atomic_poll);
- with --v27 / --compare, whether v28's instructions are v27's / the archive variant's (kernel name normalised).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
sys.path.insert(0, A16)

TILES = ("256x256x64", "256x128x64", "128x256x64", "128x128x64", "128x128x128")
ENV_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "STREAMK_PARTIALS", "STREAMK_OWNER_OPT",
            "STREAMK_PLACEMENT", "STREAMK_TILE")
# version -> (module, kernel); "x" is the archived experiment kernel with its STREAMK_OWNER_OPT variants.
NAMES = {27: ("v27_streamk_tile.matmul_kernel", "v27_streamk_tile"), 28: ("v28_streamk_owner.matmul_kernel", "v28_streamk_owner"),
         "x": ("v28_streamk_owner.experiments.matmul_kernel_variants", "v28_streamk_owner")}


def shape(tile, tiles, K):
    """A near-square M x N with exactly `tiles` tiles at the tile."""
    bm, bn = (int(x) for x in tile.split("x")[:2])
    best = None
    for a in range(1, tiles + 1):
        if tiles % a == 0:
            m, n = a * bm, tiles // a * bn
            if best is None or abs(m - n) < abs(best[0] - best[1]):
                best = (m, n)
    return best[0], best[1], K


def builds(tile):
    """(label, (M, N, K), env) of a tile's owner builds."""
    w0 = shape(tile, 64, 8192)
    w1 = shape(tile, 256 + 32, 8192)
    ow = lambda s, n: dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="owner", STREAMK_NUM_PROGRAMS=str(s * n))  # noqa
    return [
        ("w0 ow2", w0, ow(64, 2)),
        ("w0 owall", w0, ow(64, 4)),
        ("w1 ow2", w1, ow(32, 2)),
        ("w1 ow4", w1, ow(32, 4)),
        ("w1 owall", w1, ow(32, 8)),
        ("w1 2t", w1, dict(STREAMK_POLICY="two_tile")),
        ("w1 sp", w1, dict(STREAMK_POLICY="spread")),
    ]


def worker(case):
    import importlib

    import bench_ext  # noqa: F401  (paths; triton before torch)
    import torch

    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in case["env"].items() if v != "" or k != "STREAMK_OWNER_OPT"})
    os.environ["STREAMK_TILE"] = case["tile"]
    module, kernel = NAMES[case["version"]]
    mod = importlib.import_module(module)
    mod.WARMUP = True
    M, N, K = case["shape"]
    a = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((N, K), device="cuda", dtype=torch.bfloat16).T
    mod.matmul(a, b)
    fn = getattr(mod, kernel)
    ks = [k for c in fn.device_caches.values() for k in c[0].values()]
    assert len(ks) == 1, len(ks)
    text = ks[0].asm["amdgcn"]
    json.dump(dict(insts=instructions(text, kernel), vgpr=int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", text)[1]),
                   agpr=int(re.search(r"\.agpr_count:\s*(\d+)", text)[1]),
                   spills=int(re.search(r"\.vgpr_spill_count:\s*(\d+)", text)[1]), launch=mod.LAST_LAUNCH),
              open(case["out"], "w"), default=str)


def instructions(asm, name):
    out = []
    for ln in asm.splitlines():
        if ln.strip().startswith(".amdgpu_metadata"):
            break
        ln = ln.split(";")[0].strip().replace(name, "KERNEL")
        if ln and not ln.startswith("."):
            out.append(ln)
    return out


def compile_fresh(case):
    out = tempfile.mktemp(suffix=".json")
    cache = tempfile.mkdtemp(prefix="v28_asm_")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "worker", json.dumps({**case, "out": out})],
                       env={**os.environ, "TRITON_CACHE_DIR": cache}, capture_output=True, text=True)
    shutil.rmtree(cache, ignore_errors=True)
    if p.returncode:
        return {"error": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else f"exit {p.returncode}"}
    r = json.load(open(out))
    os.remove(out)
    return r


def analyse(insts):
    """(largest batch of .cv partial loads between vmcnt(0) waits, batches, polled atomics, atomic_poll loads,
    ds_write_b32 / ds_read_b32)."""
    batches, cur = [], 0
    for ln in insts:
        if re.match(r"buffer_load_dword(x\d)?\s", ln) and "sc0 sc1" in ln and " lds" not in ln:
            cur += 1
        elif ln.startswith("s_waitcnt") and re.search(r"vmcnt\(0\)", ln):
            if cur:
                batches.append(cur)
            cur = 0
    if cur:
        batches.append(cur)
    atom = sum(1 for ln in insts if ln.startswith("global_atomic_or") and " sc0" in ln)
    poll = sum(1 for ln in insts if re.match(r"global_load_dword\s", ln) and "sc1" in ln)
    ds32 = sum(1 for ln in insts if re.match(r"ds_(write|read)_b32\s", ln))
    return max(batches, default=0), len(batches), atom, poll, ds32


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="report", choices=["report", "worker"])
    ap.add_argument("case", nargs="?")
    ap.add_argument("--archive", nargs="*", default=[], help="archived experiment variants to build ('' = v27's ending)")
    ap.add_argument("--compare", help="an archive variant to check v28's instructions against")
    ap.add_argument("--tiles", nargs="+", default=list(TILES))
    ap.add_argument("--partials", default="fp32")
    ap.add_argument("--v27", action="store_true", help="also build v27 and check v28's instructions against it")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()
    if args.mode == "worker":
        worker(json.loads(args.case))
        return
    archive = list(dict.fromkeys(args.archive + ([args.compare] if args.compare is not None else [])))
    cases = []
    for tile in args.tiles:
        for label, shp, env in builds(tile):
            env = {**env, "STREAMK_PARTIALS": args.partials}
            cases.append(dict(version=28, tile=tile, label=label, shape=shp, env=env, var="v28"))
            for v in archive:
                cases.append(dict(version="x", tile=tile, label=label, shape=shp, env={**env, "STREAMK_OWNER_OPT": v},
                                  var=f"archive {v or 'base'}"))
            if args.v27:
                cases.append(dict(version=27, tile=tile, label=label, shape=shp, env=env, var="v27"))
    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(compile_fresh, cases))
    by = {(c["tile"], c["label"], c["var"]): r for c, r in zip(cases, res)}
    print(f"# v28 owner builds, partials {args.partials}\n")
    print("Batch: the largest run of .cv partial loads between vmcnt(0) waits (and how many runs); atomics: polled "
          "global_atomic_or; poll: global_load_dword sc1 (atomic_poll); ds32: ds_write_b32 + ds_read_b32.\n")
    refs = (["v27"] if args.v27 else []) + ([f"archive {args.compare or 'base'}"] if args.compare is not None else [])
    hdr = "| tile | build | shape | kernel | VGPRs | AGPRs | spills | batch (runs) | atomics | poll | ds32 |"
    hdr += "".join(f" = {r} |" for r in refs)
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    bad = 0
    for c, r in zip(cases, res):
        if c["version"] == 27:
            continue
        row = f"| {c['tile']} | {c['label']} | {'x'.join(map(str, c['shape']))} | {c['var']} |"
        if "error" in r:
            print(row + f" ERROR: {r['error'][:200]} |")
            bad += 1
            continue
        mb, nb, atom, poll, ds32 = analyse(r["insts"])
        row += f" {r['vgpr']} | {r['agpr']} | {r['spills']} | {mb} ({nb}) | {atom} | {poll} | {ds32} |"
        for ref_var in refs:
            ref = by.get((c["tile"], c["label"], ref_var))
            same = "" if c["version"] != 28 else ("yes" if ref and ref.get("insts") == r["insts"] else "**NO**")
            bad += same == "**NO**"
            row += f" {same} |"
        print(row)
    print(f"\n{len(cases)} builds, {bad} problems")


if __name__ == "__main__":
    main()
