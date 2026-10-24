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
"""v29's compiled code against v28's, build by build.

    source v29_streamk_compact/env.sh
    python v29_streamk_compact/asm_diff.py > v29_streamk_compact/results/asm_diff.md
    python v29_streamk_compact/asm_diff.py --tiles 128x128x64 --builds "w1 ow2" "w1 rs8"

Every build compiles without launching (matmul_kernel.WARMUP), in its own process with an empty Triton cache,
once for v28 (fp32 partials, its defaults otherwise) and once for v29, with the same STREAMK_* settings. The
instructions are compared with the kernel name normalised. Per tile, on near-square shapes with an exact tile
count (NP = the tile's programs):
- w0: NP / 4 tiles (no full wave), K = 8192; w1: NP + NP / 8 tiles; w1b: NP + NP / 16 tiles;
- rw1 / rw0: w1 / w0 with M, N and K ragged (rw0 has no full tile, so no persistent loop);
- dp, cap (the default rule), ow{n} / rs{n} (one-tile owner / reduce-scatter on n programs per tile, "all"
  for every program), l2 (rs with the L2 exchange), 2t (two-tile) and sp (spread); "+b" with bias.
A build the rule sends to v9_tile in both versions is reported as such and not compared.
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

TILES = ("256x256x64", "256x128x64", "128x256x64", "128x128x64", "128x128x128",
         "16x256x128", "32x128x128", "128x32x128", "16x16x256")
ENV_KEYS = ("STREAMK_POLICY", "STREAMK_FIXUP", "STREAMK_NUM_PROGRAMS", "STREAMK_PARTIALS", "STREAMK_EXCHANGE",
            "STREAMK_TILE", "STREAMK_OWNER_OPT", "STREAMK_PLACEMENT", "STREAMK_RS_READERS", "STREAMK_XCD_LOCAL")
VERSIONS = {28: ("v28_streamk_owner.matmul_kernel", "v28_streamk_owner"),
            29: ("v29_streamk_compact.matmul_kernel", "v29_streamk_compact")}


def shape(bm, bn, tiles, K):
    """A near-square M x N with exactly `tiles` bm x bn tiles."""
    best = None
    for a in range(1, tiles + 1):
        if tiles % a == 0:
            m, n = a * bm, tiles // a * bn
            if best is None or abs(m - n) < abs(best[0] - best[1]):
                best = (m, n)
    return best[0], best[1], K


def ragged(shp, bm, bn, K):
    M, N, _ = shp
    return M - max(1, bm * 3 // 32) if bm > 1 else M, N - max(1, bn * 3 // 16), K


def builds(tile):
    """(label, (M, N, K), env, bias) of a tile's builds."""
    import importlib

    import bench_ext  # noqa: F401  (paths)

    mk = importlib.import_module(VERSIONS[29][0])
    geo = mk.tile_geometry(tile)
    bm, bn = mk.parse_spec(tile)[:2]
    NP = geo["num_programs"]
    s0, s1, s1b = NP // 4, NP // 8, NP // 16
    w0, w1, w1b = shape(bm, bn, s0, 8192), shape(bm, bn, NP + s1, 8192), shape(bm, bn, NP + s1b, 8192)
    rw1, rw0 = ragged(w1, bm, bn, 8176), ragged(w0, bm, bn, 8080)

    def one(fixup, s, n, **extra):
        env = dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP=fixup, **extra)
        if n:
            env["STREAMK_NUM_PROGRAMS"] = str(s * n)
        return env

    out = [
        ("w1 dp", w1, dict(STREAMK_POLICY="dp"), 0),
        ("w1 dp+b", w1, dict(STREAMK_POLICY="dp"), 1),
        ("w0 cap", w0, {}, 0),
        ("w1 cap", w1, {}, 0),
        ("w0 ow2", w0, one("owner", s0, 2), 0),
        ("w0 owall", w0, one("owner", s0, None), 0),
        ("w1 ow2", w1, one("owner", s1, 2), 0),
        ("w1 ow2+b", w1, one("owner", s1, 2), 1),
        ("w1 owall", w1, one("owner", s1, None), 0),
        ("w1b ow16", w1b, one("owner", s1b, 16), 0),
        ("w0 rsall", w0, one("rs", s0, None), 0),
        ("w1 rs8", w1, one("rs", s1, 8), 0),
        ("w1 rsall+b", w1, one("rs", s1, None), 1),
        ("w1 l2", w1, one("rs", s1, None, STREAMK_EXCHANGE="l2"), 0),
        ("w0 l2", w0, one("rs", s0, None, STREAMK_EXCHANGE="l2"), 0),
        ("w1 2t", w1, dict(STREAMK_POLICY="two_tile"), 0),
        ("w1 2t+b", w1, dict(STREAMK_POLICY="two_tile"), 1),
        ("w1 sp", w1, dict(STREAMK_POLICY="spread"), 0),
        ("rw1 dp", rw1, dict(STREAMK_POLICY="dp"), 0),
        ("rw1 ow4", rw1, one("owner", s1, 4), 0),
        ("rw1 rsall+b", rw1, one("rs", s1, None), 1),
        ("rw1 2t", rw1, dict(STREAMK_POLICY="two_tile"), 0),
        ("rw1 sp+b", rw1, dict(STREAMK_POLICY="spread"), 1),
        ("rw0 ow2", rw0, one("owner", s0, 2), 0),
        ("rw0 rsall", rw0, one("rs", s0, None), 0),
    ]
    return out


def worker(case):
    import importlib

    import bench_ext  # noqa: F401  (paths; triton before torch)
    import torch

    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(case["env"])
    os.environ["STREAMK_TILE"] = case["tile"]
    module, kernel = VERSIONS[case["version"]]
    mod = importlib.import_module(module)
    mod.WARMUP = True
    M, N, K = case["shape"]
    a = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((N, K), device="cuda", dtype=torch.bfloat16).T
    bias = torch.empty((N,), device="cuda", dtype=torch.bfloat16) if case["bias"] else None
    mod.LAST_LAUNCH.clear()
    mod.matmul(a, b, bias=bias)
    fn = getattr(mod, kernel)
    ks = {id(k): k for c in fn.device_caches.values() for k in c[0].values()}
    if not ks:
        # WARMUP returns before LAST_LAUNCH is set, except on the v9_tile fallback.
        json.dump(dict(fallback=mod.LAST_LAUNCH.get("kernel")), open(case["out"], "w"), default=str)
        return
    assert len(ks) == 1, len(ks)
    k = next(iter(ks.values()))
    text = k.asm["amdgcn"]
    if case.get("dump"):
        os.makedirs(case["dump"], exist_ok=True)
        stem = os.path.join(case["dump"], f"{case['tile']}_{case['label'].replace(' ', '_')}_v{case['version']}")
        for ext in ("ttgir", "llir", "amdgcn"):
            with open(f"{stem}.{ext}", "w") as f:
                f.write(k.asm[ext].replace(kernel, "KERNEL"))
    json.dump(dict(insts=instructions(text, kernel), vgpr=int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", text)[1]),
                   spills=int(re.search(r"\.vgpr_spill_count:\s*(\d+)", text)[1])),
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
    cache = tempfile.mkdtemp(prefix="v29_asm_")
    p = subprocess.run([sys.executable, "-W", "ignore", os.path.abspath(__file__), "worker", json.dumps({**case, "out": out})],
                       env={**os.environ, "TRITON_CACHE_DIR": cache}, capture_output=True, text=True)
    shutil.rmtree(cache, ignore_errors=True)
    if p.returncode:
        lines = p.stderr.strip().splitlines()
        return {"error": " / ".join(lines[-3:]) if lines else f"exit {p.returncode}"}
    r = json.load(open(out))
    os.remove(out)
    return r


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i}: `{x}` / `{y}`"
    return f"lengths {len(a)} / {len(b)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="report", choices=["report", "worker"])
    ap.add_argument("case", nargs="?")
    ap.add_argument("--tiles", nargs="+", default=list(TILES))
    ap.add_argument("--builds", nargs="*", help="only these build labels")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--dump", help="also write each build's ttgir, llir and amdgcn to this directory")
    args = ap.parse_args()
    if args.mode == "worker":
        worker(json.loads(args.case))
        return
    cases = []
    for tile in args.tiles:
        for label, shp, env, bias in builds(tile):
            if args.builds and label not in args.builds:
                continue
            for v in (28, 29):
                cases.append(dict(version=v, tile=tile, label=label, shape=shp, env=env, bias=bias,
                                  dump=os.path.abspath(args.dump) if args.dump else None))
    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(compile_fresh, cases))
    by = {(c["tile"], c["label"], c["version"]): r for c, r in zip(cases, res)}
    print("# v29 against v28: compiled code per build\n")
    print("| tile | build | shape | v28 VGPRs / spills | v29 VGPRs / spills | identical |")
    print("|---|---|---|---|---|---|")
    same = differ = fallback = errors = 0
    seen = []
    for c in cases:
        key = (c["tile"], c["label"])
        if key in seen:
            continue
        seen.append(key)
        r28, r29 = by[(*key, 28)], by[(*key, 29)]
        shp = "x".join(map(str, c["shape"]))
        if "error" in r28 or "error" in r29:
            errors += 1
            print(f"| {key[0]} | {key[1]} | {shp} | {r28.get('error', '')[:200]} | {r29.get('error', '')[:200]} | ERROR |")
            continue
        if "fallback" in r28 or "fallback" in r29:
            ok = r28.get("fallback") == "v9_tile" and r29.get("fallback") == "v9_tile"
            fallback += ok
            errors += not ok
            print(f"| {key[0]} | {key[1]} | {shp} | {r28.get('fallback', 'stream-K')} | {r29.get('fallback', 'stream-K')} | "
                  f"{'v9_tile in both' if ok else 'MISMATCH'} |")
            continue
        ident = r28["insts"] == r29["insts"]
        same += ident
        differ += not ident
        print(f"| {key[0]} | {key[1]} | {shp} | {r28['vgpr']} / {r28['spills']} | {r29['vgpr']} / {r29['spills']} | "
              f"{'yes' if ident else '**NO**: ' + first_diff(r28['insts'], r29['insts'])} |")
    print(f"\n{len(seen)} builds: {same} identical, {differ} different, {fallback} v9_tile in both, {errors} errors")


if __name__ == "__main__":
    main()
