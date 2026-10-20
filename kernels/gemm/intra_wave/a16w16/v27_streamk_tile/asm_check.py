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
"""v27's compiled code: at 256x256x64 it must be v26's; on the other tiles, its registers.

    source v27_streamk_tile/env.sh
    python v27_streamk_tile/asm_check.py gate > v27_streamk_tile/results/asm_gate.md
    python v27_streamk_tile/asm_check.py tiles > v27_streamk_tile/results/asm_tiles.md

gate: every configuration of v26's asm check (v24's SAME + OWN), on 4096x8192xK with GEMM_TILE_LIMIT =
256 + S, at K, K - 16 and K - 112 (EDGE_K), and at K - 112 with STREAMK_FORCE_EDGE=1 (all three edges),
built by v26 and by v27 at its default tile, each in its own process with an empty Triton cache. The
instruction streams (kernel name replaced, comments, directives and argument metadata dropped) and the
VGPR, AGPR and spill counts must match.

tiles: each quadrant tile and the small set, on a shape with at least NUM_PROGRAMS + S tiles and
GEMM_TILE_LIMIT = NUM_PROGRAMS + S: VGPRs, AGPRs, spills and LDS of every ending, partial type,
exchange, placement and split the host allows, aligned and with every edge forced (K - 112).
"""

import difflib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import bench_ext as bench  # noqa: E402,F401  (imports triton before torch)
import torch  # noqa: E402
from v24_streamk_combined import asm_check as a24  # noqa: E402
from v26_streamk_ragged_k import matmul_kernel as v26  # noqa: E402
from v27_streamk_tile import matmul_kernel as v27  # noqa: E402

CASES = a24.SAME + a24.OWN
ENV_KEYS = a24.ENV_KEYS + ("STREAMK_FORCE_EDGE", "STREAMK_TILE", "STREAMK_EXCHANGE")
QUADRANT_TILES = ["256x256", "256x128", "128x256", "128x128x128", "128x128", "64x64x128", "64x256", "256x64x64"]
SMALL_TILES = ["16x256x128", "16x64x128", "32x128x128", "1x256x128", "256x16x128"]


def compile_fresh(version, M, N, K, limit, env, force_edge=False, gpu=None):
    """compile_case in a new process with an empty Triton cache: (instructions, (VGPRs, AGPRs, spills)).
    Builds that share a process can differ in the order of a commutative operation (the lowering
    depends on what the process compiled before), so the gate compares fresh builds only."""
    import json
    import subprocess
    import tempfile

    import shutil

    out = tempfile.mktemp(suffix=".json")
    cache = tempfile.mkdtemp(prefix="v27_asm_")
    subprocess.run([sys.executable, os.path.abspath(__file__), "worker", json.dumps(
        dict(version=version, M=M, N=N, K=K, limit=limit, env=env, force_edge=force_edge, out=out))],
        env={**os.environ, "TRITON_CACHE_DIR": cache, **({"HIP_VISIBLE_DEVICES": gpu} if gpu else {})}, check=True,
        capture_output=True)
    r = json.load(open(out))
    os.remove(out)
    shutil.rmtree(cache, ignore_errors=True)
    return r["insts"], tuple(r["regs"]), r["grid"]


def worker(case):
    import json

    mod = v26 if case["version"] == 26 else v27
    k = compile_case(mod, case["M"], case["N"], case["K"], case["limit"], case["env"], case["force_edge"])
    name = "v26_streamk_ragged_k" if mod is v26 else "v27_streamk_tile"
    # The launched grid: v27's compact launch must not apply (every gate configuration has full tiles).
    grid = v27.LAST_LAUNCH["grid"] if mod is v27 else 256
    json.dump(dict(insts=a24.instructions(k.asm["amdgcn"], name), regs=regs(k), grid=grid), open(case["out"], "w"))


def compile_case(mod, M, N, K, limit, env, force_edge=False):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    os.environ["GEMM_TILE_LIMIT"] = str(limit)
    if force_edge:
        os.environ["STREAMK_FORCE_EDGE"] = "1"
    fn = mod.v26_streamk_ragged_k if mod is v26 else mod.v27_streamk_tile
    for c in fn.device_caches.values():
        c[0].clear()
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).T
    mod.matmul(a, b)
    torch.cuda.synchronize()
    ks = [k for c in fn.device_caches.values() for k in c[0].values()]
    assert len(ks) == 1, len(ks)
    return ks[0]


def changed_lines(x, y):
    return sum(1 for d in difflib.unified_diff(x, y, lineterm="", n=0)
               if d[:1] in "+-" and not d.startswith(("+++", "---")))


def regs(k):
    """(VGPRs, AGPRs, VGPR spills); VGPRs as the loaded kernel's n_regs (arch + acc, allocated), which a
    kernel compiled without launching does not have yet: then from the .amdgcn."""
    text = k.asm["amdgcn"]
    agpr = int(re.search(r"\.agpr_count:\s*(\d+)", text).group(1))
    if hasattr(k, "n_regs"):
        return k.n_regs, agpr, k.n_spills
    vgpr = int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", text).group(1))
    spills = int(re.search(r"\.vgpr_spill_count:\s*(\d+)", text).group(1))
    return vgpr, agpr, spills


def gate():
    import queue
    from concurrent.futures import ThreadPoolExecutor

    # Each build launches its kernel, so a job holds a GPU of its own (GATE_GPUS, default the visible one):
    # two stream-K launches on one GPU need not all be resident.
    gpus = queue.Queue()
    for g in os.environ.get("GATE_GPUS", os.environ.get("HIP_VISIBLE_DEVICES", "0")).split(","):
        gpus.put(g)

    def fresh(*a):
        g = gpus.get()
        try:
            return compile_fresh(*a, gpu=g)
        finally:
            gpus.put(g)

    ok = True
    M, N = a24.M, a24.N
    print("# v27 gate: 256x256x64 against v26\n")
    print(f"{M}x{N}xK, GEMM_TILE_LIMIT = 256 + S; v26 and v27 (default tile) on the same configuration. Instruction "
          "lines (kernel name replaced, comments and directives dropped), VGPRs / AGPRs / spills, and v27's launched "
          "grid (must be all 256 programs: the compact launch never applies with full tiles).\n")
    print("| K | S | configuration | build | v26 VGPRs / AGPRs / spills | v27 | instructions | lines changed | grid | "
          "result |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    builds = [(K, S, env, label, k, forced) for K, S, env in CASES
              for label, k, forced in (("K", K, False), ("K - 16", K - 16, False), ("K - 112", K - 112, False),
                                       ("K - 112, all edges", K - 112, True))]
    # Every build is its own process with its own empty cache, so they can run side by side.
    with ThreadPoolExecutor(gpus.qsize()) as ex:
        futs = [(b, ex.submit(fresh, 26, M, N, b[4], 256 + b[1], b[2], b[5]),
                 ex.submit(fresh, 27, M, N, b[4], 256 + b[1], b[2], b[5])) for b in builds]
        for (K, S, env, label, _, _), f26, f27 in futs:
            (i26, r26, _), (i27, r27, grid) = f26.result(), f27.result()
            diff = changed_lines(i26, i27)
            same = diff == 0 and r26 == r27 and grid == 256
            ok &= same
            print(f"| {K} | {S} | {a24.label(env)} | {label} | {' / '.join(map(str, r26))} | "
                  f"{' / '.join(map(str, r27))} | {len(i27)} | {diff} | {grid} | {'identical' if same else 'FAIL'} |",
                  flush=True)
    print("\nPASS" if ok else "\nFAIL")
    return ok


def tile_configs(geo):
    """(label, env) of every ending / partial type / exchange / placement / split the host takes for
    a tile: S = 8 one-tile at 16 and 12.5 programs per tile (EXTRA > 0), and the end-to-end splits."""
    out = [("dp", dict(STREAMK_POLICY="dp")), ("two_tile", dict(STREAMK_POLICY="two_tile")),
           ("spread", dict(STREAMK_POLICY="spread"))]
    parts = ["fp32", "bf16"] + (["bf16_pairs"] if not geo["small"] and geo["block_n"] >= 128 else [])
    for fixup in ("rs", "owner", "atomic_iz"):
        for partials in (parts if fixup != "atomic_iz" else ["fp32"]):
            for snp in ("128", "100"):
                for placement in ("tile", "chunk"):
                    out.append((f"{fixup} {partials} n={snp} {placement}",
                                dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP=fixup, STREAMK_PARTIALS=partials,
                                     STREAMK_NUM_PROGRAMS=snp, STREAMK_PLACEMENT=placement)))
    for partials in parts:
        out.append((f"rs {partials} l2", dict(STREAMK_POLICY="one_tile", STREAMK_FIXUP="rs", STREAMK_PARTIALS=partials,
                                              STREAMK_NUM_PROGRAMS="128", STREAMK_EXCHANGE="l2")))
    return out


def tile_worker(case):
    """Compile one tiles() case without launching it (matmul_kernel.WARMUP): regs and LDS."""
    import json

    v27.WARMUP = True
    k = compile_case(v27, case["M"], case["N"], case["K"], case["limit"], case["env"], case["force_edge"])
    json.dump(dict(regs=regs(k), shared=k.metadata.shared), open(case["out"], "w"))


def compile_warm(case):
    import json
    import subprocess
    import tempfile

    out = tempfile.mktemp(suffix=".json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "tile_worker", json.dumps({**case, "out": out})],
                       capture_output=True, text=True)
    if p.returncode:
        return None, (p.stderr.strip().splitlines() or ["?"])[-1][:120]
    r = json.load(open(out))
    os.remove(out)
    return r, None


def tiles(jobs=32):
    from concurrent.futures import ThreadPoolExecutor

    print("# v27 registers per tile\n")
    print("GEMM_TILE_LIMIT = NUM_PROGRAMS + 8 (S = 8), K = 8192 (aligned) and 8080 with STREAMK_FORCE_EDGE=1 "
          "(all three edges). VGPRs / AGPRs / spills; LDS in KB. Compiled without launching (matmul_kernel.WARMUP).\n")
    print("| tile | shape | configuration | aligned | all edges | LDS |")
    print("|---|---|---|---|---|---|")
    cases = []
    for t in QUADRANT_TILES + SMALL_TILES:
        geo = v27.tile_geometry(t)
        # At least 1024 tiles: 128 rows of small tiles (C^T's columns when swapped), 256 tiles along N.
        if geo["swap"]:
            M, N = 256 * geo["block_n"], 128
        elif geo["small"]:
            M, N = 128, 256 * geo["block_n"]
        else:
            M, N = 4096, 8192
        for label, env in tile_configs(geo):
            env = {**env, "STREAMK_TILE": t}
            base = dict(M=M, N=N, limit=geo["num_programs"] + 8, env=env)
            cases.append((t, M, N, label, dict(base, K=8192, force_edge=False), dict(base, K=8080, force_edge=True)))
    worst = 0
    with ThreadPoolExecutor(jobs) as ex:
        futs = [(c, ex.submit(compile_warm, c[4]), ex.submit(compile_warm, c[5])) for c in cases]
        for (t, M, N, label, _, _), fa, ff in futs:
            (ra, ea), (rf, ef) = fa.result(), ff.result()
            cells = []
            for r, e in ((ra, ea), (rf, ef)):
                if r is None:
                    cells.append(f"error: {e}")
                else:
                    worst = max(worst, r["regs"][2])
                    cells.append(" / ".join(map(str, r["regs"])))
            lds = f"{ra['shared'] / 1024:.1f}" if ra else ""
            print(f"| {t} | {M}x{N} | {label} | {cells[0]} | {cells[1]} | {lds} |", flush=True)
    print(f"\nlargest spill count: {worst}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if what == "worker":
        import json

        worker(json.loads(sys.argv[2]))
    elif what == "tile_worker":
        import json

        tile_worker(json.loads(sys.argv[2]))
    elif what == "gate":
        sys.exit(0 if gate() else 1)
    else:
        tiles()
