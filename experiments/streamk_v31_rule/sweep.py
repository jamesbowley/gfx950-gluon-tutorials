"""The v31 rule sweep: every point of grid.py timed under rocprofv3 on one GPU.

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh; cd $REPO
    python experiments/streamk_v31_rule/sweep.py warm --points GRID --jobs 64
    python experiments/streamk_v31_rule/sweep.py timing --points GRID --out experiments/streamk_v31_rule/results/grid_r1.csv

Each point is checked against torch (fp32 reference), then replayed from a HIP graph
of GRAPH_LEN launches over ROTATING_MB of rotating inputs (cold caches, back to
back) under rocprofv3 --kernel-trace: the median of the last 100 dispatches. Points
run in chunks, one rocprofv3 process per chunk, shuffled within a shape's block so
that one shape's configurations are timed together. Before each chunk the GPU must
be free of foreign processes and after it the chunk is checked again: a foreign
process stops the run (exit 4) and the chunk's rows are dropped. The CSV is appended
chunk by chunk and a rerun resumes. warm compiles every point's build into Triton's
cache (JITFunction.warmup: no launches) in parallel processes.
"""

import argparse
import csv
import glob
import json
import os
import queue
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid  # noqa: E402
import gpu_guard  # noqa: E402

TMP = os.path.join(HERE, "tmp")
GRAPH_LEN = 100
ROTATING_MB = 2048
SMI_INDEX = int(os.environ.get("V31_SMI_INDEX", "1"))
KERNEL = "_gemm_a16w16_compute_bound_kernel"
FIELDS = ("point", "kind", "tile", "M", "N", "K", "T", "S", "W", "k", "pairs", "policy", "fixup",
          "split_n", "exchange", "persistent", "grid", "correct", "error", "us_median", "tflops",
          "iters", "round", "chunk", "pos", "ts")


def points_for(name):
    if name == "GRID":
        return grid.coarse_points()
    if name == "HELDOUT":
        return grid.heldout_points()
    if name == "TILESWEEP":
        return grid.tilesweep_points()
    if name == "GAP":
        return grid.gap_points()
    if name == "FORMULA":
        return grid.formula_points()
    if name == "JOINT_TRAIN":
        return grid.joint_points(grid.tilesweep_shapes())
    if name == "JOINT_HELDOUT":
        return grid.joint_points(grid.heldout_shapes())
    if name.startswith("@"):
        return [line.strip() for line in open(name[1:]) if line.strip()]
    return name.split(",")


def run_fn(kind, tile):
    from v31_streamk_aiter import matmul_kernel as mk

    config = grid.config_for(kind, tile)
    G = grid.streamk_grid(kind)

    def run(x, w, y):
        mk.gemm_a16w16(x, w, dtype=y.dtype, y=y, config=config, streamk_grid=G)

    return run


def worker(args):
    import bench_ext as bench
    import torch

    dt = torch.bfloat16
    manifest, cache = [], {}
    for p in args.points.split(","):
        kind, tile, M, N, K = grid.parse_point(p)
        iters = grid.shape_iters(tile, M, N, K, GRAPH_LEN)
        run = run_fn(kind, tile)
        if cache.get("shape") != (M, N, K):
            cache.clear()
            torch.cuda.empty_cache()
            x = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
            w = torch.rand((N, K), device="cuda", dtype=dt) - 0.5
            ref = (x.float() @ w.float().T).to(dt)
            rot = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
            cache.update(shape=(M, N, K), x=x, w=w, ref=ref, rot=rot)
        x, w, ref = cache["x"], cache["w"], cache["ref"]
        a_l, b_l, c_l, copies = cache["rot"]
        y = torch.empty((M, N), device="cuda", dtype=dt)
        run(x, w, y)
        torch.cuda.synchronize()
        correct = bool(torch.allclose(y, ref, atol=1e-1, rtol=1e-2))
        for i in range(copies):
            run(a_l[i], b_l[i].T, c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies].T, c_l[i % copies])
        for _ in range(iters // GRAPH_LEN):
            g.replay()
        torch.cuda.synchronize()
        del g
        manifest.append({"point": p, "dispatches": 1 + copies + (iters // GRAPH_LEN) * GRAPH_LEN,
                         "correct": correct, "iters": iters})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    json.dump(manifest, open(args.manifest, "w"), indent=1)


class Warmup:
    """Stands in for the kernel: compiles at the launch's arguments, launches nothing."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid_):
        return lambda *a, **k: self.kernel.warmup(*a, grid=grid_, **k)


def warm_worker(args):
    import torch
    from v31_streamk_aiter import matmul_kernel as mk

    mk._gemm_a16w16_compute_bound_kernel = Warmup(mk._gemm_a16w16_compute_bound_kernel)
    for p in args.points.split(","):
        kind, tile, M, N, K = grid.parse_point(p)
        try:
            x = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
            w = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
            run_fn(kind, tile)(x, w, torch.empty((M, N), device="cuda", dtype=torch.bfloat16))
            print(f"WARM_OK {p}", flush=True)
        except Exception as e:  # noqa: BLE001  (reported; the point fails again when timed)
            print(f"WARM_FAIL {p} {type(e).__name__}: {e}", flush=True)


def warm(points, jobs):
    todo = sorted(set(points))
    per = max(1, -(-len(todo) // (4 * jobs)))
    chunks = [todo[i:i + per] for i in range(0, len(todo), per)]
    fails, done = [], [0]
    lock = threading.Lock()

    def one(i):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "warm_worker", "--points",
                            ",".join(chunks[i])], capture_output=True, text=True)
        with lock:
            fails.extend(x for x in p.stdout.splitlines() if x.startswith("WARM_FAIL"))
            if p.returncode:
                fails.append(f"WARM_FAIL chunk {i} exited {p.returncode}: {p.stderr[-800:]}")
            done[0] += sum(x.startswith("WARM_OK") for x in p.stdout.splitlines())
            print(f"warm: {done[0]} / {len(todo)}", flush=True)

    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, range(len(chunks))))
    for f in fails:
        print(f, flush=True)
    print(f"warm: {done[0]} of {len(todo)} compiled, {len(fails)} failures", flush=True)


def timing(points, out_dir, point_timeout=600):
    """Time points in one rocprofv3 process: (rows, None) or (None, index of the point
    that failed or hung)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", KERNEL, "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--manifest", manifest, "--points", ",".join(points)]
    proc = subprocess.Popen(cmd, env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout] + [lines.put(None)], daemon=True).start()
    done, tail, last, finished = 0, [], time.monotonic(), []
    while True:
        try:
            line = lines.get(timeout=max(0.0, last + point_timeout - time.monotonic()))
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            print(f"timeout after {point_timeout} s on {points[min(done, len(points) - 1)]}", file=sys.stderr)
            return None, min(done, len(points) - 1)
        if line is None:
            break
        tail = (tail + [line])[-40:]
        if " copies=" in line:
            done, last = done + 1, time.monotonic()
            finished.append(time.time())
    if proc.wait():
        print("".join(tail), file=sys.stderr)
        return None, min(done, len(points) - 1)
    man = json.load(open(manifest))
    files = glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)
    rows = [r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"].startswith(KERNEL)]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    expected = sum(m["dispatches"] for m in man)
    if len(rows) != expected:
        print(f"{len(rows)} dispatches in the trace, expected {expected}", file=sys.stderr)
        return None, len(points) - 1
    res, pos = [], 0
    for i, m in enumerate(man):
        chunk = rows[pos:pos + m["dispatches"]]
        pos += m["dispatches"]
        us = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in chunk[-100:]]
        med = statistics.median(us)
        r = row(m["point"], correct=m["correct"])
        _, _, M, N, K = grid.parse_point(m["point"])
        r.update(us_median=round(med, 3), tflops=round(2 * M * N * K / (med * 1e-6) / 1e12, 1),
                 iters=m["iters"], ts=round(finished[i], 1) if i < len(finished) else "")
        res.append(r)
    return res, None


def row(p, correct="", error=""):
    kind, tile, M, N, K = grid.parse_point(p)
    sh = grid.shape_info(tile, M, N, K)
    c = grid.config_for(kind, tile)
    r = dict.fromkeys(FIELDS, "")
    r.update(point=p, kind=kind, tile=tile, M=M, N=N, K=K, T=sh["T"], S=sh["S"], W=sh["W"], k=sh["k"],
             pairs=sh["pairs"], policy=c.get("STREAMK_POLICY", "default"),
             fixup=c.get("STREAMK_FIXUP", ""), split_n=c.get("SPLIT_N") or 1,
             exchange=c.get("STREAMK_EXCHANGE", ""), grid=grid.streamk_grid(kind) or "",
             persistent=1,
             correct=correct, error=error)
    return r


def order_points(points, seed):
    """Group by shape (one shape's points back to back), shapes and points within a
    shape shuffled by seed."""
    rng = random.Random(seed)
    blocks = {}
    for p in dict.fromkeys(points):
        blocks.setdefault(p.split(":", 2)[2] + ":" + p.split(":")[1], []).append(p)
    keys = sorted(blocks)
    rng.shuffle(keys)
    out = []
    for k in keys:
        b = sorted(blocks[k])
        rng.shuffle(b)
        out += b
    return out


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run_points(points, out, chunk_size, round_, seed):
    done, chunk_id = set(), 0
    if os.path.exists(out):
        prev = list(csv.DictReader(open(out)))
        done = {r["point"] for r in prev}
        chunk_id = 1 + max((int(r["chunk"]) for r in prev if r.get("chunk")), default=-1)
    todo = [p for p in order_points(points, seed) if p not in done]
    total = len(todo)
    out_dir = os.path.join(TMP, f"sweep_{os.path.splitext(os.path.basename(out))[0]}")
    me = os.getpid()
    while todo:
        if gpu_guard.foreign_seen(SMI_INDEX, me, 3, 2.0):
            log(f"FOREIGN process on rocm-smi GPU {SMI_INDEX} before chunk {chunk_id}; stopping")
            sys.exit(4)
        chunk = todo[:chunk_size]
        # A process's first timed point comes out a few percent fast, so each chunk times its
        # first point once more and drops it.
        res, bad = timing(chunk[:1] + chunk, out_dir)
        res, bad = (res[1:] if res else res), (None if bad is None else max(bad - 1, 0))
        rest = todo[len(chunk):]
        if res is None:
            p = chunk[bad]
            res, _ = timing([p, p], out_dir)
            res = res[1:] if res else [row(p, error="failed or hung twice")]
            rest = todo[:bad] + todo[bad + 1:]
        if gpu_guard.foreign_seen(SMI_INDEX, me, 3, 2.0):
            log(f"FOREIGN process on rocm-smi GPU {SMI_INDEX} after chunk {chunk_id}; "
                f"its {len(res)} rows dropped; stopping")
            sys.exit(4)
        todo = rest
        for pos, r in enumerate(res):
            r.update(round=round_, chunk=chunk_id, pos=pos)
        chunk_id += 1
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(res)
        log(f"{total - len(todo)} / {total}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["timing", "worker", "warm", "warm_worker", "list"])
    ap.add_argument("--points", required=True, help="GRID, @FILE or a comma-separated list")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()
    # warm and timing processes share one cache, so timing never compiles.
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(TMP, "triton"))
    if args.mode == "worker":
        return worker(args)
    if args.mode == "warm_worker":
        return warm_worker(args)
    points = points_for(args.points)
    if args.mode == "list":
        print("\n".join(points))
        return
    if args.mode == "warm":
        return warm(points, args.jobs)
    run_points(points, args.out, args.chunk, args.round, seed=3100 + args.round)


if __name__ == "__main__":
    main()
