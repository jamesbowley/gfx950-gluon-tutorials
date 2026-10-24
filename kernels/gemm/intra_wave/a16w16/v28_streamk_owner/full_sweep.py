#!/usr/bin/env python3
"""Compile and time the full v28 sweep in restartable, cache-bounded batches.

Round 1 times every canonical launch.  Each batch's points within 5% of the
shape best are then timed in a fresh worker process for round 2 before that
batch's Triton cache is removed.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
sys.path.insert(0, A16)

import v28_streamk_owner.sweep as sweep  # noqa: E402

DEFAULT_POINTS = os.path.join(HERE, "results", "full_points.csv")
DEFAULT_RUN = "/home/jbowley/repos/aiter/streamk_study/2026-10-22_full_v28/results"
FROZEN = tuple(sweep.FROZEN_FILES) + (
    "v28_streamk_owner/full_points.py",
    "v28_streamk_owner/full_sweep.py",
    "v28_streamk_owner/full_report.py",
)
RAW_FIELDS = (
    "point", "round", "kind", "tile", "M", "N", "K", "correct", "mean_ratio", "error",
    "kernel", "policy", "fixup", "streamk_tiles", "tiles", "S", "waves", "k_steps", "n",
    "n_hi", "ending", "programs", "pairs_per_prog", "busy_cus", "grid", "p10_us",
    "median_us", "p90_us", "tflops", "tbps", "warm_launches", "timed_launches", "copies",
    "gpu", "batch", "chunk", "pos", "ts",
)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


def region_mean(x, rows, cols):
    import torch
    import torch.nn.functional as F

    M, N = x.shape
    mt, nt = -(-M // rows), -(-N // cols)
    pad = (0, nt * cols - N, 0, mt * rows - M)
    total = F.pad(x, pad).view(mt, rows, nt, cols).sum((1, 3))
    count = F.pad(torch.ones_like(x), pad).view(mt, rows, nt, cols).sum((1, 3))
    return total / count


def mean_ratio(out, ref):
    import torch

    rows = min(out.shape[0], 32)
    cols = min(out.shape[1], max(32, -(-1024 // rows)))
    err = region_mean((out.float() - ref).abs(), rows, cols)
    rnd = region_mean((ref.to(torch.bfloat16).float() - ref).abs(), rows, cols)
    if torch.isnan(out).any():
        return float("inf")
    return float((err / rnd.clamp_min(1e-6)).max().item())


def launch_counts(M, N, K):
    """Two 100-launch windows, shortened only for estimated long kernels."""
    est = sweep.est_us("256x256x64", M, N, K)
    if est <= 250:
        return 100, 100
    total = min(200, max(80, int(250_000 / est)))
    warm = max(30, min(100, total // 2))
    timed = max(50, total - warm)
    return warm, timed


def worker(args):
    import bench_ext as bench
    import torch

    requests = json.load(open(args.request))
    v28 = sweep.v27_module()
    manifest, cache = [], {}
    for item in requests:
        point = item["point"]
        kind, tile, M, N, K = sweep.parse_point(point)
        run = sweep.runner(kind, tile, M, N, K)
        dt = torch.bfloat16
        if cache.get("shape") != (M, N, K):
            cache.clear()
            torch.cuda.empty_cache()
            a = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
            b = bench.make_b(N, K, dt, False, lambda t: t.uniform_(-0.5, 0.5))
            ref = a.float() @ b.float()
            cache.update(
                shape=(M, N, K),
                a=a,
                b=b,
                ref=ref,
                rot=bench.gen_rotating_tensors(M, N, K, dt, sweep.ROTATING_MB),
            )
        a, b, ref = cache["a"], cache["b"], cache["ref"]
        out = bench.make_c(M, N, dt, False)
        run(a, b, out)
        launched = dict(v28.LAST_LAUNCH) if kind != "dp" else {"kernel": "v9_tile"}
        ratio = mean_ratio(out, ref)
        split = sweep.split_kind(kind)
        rounded = launched.get("fixup") == "atomic_iz" or (split is not None and split[2] != "fp32")
        if rounded and launched.get("streamk_tiles"):
            from v28_streamk_owner.check_streamk import tile_mean

            bm, bn = v28.parse_spec(tile)[:2]
            reps = -(-1024 // (min(M, bm) * min(N, bn)))
            bm, bn = (bm, bn * reps) if bn >= bm else (bm * reps, bn)
            sk = launched["streamk_tiles"]
            n_max = min(
                launched["streamk_programs"] // sk + 1,
                -(-K // (2 * v28.tile_geometry(tile)["block_k"])),
            )
            rnd = tile_mean((ref.to(dt).float() - ref).abs(), bm, bn)
            err = tile_mean((out.float() - ref).abs(), bm, bn)
            correct = not bool(torch.isnan(out).any() or (err > (1 + n_max**0.5) * rnd + 1e-3).any())
        else:
            correct = bool(torch.allclose(out, ref.to(dt), atol=0.1, rtol=0.01))

        a_l, b_l, c_l, copies = cache["rot"]
        warm_n, timed_n = launch_counts(M, N, K)

        def graph_window(start, count):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(count):
                    j = (start + i) % copies
                    run(a_l[j], b_l[j], c_l[j])
            return graph

        warm_graph = graph_window(0, warm_n)
        timed_graph = graph_window(warm_n, timed_n)
        warm_graph.replay()
        timed_graph.replay()
        torch.cuda.synchronize()
        del warm_graph, timed_graph, out
        manifest.append(
            {
                "point": point,
                "dispatches": 1 + warm_n + timed_n,
                "warm_launches": warm_n,
                "timed_launches": timed_n,
                "copies": copies,
                "correct": correct,
                "mean_ratio": ratio,
                "launch": {
                    key: launched.get(key)
                    for key in (
                        "kernel", "policy", "fixup", "streamk_tiles", "streamk_programs",
                        "grid", "num_programs",
                    )
                },
            }
        )
        print(f"DONE {point} copies={copies} correct={correct}", flush=True)
    json.dump(manifest, open(args.manifest, "w"), indent=1)


def trace_rows(out_dir):
    files = glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)
    if len(files) != 1:
        raise RuntimeError(f"expected one kernel trace in {out_dir}, found {files}")
    rows = [r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"] in sweep.KERNELS]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    return rows


def timing(points, out_dir, point_timeout):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    request = os.path.join(out_dir, "request.json")
    manifest_path = os.path.join(out_dir, "manifest.json")
    json.dump([{"point": p} for p in points], open(request, "w"))
    cmd = [
        "rocprofv3", "--kernel-trace", "--kernel-include-regex", "|".join(sweep.KERNELS),
        "-d", out_dir, "-o", "run", "--output-format", "csv", "--",
        sys.executable, os.path.abspath(__file__), "worker",
        "--request", request, "--manifest", manifest_path,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=A16,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    lines, tail = queue.Queue(), []
    threading.Thread(target=lambda: [lines.put(x) for x in proc.stdout] + [lines.put(None)], daemon=True).start()
    done, last, finished = 0, time.monotonic(), []
    while True:
        try:
            line = lines.get(timeout=max(0.0, last + point_timeout - time.monotonic()))
        except queue.Empty:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return None, min(done, len(points) - 1), f"timeout after {point_timeout}s"
        if line is None:
            break
        tail = (tail + [line])[-40:]
        if line.startswith("DONE "):
            done += 1
            last = time.monotonic()
            finished.append(time.time())
    if proc.wait():
        return None, min(done, len(points) - 1), "".join(tail)[-2000:]
    manifest = json.load(open(manifest_path))
    chunks = sweep.base.chunks(trace_rows(out_dir), manifest)
    result = []
    for i, (meta, dispatches) in enumerate(zip(manifest, chunks)):
        point = meta["point"]
        kind, tile, M, N, K = sweep.parse_point(point)
        count = meta["timed_launches"]
        us = [
            (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
            for r in dispatches[-count:]
        ]
        median = statistics.median(us)
        row = sweep.row(point, meta["launch"], correct=meta["correct"])
        row.update(
            point=point,
            mean_ratio=f"{meta['mean_ratio']:.6g}",
            p10_us=f"{percentile(us, 0.1):.3f}",
            median_us=f"{median:.3f}",
            p90_us=f"{percentile(us, 0.9):.3f}",
            tflops=f"{2 * M * N * K / (median * 1e-6) / 1e12:.3f}",
            tbps=f"{2 * (M * K + K * N + M * N) / (median * 1e-6) / 1e12:.6f}",
            warm_launches=meta["warm_launches"],
            timed_launches=count,
            copies=meta["copies"],
            ts=f"{finished[i]:.1f}" if i < len(finished) else "",
        )
        result.append(row)
    return result, None, ""


def append_rows(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def completed(path, round_no):
    if not os.path.exists(path):
        return set()
    return {
        r["point"]
        for r in csv.DictReader(open(path))
        if int(r["round"]) == round_no
    }


def run_points(points, raw_path, round_no, batch, chunk_size, point_timeout, frozen):
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    done = completed(raw_path, round_no)
    todo = [p for p in points if p not in done]
    chunk_id = 1 + max(
        (
            int(r["chunk"])
            for r in csv.DictReader(open(raw_path))
            if r.get("chunk")
        ),
        default=-1,
    ) if os.path.exists(raw_path) else 0
    tmp = os.path.join(os.path.dirname(raw_path), "tmp", f"b{batch}_r{round_no}")
    total = len(todo)
    while todo:
        check_frozen(frozen)
        chunk = todo[:chunk_size]
        with sweep.gpu_lock(gpu):
            sweep.wait_for_gpu(gpu, 1800, lambda x: print(time.strftime("%H:%M:%S"), x, flush=True))
            # A duplicate first point absorbs the profiler process's consistently fast first result.
            result, bad, why = timing(chunk[:1] + chunk, tmp, point_timeout)
            result = result[1:] if result else result
            bad = None if bad is None else max(0, bad - 1)
            rest = todo[len(chunk):]
            if result is None:
                point = chunk[bad]
                result, _, second = timing([point, point], tmp, point_timeout)
                if result:
                    result = result[1:]
                else:
                    r = sweep.row(point, None, error=(second or why or "failed or hung twice")[:1000])
                    r["point"] = point
                    result = [r]
                rest = todo[:bad] + todo[bad + 1:]
            foreign = sweep.foreign_pids(gpu)
        if foreign:
            print(f"FOREIGN pids {foreign}; dropping chunk {chunk_id}", flush=True)
            continue
        for pos, row in enumerate(result):
            row.update(round=round_no, gpu=gpu, batch=batch, chunk=chunk_id, pos=pos)
        append_rows(raw_path, result)
        todo = rest
        chunk_id += 1
        print(f"batch {batch} round {round_no}: {total - len(todo)}/{total}", flush=True)


def load_point_rows(path, batches=None):
    rows = [
        r for r in csv.DictReader(open(path))
        if not r["alias"] and not r["reject_reason"]
    ]
    if batches is not None:
        rows = [r for r in rows if int(r["batch"]) in batches]
    return rows


def near_best(points, raw_path, batch):
    wanted = set(points)
    rows = [
        r for r in csv.DictReader(open(raw_path))
        if int(r["round"]) == 1 and int(r["batch"]) == batch
        and r["point"] in wanted and not r["error"] and r["correct"] == "True"
    ]
    best = {}
    for row in rows:
        shape = (row["M"], row["N"], row["K"])
        best[shape] = min(best.get(shape, float("inf")), float(row["median_us"]))
    return [
        row["point"]
        for row in rows
        if float(row["median_us"]) <= 1.05 * best[(row["M"], row["N"], row["K"])]
    ]


def relative_hashes():
    return {
        path: hashlib.sha256(open(os.path.join(A16, path), "rb").read()).hexdigest()
        for path in FROZEN
    }


def write_manifest(path, args):
    data = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": relative_hashes(),
        "points": os.path.abspath(args.points),
        "points_sha256": hashlib.sha256(open(args.points, "rb").read()).hexdigest(),
        "timing_gpu": str(args.timing_gpu),
        "compile_gpu": str(args.compile_gpu),
        "batch_shapes": 100,
    }
    json.dump(data, open(path, "w"), indent=2)


def check_frozen(path):
    data = json.load(open(path))
    now = relative_hashes()
    changed = [name for name, digest in data["files"].items() if now.get(name) != digest]
    if hashlib.sha256(open(data["points"], "rb").read()).hexdigest() != data["points_sha256"]:
        changed.append(data["points"])
    if changed:
        raise SystemExit(f"FROZEN: changed during run: {', '.join(changed)}")


def device_info(gpu):
    code = (
        "import json,torch,triton; p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({'name':p.name,'arch':getattr(p,'gcnArchName',''),"
        "'bus':p.pci_bus_id,'device':p.pci_device_id,"
        "'target':str(triton.runtime.driver.active.get_current_target())}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu)),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.splitlines()[-1])


def compile_batch(args, batch):
    sweep.wait_for_gpu(
        str(args.compile_gpu),
        1800,
        lambda message: print(time.strftime("%H:%M:%S"), f"compile: {message}", flush=True),
    )
    rows = load_point_rows(args.points, {batch})
    unique, seen = [], set()
    for row in rows:
        if row["build_key"] not in seen:
            seen.add(row["build_key"])
            unique.append(row["point"])
    os.makedirs(args.cache_root, exist_ok=True)
    cache = os.path.join(args.cache_root, f"batch_{batch:03d}")
    env = dict(
        os.environ,
        HIP_VISIBLE_DEVICES=str(args.compile_gpu),
        TRITON_CACHE_DIR=cache,
    )
    code = (
        "import json,sys; from v28_streamk_owner import sweep; "
        "d=json.load(open(sys.argv[1])); sweep.warm(d, int(sys.argv[2]), [sys.argv[3]], sys.argv[4])"
    )
    request = os.path.join(args.run_dir, f"compile_{batch:03d}.json")
    failures = os.path.join(args.run_dir, f"compile_failures_{batch:03d}.json")
    json.dump(unique, open(request, "w"))
    subprocess.run(
        [sys.executable, "-c", code, request, str(args.jobs), str(args.compile_gpu), failures],
        cwd=A16,
        env=env,
        check=True,
    )


def compile_command(args):
    compile_batch(args, args.batch)


def parse_batches(text, available):
    if not text or text == "all":
        return sorted(available)
    result = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    missing = result - available
    if missing:
        raise SystemExit(f"unknown batches: {sorted(missing)}")
    return sorted(result)


def run(args):
    os.makedirs(args.run_dir, exist_ok=True)
    args.cache_root = args.cache_root or os.path.join(args.run_dir, "cache")
    raw = os.path.join(args.run_dir, "rounds.csv")
    manifest = os.path.join(args.run_dir, "manifest.json")
    all_rows = load_point_rows(args.points)
    available = {int(r["batch"]) for r in all_rows}
    batches = parse_batches(args.batches, available)
    timing_device, compile_device = device_info(args.timing_gpu), device_info(args.compile_gpu)
    if (
        timing_device["arch"] != compile_device["arch"]
        or timing_device["name"] != compile_device["name"]
        or timing_device["target"] != compile_device["target"]
    ):
        raise SystemExit(f"compile and timing devices differ: {compile_device} vs {timing_device}")
    if args.expected_bus >= 0 and timing_device["bus"] != args.expected_bus:
        raise SystemExit(
            f"HIP {args.timing_gpu} is PCI bus {timing_device['bus']}, expected {args.expected_bus}"
        )
    if os.path.exists(manifest):
        check_frozen(manifest)
    else:
        write_manifest(manifest, args)

    done1, done2 = completed(raw, 1), completed(raw, 2)
    pending_batches = []
    for batch in batches:
        points = [r["point"] for r in all_rows if int(r["batch"]) == batch]
        if set(points) <= done1 and set(near_best(points, raw, batch)) <= done2:
            print(f"batch {batch} already complete", flush=True)
        else:
            pending_batches.append(batch)
    batches = pending_batches

    compile_proc = None
    for index, batch in enumerate(batches):
        cache = os.path.join(args.cache_root, f"batch_{batch:03d}")
        if compile_proc is None and not os.path.exists(cache):
            compile_batch(args, batch)
        elif compile_proc is not None:
            if compile_proc.wait():
                raise SystemExit(f"compile of batch {batch} failed")

        if index + 1 < len(batches):
            next_batch = batches[index + 1]
            next_cache = os.path.join(args.cache_root, f"batch_{next_batch:03d}")
            if not os.path.exists(next_cache):
                compile_proc = subprocess.Popen(
                    [
                        sys.executable, os.path.abspath(__file__), "compile",
                        "--points", args.points, "--run-dir", args.run_dir,
                        "--cache-root", args.cache_root, "--compile-gpu", str(args.compile_gpu),
                        "--jobs", str(args.jobs), "--batch", str(next_batch),
                    ],
                    cwd=A16,
                    env=dict(os.environ),
                )
            else:
                compile_proc = None
        else:
            compile_proc = None

        rows = [r for r in all_rows if int(r["batch"]) == batch]
        points = [r["point"] for r in rows]
        old_visible, old_cache = os.environ.get("HIP_VISIBLE_DEVICES"), os.environ.get("TRITON_CACHE_DIR")
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.timing_gpu)
        os.environ["TRITON_CACHE_DIR"] = cache
        try:
            run_points(points, raw, 1, batch, args.chunk, args.point_timeout, manifest)
            round2 = near_best(points, raw, batch)
            run_points(round2, raw, 2, batch, args.chunk, args.point_timeout, manifest)
        finally:
            if old_visible is None:
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
            else:
                os.environ["HIP_VISIBLE_DEVICES"] = old_visible
            if old_cache is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = old_cache
        if not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
        print(f"batch {batch} complete; round 2 has {len(round2)} points", flush=True)


def add_common(ap):
    ap.add_argument("--points", default=DEFAULT_POINTS)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--cache-root")
    ap.add_argument("--compile-gpu", default="1")
    ap.add_argument("--jobs", type=int, default=64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    run_ap = sub.add_parser("run")
    add_common(run_ap)
    run_ap.add_argument("--timing-gpu", default="3")
    run_ap.add_argument("--expected-bus", type=int, default=21)
    run_ap.add_argument("--batches", default="all")
    run_ap.add_argument("--chunk", type=int, default=2000)
    run_ap.add_argument("--point-timeout", type=int, default=300)
    run_ap.add_argument("--keep-cache", action="store_true")
    compile_ap = sub.add_parser("compile")
    add_common(compile_ap)
    compile_ap.add_argument("--batch", type=int, required=True)
    worker_ap = sub.add_parser("worker")
    worker_ap.add_argument("--request", required=True)
    worker_ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args)
    elif args.mode == "compile":
        args.cache_root = args.cache_root or os.path.join(args.run_dir, "cache")
        os.makedirs(args.run_dir, exist_ok=True)
        compile_command(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
