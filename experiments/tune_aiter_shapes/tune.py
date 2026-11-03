"""The 380 AITER shapes (shapes.py): every valid v31 tile spec and Stream-K method,
every v9 tile and torch.mm (hipBLASLt), each timed over 1000 launches under rocprofv3.

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh; cd $REPO
    bash kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/run_guarded.sh \
        experiments/tune_aiter_shapes/results/tune.log python experiments/tune_aiter_shapes/tune.py all

Points are ``KIND:BMxBNxBK:MxNxK`` as in streamk_v31_rule/grid.py (v31 kinds dp, ow{n},
rs{n}, l2rs{n}, 2t, sp), plus ``v9:BMxBNx64:...`` and ``torch:hbl:...``. Each gluon
point is checked against an fp32 torch reference, run once over every rotating copy,
then a HIP graph of GRAPH_LEN launches over ROTATING_MB of rotating inputs (cold
caches) is replayed REPLAYS times: all TIMED traced dispatches go into the stats.

Gluon points run shape by shape (shapes contiguous, points shuffled within a shape) in
chunks, one rocprofv3 process per chunk, each chunk's first point timed once more and
dropped; the torch chunks run first. A ``tcheck`` pseudo-point in the gluon process checks torch.mm against the
same reference (its kernels are outside the gluon trace). torch points run in their
own processes traced on hipBLASLt's ``Cijk_`` kernels with no reference computed, so
every traced dispatch is a timed call; kernels per call (k) come from the period of
the kernel names and are summed per call.

timing appends to results/raw.csv (and results/torch_checks.csv) and resumes from
them; report writes results/all_points.csv (with non_pow2_tile, best and
best_in_version) and results/summary.md.
"""

import argparse
import csv
import functools
import glob
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RULE = os.path.abspath(os.path.join(HERE, "..", "streamk_v31_rule"))
sys.path.insert(0, HERE)
sys.path.insert(0, RULE)
import candidates  # noqa: E402
import gpu_guard  # noqa: E402
import grid  # noqa: E402  (puts kernels/gemm/intra_wave/a16w16 on sys.path)
import shapes  # noqa: E402
import sweep  # noqa: E402

GRAPH_LEN = 100
REPLAYS = 10
TIMED = GRAPH_LEN * REPLAYS
ROTATING_MB = 2048
CHUNK_POINTS = 200
CHUNK_EST_S = 400
TORCH_CHUNK = 20
SEED = 380
SMI_INDEX = int(os.environ.get("V31_SMI_INDEX", "1"))
RESULTS = os.path.join(HERE, "results")
TMP = os.path.join(HERE, "tmp")
RAW = os.path.join(RESULTS, "raw.csv")
CHECKS = os.path.join(RESULTS, "torch_checks.csv")
GLUON_KERNEL = "_gemm_a16w16_compute_bound_kernel"
HBL_KERNEL = "Cijk_"
FLY_KERNEL = "hgemm_"
FLY_DIR = os.path.abspath(os.path.join(HERE, "..", "tune_flydsl"))
FLY_TOP = 5
FLY_CHUNK = 40
# FlyDSL's large-GEMM config (get_flydsl_a16w16_configs keeps only it when M, N and K are
# all >= 4096); timed directly where the policy's pruner removed it first.
FLY_FIXED = "flydsl_hgemm_abf16_wbf16_bf16_t256x256x64x2_ks1_w2x4x1_bias0_ktail0_gm0_pht_gfx950"
FLY_NAME = re.compile(r"flydsl_hgemm_abf16_wbf16_bf16_t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)"
                      r"x(?P<stages>\d+)_ks(?P<split_k>\d+)_w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)"
                      r"_bias0_ktail0_gm(?P<group_m>\d+)_p(?P<policy>ft|ht|hti)_gfx950")
V9_TILES = [f"{bm}x{bn}x64" for bm in (256, 128, 64) for bn in (256, 128, 64)]
RAW_FIELDS = ("point", "M", "N", "K", "version", "tile", "method", "policy", "fixup", "split_n", "exchange",
              "T", "S", "correct", "us_median", "us_mean", "us_min", "us_p90", "tflops", "timed",
              "kernels_per_call", "hbl_tile", "hbl_kernel", "formula_pick", "error", "chunk", "ts")
# Per row: shape (the MxNxK key), nk_group, method_family (method without its split),
# waves (T / P, output tiles per wave of programs), us_over_best (the row's time over the
# shape's fastest), speedup_vs_hbl (hipBLASLt's time over the row's).
# Per shape, repeated on every row: hbl_tile_shape, hbl_non_pow2, hbl_kernels_per_call,
# hbl_tiles / hbl_waves (hipBLASLt's output tiles and tiles / P),
# hbl_us, best_ours_us, best_ours_speedup, ours_win (best_ours beats hipBLASLt).
# Flags: best (fastest of all), best_in_version, best_ours (fastest v31 or v9 point).
# FlyDSL rows: fly_kernel, aiter_us / aiter_rank / aiter_err_ratio (AITER tuner's time, rank
# and error ratio), outside_catalog (FLY_FIXED where AITER's catalog is empty). Per shape:
# flydsl_tile / flydsl_us (fastest FlyDSL passing our check), flydsl_us_any (fastest
# FlyDSL, failing or not), speedup_vs_flydsl (flydsl_us over the row's time).
ALL_FIELDS = (("shape", "nk_group") + RAW_FIELDS[:-2]
              + ("method_family", "waves", "non_pow2_tile", "hbl_tile_shape", "hbl_non_pow2",
                 "hbl_kernels_per_call", "hbl_tiles", "hbl_waves", "best", "best_in_version", "best_ours", "ours_win", "hbl_us",
                 "best_ours_us", "best_ours_speedup", "speedup_vs_hbl", "us_over_best",
                 "fly_kernel", "aiter_us", "aiter_rank", "aiter_err_ratio", "outside_catalog", "flydsl_tile",
                 "flydsl_us", "flydsl_us_any", "speedup_vs_flydsl",
                 "tile_waves_launched", "tile_fill", "tile_util", "hbl_waves_launched", "hbl_fill", "hbl_util",
                 "util_gap_vs_hbl", "target_waves", "tile_extra_waves", "tile_target_fill", "hbl_extra_waves",
                 "hbl_target_fill"))
# Target fill: target_waves = ceil(M N / (P A_MAX)) (at least 1), the waves the largest tile
# (A_MAX elements) needs: 1 below a wave, n + 1 for n.x waves. extra_waves = W - target_waves.
# target_fill = min(T / (P target_waves), 1) * target_waves / W: 1 when the tiles keep every
# CU busy for exactly the target waves; a tile that leaves CUs idle scores its share of
# them, one that needs extra waves is scaled by target_waves / W.
A_MAX = 224 * 384  # the largest hipBLASLt tile on these shapes
# Tile spread metrics, for a BM x BN tile on P CUs with T = ceil(M/BM) ceil(N/BN) tiles:
# waves_launched W = ceil(T/P); fill = T / (P W), the share of launched CU slots holding a
# tile; util = M N / (P W BM BN), fill times padding efficiency: the useful share of the
# tile area the launched waves compute (a data-parallel launch takes about W tile-times).
# tile_* for the row's own tile (hipBLASLt's for torch rows), hbl_* for the shape's
# hipBLASLt tile, util_gap_vs_hbl = hbl_util - tile_util.


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def shape_of(p):
    _, _, M, N, K = grid.parse_point(p)
    return M, N, K


def v31_kinds(M, N, K, tile):
    """Every valid dp / owner / rs / l2rs split, two_tile and spread."""
    kinds = (["dp"] + [f"ow{n}" for n in grid.OWNER_N] + [f"rs{n}" for n in grid.RS_N]
             + [f"l2rs{n}" for n in grid.RS_N] + ["2t", "sp"])
    return [k for k in kinds if grid.valid(k, tile, M, N, K)]


def v9_valid(M, N, K):
    return M >= 1 and N % 8 == 0 and K % 8 == 0 and K >= 256


def shape_points(M, N, K):
    pts = [grid.point_name(k, t, M, N, K) for t in grid.ALL_SPECS for k in v31_kinds(M, N, K, t)]
    if v9_valid(M, N, K):
        pts += [grid.point_name("v9", t, M, N, K) for t in V9_TILES]
    return pts + [grid.point_name("torch", "hbl", M, N, K)]


@functools.cache
def rule():
    return json.load(open(os.path.join(RULE, "results", "streamk_rule.json")))


@functools.cache
def formula_kinds(M, N, K):
    """The v31 default rule's plan at each spec, as a point kind."""
    out = {}
    for spec in candidates.SPECS:
        c = candidates.streamk_default(M, N, K, spec, grid.P, rule())
        if c["STREAMK_POLICY"] == "dp":
            kind = "dp"
        else:
            kind = ("ow" if c["STREAMK_FIXUP"] == "owner" else "rs") + str(c["SPLIT_N"])
        out[candidates.spec_name(spec)] = kind
    return out


def fly_config(name):
    m = FLY_NAME.fullmatch(name)
    if not m:
        raise ValueError(f"not a FlyDSL hgemm kernel name: {name}")
    c = {k: (v if k == "policy" else int(v)) for k, v in m.groupdict().items()}
    return c


def fly_tile(name):
    c = fly_config(name)
    return f"{c['block_m']}x{c['block_n']}x{c['block_k']}"


def fly_method(name):
    c = fly_config(name)
    return (f"flydsl ks{c['split_k']} w{c['m_waves']}x{c['n_waves']}x{c['k_waves']} st{c['stages']} "
            f"gm{c['group_m']} {c['policy']}")


def fly_family(name):
    c = fly_config(name)
    return "flydsl" + (" split-K" if c["split_k"] > 1 else "") + (" k_waves=2" if c["k_waves"] > 1 else "")


@functools.cache
def fly_meta():
    """{point: AITER tuner info} for the FlyDSL points: each shape's FLY_TOP fastest
    candidates in AITER's profile (tune_flydsl/runs/*_profile.csv) among those passing
    AITER's own check (err_ratio <= 0.05, as its tuner requires), so fly1 is AITER's pick;
    plus FLY_FIXED as fly0 on shapes whose FlyDSL catalog is empty."""
    meta = {}
    for path in sorted(glob.glob(os.path.join(FLY_DIR, "runs", "*_profile.csv"))):
        rows = []
        for r in csv.DictReader(open(path)):
            us = float(r["us"])
            if 0 < us < float("inf") and float(r["err_ratio"]) <= 0.05 and FLY_NAME.fullmatch(r["kernelName"]):
                rows.append((us, r))
        rows.sort(key=lambda x: x[0])
        for rank, (us, r) in enumerate(rows[:FLY_TOP], 1):
            p = grid.point_name(f"fly{rank}", r["kernelName"], int(r["M"]), int(r["N"]), int(r["K"]))
            meta[p] = dict(aiter_us=r["us"], aiter_rank=rank, aiter_err_ratio=r["err_ratio"], outside_catalog="no")
    sizes = os.path.join(FLY_DIR, "catalog_sizes.csv")
    if os.path.exists(sizes):
        for r in csv.DictReader(open(sizes)):
            if int(r["candidates"]) == 0:
                p = grid.point_name("fly0", FLY_FIXED, int(r["M"]), int(r["N"]), int(r["K"]))
                meta[p] = dict(aiter_us="", aiter_rank="", aiter_err_ratio="", outside_catalog="yes")
    return meta


def is_fly(p_or_kind):
    return p_or_kind.startswith("fly")


def version_of(kind):
    if is_fly(kind):
        return "flydsl"
    return kind if kind in ("v9", "torch") else "v31"


def method_of(kind):
    if kind == "torch":
        return "hipBLASLt"
    if kind in ("v9", "dp"):
        return "dp"
    if kind == "2t":
        return "two_tile"
    if kind == "sp":
        return "spread"
    for prefix, name in (("l2rs", "one_tile rs (l2)"), ("rs", "one_tile rs"), ("ow", "one_tile owner")):
        if kind.startswith(prefix):
            return f"{name} n={kind[len(prefix):]}"
    raise ValueError(kind)


def run_fn(kind, tile):
    if kind in ("torch", "tcheck"):
        import torch

        return lambda x, w, y: torch.mm(x, w.t(), out=y)
    if is_fly(kind):
        import torch
        from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

        c = fly_config(tile)
        kw = {k: c[k] for k in ("block_m", "block_n", "block_k", "stages", "split_k", "m_waves", "n_waves",
                                "k_waves", "group_m")}
        return lambda x, w, y: flydsl_hgemm(x, w, out=y, policy=c["policy"], out_dtype=torch.bfloat16, **kw)
    if kind == "v9":
        from v31_streamk_aiter import dp_ab

        v9 = dp_ab.v9_module()
        bm, bn, _ = grid.parse_tile(tile)
        return lambda x, w, y: v9.gemm_a16w16_compute_bound(x, w, y, tile=(bm, bn))
    return sweep.run_fn(kind, tile)


def worker(args):
    """Times args.points in order; the shape's reference (unless --no-ref) and rotating
    inputs are allocated once per run of same-shape points."""
    import bench_ext as bench
    import torch

    dt = torch.bfloat16
    manifest, cache = [], {}
    for p in args.points.split(","):
        kind, tile, M, N, K = grid.parse_point(p)
        if cache.get("shape") != (M, N, K):
            cache.clear()
            torch.cuda.empty_cache()
            x = torch.rand((M, K), device="cuda", dtype=dt) - 0.5
            w = torch.rand((N, K), device="cuda", dtype=dt) - 0.5
            ref = None if args.no_ref else (x.float() @ w.float().T).to(dt)
            rot = bench.gen_rotating_tensors(M, N, K, dt, ROTATING_MB)
            torch.cuda.synchronize()
            cache.update(shape=(M, N, K), x=x, w=w, ref=ref, rot=rot)
        x, w, ref = cache["x"], cache["w"], cache["ref"]
        a_l, b_l, c_l, copies = cache["rot"]
        run = run_fn(kind, tile)
        y = torch.empty((M, N), device="cuda", dtype=dt)
        run(x, w, y)
        torch.cuda.synchronize()
        correct = "" if ref is None else bool(torch.allclose(y, ref, atol=1e-1, rtol=1e-2))
        if kind == "tcheck":
            manifest.append({"point": p, "calls": 0, "correct": correct})
            print(f"{p} copies=0 correct={correct}", flush=True)
            continue
        for i in range(copies):
            run(a_l[i], b_l[i].T, c_l[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GRAPH_LEN):
                run(a_l[i % copies], b_l[i % copies].T, c_l[i % copies])
        for _ in range(REPLAYS):
            g.replay()
        torch.cuda.synchronize()
        del g, y
        manifest.append({"point": p, "calls": 1 + copies + TIMED, "correct": correct})
        print(f"{p} copies={copies} correct={correct}", flush=True)
    json.dump(manifest, open(args.manifest, "w"), indent=1)


def warm_worker(args):
    import torch
    from v31_streamk_aiter import dp_ab
    from v31_streamk_aiter import matmul_kernel as mk

    v9 = dp_ab.v9_module()
    mk._gemm_a16w16_compute_bound_kernel = sweep.Warmup(mk._gemm_a16w16_compute_bound_kernel)
    v9._gemm_a16w16_compute_bound_kernel = sweep.Warmup(v9._gemm_a16w16_compute_bound_kernel)
    cache = {}
    for p in args.points.split(","):
        kind, tile, M, N, K = grid.parse_point(p)
        if cache.get("shape") != (M, N, K):
            cache = dict(shape=(M, N, K), x=torch.empty((M, K), device="cuda", dtype=torch.bfloat16),
                         w=torch.empty((N, K), device="cuda", dtype=torch.bfloat16),
                         y=torch.empty((M, N), device="cuda", dtype=torch.bfloat16))
        try:
            run_fn(kind, tile)(cache["x"], cache["w"], cache["y"])
            print(f"WARM_OK {p}", flush=True)
        except Exception as e:  # noqa: BLE001  (reported; the point fails again when timed)
            print(f"WARM_FAIL {p} {type(e).__name__}: {e}", flush=True)


def warm(pts, jobs):
    # torch needs no compile; FlyDSL compiles on first call into its own disk cache.
    todo = [p for p in pts if not p.startswith("torch:") and not is_fly(p)]
    if not todo:
        return
    per = max(1, -(-len(todo) // (4 * jobs)))
    chunks = [todo[i:i + per] for i in range(0, len(todo), per)]
    lines, done = [], [0]

    def one(c):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "warm_worker", "--points", ",".join(c)],
                           capture_output=True, text=True)
        out = r.stdout.splitlines()
        if r.returncode:
            out.append(f"WARM_FAIL chunk exit {r.returncode}: {r.stderr[-800:]}")
        before = done[0]
        done[0] += sum(x.startswith("WARM_OK") for x in out)
        if done[0] * 20 // len(todo) != before * 20 // len(todo):
            log(f"warm: {done[0]} / {len(todo)}")
        return out

    with ThreadPoolExecutor(jobs) as ex:
        for out in ex.map(one, chunks):
            lines += out
    fails = [x for x in lines if x.startswith("WARM_FAIL")]
    for f in fails:
        print(f, flush=True)
    log(f"warm: {sum(x.startswith('WARM_OK') for x in lines)} of {len(todo)} compiled, {len(fails)} failures")


def per_call(rows, calls, ks=range(1, 5)):
    """(per-call us of the last TIMED calls, k) for rows holding calls x k dispatches
    that repeat with period k. A call can repeat its own kernels (hipBLASLt runs GEMM +
    PostGSU twice for some shapes), so a trace of one point passes its exact k."""
    for k in ks:
        if len(rows) < calls * k:
            break
        seg = rows[:calls * k]
        names = [r["Kernel_Name"] for r in seg]
        if all(names[i] == names[i % k] for i in range(len(names))):
            timed = seg[-TIMED * k:]
            us = [sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in timed[i:i + k]) / 1e3
                  for i in range(0, len(timed), k)]
            return us, k
    raise RuntimeError(f"{len(rows)} dispatches left for {calls} calls: no consistent kernels per call")


def run_rocprof(pts, out_dir, timeout=3600):
    """Run the worker on pts under rocprofv3: (manifest, traced rows in dispatch order)."""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    manifest = os.path.join(out_dir, "manifest.json")
    torch_run = pts[0].startswith("torch:")
    regex = HBL_KERNEL if torch_run else FLY_KERNEL if is_fly(pts[0]) else GLUON_KERNEL
    cmd = ["rocprofv3", "--kernel-trace", "--kernel-include-regex", regex, "-d", out_dir, "-o", "run",
           "--output-format", "csv", "--", sys.executable, os.path.abspath(__file__), "worker",
           "--manifest", manifest, "--points", ",".join(pts)] + (["--no-ref"] if torch_run else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise RuntimeError(f"timeout after {timeout} s")
    if proc.returncode or not os.path.exists(manifest):
        print(out[-3000:], file=sys.stderr)
        raise RuntimeError(f"rocprofv3 run exited {proc.returncode}")
    man = json.load(open(manifest))
    files = glob.glob(os.path.join(out_dir, "**", "*kernel_trace.csv"), recursive=True)
    rows = []
    if files:
        rows = sorted((r for r in csv.DictReader(open(files[0])) if r["Kernel_Name"].startswith(regex)),
                      key=lambda r: int(r["Dispatch_Id"]))
    return man, rows


def timing(pts, out_dir):
    """Time pts in one rocprofv3 process: (result rows, torch checks {shape: correct})."""
    man, rows = run_rocprof(pts, out_dir)
    expected = sum(m["calls"] for m in man)
    if not pts[0].startswith("torch:") and not is_fly(pts[0]) and len(rows) != expected:
        raise RuntimeError(f"{len(rows)} gluon dispatches in the trace, expected {expected}")
    out, checks, pos = [], {}, 0
    names_of = {}
    timed_points = [m for m in man if m["calls"]]
    exact = len(timed_points) == 1 and len(rows) % timed_points[0]["calls"] == 0
    for m in man:
        if m["calls"] == 0:
            checks[shape_of(m["point"])] = m["correct"]
            continue
        us, k = per_call(rows[pos:], m["calls"], [len(rows) // m["calls"]] if exact else range(1, 5))
        names_of[m["point"]] = [r["Kernel_Name"] for r in rows[pos:pos + k]]
        pos += m["calls"] * k
        out.append(result_row(m["point"], m["correct"], us, k, names_of[m["point"]]))
    if pos != len(rows):
        raise RuntimeError(f"{len(rows) - pos} traced dispatches not accounted for")
    return out, checks


def hbl_tile(name):
    mt = re.search(r"_MT(\d+)x(\d+)x(\d+)_", name)
    sk = re.search(r"_(SK\d+)_", name)
    gsu = re.search(r"_GSU(\d+)_", name)
    s = f"{mt.group(1)}x{mt.group(2)}x{mt.group(3)}" if mt else "?"
    return s + (f" {sk.group(1)}" if sk else "") + (f" GSU{gsu.group(1)}" if gsu else "")


def base_row(p):
    kind, tile, M, N, K = grid.parse_point(p)
    r = dict.fromkeys(RAW_FIELDS, "")
    if is_fly(kind):
        c = fly_config(tile)
        tile = fly_tile(tile)
        r.update(point=p, M=M, N=N, K=K, version="flydsl", tile=tile, method=fly_method(grid.parse_point(p)[1]),
                 split_n=c["split_k"])
    else:
        r.update(point=p, M=M, N=N, K=K, version=version_of(kind), tile=tile, method=method_of(kind))
    if kind != "torch":
        sh = grid.shape_info(tile, M, N, K)
        r.update(T=sh["T"], S=sh["S"])
    if r["version"] == "v31":
        c = grid.config_for(kind, tile)
        r.update(policy=c["STREAMK_POLICY"], fixup=c["STREAMK_FIXUP"], split_n=c["SPLIT_N"] or 1,
                 exchange=c["STREAMK_EXCHANGE"], formula_pick=int(formula_kinds(M, N, K).get(tile) == kind))
    return r


def result_row(p, correct, us, k, names):
    r = base_row(p)
    M, N, K = shape_of(p)
    med = statistics.median(us)
    r.update(correct=correct, us_median=round(med, 3), us_mean=round(statistics.fmean(us), 3),
             us_min=round(min(us), 3), us_p90=round(sorted(us)[int(0.9 * len(us))], 3),
             tflops=round(2 * M * N * K / (med * 1e-6) / 1e12, 1), timed=len(us), kernels_per_call=k)
    if p.startswith("torch:"):
        r.update(hbl_tile=hbl_tile(names[0]), hbl_kernel=" | ".join(names))
    return r


def check_gpu(when):
    if gpu_guard.foreign_seen(SMI_INDEX, os.getpid(), 3, 2.0):
        log(f"FOREIGN process on rocm-smi GPU {SMI_INDEX} {when}; stopping")
        sys.exit(4)


def est_s(p):
    kind, tile, M, N, K = grid.parse_point(p)
    t = grid.est_us("256x256x64" if kind == "torch" else fly_tile(tile) if is_fly(kind) else tile, M, N, K)
    return (TIMED + 20) * t / 1e6


def plan_batches(done):
    """Torch chunks first (the baseline for every shape), then gluon chunks (shapes
    contiguous, at most CHUNK_POINTS points or CHUNK_EST_S s of kernel time; a shape's
    tcheck rides with its first chunk)."""
    rng = random.Random(SEED)
    order = list(shapes.SHAPES)
    rng.shuffle(order)
    gluon, torch_pts = [], []
    for M, N, K in order:
        pts = shape_points(M, N, K)
        g = [p for p in pts if not p.startswith("torch:") and p not in done]
        rng.shuffle(g)
        if g:
            gluon.append((g, grid.point_name("tcheck", "hbl", M, N, K)))
        torch_pts += [p for p in pts if p.startswith("torch:") and p not in done]
    batches, cur, cur_s = [], [], 0.0
    for g, tc in gluon:
        first = True
        for p in g:
            if cur and (len(cur) >= CHUNK_POINTS or cur_s + est_s(p) > CHUNK_EST_S):
                batches.append(cur)
                cur, cur_s = [], 0.0
            cur.append(p)
            cur_s += est_s(p)
            if first:
                cur.append(tc)
                first = False
    if cur:
        batches.append(cur)
    shape_rank = {s: i for i, s in enumerate(order)}
    fly = sorted((p for p in fly_meta() if p not in done), key=lambda p: (shape_rank.get(shape_of(p), 1e9), p))
    return ([torch_pts[i:i + TORCH_CHUNK] for i in range(0, len(torch_pts), TORCH_CHUNK)] + batches
            + [fly[i:i + FLY_CHUNK] for i in range(0, len(fly), FLY_CHUNK)])


def append(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def time_one(pts, out_dir):
    """A chunk; on failure its points one by one; a point failing twice gets an error row."""
    try:
        # A process's first timed point comes out a few percent fast: time it twice, drop the first.
        res, checks = timing(pts[:1] + pts, out_dir)
        return res[1:], checks
    except RuntimeError as e:
        log(f"chunk of {len(pts)} failed ({e}); timing its points one by one")
    res, checks = [], {}
    for p in pts:
        try:
            if p.startswith("tcheck:"):
                checks.update(timing([p], out_dir)[1])
                continue
            r, c = timing([p, p], out_dir)
            res += r[1:]
            checks.update(c)
        except RuntimeError as e1:
            log(f"{p} failed: {e1}")
            res.append(dict(base_row(p), error=str(e1)[:200]))
    return res, checks


def time_all():
    done = {r["point"] for r in csv.DictReader(open(RAW))} if os.path.exists(RAW) else set()
    batches = plan_batches(done)
    total = sum(len([p for p in b if not p.startswith("tcheck:")]) for b in batches)
    chunk0 = 1 + max((int(r["chunk"]) for r in csv.DictReader(open(RAW)) if r["chunk"]), default=-1) \
        if os.path.exists(RAW) else 0
    log(f"{len(done)} points already timed; {total} to time in {len(batches)} chunks")
    out_dir = os.path.join(TMP, "trace")
    n = 0
    for i, b in enumerate(batches):
        check_gpu(f"before chunk {i}")
        res, checks = time_one(b, out_dir)
        check_gpu(f"after chunk {i} (its rows dropped)")
        ts = round(time.time(), 1)
        for r in res:
            r.update(chunk=chunk0 + i, ts=ts)
        append(RAW, RAW_FIELDS, res)
        if checks:
            append(CHECKS, ("M", "N", "K", "correct"),
                   [dict(M=s[0], N=s[1], K=s[2], correct=c) for s, c in checks.items()])
        n += len(res)
        log(f"chunk {i + 1} / {len(batches)}: {n} / {total} points")


# ---------------------------------------------------------------- report

def is_pow2(x):
    return x > 0 and x & (x - 1) == 0


def non_pow2(r):
    if r["version"] == "torch":
        m = re.match(r"(\d+)x(\d+)x", r["hbl_tile"])
        if not m:
            return ""
        return "no" if is_pow2(int(m.group(1))) and is_pow2(int(m.group(2))) else "yes"
    bm, bn, _ = grid.parse_tile(r["tile"])
    return "no" if is_pow2(bm) and is_pow2(bn) else "yes"


def hbl_waves(hbl):
    """(output tiles, tiles / P) of hipBLASLt's tile. torch hands hipBLASLt the column-major
    problem C^T = W X^T, so MT{a}x{b} tiles N by a and M by b."""
    m = re.match(r"(\d+)x(\d+)x", hbl["hbl_tile"]) if hbl else None
    if not m:
        return "", ""
    tiles = grid.cdiv(int(hbl["N"]), int(m.group(1))) * grid.cdiv(int(hbl["M"]), int(m.group(2)))
    return tiles, f"{tiles / grid.P:.3f}"


def tile_spread(M, N, bm, bn):
    """(waves launched, fill, util) of a data-parallel launch of bm x bn tiles."""
    T = grid.cdiv(M, bm) * grid.cdiv(N, bn)
    W = grid.cdiv(T, grid.P)
    return W, T / (grid.P * W), M * N / (grid.P * W * bm * bn)


def row_tile_mn(r):
    """(BM, BN) of the row's tile in M x N order; hipBLASLt's MT{a}x{b} tiles N by a, M by b."""
    if r["version"] == "torch":
        m = re.match(r"(\d+)x(\d+)x", r["hbl_tile"])
        return (int(m.group(2)), int(m.group(1))) if m else None
    bm, bn, _ = grid.parse_tile(r["tile"])
    return bm, bn


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs)) if xs else float("nan")


def fly_section(summary, timed_shapes, wins):
    """FlyDSL against ours (best v31 or v9) and hipBLASLt, and AITER's pick against ours."""
    def us(r):
        return float(r["us_median"])

    have = [s for s in timed_shapes if summary[s]["flydsl_any"]]
    if not have:
        return []
    ours = {s: min((summary[s][v] for v in ("v31", "v9") if summary[s][v]), key=us, default=None)
            for s in have}
    ok_fly = [s for s in have if summary[s]["flydsl"]]
    vs_fly = [us(summary[s]["flydsl"]) / us(ours[s]) for s in ok_fly if ours[s]]
    vs_fly_any = [us(summary[s]["flydsl_any"]) / us(ours[s]) for s in have if ours[s]]
    fly_hbl = [us(summary[s]["torch"]) / us(summary[s]["flydsl"]) for s in ok_fly if summary[s]["torch"]]
    meta = fly_meta()
    agree = n_pick = 0
    for s in have:
        pick = next((p for p, m in meta.items() if m["aiter_rank"] == 1 and shape_of(p) == s), None)
        fa = summary[s]["flydsl_any"]
        if pick and fa:
            n_pick += 1
            agree += fa["point"] == pick
    outside = [s for s in have if fly_meta().get(summary[s]["flydsl_any"]["point"], {}).get("outside_catalog") == "yes"]
    L = ["## FlyDSL (AITER's tuner, re-timed here)", "",
         f"AITER's a16w16 tuner with `--libtype flydsl` on every shape (experiments/tune_flydsl), then each "
         f"shape's {FLY_TOP} fastest candidates by AITER's time re-timed in this harness; on the "
         f"{len([1 for m in meta.values() if m['outside_catalog'] == 'yes'])} shapes whose FlyDSL catalog is "
         "empty, the policy's large-GEMM config instead. FlyDSL timed on "
         f"{len(have)} shapes; a FlyDSL point passes our check (atol 0.1, rtol 1%) on {len(ok_fly)} of them.", "",
         f"- Fastest overall (passing points): FlyDSL on {wins['flydsl']} shapes.",
         f"- Ours (best v31 or v9) over FlyDSL's best passing point: geomean speedup {gm(vs_fly):.3f}, "
         f"ours faster on {sum(x > 1 for x in vs_fly)} of {len(vs_fly)}.",
         f"- Ours over FlyDSL's best point regardless of our check: geomean {gm(vs_fly_any):.3f}, ours faster on "
         f"{sum(x > 1 for x in vs_fly_any)} of {len(vs_fly_any)}.",
         f"- FlyDSL (best passing) over hipBLASLt: geomean speedup {gm(fly_hbl):.3f}, FlyDSL faster on "
         f"{sum(x > 1 for x in fly_hbl)} of {len(fly_hbl)}.",
         f"- AITER's pick (its rank 1) is also the fastest FlyDSL point here on {agree} of {n_pick} shapes.",
         f"- Outside AITER's catalog (large-GEMM config timed directly): {len(outside)} shapes.", ""]
    return L


def report():
    raw = list(csv.DictReader(open(RAW)))
    checks = {}
    if os.path.exists(CHECKS):
        for c in csv.DictReader(open(CHECKS)):
            checks[(int(c["M"]), int(c["N"]), int(c["K"]))] = c["correct"]
    by_shape = {}
    for r in raw:
        if r["version"] == "torch":
            r["correct"] = checks.get((int(r["M"]), int(r["N"]), int(r["K"])), "")
        r["non_pow2_tile"] = non_pow2(r)
        r["best"] = r["best_in_version"] = r["best_ours"] = "no"
        r["shape"] = f"{r['M']}x{r['N']}x{r['K']}"
        r["nk_group"] = f"N={r['N']} K={r['K']}"
        r["method_family"] = re.sub(r" n=\d+$", "", r["method"])
        r["waves"] = f"{int(r['T']) / grid.P:.3f}" if r["T"] else ""
        if r["version"] == "flydsl":
            name = grid.parse_point(r["point"])[1]
            r["fly_kernel"] = name
            r["method_family"] = fly_family(name)
            r.update(fly_meta().get(r["point"], {}))
        by_shape.setdefault((int(r["M"]), int(r["N"]), int(r["K"])), []).append(r)

    def ok(r):
        # A torch point's check runs with its shape's first gluon chunk; until then it is pending.
        pending = r["version"] == "torch" and r["correct"] == ""
        return r["us_median"] and (r["correct"] == "True" or pending)

    summary = {}
    for s, rs in by_shape.items():
        good = sorted((r for r in rs if ok(r)), key=lambda r: float(r["us_median"]))
        if good:
            good[0]["best"] = "yes"
        best = {}
        for ver in ("v31", "v9", "torch", "flydsl"):
            b = next((r for r in good if r["version"] == ver), None)
            if b:
                b["best_in_version"] = "yes"
            best[ver] = b
        fly_any = min((r for r in rs if r["version"] == "flydsl" and r["us_median"]),
                      key=lambda r: float(r["us_median"]), default=None)
        best["flydsl_any"] = fly_any
        ours = next((r for r in good if r["version"] in ("v31", "v9")), None)
        if ours:
            ours["best_ours"] = "yes"
        hbl = next((r for r in rs if r["version"] == "torch" and r["us_median"]), None)
        best_us = float(good[0]["us_median"]) if good else None
        hbl_mn = row_tile_mn(hbl) if hbl else None
        hbl_spread = tile_spread(s[0], s[1], *hbl_mn) if hbl_mn else None
        w_t = max(1, grid.cdiv(s[0] * s[1], grid.P * A_MAX))

        def target_fill(W, fill):
            return f"{min(W * fill / w_t, 1.0) * w_t / W:.4f}"

        for r in rs:
            r["target_waves"] = w_t
            mn = row_tile_mn(r)
            if mn:
                W, fill, util = tile_spread(s[0], s[1], *mn)
                r.update(tile_waves_launched=W, tile_fill=f"{fill:.4f}", tile_util=f"{util:.4f}",
                         tile_extra_waves=W - w_t, tile_target_fill=target_fill(W, fill))
                if hbl_spread:
                    r["util_gap_vs_hbl"] = f"{hbl_spread[2] - util:.4f}"
            if hbl_spread:
                r.update(hbl_waves_launched=hbl_spread[0], hbl_fill=f"{hbl_spread[1]:.4f}",
                         hbl_util=f"{hbl_spread[2]:.4f}", hbl_extra_waves=hbl_spread[0] - w_t,
                         hbl_target_fill=target_fill(*hbl_spread[:2]))
        for r in rs:
            r["hbl_tile_shape"] = hbl["hbl_tile"].split()[0] if hbl else ""
            r["hbl_tiles"], r["hbl_waves"] = hbl_waves(hbl)
            r["hbl_non_pow2"] = hbl["non_pow2_tile"] if hbl else ""
            r["hbl_kernels_per_call"] = hbl["kernels_per_call"] if hbl else ""
            r["hbl_us"] = hbl["us_median"] if hbl else ""
            r["speedup_vs_hbl"] = (f"{float(hbl['us_median']) / float(r['us_median']):.4f}"
                                   if hbl and r["us_median"] else "")
            r["us_over_best"] = f"{float(r['us_median']) / best_us:.4f}" if best_us and r["us_median"] else ""
            r["best_ours_us"] = ours["us_median"] if ours else ""
            r["best_ours_speedup"] = (f"{float(hbl['us_median']) / float(ours['us_median']):.4f}"
                                      if hbl and ours else "")
            r["ours_win"] = ("yes" if float(hbl["us_median"]) > float(ours["us_median"]) else "no") \
                if hbl and ours else ""
            fly = best["flydsl"]
            r["flydsl_tile"] = fly["tile"] if fly else ""
            r["flydsl_us"] = fly["us_median"] if fly else ""
            r["flydsl_us_any"] = fly_any["us_median"] if fly_any else ""
            r["speedup_vs_flydsl"] = (f"{float(fly['us_median']) / float(r['us_median']):.4f}"
                                      if fly and r["us_median"] else "")
        fp = [r for r in good if r["formula_pick"] == "1"]
        fp_best = min(fp, key=lambda r: float(r["us_median"])) if fp else None
        # The formula's pick at the best v31 tile.
        fp_tile = next((r for r in fp if best["v31"] and r["tile"] == best["v31"]["tile"]), None)
        summary[s] = dict(overall=good[0] if good else None, fp_best=fp_best, fp_tile=fp_tile, **best)
    order = {s: i for i, s in enumerate(shapes.SHAPES)}
    with open(os.path.join(RESULTS, "all_points.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(raw, key=lambda r: (order.get((int(r["M"]), int(r["N"]), int(r["K"])), 1e9),
                                               float(r["us_median"] or 1e18))))

    def us(r):
        return float(r["us_median"])

    def cell(r, method=True, tile=True):
        if r is None:
            return " | ".join([""] * (2 + method + tile))
        label = f"hipBLASLt {r['hbl_tile']}" if r["version"] == "torch" else r["tile"]
        parts = ([label] if tile else []) + ([r["method"]] if method else []) + [r["us_median"], r["tflops"]]
        return " | ".join(parts)

    timed_shapes = [s for s in shapes.SHAPES if s in summary]
    all_t, all_9, wins = [], [], {"v31": 0, "v9": 0, "torch": 0, "flydsl": 0}
    for s in timed_shapes:
        x = summary[s]
        if x["v31"] and x["torch"]:
            all_t.append(us(x["torch"]) / us(x["v31"]))
        if x["v31"] and x["v9"]:
            all_9.append(us(x["v9"]) / us(x["v31"]))
        if x["overall"]:
            wins[x["overall"]["version"]] += 1
    L = ["# The 380 AITER shapes: v31 Stream-K, v9 and hipBLASLt (torch)", "",
         f"bf16, x (M, K) and w (N, K) row-major, physical GPU 1 (rocm-smi 1). Every valid v31 tile spec and "
         f"Stream-K method, all 9 v9 tiles and torch.mm (hipBLASLt). Each point: {TIMED} launches traced with "
         f"rocprofv3 (a graph of {GRAPH_LEN} replayed {REPLAYS} times) over {ROTATING_MB} MB of rotating inputs; "
         f"time is the median of all {TIMED} in us, TFLOPS from the median. Every point is in "
         "[all_points.csv](all_points.csv) (with `non_pow2_tile`, `best` and `best_in_version`).", "",
         f"**{len(timed_shapes)} shapes.** Fastest: v31 on {wins['v31']}, v9 on {wins['v9']}, hipBLASLt on "
         f"{wins['torch']}. v31 (its best point) over hipBLASLt: geomean speedup {gm(all_t):.3f} "
         f"(above 1 is faster than torch), {sum(x > 1 for x in all_t)} of {len(all_t)} faster; over v9's best "
         f"tile: {gm(all_9):.3f}.", "",
         "Columns: `v31/torch` is torch's time over v31's best; `formula/best` is the v31 default rule's plan "
         "at the best v31 tile over the best v31 point (1.000 when the rule picks it).", ""]
    L += fly_section(summary, timed_shapes, wins)
    for (N, K), ms in shapes.GROUPS.items():
        rows_g = [((M, N, K), summary.get((M, N, K))) for M in ms]
        t = [us(x["torch"]) / us(x["v31"]) for _, x in rows_g if x and x["v31"] and x["torch"]]
        n9 = [us(x["v9"]) / us(x["v31"]) for _, x in rows_g if x and x["v31"] and x["v9"]]
        w31 = sum(1 for _, x in rows_g if x and x["overall"] and x["overall"]["version"] == "v31")
        L += [f"## N={N}, K={K} ({len(ms)} shapes)", "",
              "| M | best | best tile | best method | us | TFLOPS | v31 tile | v31 method | v31 us | v31 TFLOPS | "
              "v9 tile | v9 us | v9 TFLOPS | hipBLASLt tile | non-pow2 | torch us | torch TFLOPS | v31/torch | "
              "formula/best | FlyDSL tile | FlyDSL us | FlyDSL us (any) | v31/FlyDSL |",
              "|" + "---|" * 23]
        for (M, _, _), x in rows_g:
            if not x:
                L.append(f"| {M} | not timed |" + " |" * 21)
                continue
            fl, fa = x["flydsl"], x["flydsl_any"]
            fly_ratio = f"{us(fl) / us(x['v31']):.3f}" if fl and x["v31"] else ""
            fly_cells = (f"{(fl or fa or {}).get('tile', '')} | {fl['us_median'] if fl else ''} | "
                         f"{fa['us_median'] if fa else ''} | {fly_ratio}")
            o, t9, tt = x["overall"], x["v9"], x["torch"]
            ratio = f"{us(tt) / us(x['v31']):.3f}" if tt and x["v31"] else ""
            fr = f"{us(x['fp_tile']) / us(x['v31']):.3f}" if x["fp_tile"] else ""
            hbl = (f"{tt['hbl_tile']} | {tt['non_pow2_tile']} | {tt['us_median']} | {tt['tflops']}" if tt
                   else " | | | ")
            L.append(f"| {M} | {o['version'] if o else ''} | {cell(o)} | {cell(x['v31'])} | "
                     f"{cell(t9, method=False)} | {hbl} | {ratio} | {fr} | {fly_cells} |")
        L += ["", f"v31 fastest on {w31} of {len(ms)}; v31 over hipBLASLt geomean {gm(t):.3f}; "
              f"v31 over v9 geomean {gm(n9):.3f}.", ""]
    bad = [r for r in raw if not ok(r)]
    if bad:
        L += [f"## Failed or incorrect points ({len(bad)})", ""]
        L += [f"- {r['point']}: correct={r['correct']} {r['error']}" for r in bad]
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(L) + "\n")
    log(f"report: {len(raw)} points, {len(timed_shapes)} shapes, {len(bad)} failed or incorrect")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["all", "list", "warm", "timing", "report", "worker", "warm_worker"])
    ap.add_argument("--points")
    ap.add_argument("--manifest")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--jobs", type=int, default=64)
    args = ap.parse_args()
    # warm and timing processes share one cache, so timing never compiles.
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(TMP, "triton"))
    if args.mode == "worker":
        return worker(args)
    if args.mode == "warm_worker":
        return warm_worker(args)
    os.makedirs(RESULTS, exist_ok=True)
    if args.mode == "list":
        pts = [p for s in shapes.SHAPES for p in shape_points(*s)]
        print("\n".join(pts))
        print(f"{len(pts)} points over {len(shapes.SHAPES)} shapes", file=sys.stderr)
        return
    if args.mode in ("all", "warm"):
        done = {r["point"] for r in csv.DictReader(open(RAW))} if os.path.exists(RAW) else set()
        warm([p for b in plan_batches(done) for p in b if not p.startswith("tcheck:")], args.jobs)
    if args.mode in ("all", "timing"):
        time_all()
    if args.mode in ("all", "report"):
        report()


if __name__ == "__main__":
    main()
