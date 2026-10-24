#!/usr/bin/env python3
"""Enumerate the full v28 tuning space and capture each distinct host launch.

This is intentionally a host-side operation: kernel launchers are replaced by
recording stubs, so no Triton build or GPU kernel launch occurs.
"""

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
AIS = os.path.abspath(os.path.join(A16, "..", "..", "..", "..", "experiments", "aiter_lixun_shapes"))
sys.path.insert(0, A16)

import bench_ext  # noqa: E402,F401  (sets the kernel import paths)
import torch  # noqa: E402

sweep = importlib.import_module("v28_streamk_owner.sweep")

FIELDS = (
    "M", "N", "K", "scope", "models", "bf16_reduce", "kind", "tile", "point", "options",
    "alias", "reject_reason", "kernel", "grid", "launch", "build_key", "batch",
)
CAPTURE_VERSION = 1


def transpose(tile):
    bm, bn, *rest = tile.split("x")
    return "x".join([bn, bm] + rest)


def candidate_tiles():
    """The 34 REFIT tiles plus accepted transposes of small tiles."""
    v28 = sweep.v27_module()
    out = []
    for tile in sweep.REFIT_TILES:
        for spec in (tile, transpose(tile)):
            if spec in out:
                continue
            try:
                geo = v28.tile_geometry(spec)
            except AssertionError:
                continue
            if spec == tile or geo["small"]:
                out.append(spec)
    return out


def pow2(x):
    return 1 << max(0, (x - 1).bit_length())


def tile_allowed(tile, M, N, K, max_pad):
    if sweep.shape_info(tile, M, N, K)["pairs"] < 2:
        return False
    bm, bn = (int(x) for x in tile.split("x")[:2])
    return bm <= max_pad * max(16, pow2(M)) and bn <= max_pad * max(16, pow2(N))


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return {
            "tensor": str(value.dtype).removeprefix("torch."),
            "stride": list(value.stride()),
            "ndim": value.ndim,
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return repr(value)


class CaptureKernel:
    def __init__(self, name, sink):
        self.name = name
        self.sink = sink

    def _record(self, args, grid, kwargs):
        launch = {
            "version": CAPTURE_VERSION,
            "kernel": self.name,
            "args": jsonable(args),
            "grid": jsonable(grid),
            "kwargs": jsonable(kwargs),
        }
        self.sink.clear()
        self.sink.update(launch)
        return types.SimpleNamespace(metadata=types.SimpleNamespace(shared=0))

    def warmup(self, *args, grid=None, **kwargs):
        return self._record(args, grid, kwargs)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self._record(args, grid, kwargs)


class HostCapture:
    def __init__(self):
        self.launch = {}
        self.v28 = sweep.v27_module()
        self.v9 = importlib.import_module("v9_tile.matmul_kernel")
        self.v28.v28_streamk_owner = CaptureKernel("v28_streamk_owner", self.launch)
        self.v28.check_residency = lambda *args, **kwargs: 0
        self.v28.WARMUP = True
        for name in ("v9_tile", "v9_tile_small", "v9_tile_2r"):
            setattr(self.v9, name, CaptureKernel(name, self.launch))

    def __call__(self, kind, tile, M, N, K):
        self.launch.clear()
        a = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((N, K), device="cuda", dtype=torch.bfloat16).T
        c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        run = sweep.runner(kind, tile, M, N, K)
        run(a, b, c)
        if not self.launch:
            # WARMUP deliberately returns before a v28 data-parallel fallback
            # calls v9. Capture that fallback explicitly.
            last = dict(self.v28.LAST_LAUNCH)
            if last.get("kernel") != "v9_tile":
                raise RuntimeError("host launcher returned without a captured launch")
            self.v9.matmul(a, b, c, tile=last["tile"])
        return dict(self.launch)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_signature(launch):
    return {
        "version": launch["version"],
        "kernel": launch["kernel"],
        "args": launch["args"],
        "kwargs": launch["kwargs"],
    }


def read_shapes(shapes_path, picks_path):
    shapes = list(csv.DictReader(open(shapes_path)))
    picked = {
        (int(r["M"]), int(r["N"]), int(r["K"]))
        for r in csv.DictReader(open(picks_path))
    }
    return [r for r in shapes if (int(r["M"]), int(r["N"]), int(r["K"])) in picked]


def enumerate_points(args):
    shapes = read_shapes(args.shapes, args.picks)
    if args.shape:
        wanted = {tuple(int(x) for x in value.split("x")) for value in args.shape}
        shapes = [r for r in shapes if tuple(int(r[x]) for x in ("M", "N", "K")) in wanted]
        found = {tuple(int(r[x]) for x in ("M", "N", "K")) for r in shapes}
        if found != wanted:
            raise SystemExit(f"requested shapes not found: {sorted(wanted - found)}")
    if args.sample_shapes:
        if args.sample_shapes > len(shapes):
            raise SystemExit(f"--sample-shapes {args.sample_shapes} exceeds {len(shapes)} available shapes")
        indices = {
            round(i * (len(shapes) - 1) / max(1, args.sample_shapes - 1))
            for i in range(args.sample_shapes)
        }
        shapes = [shape for i, shape in enumerate(shapes) if i in indices]
    if args.limit_shapes:
        shapes = shapes[: args.limit_shapes]
    tiles = candidate_tiles()
    if args.tiles:
        requested = args.tiles.split(",")
        unknown = set(requested) - set(tiles)
        if unknown:
            raise SystemExit(f"unknown candidate tiles: {sorted(unknown)}")
        tiles = requested
    fp32_kinds = [
        kind for kind in sweep.WORST10_KINDS
        if not sweep.split_kind(kind) or sweep.split_kind(kind)[2] == "fp32"
    ]
    selected_kinds = args.kinds.split(",") if args.kinds else None
    if selected_kinds:
        unknown = set(selected_kinds) - set(sweep.WORST10_KINDS)
        if unknown:
            raise SystemExit(f"unknown WORST10 kinds: {sorted(unknown)}")
    capture = None if args.no_capture else HostCapture()
    rows = []
    for shape_index, shape in enumerate(shapes):
        M, N, K = (int(shape[x]) for x in ("M", "N", "K"))
        kinds = sweep.WORST10_KINDS if shape.get("bf16_reduce") else fp32_kinds
        if selected_kinds:
            kinds = [kind for kind in kinds if kind in selected_kinds]
        seen = {}
        shape_points = 0
        for tile in tiles:
            if not tile_allowed(tile, M, N, K, args.max_pad):
                continue
            for kind in kinds:
                if not sweep.valid(kind, tile, M, N, K):
                    continue
                point = sweep.point_name(kind, tile, M, N, K)
                options = {} if kind == "dp" else sweep.kind_env(kind, tile, M, N, K)
                row = {
                    "M": M,
                    "N": N,
                    "K": K,
                    "scope": shape.get("scope", ""),
                    "models": shape.get("models", ""),
                    "bf16_reduce": shape.get("bf16_reduce", ""),
                    "kind": kind,
                    "tile": tile,
                    "point": point,
                    "options": json.dumps(options, sort_keys=True, separators=(",", ":")),
                    "alias": "",
                    "reject_reason": "",
                    "kernel": "",
                    "grid": "",
                    "launch": "",
                    "build_key": "",
                    "batch": shape_index // args.batch_shapes,
                }
                if capture is not None:
                    try:
                        launch = capture(kind, tile, M, N, K)
                        key = json.dumps(launch, sort_keys=True, separators=(",", ":"))
                        row.update(
                            kernel=launch["kernel"],
                            grid=json.dumps(launch["grid"], separators=(",", ":")),
                            launch=key,
                            build_key=digest(build_signature(launch)),
                        )
                        if key in seen:
                            row["alias"] = seen[key]
                        else:
                            seen[key] = point
                    except (AssertionError, RuntimeError) as exc:
                        row["reject_reason"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                shape_points += 1
        print(
            f"{shape_index + 1}/{len(shapes)} {M}x{N}x{K}: {shape_points} points",
            flush=True,
        )
    return rows, tiles, len(shapes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default=os.path.join(AIS, "shapes_origami.csv"))
    ap.add_argument("--picks", default=os.path.join(AIS, "picks_origami.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "full_points.csv"))
    ap.add_argument("--max-pad", type=float, default=2.0)
    ap.add_argument("--batch-shapes", type=int, default=100)
    ap.add_argument("--limit-shapes", type=int, default=0)
    ap.add_argument("--sample-shapes", type=int, default=0, help="select evenly spaced shapes across the input")
    ap.add_argument("--shape", action="append", help="select one MxNxK shape; may be repeated")
    ap.add_argument("--tiles", help="comma-separated subset of the 34 candidate tiles")
    ap.add_argument("--kinds", help="comma-separated subset of sweep.WORST10_KINDS")
    ap.add_argument("--no-capture", action="store_true", help="enumerate only; useful for a CPU-side smoke check")
    args = ap.parse_args()
    if args.max_pad <= 0 or args.batch_shapes <= 0:
        ap.error("--max-pad and --batch-shapes must be positive")

    rows, tiles, nshapes = enumerate_points(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    canonical = sum(not r["alias"] and not r["reject_reason"] for r in rows)
    aliases = sum(bool(r["alias"]) for r in rows)
    rejects = sum(bool(r["reject_reason"]) for r in rows)
    print(
        f"{len(rows):,} points on {nshapes:,} shapes and {len(tiles)} tiles: "
        f"{canonical:,} timed, {aliases:,} aliases, {rejects:,} host rejects -> {args.out}"
    )


if __name__ == "__main__":
    main()
