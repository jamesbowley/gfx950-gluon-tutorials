#!/usr/bin/env python3
"""The library kernels that beat streamk_wip, with their tile, waves, k-steps and approach.

    python beaten_by.py select --warm results/details_v24.csv --cold results/details_v24_rot2048.csv \
        --out results/beaten_v24_shapes.csv
    V20CMP_GPU=3 ./run_rocm724_v20.sh python beaten_by.py probe --shapes results/beaten_v24_shapes.csv \
        --out results/beaten_v24_probe.json
    python beaten_by.py report --warm results/details_v24.csv --cold results/details_v24_rot2048.csv \
        --probe results/beaten_v24_probe.json --out results/beaten_v24.md

A shape is listed when the best library (the faster of AITER's pick and hipBLASLt, contiguous) is
more than 1% faster than streamk_wip and at least one of:
- the library's tile is 256x256x64;
- streamk_wip's 256x256 grid has at least one full wave (tiles // 256 >= 1);
- the library's own tile grid fills at least one wave of 256 CUs.

probe runs in the AITER worker's environment: per listed shape it records AITER's tuned config
(libtype, solidx, splitK) and, for AITER's pick and hipBLASLt, every kernel one call launches with
its grid and block from the torch profiler's trace. The trace's hipBLASLt grids are in work-items
(hipExtModuleLaunchKernel), the other libraries' in workgroups.

The approach is decoded from the kernel names and AITER's config:
- hipBLASLt (TensileLite solution names): MT tile; SK stream-K mode (0 data-parallel, 3 two-tile
  stream-K with the data-parallel tiles first, 4 per-XCD work queues, 5 hybrid); SKXCCM XCC
  remapping; SKFTR tree fix-up; GSU the solution's global split-U and GSUA its algorithm (SB single
  buffer with atomics, MB multiple buffers reduced by a second kernel, MBSK by the same kernel).
  StreamKAtomic is not in the name; TensileLite's default is 0 (partials in a workspace,
  deterministic fix-up in the same kernel). hipBLASLt can set GSU per problem at run time, so with
  probe data what the launch did comes from the launched kernels instead: a PostGSU kernel means a
  split K summed by that second kernel; otherwise the GEMM kernel's workgroups against its tiles
  give one per tile, or a persistent grid with its leftover tiles.
- FlyDSL hgemm: t{M}x{N}x{K}x{stages}, ks{K split}, w{M}x{N}x{K} warps; the K count is the waves
  that split K inside a workgroup (k-wave).
- opus mono tile: {threads}x{M}x{N}x{K}, then the warp layout and the MFMA shape; split-K from AITER's
  config.
- AITER asm bf16gemm: tile MxN with K steps of 64; splitk kernels split K over splitK workgroups
  that add into C through a semaphore workspace, the clean variant also zeroing C itself.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile

from pad_report import best_fields
from run_v20_vs_aiter import AITER, HERE, HVR, NUM_CUS, TRITON_PY, fnum, key, pct, read_rows, run_paths

TIE = 1.0


def first_cijk(names):
    return next((k for k in (names or "").split(" | ") if "Cijk_" in k), "")


def hbl_tokens(name):
    """TensileLite solution-name tokens, e.g. {"MT": "256x256x64", "SK": "3", "GSU": "0"}."""
    out = {}
    for tok in name.split("_"):
        m = re.fullmatch(r"(MT|MI|SKXCCM|SKFTR|SK|GSUC|GSU|MIWT|WGM|WG|NTC|NTD|DTVA|DTVB)(.+)", tok)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    m = re.search(r"_GSUA(MBSK|MB|SB)_", name)
    if m:
        out["GSUA"] = m.group(1)
    m = re.search(r"_WG(\d+)_(\d+)_(\d+)", name)
    if m:
        out["WG"] = "x".join(m.groups())
    m = re.search(r"_MIWT(\d+)_(\d+)_", name)
    if m:
        out["MIWT"] = "x".join(m.groups())
    return out


def lib_kernel(r, lib):
    """(family, kernel name) of the best library on row r."""
    if lib == "hipBLASLt":
        return "hipBLASLt", first_cijk(r.get("hbl_kernels_torch", ""))
    pick = r["aiter_pick"]
    if pick in ("torch", "hipblaslt"):
        return "hipBLASLt", first_cijk(r.get("pick_kernels_launched", "")) or first_cijk(r.get("hbl_kernels", ""))
    return pick, r["aiter_kernel"]


def decode(family, name, M, N, K, cfg=None, ks=None):
    """Tile (tm, tn, tk) in torch's M and N, and a dict of approach fields; ks: launched()'s kernels."""
    cfg = cfg or {}
    split = cfg.get("splitK")
    split = int(split) if split not in (None, "", "nan") and split == split else None
    d = dict(splitk="-", kwave="-", streamk="-", atomics="-", other="")
    tm = tn = tk = None
    if family == "hipBLASLt":
        t = hbl_tokens(name)
        if "MT" in t:
            # torch runs C^T = W X^T in hipBLASLt's column-major terms: MT0 tiles torch's N, MT1 its M.
            mt0, mt1, tk = (int(x) for x in t["MT"].split("x"))
            tm, tn = mt1, mt0
        sk = t.get("SK", "0")
        d["streamk"] = {"0": "none", "3": "SK3 (two-tile, DP first)", "4": "SK4 (XCD queues)",
                        "5": "SK5 (hybrid)"}.get(sk, f"SK{sk}")
        if sk != "0":
            d["streamk"] += f", XCC remap {t.get('SKXCCM', '?')}, " + ("tree" if t.get("SKFTR") == "1" else "linear") \
                + " fix-up"
        gsu = t.get("GSU", "0")
        d["splitk"] = "none (GSU0)" if gsu in ("0", "1") else f"GSU{gsu} ({t.get('GSUA', '?')})"
        if gsu not in ("0", "1") and t.get("GSUA") == "SB":
            d["atomics"] = "yes (GSU single buffer)"
        elif sk != "0":
            d["atomics"] = "not in name (default: workspace partials)"
        else:
            d["atomics"] = "no"
        d["kwave"] = "no"
        d["other"] = f"WG {t.get('WG', '?')}, MFMA {t.get('MI', '?')}, wave tile {t.get('MIWT', '?')}"
    elif family == "flydsl":
        m = re.search(r"_t(\d+)x(\d+)x(\d+)x(\d+)_ks(\d+)_w(\d+)x(\d+)x(\d+)", name)
        if m:
            tm, tn, tk, stages, ks, wm, wn, wk = (int(x) for x in m.groups())
            d.update(splitk=f"ks{ks}" + (" (none)" if ks == 1 else ""), kwave=f"{wk}" + (" (none)" if wk == 1 else ""),
                     streamk="none", atomics="no" if ks == 1 else "split-K adds",
                     other=f"{stages} stages, warps {wm}x{wn}x{wk}" + (", gm4" if "_gm4_" in name else ""))
    elif family == "opus":
        m = re.search(r"opus_gemm(?:_[a-z_]+?)?_(\d+)x(\d+)x(\d+)x(\d+)_(\d+)x(\d+)_(\d+x\d+x\d+)", name)
        if m:
            threads, tm, tn, tk, wm, wn, mfma = int(m.group(1)), int(m.group(2)), int(m.group(3)), \
                int(m.group(4)), m.group(5), m.group(6), m.group(7)
            d.update(splitk=f"{split}" if split and split > 1 else "none", kwave="no",
                     streamk="none", atomics="split-K adds" if split and split > 1 else "no",
                     other=f"{threads} threads, warps {wm}x{wn}, MFMA {mfma}" + (", 4g_safe" if "4g_safe" in name else ""))
    elif family == "asm":
        m = re.search(r"tn_(\d+)x(\d+)", name)
        if m:
            tm, tn, tk = int(m.group(1)), int(m.group(2)), 64
            sk = "splitk" in name
            d.update(splitk=(f"{split}" if split else "yes") if sk else "none", kwave="no", streamk="none",
                     atomics=("yes (semaphore" + (", C zeroed in-kernel)" if "clean" in name else ")")) if sk else "no",
                     other="B preshuffled" if "bshuffle" in name else "")
    if tm:
        tiles = math.ceil(M / tm) * math.ceil(N / tn)
        d.update(tile=f"{tm}x{tn}x{tk}", lib_tiles=tiles, lib_waves=round(tiles / NUM_CUS, 2),
                 lib_k_steps=math.ceil(K / tk))
    else:
        d.update(tile="?", lib_tiles="", lib_waves="", lib_k_steps="")
    if family == "hipBLASLt" and ks and tm:
        # The solution name does not say what this launch did: hipBLASLt sets GSU per problem at run
        # time. A PostGSU kernel means K was split over buffers that a second kernel sums.
        main = next((w for n, w, _ in ks if "_MT" in n and w), None)
        post = next((n for n, _, _ in ks if "PostGSU" in n), None)
        mode = "SK" + hbl_tokens(name).get("SK", "0")
        if main and post:
            ways = main / tiles
            d.update(splitk=f"runtime, {ways:.0f} ways ({main} WGs for {tiles} tiles)" if ways == int(ways) else
                     f"runtime ({main} WGs for {tiles} tiles)", atomics="no (buffers summed by a 2nd kernel)",
                     streamk=f"{mode} kernel, split-U instead")
            post_name = re.search(r"PostGSU\d*", post).group(0)
            d["other"] += f"; 2nd kernel {post_name}"
        elif main:
            d["splitk"] = "none"
            if main < tiles:
                left = tiles % main
                d["streamk"] = (f"{mode}: persistent, {main} WGs, "
                                + (f"{left} leftover tiles" if left else "no leftover tiles"))
            elif main == tiles:
                d["streamk"] = f"{mode} kernel, one WG per tile: no K split"
            else:
                d["streamk"] = f"{mode}: {main} WGs for {tiles} tiles, K split"
    return d


def selected(rows):
    """Rows where the best library is more than 1% ahead and the selection rule holds."""
    out = []
    for r in rows:
        b = best_fields(r, "")
        v = fnum(r["v20_tflops"])
        gap = pct(v, b["best"])
        if not gap < -TIE:
            continue
        family, name = lib_kernel(r, "hipBLASLt" if b["best_lib"] == "hipBLASLt" else "aiter")
        d = decode(family, name, int(r["M"]), int(r["N"]), int(r["K"]))
        same_tile = d["tile"] == "256x256x64"
        ours = int(r["waves"]) >= 1
        theirs = d["lib_tiles"] != "" and d["lib_tiles"] >= NUM_CUS
        if same_tile or ours or theirs:
            why = ", ".join(w for w, ok in (("same tile", same_tile), ("our grid >= 1 wave", ours),
                                            ("their grid >= 1 wave", theirs)) if ok)
            out.append((r, b, gap, family, name, why))
    return out


def cmd_select(args):
    keys = {}
    for state, path in (("warm", args.warm), ("cold", args.cold)):
        for r, *_ in selected(read_rows(path)):
            keys.setdefault(key(r["M"], r["N"], r["K"], r["out"]), set()).add(state)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "out", "states"])
        for (M, N, K, out), st in sorted(keys.items()):
            w.writerow([M, N, K, out, "+".join(sorted(st))])
    print(f"wrote {args.out}: {len(keys)} shapes")


def probe_child(args):
    import torch
    from torch.profiler import ProfilerActivity, profile

    import aiter.tuned_gemm as tg
    from aiter.tuned_gemm import get_GEMM_A16W16_config
    from run_v20_vs_aiter import Inputs, device_info, hbl_call

    info = device_info()
    res = dict(device=info, shapes={})
    for s in read_rows(args.shapes):
        inp = Inputs(dict(M=s["M"], N=s["N"], K=s["K"], out=s["out"]))
        cfg = get_GEMM_A16W16_config(M=inp.M, N=inp.N, K=inp.K, bias=False, dtype=str(inp.x.dtype),
                                     otype=str(inp.otype), scaleAB=False, bpreshuffle=False)
        cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in dict(cfg).items()}
        hbl = hbl_call(inp.otype)
        fns = dict(pick=lambda: tg.gemm_a16w16(inp.x, inp.w, None, inp.otype), hbl=lambda: hbl(inp.x, inp.w, None))
        entry = dict(cfg={k: v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
                          for k, v in cfg.items()})
        for n, fn in fns.items():
            fn()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
                path = tf.name
            prof.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
            os.unlink(path)
            ks = [dict(name=e["name"], dur_us=e.get("dur"), args=e.get("args", {}))
                  for e in trace.get("traceEvents", []) if str(e.get("cat", "")).lower() == "kernel"]
            entry[n] = ks
        res["shapes"][f"{s['M']}x{s['N']}x{s['K']}:{s['out']}"] = entry
        print(f"{s['M']}x{s['N']}x{s['K']} {s['out']}: {cfg.get('libtype')} splitK={cfg.get('splitK')} "
              + " | ".join(f"{n}: " + ", ".join(f"{k['name'][:40]} grid={k['args'].get('grid')} "
                                               f"block={k['args'].get('block')}" for k in entry[n]) for n in fns),
              flush=True)
        del inp
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {args.out}: {len(res['shapes'])} shapes")


def cmd_probe(args):
    if os.environ.get("BEATEN_CHILD"):
        return probe_child(args)
    assert os.environ.get("V20CMP_ENV") == "rocm724", "run through run_rocm724_v20.sh"
    env = dict(os.environ, BEATEN_CHILD="1", V20CMP_HBL="1",
               PYTHONPATH=f"{HVR}/shim:{TRITON_PY}:{AITER}:{HERE}",
               AITER_JIT_DIR=run_paths("", args.jit_tag)["jit"], TRITON_CACHE_DIR=f"{HERE}/cache/aiter")
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env, cwd=HERE))


def launched(entry, n):
    """The kernels one call launched: [(short name, workgroups, block threads)]."""
    out = []
    for k in (entry or {}).get(n, []):
        a = k["args"]
        grid, block = a.get("grid"), a.get("block")
        wgs = threads = None
        if isinstance(grid, list) and isinstance(block, list) and all(block):
            threads = math.prod(block)
            # hipBLASLt launches through hipExtModuleLaunchKernel, whose grid is in work-items; the
            # other libraries' grids are in workgroups.
            wgs = math.prod(grid) // threads if "Cijk_" in k["name"] else math.prod(grid)
        out.append((k["name"], wgs, threads))
    return out


def main_wgs(ks, name):
    """Workgroups of the GEMM kernel: the launched kernel that matches name, else the first."""
    m = next((w for n, w, _ in ks if name and (n == name or n in name or name in n)), None)
    return m or next((w for n, w, _ in ks if "PostGSU" not in n), None)


def short(name):
    m = re.search(r"_MT\d+x\d+x\d+", name)
    if "Cijk_" in name:
        return "Cijk ... " + (m.group(0)[1:] if m else "") + "".join(
            f"_{t}" for t in re.findall(r"_(SK\d|SKXCCM\d|GSU\d+)(?=_)", name))
    return name.replace("_ZN5aiter", "").rstrip("E")[:70]


HEAD = ("| # | Shape (MxNxK) | out | v24 vs best | Best library | Kernel | Tile (MxNxK) | Their tiles / waves "
        "| Launched WGs x threads | Their k-steps: per tile / per WG | Our tiles / full waves / S | Our k-steps "
        "| Our config | Stream-K | Split-K | K-wave | Atomics | Other |")
SEP = "|---:|---|---|---:|---|---|---|---|---|---|---|---:|---|---|---|---|---|---|"


def row_md(i, r, b, gap, family, name, probe):
    e = probe.get(f"{r['M']}x{r['N']}x{r['K']}:{r['out']}", {})
    src = "hbl" if b["best_lib"] == "hipBLASLt" else "pick"
    ks = [k for k in launched(e, src) if k[1]]
    d = decode(family, name, int(r["M"]), int(r["N"]), int(r["K"]), e.get("cfg") if src == "pick" else None, ks)
    wg = "; ".join((f"{short(n)[:28]}: " if len(ks) > 1 else "") + f"{w} x {t}" for n, w, t in ks) or "-"
    main = main_wgs(ks, name)
    # Average k-steps a GEMM workgroup runs: all tiles' k-steps over the workgroups launched.
    per_wg = f"{d['lib_tiles'] * d['lib_k_steps'] / main:.1f}" if main and d["lib_tiles"] != "" else ""
    lib = b["best_lib"] + (f" (splitK {e['cfg'].get('splitK')})" if src == "pick" and e.get("cfg") else "")
    handoff = ", handed to v9" if str(r.get("v20_v9_delegated")) == "True" else ""
    return (f"| {i} | {r['M']}x{r['N']}x{r['K']} | {r['out'][:4]} | {gap:+.1f}% | {lib} | `{short(name)}` | "
            f"{d['tile']} | {d['lib_tiles']} / {d['lib_waves']} | {wg} | {d['lib_k_steps']} / {per_wg or '-'} | "
            f"{r['tiles']} / {r['waves']} / {r['S']} | {r['k_steps']} | {r['v20_label']}{handoff} | "
            f"{d['streamk']} | {d['splitk']} | {d['kwave']} | {d['atomics']} | {d['other']} |")


def table(state, rows, probe):
    sel = selected(rows)
    sel.sort(key=lambda x: x[2])
    same = [x for x in sel if "same tile" in x[5]]
    rest = [x for x in sel if "same tile" not in x[5]]
    L = [f"**{state.capitalize()}: {len(sel)} shapes, {len(same)} with the library on a 256x256x64 tile**", ""]
    for title, part in (("Library on a 256x256x64 tile", same),
                        ("Library on another tile, with our grid or theirs at least one wave", rest)):
        if not part:
            continue
        L += [f"*{title} ({len(part)}), worst first:*", "", HEAD, SEP]
        L += [row_md(i, r, b, gap, family, name, probe) for i, (r, b, gap, family, name, why) in enumerate(part, 1)]
        L += [""]
    return L, sel


def cmd_report(args):
    probe = {}
    if args.probe and os.path.exists(args.probe):
        with open(args.probe) as f:
            probe = json.load(f)["shapes"]
    tables, sels = [], {}
    for state, path in (("warm", args.warm), ("cold", args.cold)):
        t, sel = table(state, read_rows(path), probe)
        sels[state] = sel
        tables += t
    L = ["## Library kernels that beat v24 with the same tile or at least one wave", ""]
    L += ["**Patterns.** By the winning kernel's approach and launched grid (workgroups of its GEMM kernel), "
          "with the median gap and how many of those shapes are under one wave of our 256x256 tiles:", "",
          "| Caches | Approach | Launched grid | Shapes | v24 vs best: median | Our grid under one wave | "
          "Their tile area vs 256x256: median |", "|---|---|---|---:|---:|---:|---:|"]
    for state, sel in sels.items():
        pat = {}
        for r, b, gap, family, name, why in sel:
            e = probe.get(f"{r['M']}x{r['N']}x{r['K']}:{r['out']}", {})
            src = "hbl" if b["best_lib"] == "hipBLASLt" else "pick"
            ks = [k for k in launched(e, src) if k[1]]
            d = decode(family, name, int(r["M"]), int(r["N"]), int(r["K"]), e.get("cfg") if src == "pick" else None, ks)
            main = main_wgs(ks, name)
            if family == "hipBLASLt":
                appr = ("hipBLASLt, runtime split-K + 2nd kernel" if d["splitk"].startswith("runtime") else
                        "hipBLASLt SK3 kernel")
            elif family == "asm":
                appr = "AITER asm split-K" if "splitk" in name else "AITER asm"
            else:
                appr = f"AITER {family}" + (f", split-K {d['splitk']}" if d["splitk"] not in ("none", "ks1 (none)", "-")
                                            else "") + (f", K-wave {d['kwave'][0]}" if d["kwave"][:1].isdigit()
                                                        and d["kwave"][0] != "1" else "")
            tiles = d["lib_tiles"]
            if main is None or tiles == "":
                grid = "-"
            elif main == tiles:
                grid = "one WG per tile" + (", exactly 256" if main == NUM_CUS else "")
            elif main < tiles:
                grid = f"persistent, {'256' if main == NUM_CUS else 'under 256'} WGs, more tiles"
            else:
                grid = "more WGs than tiles (K split)"
            tm, tn = (int(x) for x in d["tile"].split("x")[:2]) if d["tile"] != "?" else (0, 0)
            pat.setdefault((appr, grid), []).append((gap, int(r["tiles"]) < NUM_CUS, tm * tn / 65536))
        for (appr, grid), v in sorted(pat.items(), key=lambda kv: -len(kv[1])):
            L.append(f"| {state} | {appr} | {grid} | {len(v)} | {statistics.median(g for g, _, _ in v):+.1f}% | "
                     f"{sum(u for _, u, _ in v)} | {statistics.median(a for _, _, a in v):.2f} |")
    L += [""] + tables
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("select")
    s.add_argument("--warm", required=True)
    s.add_argument("--cold", required=True)
    s.add_argument("--out", required=True)
    q = sub.add_parser("probe")
    q.add_argument("--shapes", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--jit-tag", default="main")
    r = sub.add_parser("report")
    r.add_argument("--warm", required=True)
    r.add_argument("--cold", required=True)
    r.add_argument("--probe", default=None)
    r.add_argument("--out", required=True)
    args = p.parse_args()
    dict(select=cmd_select, probe=cmd_probe, report=cmd_report)[args.mode](args)


if __name__ == "__main__":
    main()
