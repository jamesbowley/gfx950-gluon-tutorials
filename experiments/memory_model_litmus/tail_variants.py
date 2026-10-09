"""Measured stream-K tails of every variant, S = 1-255, on one matrix, against the model (no GPU).

    python tail_variants.py > results/tail_variants.md   # also images/tail_variants*.png

Reads experiments/streamk_split_count/results/sweep_c.jsonl: rocprof kernel times on 4096x8192xK
with GEMM_TILE_LIMIT = 256 + S, so every S has the same one full wave, the same A and B and the same
full-wave tiles. The tail is the kernel time minus the same matrix's with 256 tiles (the full wave
alone), for every variant alike:
- data-parallel (v13): the S leftover tiles as a partial last wave;
- one-tile, end-to-end cuts (v16): the leftover tiles' k-steps laid end to end and cut into S n
  pieces, so programs can cross tile boundaries;
- one-tile, tile-aligned cuts (v20 one_tile), with the owner or the reduce-scatter ending: n
  programs per tile, none crossing tiles; at the default (all 256 programs) a tile gets 256 // S;
- one-tile, end-to-end reversed (v20 spread), all 256 programs;
- two-tile, forward (v16 STREAMK_POLICY=two_tile) and reversed (v19): the leftover tiles plus the
  full wave, end to end over all 256 programs.
The model (split_model.py: MAC + synthetic exchange) is drawn over the tile-aligned splits.
"""

import ast
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import split_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
SPLITS = {
    "v16": ("one-tile, end-to-end cuts (v16)", "#4878a8", "o"),
    "v20 spread": ("one-tile, end-to-end reversed, capped (v20 spread)", "#e08a1e", "D"),
    "v20 owner": ("one-tile, tile-aligned, owner ending (v20)", "#c0504d", "s"),
    "v20 rs": ("one-tile, tile-aligned, reduce-scatter (v20)", "#6aa56a", "^"),
    "v22 atomic": ("one-tile, tile-aligned, bf16 atomics into C (v22)", "#8c564b", "v"),
    "v22 atomic_iz": ("one-tile, tile-aligned, bf16 atomics, C zeroed in the GEMM (v22)", "#17becf", "P"),
}
MAIN_K = (2048, 4096, 8192, 16384)
WHOLES = {
    "v13": ("data-parallel (v13)", "0.25", (0, (1, 1.5))),
    "v16 two_tile": ("two-tile, forward (v16)", "#8064a2", (0, (5, 2))),
    "v19": ("two-tile, reversed (v19)", "#8064a2", "-"),
    "v20 spread": ("one-tile, end-to-end reversed (v20 spread)", "#e08a1e", (0, (4, 1.5, 1, 1.5))),
}
SHORT = {"v13": "dp", "v16": "e2e", "v20 owner": "owner", "v20 rs": "rs", "v16 two_tile": "2T fwd", "v19": "2T rev",
         "v20 spread": "spread", "v22 atomic": "atomic", "v22 atomic_iz": "atomic iz"}
ATOMICS = ("v22 atomic", "v22 atomic_iz")
NS = (1, 2, 3, 4, 6, 8, 12, 16, 32, 64)
GRID_S = (2, 8, 32, 128)


def load():
    rows = split_model.sweep_rows()
    base = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base" and r.get("limit") == 256}
    lim = [dict(r, tail=r["us"] - base[r["K"]]) for r in rows if r.get("limit") and r["K"] in base
           and r["config"] != "v13 base"]
    real = [r for r in rows if not r.get("limit")]
    two = {r["K"]: r["us"] for r in rows if r["config"] == "v13 base" and r.get("limit") == 512}
    return lim, real, base, two


def pick(rows, S, K, config, n=None):
    """The row of one case; n = None matches the one-run configs (n "-") and anything else n exactly."""
    want = "-" if n is None else n
    for r in rows:
        if r["S"] == S and r["K"] == K and r["config"] == config and r["n"] == want:
            return r
    if n is None:  # a config with no "-" row (the splits): its first row
        return next((r for r in rows if r["S"] == S and r["K"] == K and r["config"] == config), None)
    return None


def best_split(rows, S, K, config):
    rr = [r for r in rows if r["S"] == S and r["K"] == K and r["config"] == config]
    return min(rr, key=lambda r: r["tail"]) if rr else None


def n_of(r):
    """The programs per tile of a split row: n, or 256 / S for the default."""
    return 256 / r["S"] if r["n"] == "default" else r["n"]


V20_KERNEL = os.path.join(split_model.REPO, "kernels", "gemm", "intra_wave", "a16w16", "v20_streamk_reduce_scatter",
                          "matmul_kernel.py")


def load_capped_policy():
    """v20's capped_policy (the default host rule), taken from the kernel's source so that the replay
    is the rule itself; it needs only math, so Triton is not imported."""
    src = open(V20_KERNEL).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "capped_policy")
    ns = {"math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), V20_KERNEL, "exec"), ns)
    return ns["capped_policy"]


def single_rounding_best(rows, S, K):
    """The fastest measured configuration that rounds C once (data-parallel included)."""
    cfgs = [c for c in [*SPLITS, *WHOLES] if c not in ATOMICS]
    return min((b for c in cfgs if (b := best_split(rows, S, K, c))), key=lambda r: r["tail"])


def grid_figure(rows, model, Ks):
    fig, axes = plt.subplots(len(GRID_S), len(Ks), figsize=(4.3 * len(Ks), 3.5 * len(GRID_S)), squeeze=False)
    for i, S in enumerate(GRID_S):
        for j, K in enumerate(Ks):
            ax = axes[i][j]
            ymax = 0
            for cfg, (label, color, ls) in WHOLES.items():
                r = pick(rows, S, K, cfg)
                if r:
                    ax.axhline(r["tail"], color=color, ls=ls, lw=1.4, label=label)
                    ymax = max(ymax, r["tail"])
            for cfg, (label, color, mk) in SPLITS.items():
                rr = sorted((r for r in rows if r["S"] == S and r["K"] == K and r["config"] == cfg
                             and isinstance(r["n"], int)), key=lambda r: r["n"])
                if rr:
                    ax.plot([r["n"] for r in rr], [r["tail"] for r in rr], marker=mk, color=color, lw=1.8, ms=4.5,
                            label=label)
                d = pick(rows, S, K, cfg, "default")
                if d:
                    ax.plot([n_of(d)], [d["tail"]], marker="*" if cfg != "v16" else "D", color=color, ms=12 if cfg != "v16" else 7,
                            mec="k", mew=0.6, ls="none", zorder=5,
                            label="default split, 256 / S per tile" if (i, j, cfg) == (0, 0, "v20 owner") else None)
            for p, color in split_model.PATTERNS.items():
                ns = model.splits(S, K)
                if ns:
                    ax.plot(ns, [model.tail(p, S, n, K)[0] for n in ns], ls="--", color=color, lw=1, alpha=0.8,
                            label=f"model: MAC + exchange, {p}")
            dp = pick(rows, S, K, "v13")
            top = max(ymax, dp["tail"] if dp else 0) * 1.5
            ax.set_ylim(0, top)
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS], fontsize=7)
            ax.set_xlim(0.8, 80)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"K = {K} ({K // 64} k-steps per tile)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"S = {S}: tail us")
            if i == len(GRID_S) - 1:
                ax.set_xlabel("programs per tile n")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Measured stream-K tails on one 4096x8192 matrix (first 256 + S tiles): kernel time minus the full "
                 "wave alone. Dashed: the model.", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = os.path.join(IMG, "tail_variants.png")
    fig.savefig(out, dpi=130)
    return out


def summary_figure(rows, Ss, Ks):
    fig, axes = plt.subplots(1, 3, figsize=(21, 0.42 * len(Ss) + 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    cfgs = list(SPLITS) + [c for c in WHOLES if c != "v13" and c not in SPLITS]
    rel = {c: np.full((len(Ss), len(Ks)), np.nan) for c in cfgs}
    for ax, use, title in ((axes[0], cfgs, "Fastest tail, any variant"),
                           (axes[1], [c for c in cfgs if c not in ATOMICS],
                            "Fastest tail rounding once (no atomic ending)")):
        grid = np.zeros((len(Ss), len(Ks)))
        text = [[""] * len(Ks) for _ in Ss]
        for i, S in enumerate(Ss):
            for j, K in enumerate(Ks):
                dp = pick(rows, S, K, "v13")
                if not dp:
                    continue
                cands = [b for c in use if (b := best_split(rows, S, K, c))]
                for b in cands:
                    rel[b["config"]][i, j] = dp["tail"] / b["tail"]
                b = min(cands, key=lambda r: r["tail"])
                if b["tail"] >= dp["tail"]:
                    grid[i, j], text[i][j] = 1.0, "dp"
                else:
                    grid[i, j] = dp["tail"] / b["tail"]
                    n = "" if b["n"] == "-" else (" d" if b["n"] == "default" else f" {b['n']}")
                    text[i][j] = f"{SHORT[b['config']]}{n}"
        im = ax.imshow(grid, cmap="YlGn", norm=LogNorm(vmin=1, vmax=16), aspect="auto")
        for i in range(len(Ss)):
            for j in range(len(Ks)):
                ax.text(j, i, f"{text[i][j]}\n{grid[i, j]:.2f}x", ha="center", va="center", fontsize=7,
                        color="w" if grid[i, j] > 5 else "k")
        ax.set_xticks(range(len(Ks)), [str(K) for K in Ks])
        ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
        ax.set_xlabel("K")
        ax.set_ylabel("leftover tiles S")
        ax.set_title(f"{title}: variant and n (d = default), speed-up over data-parallel", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    ax = axes[2]
    width = 0.8 / len(cfgs)
    for k, c in enumerate(cfgs):
        label, color = (SPLITS.get(c) or WHOLES[c])[:2]
        vals = np.nanmedian(np.where(np.isfinite(rel[c]), rel[c], np.nan), axis=1)
        ax.barh(np.arange(len(Ss)) + (k - len(cfgs) / 2) * width, vals, height=width, color=color,
                alpha=0.55 + 0.45 * (k % 2), label=label)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
    ax.invert_yaxis()
    ax.set_xlabel("data-parallel tail / variant's tail at its best n (median over K)")
    ax.set_title("Each variant at its best n, against data-parallel", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = os.path.join(IMG, "tail_variants_summary.png")
    fig.savefig(out, dpi=130)
    return out


def high_s_figure(rows, Ks):
    Ss = sorted({r["S"] for r in rows if r["S"] > 128})
    if not Ss:
        return None
    fig, axes = plt.subplots(1, len(Ks), figsize=(5 * len(Ks), 4), squeeze=False)
    cfgs = {**{k: v for k, v in WHOLES.items()}, "v16": ("one-tile, end-to-end cuts (v16), 256 programs", "#4878a8", "-")}
    for ax, K in zip(axes[0], Ks):
        for c, (label, color, ls) in cfgs.items():
            pts = [(S, r["tail"]) for S in Ss if (r := pick(rows, S, K, c))]
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=4, color=color, ls=ls if c != "v16" else "-", lw=1.6, label=label)
        ax.set_title(f"K = {K}", fontsize=10)
        ax.set_xlabel("leftover tiles S")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("tail us")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("S > 128: the tile-aligned split has one program per tile (data-parallel), so only end-to-end cuts "
                 "and two-tile can split", fontsize=10)
    fig.tight_layout()
    out = os.path.join(IMG, "tail_variants_high_s.png")
    fig.savefig(out, dpi=130)
    return out


def main():
    lim, real, base, two = load()
    model = split_model.Model()
    Ss = sorted({r["S"] for r in lim if r["S"] <= 128})
    Ks = sorted({r["K"] for r in lim if r["S"] <= 128})
    Ks_hi = sorted({r["K"] for r in lim if r["S"] > 128})
    Ks_main = [K for K in Ks if K in MAIN_K]
    os.makedirs(IMG, exist_ok=True)
    outs = [grid_figure(lim, model, Ks_main), summary_figure(lim, Ss, Ks), high_s_figure(lim, Ks_hi)]

    print("# Stream-K tails of every variant on one matrix (sweep C)\n")
    print("rocprof kernel times (1000 launches, rotating inputs, median) on 4096x8192xK with GEMM_TILE_LIMIT = "
          "256 + S. Tail = kernel time minus the same matrix's with 256 tiles. Figures: "
          + ", ".join(f"`images/{os.path.basename(o)}`" for o in outs if o) + ".\n")
    print("## Baselines\n")
    print("| K | full wave alone (T = 256) us | two full waves (T = 512) us | second wave us |")
    print("|---|---|---|---|")
    for K in sorted(base):
        print(f"| {K} | {base[K]:.1f} | {two.get(K, float('nan')):.1f} | {two.get(K, float('nan')) - base[K]:.1f} |")

    print("\n## The limit against real shapes (K = 8192, kernel us)\n")
    print("| S | real shape | config | real | limited matrix | limited / real |")
    print("|---|---|---|---|---|---|")
    ratios = []
    for r in sorted(real, key=lambda r: (r["S"], r["config"])):
        lr = pick(lim, r["S"], r["K"], r["config"], r["n"] if r["n"] != "-" else None)
        if lr:
            ratios.append(lr["us"] / r["us"])
            print(f"| {r['S']} | {r['M']}x{r['N']} | {r['config']}{'' if r['n'] == '-' else ' ' + str(r['n'])} | "
                  f"{r['us']:.1f} | {lr['us']:.1f} | {lr['us'] / r['us']:.3f} |")
    if ratios:
        print(f"\nlimited / real: median {statistics.median(ratios):.3f}, range {min(ratios):.3f}-{max(ratios):.3f}\n")

    print("## Tails, us: each variant at its best n (n in brackets; d = default, 256 / S per tile)\n")
    print("Spread is the best of its default (all 256 programs, no n) and its capped splits. The atomic ending's "
          "tail includes its zeroing launch and the gap before the GEMM.\n")
    print("| S | K | data-parallel | end-to-end (v16) | tile-aligned owner | tile-aligned rs | tile-aligned atomic (v22) "
          "| atomic, zeroed in the GEMM | default split owner / rs | spread | two-tile fwd | two-tile rev | fastest |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for S in sorted({r["S"] for r in lim}):
        for K in sorted({r["K"] for r in lim if r["S"] == S}):
            dp = pick(lim, S, K, "v13")

            def cell(c):
                b = best_split(lim, S, K, c)
                if not b:
                    return "-"
                n = "" if b["n"] == "-" else f" ({'d' if b['n'] == 'default' else b['n']})"
                return f"{b['tail']:.1f}{n}"

            defs = [pick(lim, S, K, c, "default") for c in ("v20 owner", "v20 rs")]
            cands = [b for c in [*SPLITS, *WHOLES] if (b := best_split(lim, S, K, c))]
            f = min(cands, key=lambda r: r["tail"]) if cands else None
            fn = "" if not f or f["n"] == "-" else f" {'d' if f['n'] == 'default' else f['n']}"
            fast = f"{SHORT[f['config']]}{fn} ({dp['tail'] / f['tail']:.2f}x)" if f and dp else "-"
            print(f"| {S} | {K} | {dp['tail']:.1f} | {cell('v16')} | {cell('v20 owner')} | {cell('v20 rs')} | "
                  f"{cell('v22 atomic')} | {cell('v22 atomic_iz')} | "
                  f"{' / '.join('-' if d is None else f'{d['tail']:.1f}' for d in defs)} | {cell('v20 spread')} | "
                  f"{cell('v16 two_tile')} | {cell('v19')} | {fast} |")

    print("\n## The model against the measured tile-aligned tails\n")
    print("Residual = measured tail - model (MAC + synthetic exchange). It holds what the model leaves out: the C "
          "store and the GEMM's own collection code, and the programs' spread of stream-K start times.\n")
    print("| ending | cases | median residual us | 10th-90th percentile us | model's best n = measured best n |")
    print("|---|---|---|---|---|")
    for cfg, p in (("v20 rs", "reduce-scatter"), ("v20 owner", "owner")):
        res, agree, tot = [], 0, 0
        for S in Ss:
            for K in Ks_main:
                rr = [r for r in lim if r["S"] == S and r["K"] == K and r["config"] == cfg and r["n"] != "default"
                      and r["n"] in model.splits(S, K)]
                if not rr:
                    continue
                res += [r["tail"] - model.tail(p, S, r["n"], K)[0] for r in rr]
                mb = min(rr, key=lambda r: model.tail(p, S, r["n"], K)[0])["n"]
                agree += mb == min(rr, key=lambda r: r["tail"])["n"]
                tot += 1
        q = np.percentile(res, [10, 90]) if res else (float("nan"),) * 2
        print(f"| {p} | {len(res)} | {statistics.median(res):.1f} | {q[0]:.1f} to {q[1]:.1f} | {agree} of {tot} |")
    old_path = split_model.SWEEP_C.replace("sweep_c.jsonl", "sweep_c_amdgcnas.jsonl")
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = {(r["S"], r["K"], r["config"], str(r["n"]), r["M"], r.get("limit")): r["us"] for r in map(json.loads, f)}
        print("\n## Against the same sweep with the amdgcnas hook\n")
        print("The first run of sweep C added the repository's amdgcnas post-assembly hook to llirSched; this one "
              "uses llirSched alone, as AITER does (results/sweep_c_amdgcnas.jsonl). Kernel time without the hook / "
              "with it, and the tail difference (without - with), medians over the matching cases.\n")
        print("| config | cases | kernel time ratio (10th-90th percentile) | tail difference us (10th-90th percentile) |")
        print("|---|---|---|---|")
        rows_all = split_model.sweep_rows()
        base_old = {k[1]: v for k, v in old.items() if k[2] == "v13 base" and k[5] == 256}
        for cfg in ["v13", *[c for c in SPLITS if c != "v22 atomic"], "v16 two_tile", "v19"]:
            rat, dif = [], []
            for r in rows_all:
                if r["config"] != cfg or not r.get("limit"):
                    continue
                o = old.get((r["S"], r["K"], cfg, str(r["n"]), r["M"], r.get("limit")))
                if o is None or r["K"] not in base_old or r["K"] not in base:
                    continue
                rat.append(r["us"] / o)
                dif.append((r["us"] - base[r["K"]]) - (o - base_old[r["K"]]))
            if rat:
                q, d = np.percentile(rat, [10, 50, 90]), np.percentile(dif, [10, 50, 90])
                print(f"| {cfg} | {len(rat)} | {q[1]:.3f} ({q[0]:.3f}-{q[2]:.3f}) | {d[1]:+.1f} ({d[0]:+.1f} to {d[2]:+.1f}) |")
        print(f"\nFull wave alone (T = 256), without / with the hook: " + ", ".join(
            f"K = {K}: {base[K]:.1f} / {base_old[K]:.1f} us" for K in sorted(base) if K in base_old))

    at = [r for r in lim if r["config"] == "v22 atomic" and "pre_us" in r]
    if at:
        print("\n## The atomic ending's zeroing launch\n")
        print("v22 zeroes the S stream-K tiles of C in a launch of its own before the GEMM. Its tail is the call: "
              "from that launch's start to the GEMM's end (medians over the dispatches).\n")
        print("| S | zero kernel us | gap and zero kernel (call - GEMM) us, median over K and n | GEMM-only tail at the "
              "best n, K = 8192 | call tail at that n |")
        print("|---|---|---|---|---|")
        for S in sorted({r["S"] for r in at}):
            rr = [r for r in at if r["S"] == S]
            b = best_split([r for r in rr if r["K"] == 8192], S, 8192, "v22 atomic")
            gem = f"{b['gemm_us'] - base[8192]:.1f} (n = {b['n']})" if b else "-"
            print(f"| {S} | {statistics.median(r['pre_us'] for r in rr):.1f} | "
                  f"{statistics.median(r['us'] - r['gemm_us'] for r in rr):.1f} | {gem} | "
                  f"{b['tail']:.1f} |" if b else f"| {S} | - | - | - | - |")

    iz = [r for r in lim if r["config"] == "v22 atomic_iz" and r["K"] in MAIN_K]
    if iz:
        print("\n## Zeroing C inside the GEMM (atomic_iz) against a zeroing launch (atomic)\n")
        print("atomic_iz: each of a tile's n programs zeroes 1/n of it before its K loop and counts itself on the "
              "tile's counter; the adds wait for n. One launch. Tail ratio atomic_iz / atomic at the same S, K "
              "and n, and each at its best n.\n")
        print("| S | ratio at the same n, median (min-max) over K and n | best atomic_iz / best atomic, K = 2048 / "
              "4096 / 8192 / 16384 |")
        print("|---|---|---|")
        for S in sorted({r["S"] for r in iz}):
            rat = [r["tail"] / a["tail"] for r in iz if r["S"] == S
                   and (a := pick(lim, S, r["K"], "v22 atomic", r["n"]))]
            bests = []
            for K in MAIN_K:
                bi, ba = best_split(lim, S, K, "v22 atomic_iz"), best_split(lim, S, K, "v22 atomic")
                bests.append(f"{bi['tail']:.1f} / {ba['tail']:.1f}" if bi and ba else "-")
            print(f"| {S} | {statistics.median(rat):.3f} ({min(rat):.3f}-{max(rat):.3f}) | " + " ; ".join(bests) + " |")

    small = sorted({r["K"] for r in lim if r["K"] < min(MAIN_K)})
    if small:
        print("\n## Small K (K = " + ", ".join(map(str, small)) + ")\n")
        print("Tails us at each ending's best n (n <= K / 128), against data-parallel; the fastest, and its "
              "speed-up.\n")
        print("| S | K | data-parallel | owner | rs | atomic | atomic, zeroed in the GEMM | fastest |")
        print("|---|---|---|---|---|---|---|---|")
        for K in small:
            for S in Ss:
                dp = pick(lim, S, K, "v13")
                if not dp:
                    continue
                cells, cands = [], []
                for c in ("v20 owner", "v20 rs", "v22 atomic", "v22 atomic_iz"):
                    b = best_split(lim, S, K, c)
                    cells.append("-" if not b else f"{b['tail']:.1f} ({'d' if b['n'] == 'default' else b['n']})")
                    if b:
                        cands.append(b)
                f = min(cands, key=lambda r: r["tail"])
                fast = "data-parallel" if f["tail"] >= dp["tail"] else (
                    f"{SHORT[f['config']]} {f['n']} ({dp['tail'] / f['tail']:.2f}x)")
                print(f"| {S} | {K} | {dp['tail']:.1f} | " + " | ".join(cells) + f" | {fast} |")

    packed = [r for r in lim if r["config"].endswith(" 1xcd")]
    if packed:
        print("\n## Packing the active programs onto few XCDs (STREAMK_BALANCE_XCDS=0)\n")
        print("By default the S n active programs are spread evenly over the 8 XCDs, so for S < 8 a tile's n "
              "programs span several XCDs. Packed, the active programs are the first S n stream-K ids: a tile's "
              "programs share an XCD (for n <= 32) and the tail runs on ceil(S n / 32) XCDs. Tails at each "
              "configuration's best n, spread / packed.\n")
        print("| S | K | rs | atomic | atomic, zeroed in the GEMM |")
        print("|---|---|---|---|---|")
        for S in sorted({r["S"] for r in packed}):
            for K in MAIN_K:
                cells = []
                for c in ("v20 rs", "v22 atomic", "v22 atomic_iz"):
                    a = min((r for r in lim if r["S"] == S and r["K"] == K and r["config"] == c
                             and isinstance(r["n"], int) and r["n"] >= 2), key=lambda r: r["tail"], default=None)
                    b = best_split(lim, S, K, c + " 1xcd")
                    cells.append("-" if not (a and b) else
                                 f"{a['tail']:.1f} ({a['n']}) / {b['tail']:.1f} ({b['n']})")
                print(f"| {S} | {K} | " + " | ".join(cells) + " |")

    capped_policy = load_capped_policy()
    print("\n## v20's capped_policy against the measurements\n")
    print("`capped_policy`, v20's default host rule (`STREAMK_POLICY=capped`), replayed on every cell with S <= 128: "
          "its choice and the measured tail of that choice, against the fastest measured configuration that "
          "rounds C once. `~n`: the rule's n was not measured, and the nearest measured n stands in for it. "
          "`all`: the rule's n is at least 256 // S, so it leaves STREAMK_NUM_PROGRAMS unset and the tile gets the "
          "default split. Kernel ratio = (full wave + rule's tail) / (full wave + best tail).\n")
    print("| S | K | rule | rule's tail | best | best tail | tail ratio | kernel ratio |")
    print("|---|---|---|---|---|---|---|---|")
    kr_all, short = [], []
    for K in sorted({r["K"] for r in lim if r["S"] <= 128}):
        for S in Ss:
            dp = pick(lim, S, K, "v13")
            if not dp:
                continue
            pol, fixup, snp = capped_policy(256 + S, K // 64, 256)
            if pol == "dp":
                r, lab = dp, "dp"
            else:
                cfg = "v20 rs" if fixup == "rs" else "v20 owner"
                n = 256 // S if snp is None else snp // S
                if snp is None:
                    r, lab = pick(lim, S, K, cfg, "default"), f"{fixup} all"
                else:
                    r, lab = pick(lim, S, K, cfg, n), f"{fixup} {n}"
                    if r is None:
                        cand = [x for x in lim if x["S"] == S and x["K"] == K and x["config"] == cfg
                                and isinstance(x["n"], int)]
                        r = min(cand, key=lambda x: abs(x["n"] - n))
                        lab += f" ~{r['n']}"
            b = single_rounding_best(lim, S, K)
            if dp["tail"] <= b["tail"]:
                b = dp
            bn = "" if b["n"] == "-" else f" {'d' if b['n'] == 'default' else b['n']}"
            kr = (base[K] + r["tail"]) / (base[K] + b["tail"])
            kr_all.append(kr)
            if kr > 1.02:
                short.append((S, K, lab, kr))
            print(f"| {S} | {K} | {lab} | {r['tail']:.1f} | {SHORT[b['config']]}{bn} | {b['tail']:.1f} | "
                  f"{r['tail'] / b['tail']:.2f} | {kr:.3f} |")
    print(f"\nKernel ratio over {len(kr_all)} cells: median {statistics.median(kr_all):.3f}, 90th percentile "
          f"{np.percentile(kr_all, 90):.3f}, worst {max(kr_all):.3f}. More than 2% short in {len(short)} cells: "
          + "; ".join(f"S = {S}, K = {K} ({lab}, {kr:.3f})" for S, K, lab, kr in short) + ".")

    print("\n## Choosing n from the model\n")
    print("For each ending and (S, K) cell (K = 2048-16384), the n the model (split_model.py) ranks first among the "
          "measured n, against the measured best n; and how far each measured tail is from the model.\n")
    print("| ending | cells | kernel cost of the model's n: median / 90th percentile / worst | "
          "abs(measured - model): median us, median and 90th percentile share of the tail |")
    print("|---|---|---|---|")
    for cfg, p in (("v20 rs", "reduce-scatter"), ("v20 owner", "owner")):
        kc, err, rel = [], [], []
        for S in Ss:
            for K in Ks_main:
                rr = [r for r in lim if r["S"] == S and r["K"] == K and r["config"] == cfg
                      and isinstance(r["n"], int) and r["n"] in model.splits(S, K)]
                if not rr:
                    continue
                mb = min(rr, key=lambda r: model.tail(p, S, r["n"], K)[0])
                b = min(rr, key=lambda r: r["tail"])
                kc.append((base[K] + mb["tail"]) / (base[K] + b["tail"]))
                for r in rr:
                    e = abs(r["tail"] - model.tail(p, S, r["n"], K)[0])
                    err.append(e)
                    rel.append(e / r["tail"])
        print(f"| {p} | {len(kc)} | {statistics.median(kc):.3f} / {np.percentile(kc, 90):.3f} / {max(kc):.3f} | "
              f"{statistics.median(err):.1f} us, {100 * statistics.median(rel):.0f}% / "
              f"{100 * np.percentile(rel, 90):.0f}% |")

    print("\n## Where the default split's programs run\n")
    print("With STREAMK_NUM_PROGRAMS unset, a tile gets min(256 // S, K / 128) programs (at most one per pair of "
          "k-steps), and the work goes to the lowest stream-K ids, 32 consecutive ids per XCD. Where the pair limit "
          "binds, the active programs therefore sit on the first XCDs: the default is the packed placement. Tails at "
          "that n: the default split, the capped split spread evenly over the XCDs (STREAMK_NUM_PROGRAMS = S n), and "
          "packed (the same, with STREAMK_BALANCE_XCDS=0).\n")
    print("| S | K | n per tile | active programs | XCDs used | rs: default / spread / packed | "
          "atomic: default / spread / packed | atomic_iz: default / spread / packed |")
    print("|---|---|---|---|---|---|---|---|")
    for K in Ks_main:
        for S in [s for s in Ss if s <= 8]:
            pairs = K // 128
            if 256 // S <= pairs:
                continue
            n = pairs
            cells = []
            for c in ("v20 rs", "v22 atomic", "v22 atomic_iz"):
                vals = [pick(lim, S, K, c, "default"), pick(lim, S, K, c, n), pick(lim, S, K, c + " 1xcd", n)]
                cells.append(" / ".join("-" if v is None else f"{v['tail']:.1f}" for v in vals))
            print(f"| {S} | {K} | {n} | {S * n} | {-(-S * n // 32)} | " + " | ".join(cells) + " |")

    print("\n## Where the tile-aligned tail goes, and what the old skew term was\n")
    print("Stamped phases from v20's fixup_timing.py --tile-limit on the same matrix (medians; the stamps add a barrier "
          "and a drain each, so they run a little long), against the kernel-time tail of the same case. 'stamped tail' "
          "runs from the first program's stream-K start to the last done, as the old split model's tail did; "
          "'start spread' is the last start minus the first. The kernel-time tail is shorter than MAC + ending "
          "whenever the split programs start before the full wave's last tile ends: they overlap it.\n")
    print("| K | S | n | ending | MAC | ending | MAC + ending | stamped tail | start spread | kernel-time tail | "
          "stamped tail - kernel-time tail |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    with open(split_model.MAC_ROWS) as f:
        stamped = [json.loads(line) for line in f]
    diffs = []
    for s in stamped:
        r = pick(lim, s["S"], s["K"], "v20 rs" if s["fixup"] == "rs" else "v20 owner", s["n"])
        if not r:
            continue
        diffs.append(s["tail"] - r["tail"])
        if s["S"] in (1, 8, 32, 128) and s["n"] in (2, 8, 32):
            print(f"| {s['K']} | {s['S']} | {s['n']} | {s['fixup']} | {s['mac']:.1f} | {s['ending']:.1f} | "
                  f"{s['mac'] + s['ending']:.1f} | {s['tail']:.1f} | {s['start_spread']:.1f} | {r['tail']:.1f} | "
                  f"{s['tail'] - r['tail']:.1f} |")
    if diffs:
        q = np.percentile(diffs, [10, 50, 90])
        print(f"\nStamped tail minus kernel-time tail over all {len(diffs)} stamped cases: median {q[1]:.1f} us, "
              f"10th-90th percentile {q[0]:.1f} to {q[2]:.1f} us.")
    l2_path = os.path.join(os.path.dirname(split_model.SWEEP_C), "l2_c.jsonl")
    if os.path.exists(l2_path):
        with open(l2_path) as f:
            l2 = [json.loads(line) for line in f]
        base_l2 = next((r for r in l2 if r["config"] == "v13 base"), None)
        print("\n## L2 hit rate and memory reads (K = 8192, warm, rocprof_check.l2)\n")
        print("TCC hit rate and TCC_EA0_RDREQ (memory read requests) per launch, medians of launches 3-10 on the "
              "same inputs. Extra reads: against the full wave alone (T = 256), so they are the tail's own.\n")
        print("| S | config | n | L2 hit rate | memory reads (M) | extra reads over the full wave (M) |")
        print("|---|---|---|---|---|---|")
        for r in l2:
            if r["config"] == "v13 base":
                continue
            extra = (r["reads"] - base_l2["reads"]) / 1e6 if base_l2 else float("nan")
            print(f"| {r['S']} | {r['config']} | {r['n']} | {100 * r['hit']:.1f}% | {r['reads'] / 1e6:.2f} | "
                  f"{extra:.2f} |")
        if base_l2:
            print(f"\nFull wave alone: {100 * base_l2['hit']:.1f}% hits, {base_l2['reads'] / 1e6:.2f} M reads.")
    for o in outs:
        if o:
            print(f"\n{o}", flush=True)


if __name__ == "__main__":
    main()
