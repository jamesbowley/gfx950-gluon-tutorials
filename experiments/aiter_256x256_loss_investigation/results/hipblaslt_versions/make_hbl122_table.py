"""Write results/aiter_live_compare_hbl122.md from aiter_live_compare_hbl122.csv and the kernels
AITER launched (aiter_pick_kernels_hbl122.log). Replaces run_aiter_compare.py's generic summary.

    python make_hbl122_table.py
"""

import csv
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
LIVE = os.path.join(RESULTS, "aiter_live_compare_hbl122.csv")
PREV = os.path.join(RESULTS, "aiter_live_compare.csv")
KERNELS = os.path.join(HERE, "aiter_pick_kernels_hbl122.log")
OUT = os.path.join(RESULTS, "aiter_live_compare_hbl122.md")
GLUON = ["v9", "v10", "v11", "v12", "v13"]
BLOCK_K = 64


def key(r):
    return (int(r["M"]), int(r["N"]), int(r["K"]))


def load(path):
    with open(path, newline="") as f:
        return {key(r): r for r in csv.DictReader(f)}


def pick_kernels():
    """(M, N, K) -> short kernel label, from the timed call of each shape."""
    labels, cur, timed = {}, None, False
    for line in open(KERNELS):
        if line.startswith("@@WARMUP"):
            timed = False
        elif line.startswith("@@SHAPE"):
            _, M, N, K, lib, name = line.split()
            cur, timed = (int(M), int(N), int(K)), True
            if lib == "asm":
                tile = re.search(r"_(\d+x\d+)E$", name).group(1)
                labels[cur] = f"asm {tile}x{BLOCK_K}"
        elif timed and "MT" in line:
            mt = re.search(r"MT(\d+x\d+x\d+)", line).group(1)
            sk = re.search(r"_SK(\d)", line).group(1)
            labels[cur] = f"hipBLASLt {mt} SK{sk}"
    return labels


def pct(a, b):
    return 100 * (a / b - 1)


def stats(xs):
    return (st.median(xs),)


def main():
    live, prev, kern = load(LIVE), load(PREV), pick_kernels()
    rows = sorted(live.values(), key=lambda r: (int(r["tiles"]), int(r["K"])))

    for r in rows:
        tf = {v: float(r[f"{v}_tflops"]) for v in GLUON}
        r["_best"] = max(tf, key=tf.get)
        r["_pick"] = float(r["pick_tflops"])
        r["_vs_v9"] = pct(r["_pick"], tf["v9"])
        r["_vs_best"] = pct(r["_pick"], tf[r["_best"]])
        r["_ok"] = all(r[f"{v}_ok"] == "True" for v in ["pick", *GLUON])

    tiles_ok = all(re.search(r"256x256x64", kern[key(r)]) for r in rows)
    n_hbl = sum(kern[key(r)].startswith("hipBLASLt") for r in rows)
    n_asm = len(rows) - n_hbl
    rerun = [
        pct(float(live[k][f"{v}_tflops"]), float(prev[k][f"{v}_tflops"]))
        for k in live
        for v in GLUON
    ]
    rerun += [
        pct(float(live[k]["pick_tflops"]), float(prev[k]["pick_tflops"]))
        for k in live
        if live[k]["live_libtype"] == "asm"
    ]
    vs_v9 = stats(r["_vs_v9"] for r in rows)
    vs_best = stats(r["_vs_best"] for r in rows)
    aiter_wins_v9 = sum(r["_vs_v9"] > 0 for r in rows)
    aiter_wins_best = sorted(
        (r for r in rows if r["_vs_best"] > 0), key=lambda r: -r["_vs_best"]
    )
    few_tiles = [r for r in aiter_wins_best if int(r["tiles"]) < 256]
    close = [r for r in aiter_wins_best if int(r["tiles"]) >= 256]
    wins = {}
    for r in rows:
        wins[r["_best"]] = wins.get(r["_best"], 0) + 1

    def shapes(rs):
        return ", ".join(f"{r['M']}x{r['N']}x{r['K']} ({r['_vs_best']:+.1f}%)" for r in rs)

    L = []
    L += [
        "# Gluon v9–v13 vs AITER on the 256x256 loss shapes",
        "",
        f"{len(rows)} of the 38 shapes from the 2026-09-28 status update for "
        "[ROCm/aiter#5897](https://github.com/ROCm/aiter/pull/5897) where AITER's pick beat "
        "Gluon v9 with the same tile, re-measured on hipBLASLt 1.2.2 (ROCm 7.2.4).",
        "",
        (
            f"- **Same tile:** every AITER pick uses 256x256x64: hipBLASLt `MT256x256x64 SK3` "
            f"(stream-K) on {n_hbl} shapes, AITER's asm `bf16gemm_bf16_tn_256x256` on {n_asm}."
            if tiles_ok
            else "- **Not every AITER pick uses 256x256x64; see the AITER kernel column.**"
        ),
        f"- **AITER vs v9:** median {vs_v9[0]:+.1f}%; AITER is faster on {aiter_wins_v9} of "
        f"{len(rows)}.",
        f"- **AITER vs the best Gluon version:** median {vs_best[0]:+.1f}%; AITER is faster on "
        f"{len(aiter_wins_best)} of {len(rows)}. "
        + (
            f"{len(few_tiles)} have fewer tiles than CUs, where stream-K spreads K over the "
            f"idle CUs: {shapes(few_tiles)}. "
            if few_tiles
            else ""
        )
        + (
            f"The other {len(close)} are within "
            f"{max(r['_vs_best'] for r in close):.1f}%."
            if close
            else ""
        ),
        "- **Best Gluon version:** "
        + ", ".join(f"{v} on {n}" for v, n in sorted(wins.items(), key=lambda t: -t[1]))
        + ".",
        f"- **Noise:** the same kernels re-measured two days apart moved by at most "
        f"{max(abs(x) for x in rerun):.1f}%, so treat gaps under 1% as ties.",
        "",
        "**Method:** one MI355X; per shape one process with the same inputs for every kernel "
        "(no bias); `do_bench_cudagraph`, 3 interleaved rounds, median; every output "
        + ("matched" if all(r["_ok"] for r in rows) else "except those marked WRONG matched")
        + " an fp32 reference. torch 2.13 with ROCm 7.2.4's hipBLASLt, rocBLAS and hipBLAS; "
        "Triton `gfx950-tutorial-v2.2` with llirSched. AITER's kernels were logged from its own "
        "`tuned_gemm.gemm_a16w16` calls.",
        "",
        "**Columns:** TFLOPS per kernel. Waves = tiles / 256 CUs; K steps = K / 64. "
        "Diffs are AITER / Gluon − 1, so positive means AITER is faster.",
        "",
        "| Shape (MxNxK) | Tiles | Waves | K steps | AITER kernel | AITER | v9 | v10 | v11 | v12 "
        "| v13 | Best Gluon | AITER vs v9 | AITER vs best |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        shape = f"{r['M']}x{r['N']}x{r['K']}" + ("" if r["out"] == "bfloat16" else " (fp32 out)")
        L.append(
            f"| {shape} | {r['tiles']} | {float(r['waves']):.2f} | {int(r['K']) // BLOCK_K} "
            f"| {kern[key(r)]} | {r['_pick']:.0f} | "
            + " | ".join(f"{float(r[f'{v}_tflops']):.0f}" for v in GLUON)
            + f" | {r['_best']} | {r['_vs_v9']:+.1f}% | {r['_vs_best']:+.1f}% |"
            + ("" if r["_ok"] else " WRONG")
        )

    L += [
        "",
        "| Bucket | Shapes | AITER vs v9 (median) | AITER vs best (median) | Best Gluon faster |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in ["few tiles (< 1 wave)", "partial waves", "full waves"]:
        xs = [r for r in rows if r["bucket"] == b]
        L.append(
            f"| {b} | {len(xs)} | {st.median(r['_vs_v9'] for r in xs):+.1f}% "
            f"| {st.median(r['_vs_best'] for r in xs):+.1f}% "
            f"| {sum(r['_vs_best'] <= 0 for r in xs)} |"
        )

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
