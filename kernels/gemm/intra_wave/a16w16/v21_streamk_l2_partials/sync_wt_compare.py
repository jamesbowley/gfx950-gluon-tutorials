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
"""Before/after tables for the switch to write-through partials with relaxed flags (2026-10-08).

    cd kernels/gemm/intra_wave/a16w16
    python v21_streamk_l2_partials/sync_wt_compare.py            # print
    python v21_streamk_l2_partials/sync_wt_compare.py --write    # <version>/results/sync_wt.md

Reads the raw tables in results/sync_wt/: before, after (regress.py --streamk, rocprof_check.py
--no-l2, sync_asm_check.py, fixup_timing.py), before2 and after2 (a second interleaved round of
rocprof_check.py and regress.py on two shapes), and checks/ (the correctness runs). Every run used
GPU 0, Triton v2.2 + deepbind shim and the llirSched plugin, with TRITON_AMDGCNAS_PLUGIN unset
(results/sync_wt/env.sh).
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
RAW = os.path.join(HERE, "results", "sync_wt")
LINK = "../../v21_streamk_l2_partials/results/sync_wt"
DIRS = {15: "v15_streamk_onetile", 16: "v16_streamk_lane_partials", 17: "v17_streamk_tile_aligned",
        18: "v18_streamk_chunk_major", 19: "v19_streamk_two_tile_reversed", 20: "v20_streamk_reduce_scatter",
        21: "v21_streamk_l2_partials", 22: "v22_streamk_atomic_c"}

OWNER_CHECK = ("- `check_streamk.py`: 36 configurations (the 9 stream-K shapes, both tile orders, with and without "
               "bias, 30 rotating launches each), no wrong result, every flag re-armed.\n"
               "- `experiments/streamk_split_count/check_all.py` (`STREAMK_NUM_PROGRAMS` = S, 2 S and all): "
               "24 configurations, all pass.")
CORRECTNESS = {
    **{v: OWNER_CHECK for v in (15, 16, 17, 18, 19)},
    20: ("- `check_streamk.py`: 108 configurations (one-tile and two-tile, both endings), and 36 with "
         "`--policies capped`, no wrong result, every flag and barrier count re-armed.\n"
         "- `check_all.py`: 138 configurations (including spread, `STREAMK_BALANCE_XCDS=0` and small K), all pass.\n"
         "- Negative control, `--stale-count` (tile 0's barrier count preset to n - 1): all 24 one-tile "
         "reduce-scatter configurations hang, as they must. So the stress test still catches a broken "
         "barrier with relaxed atomics. (The owner and two-tile rows do not use the preset count; the "
         "run was stopped after 72 configurations.)"),
    21: ("- `check_streamk.py` (capped rule, reduce-scatter, all three `STREAMK_PARTIALS` modes): 66 "
         "configurations pass.\n"
         "- `--partials wt --policies one_tile two_tile --fixups rs owner --orders v9 split`: 132 "
         "configurations pass."),
    22: ("- `check_streamk.py`, the reduce-scatter and owner endings: 108 configurations (one-tile and "
         "two-tile) and 36 with `--policies capped`, no wrong result, every flag and count re-armed.\n"
         "- The atomic ending's own check, `experiments/streamk_split_count/check_tile_limit.py --versions 22` "
         "(stream-K tiles held to the mean-error bound of n bf16 roundings): 1630 of 1630 cases pass "
         "(`atomic` and `atomic_iz`, spread and on one XCD).\n"
         "- `check_streamk.py`'s plain `allclose` rejects the atomic ending on most shapes, and did so "
         "before this change too: the old code gave 30, 10 and 0 wrong launches of 30 on 4352x4096x8192, "
         f"4096x6144x8192 and 8192x7936x8192 ([checks/v22_atomic_BASELINE_code.md]({LINK}/checks/"
         "v22_atomic_BASELINE_code.md)). Its code is unchanged."),
}


def rows(path):
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path):
        if ln.startswith("| ") and not re.match(r"\| (shape|spec|config)", ln) and "---" not in ln:
            out.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return out


def num(s):
    return float(s.split()[0])


def regress(tag):
    return {(r[0], r[1]): (float(r[3]), r[5], r[7], r[8], r[2])
            for r in rows(os.path.join(RAW, tag, "regress_streamk.md")) if len(r) >= 10 and r[2] in ("yes", "NO")}


def rocprof(tag):
    return {(r[0], r[1]): float(r[2]) for r in rows(os.path.join(RAW, tag, "rocprof.md"))}


def asm(tag):
    return {r[0]: r for r in rows(os.path.join(RAW, tag, "asm.md"))}


def pct(a, b):
    return f"{100 * (b / a - 1):+.1f}%"


def table_regress(v, before, after):
    lines = ["| shape | before TFLOPS | after TFLOPS | change | VGPRs / VGPR spills / SGPR spills, before | after |",
             "|---|---|---|---|---|---|"]
    for (shape, ver), b in before.items():
        if ver == f"v{v}" and (shape, ver) in after:
            a = after[(shape, ver)]
            lines.append(f"| {shape} | {b[0]:.1f} | {a[0]:.1f} | {pct(b[0], a[0])} | {b[1]} / {b[2]} / {b[3]} | "
                         f"{a[1]} / {a[2]} / {a[3]} |")
    return lines


def table_repeat(v, before, after):
    lines = ["| shape | before TFLOPS | after TFLOPS | change |", "|---|---|---|---|"]
    for (shape, ver), b in before.items():
        if ver == f"v{v}" and (shape, ver) in after:
            a = after[(shape, ver)]
            lines.append(f"| {shape} | {b[0]:.1f} | {a[0]:.1f} | {pct(b[0], a[0])} |")
    return lines


def table_rocprof(v, runs):
    lines = ["| shape | before, rounds 1 / 2 (us) | after, rounds 1 / 2 (us) | change of the means |",
             "|---|---|---|---|"]
    label_b = "v21 partials=mem" if v == 21 else f"v{v}"
    label_a = "v21 partials=wt" if v == 21 else f"v{v}"
    for (shape, lab) in runs[0][0]:
        if lab != label_b:
            continue
        bs = [b[(shape, label_b)] for b, _ in runs if (shape, label_b) in b]
        as_ = [a[(shape, label_a)] for _, a in runs if (shape, label_a) in a]
        if not as_:
            continue
        mb, ma = sum(bs) / len(bs), sum(as_) / len(as_)
        extra = ""
        if v == 21:
            ms = [a[(shape, "v21 partials=mem")] for _, a in runs if (shape, "v21 partials=mem") in a]
            extra = f" (`mem` in the same runs: {' / '.join(f'{x:.1f}' for x in ms)})"
        lines.append(f"| {shape} | {' / '.join(f'{x:.1f}' for x in bs)} | "
                     f"{' / '.join(f'{x:.1f}' for x in as_)}{extra} | {ma - mb:+.1f} us ({pct(mb, ma)}) |")
    return lines


def table_asm(v, before, after):
    lines = ["| build | `buffer_wbl2` before | after | `buffer_inv` before | after |", "|---|---|---|---|---|"]
    for spec, b in before.items():
        if spec.startswith(f"v{v}") and spec in after:
            a = after[spec]
            lines.append(f"| {spec} | {b[1]} | {a[1]} | {b[2]} | {a[2]} |")
    return lines


def table_fixup_v20():
    b = {(r[0], r[3]): r for r in rows(os.path.join(RAW, "before", "fixup_timing_v20.md"))}
    a = {(r[0], r[3]): r for r in rows(os.path.join(RAW, "after", "fixup_timing_v20.md"))}
    lines = ["| shape | S | n | ending | flag or barrier wait, before (us) | after | tail, before (us) | after |",
             "|---|---|---|---|---|---|---|---|"]
    for k, rb in b.items():
        if k in a:
            ra = a[k]
            lines.append(f"| {rb[0]} | {rb[1]} | {rb[2]} | {rb[3]} | {num(rb[6]):.1f} | {num(ra[6]):.1f} | "
                         f"{num(rb[8]):.1f} | {num(ra[8]):.1f} |")
    return lines


def table_fixup_v21():
    r = rows(os.path.join(RAW, "after", "fixup_timing_v21.md"))
    by = {}
    for x in r:
        by.setdefault((x[0], x[1], x[2]), {})[x[3]] = x
    lines = ["| shape | S | n | barrier, `mem` / `wt` (us) | ending, `mem` / `wt` (us) | kernel with clean caches, "
             "`mem` / `wt` (us) |", "|---|---|---|---|---|---|"]
    for (shape, S, n), m in by.items():
        if "mem" in m and "wt" in m:
            M, W = m["mem"], m["wt"]
            lines.append(f"| {shape} | {S} | {n} | {num(M[7]):.1f} / {num(W[7]):.1f} | {num(M[9]):.1f} / "
                         f"{num(W[9]):.1f} | {num(M[12]):.1f} / {num(W[12]):.1f} |")
    return lines


def render(v, d, data):
    rb, ra, rb2, ra2, runs, ab, aa = data
    before = ("`STREAMK_PARTIALS=mem`, the default until 2026-10-08: write-through partials, the compiler's "
              "GPU-scope release and acquire on the barrier. After: `wt`, now the default." if v == 21 else
              "Before: write-through (`.wt`) partials, then the compiler's GPU-scope release on the flag or "
              "barrier arrival and an acquire after the wait (`buffer_wbl2 sc1`, `buffer_inv sc1`).")
    out = [
        f"# {d}: write-through partials with relaxed flags (2026-10-08)",
        "",
        before,
        "After: the same `.wt` partial stores and drain in every wave, then relaxed `gpu`-scope flags and "
        "counters, and `.cv` partial reads behind a compiler fence "
        "([memory_model_litmus WRITEUP](../../../../../../experiments/memory_model_litmus/WRITEUP.md), "
        "recommendation 1).",
        "",
        "GPU 0, Triton v2.2 + deepbind shim, llirSched plugin, **without the amdgcnas hook** "
        "(`TRITON_AMDGCNAS_PLUGIN` unset). Old and new code were measured on the same day with the same "
        f"commands, in two interleaved rounds. Raw tables: [results/sync_wt]({LINK}/); "
        "generated by `v21_streamk_l2_partials/sync_wt_compare.py`. Numbers in earlier WORKLOG entries were "
        "taken on another GPU with the amdgcnas hook, so compare only within this file.",
        "",
        "## Correctness",
        "",
        CORRECTNESS[v],
        "",
        "## Cache maintenance in the compiled kernel (`v21_streamk_l2_partials/sync_asm_check.py`, "
        "4352x4096x8192)",
        "",
        *table_asm(v, ab, aa),
        "",
        "## Cold kernel time, two interleaved rounds",
        "",
        "`v21_streamk_l2_partials/rocprof_check.py --no-l2`: 1000 launches over 512 MB of rotating inputs, "
        "median dispatch time. Repeats agree within about 1 us.",
        "",
        *table_rocprof(v, runs),
        "",
        "## do_bench (`regress.py --streamk`), bf16, default policy",
        "",
        "do_bench moves by up to 8% between runs of unchanged code at K = 16384; the second round "
        "repeats the two shapes that moved most.",
        "",
        *table_regress(v, rb, ra),
        "",
        "Second round:",
        "",
        *table_repeat(v, rb2, ra2),
        "",
    ]
    if v in (20, 22):
        out += [
            "## The ending, phase by phase (`v20_streamk_reduce_scatter/fixup_timing.py`)",
            "",
            "v20's endings, with `s_memrealtime` stamps; medians over the split programs (reduce-scatter) "
            "or the owners. v22's reduce-scatter and owner endings are the same code." if v == 22 else
            "Medians over the split programs (reduce-scatter) or the owners, `s_memrealtime` stamps. The "
            "wait is the barrier (reduce-scatter) or the owner's wait for its last peer flag.",
            "",
            *table_fixup_v20(),
            "",
        ]
    if v == 21:
        out += [
            "## The reduce-scatter ending, `mem` against `wt` in one run (`fixup_timing.py --modes mem wt`)",
            "",
            *table_fixup_v21(),
            "",
        ]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    data = (regress("before"), regress("after"), regress("before2"), regress("after2"),
            [(rocprof("before"), rocprof("after")), (rocprof("before2"), rocprof("after2"))],
            asm("before"), asm("after"))
    for v, d in DIRS.items():
        text = render(v, d, data)
        if args.write:
            with open(os.path.join(A16, d, "results", "sync_wt.md"), "w") as f:
                f.write(text)
            print(f"wrote {d}/results/sync_wt.md")
        else:
            print(text + "\n")


if __name__ == "__main__":
    main()
