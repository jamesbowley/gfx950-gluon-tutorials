"""Summaries of timing_reduce_variants.py's graph timings (WORKLOG entries 15, 16, 18).

    python v30_streamk_streamlined/reduce_variants_summary.py \
        > v30_streamk_streamlined/results/reduce_variants_summary.md
    python v30_streamk_streamlined/reduce_variants_summary.py --set e16 \
        > v30_streamk_streamlined/results/reduce_variants_rs_all_summary.md
    python v30_streamk_streamlined/reduce_variants_summary.py --set e18 \
        > v30_streamk_streamlined/results/rs_all_production_summary.md

--set e18 compares production rs (rs_all since entry 18) with the copied old rs,
the experiment's rs all and sep, and production's best of rs and owner with the
old one's.

1. Per point (fixed split): sep, the best tree variant and rs all (plain rs where
   it has no extra readers) against the better of production rs and owner, in both
   runs; rs all against rs and sep; and rs all packed against rs all.
2. Per shape: each method at its own best split against production at its best
   split (rs or owner), worse and better of the two runs.
"""

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "reduce_variants_raw")
SETS = {
    "e15": [
        ("focused", [RAW + "/run1/focused_*.jsonl", RAW + "/focused_*.jsonl"]),
        ("wide", [RAW + "/wide_a_*.jsonl", RAW + "/wide_b_*.jsonl"]),
    ],
    "e16": [
        ("focused", [RAW + "/e16/focused_a_*.jsonl", RAW + "/e16/focused_b_*.jsonl"]),
        ("wide", [RAW + "/e16/wide_a_*.jsonl", RAW + "/e16/wide_b_*.jsonl"]),
    ],
    "e18": [
        ("focused", [RAW + "/e18/focused_a_*.jsonl", RAW + "/e18/focused_b_*.jsonl"]),
        ("wide", [RAW + "/e18/wide_a_*.jsonl", RAW + "/e18/wide_b_*.jsonl"]),
    ],
}


def load(pattern):
    d = {}
    for path in glob.glob(pattern):
        for line in open(path):
            r = json.loads(line)
            m = {k.split("|")[1]: v for k, v in r["median"].items() if k.startswith("graph")}
            m["_SN"] = r["S"] * r["n"]
            d[(r["label"], r["tile"], tuple(r["shape"]), r["n"])] = m
    return d


def prod(m):
    return min(m["prod rs"], m["prod owner"])


def tree(m):
    return min(v for x, v in m.items() if x.startswith("tree"))


def rs_all(m):
    return m.get("rs all", m["rs"])


def per_point(name, runs, methods):
    A = runs[0]
    print(f"## {name}, fixed split: {len(A)} points\n")
    for var, f, base_name, base in methods:
        rows = sorted(
            (max(r), min(r), k)
            for k in A
            for r in [[f(D[k]) / base(D[k]) for D in runs]]
        )
        better = [r for r in rows if r[0] < 0.99]
        worse = [r for r in rows if r[1] > 1.01]
        geo = 1.0
        for r in rows:
            geo *= r[0]
        print(
            f"- {var} / {base_name}: more than 1% faster in both runs at {len(better)} "
            f"points, more than 1% slower at {len(worse)}; best "
            f"{rows[0][1]:.3f}-{rows[0][0]:.3f}, worst {rows[-1][1]:.3f}-{rows[-1][0]:.3f}, "
            f"geomean (worse run) {geo ** (1 / len(rows)):.3f}"
        )
        for hi, lo, k in better + worse:
            print(
                f"  - {k[1]} {'x'.join(map(str, k[2]))} n={k[3]} "
                f"(S n = {A[k]['_SN']}): {lo:.3f}-{hi:.3f}"
            )
    print()


def per_shape(name, runs, methods):
    print(f"## {name}, best split per shape (worse-better run)\n")
    head = "| tile | shape | best prod n | prod us |"
    for var, _ in methods:
        head += f" best {var} n | {var} / prod |"
    print(head)
    print("|---|---|---|---|" + "---|---|" * len(methods))
    shapes = sorted({(k[1], k[2]) for k in runs[0]})
    for sh in shapes:
        ratios = {var: [] for var, _ in methods}
        best_n = {}
        for D in runs:
            ks = [k for k in D if (k[1], k[2]) == sh]
            bp = min((prod(D[k]), k[3]) for k in ks)
            for var, f in methods:
                bv = min((f(D[k]), k[3]) for k in ks)
                ratios[var].append(bv[0] / bp[0])
                best_n[var] = bv[1]
        line = f"| {sh[0]} | {'x'.join(map(str, sh[1]))} | {bp[1]} | {bp[0]:.1f} |"
        for var, _ in methods:
            line += f" {best_n[var]} | {min(ratios[var]):.3f}-{max(ratios[var]):.3f} |"
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="e15", choices=sorted(SETS))
    args = ap.parse_args()
    e16 = args.set == "e16"
    if args.set == "e18":
        # Production rs is rs_all; "rs" is the copied pre-entry-18 rs.
        def old_prod(m):
            return min(m["rs"], m["prod owner"])

        point_methods = [
            ("prod rs", lambda m: m["prod rs"], "rs (old)", lambda m: m["rs"]),
            ("prod rs", lambda m: m["prod rs"], "rs all (experiment)", rs_all),
            ("prod rs", lambda m: m["prod rs"], "sep", lambda m: m["sep"]),
            ("prod best", prod, "old prod best", old_prod),
        ]
        print("# rs_all in production: summary (WORKLOG entry 18)\n")
        for name, patterns in SETS[args.set]:
            runs = [load(p) for p in patterns]
            per_point(name, runs, point_methods)
            per_shape(
                name,
                runs,
                [("old prod", old_prod), ("sep", lambda m: m["sep"])],
            )
        return
    point_methods = [
        ("sep", lambda m: m["sep"], "prod", prod),
        ("best tree", tree, "prod", prod),
    ]
    shape_methods = [("sep", lambda m: m["sep"]), ("tree", tree)]
    if e16:
        point_methods = [
            ("rs all", rs_all, "prod", prod),
            ("rs all", rs_all, "rs", lambda m: m["rs"]),
            ("rs all", rs_all, "sep", lambda m: m["sep"]),
            ("rs all packed", lambda m: m.get("rs all packed", m["rs"]), "rs all", rs_all),
        ] + point_methods
        shape_methods = [("rs all", rs_all)] + shape_methods
    print(f"# Reduce variants: summary ({'WORKLOG entry 16' if e16 else 'WORKLOG entry 15'})\n")
    for name, patterns in SETS[args.set]:
        runs = [load(p) for p in patterns]
        per_point(name, runs, point_methods)
        per_shape(name, runs, shape_methods)


if __name__ == "__main__":
    main()
