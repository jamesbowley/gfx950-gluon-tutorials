"""Fit the v31 default Stream-K rule to a sweep and score it.

    python experiments/streamk_v31_rule/fit.py results/grid_r1.csv [results/grid_r2.csv ...] \\
        --out experiments/streamk_v31_rule/results/fit.md --rule-out experiments/streamk_v31_rule/results/rule.json

Each (shape, tile) block's times are the median over the rounds given. The rule
(rule()) has a few constants per regime (below one wave: T < P; else one wave or
more); the fit is a grid search minimising the geometric-mean regret (time of the
rule's choice over the block's best single-rounding time) plus the mean excess of
the rule's time over dp's beyond the noise (LOSS_WEIGHT). Cross-validation
leaves out one k (K-steps per tile) bin, then one S bin, refitting each time.
"""

import argparse
import collections
import csv
import json
import math
import random
import statistics
import sys

P = 256
OWNER_N = (2, 3, 4, 6, 8, 12, 16)
RS_N = (2, 4, 6, 8, 12, 16, 32, 64)
NOISE = 1.01
LOSS_WEIGHT = float(__import__("os").environ.get("LOSS_WEIGHT", "1.0"))
RESTARTS = 40
SEED = 3102

# Search space of each regime's constants (rule()).
SPACE = {
    "k_min": (4, 8, 12, 16, 20, 24, 32),
    "owner_max_per_tile": (1, 2, 3, 4),
    "owner_c": (1, 2, 3, 4, 6, 8, 12, 16),
    "owner_k_min": (4, 8, 12, 16, 20, 24, 32, 48, 64),
    "rs_b": (4, 6, 8, 12, 16),
    "rs_gain_min": (0, 2, 3, 4, 5, 6, 7, 8, 10),
}
BASE = dict(k_min=8, owner_max_per_tile=3, owner_c=4, owner_k_min=16, rs_b=8, rs_gain_min=5)


def load(paths):
    times = collections.defaultdict(list)
    meta = {}
    bad = []
    for path in paths:
        for r in csv.DictReader(open(path)):
            if r["error"] or r["correct"] != "True":
                bad.append(r["point"])
                continue
            times[r["point"]].append(float(r["us_median"]))
            meta[r["point"]] = r
    blocks = collections.defaultdict(dict)
    for p, ts in times.items():
        r = meta[p]
        key = (r["tile"], int(r["M"]), int(r["N"]), int(r["K"]))
        blocks[key][r["kind"]] = statistics.median(ts)
        blocks[key]["_info"] = dict(T=int(r["T"]), S=int(r["S"]), W=int(r["W"]), k=int(r["k"]),
                                    pairs=int(r["pairs"]))
    return blocks, bad


def snap(n, allowed, cap):
    ok = [a for a in allowed if a <= cap]
    if not ok:
        return None
    return min(ok, key=lambda a: (abs(math.log(a / n)), a))


def rule(info, c):
    """The kind the rule picks for a block; c maps regime ("below": T < P, "after")
    to constants. S tail tiles, k K-steps per tile (pairs = k / 2), P // S programs
    per tail tile, cap = min(P // S, pairs):
    - dp when S = 0, cap < 2 or k < k_min;
    - the owner ending when P // S <= owner_max_per_tile, on
      n = sqrt(pairs / owner_c) programs (its n - 1 serial partial reads against
      pairs / n of MAC), if k >= owner_k_min (else dp);
    - else reduce-scatter on n = sqrt(pairs (P // S) / rs_b) programs: the n that
      minimises pairs / n (each program's MAC) + n S / (rs_b P) (the partials all
      S n programs store and read at once), if the split saves at least rs_gain_min
      pairs of the tail's time, pairs (1 - 1 / n) (else dp)."""
    T, S, k, pairs = info["T"], info["S"], info["k"], info["pairs"]
    if S == 0:
        return "dp"
    per_tile = P // S
    cap = min(per_tile, pairs)
    cc = c["below" if T < P else "after"]
    if cap < 2 or k < cc["k_min"]:
        return "dp"
    if per_tile <= cc["owner_max_per_tile"]:
        if k < cc["owner_k_min"]:
            return "dp"
        n = snap(max(2, math.sqrt(pairs / cc["owner_c"])), OWNER_N, cap)
        return f"ow{n}" if n else "dp"
    n = snap(max(2, math.sqrt(pairs * per_tile / cc["rs_b"])), RS_N, cap)
    if not n or pairs * (1 - 1 / n) < cc["rs_gain_min"]:
        return "dp"
    return f"rs{n}"


def best_kind(b):
    cands = {k: v for k, v in b.items() if not k.startswith("_") and k not in ("pdp",)}
    return min(cands, key=cands.get)


def score(blocks, c, keys=None):
    """(geomean regret, worst regret, blocks slower than dp beyond noise, missing)."""
    logs, worst, lose, missing = [], 1.0, 0, 0
    for key in keys or blocks:
        b = blocks[key]
        kind = rule(b["_info"], c)
        if kind not in b:
            missing += 1
            continue
        best = b[best_kind(b)]
        reg = b[kind] / best
        logs.append(math.log(reg))
        worst = max(worst, reg)
        if "dp" in b and b[kind] > NOISE * b["dp"]:
            lose += 1
    gm = math.exp(sum(logs) / len(logs)) if logs else float("nan")
    return gm, worst, lose, missing


def objective(blocks, c, keys):
    """Mean log regret plus LOSS_WEIGHT times the mean excess of the rule's time over
    dp's beyond NOISE (log), so a fallback that loses to dp costs twice."""
    total, n = 0.0, 0
    for key in keys:
        b = blocks[key]
        kind = rule(b["_info"], c)
        if kind not in b:
            total += 0.1
            n += 1
            continue
        t = b[kind]
        total += math.log(t / b[best_kind(b)])
        total += LOSS_WEIGHT * max(0.0, math.log(t / (NOISE * b["dp"])))
        n += 1
    return total / max(n, 1)


def descend(blocks, regime, base, keys, start, passes=6):
    cc = dict(start)
    best = objective(blocks, dict(base, **{regime: cc}), keys)
    for _ in range(passes):
        changed = False
        for name, values in SPACE.items():
            for v in values:
                trial = dict(cc, **{name: v})
                o = objective(blocks, dict(base, **{regime: trial}), keys)
                if o < best - 1e-12:
                    best, cc, changed = o, trial, True
        if not changed:
            break
    return best, cc


def fit_regime(blocks, regime, base, keys, restarts=RESTARTS):
    """Coordinate descent over SPACE from BASE and `restarts` seeded random starts."""
    keys = [k for k in keys if (blocks[k]["_info"]["T"] < P) == (regime == "below")]
    rng = random.Random(SEED)
    starts = [dict(BASE)] + [{n: rng.choice(v) for n, v in SPACE.items()} for _ in range(restarts)]
    return min((descend(blocks, regime, base, keys, st) for st in starts), key=lambda r: r[0])[1]


def fit(blocks, keys):
    base = {r: dict(BASE) for r in ("below", "after")}
    for regime in ("below", "after"):
        base[regime] = fit_regime(blocks, regime, base, keys)
    return base


def cross_validate(blocks, by):
    groups = collections.defaultdict(list)
    for key in blocks:
        groups[blocks[key]["_info"][by]].append(key)
    regrets = []
    for g, held in sorted(groups.items()):
        train = [k for k in blocks if k not in set(held)]
        c = fit(blocks, train)
        for key in held:
            b = blocks[key]
            kind = rule(b["_info"], c)
            if kind in b:
                regrets.append(b[kind] / b[best_kind(b)])
    return regrets


def summary(rs):
    rs = sorted(rs)
    if not rs:
        return "-"
    q = lambda f: rs[min(len(rs) - 1, int(f * len(rs)))]
    return f"{statistics.geometric_mean(rs):.4f} / {q(0.5):.3f} / {q(0.9):.3f} / {rs[-1]:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--rule-out")
    ap.add_argument("--no-cv", action="store_true", help="skip the cross-validation refits")
    args = ap.parse_args()
    blocks, bad = load(args.csv)
    blocks = {k: b for k, b in blocks.items() if "dp" in b}
    c = fit(blocks, list(blocks))
    lines = [f"# v31 rule fit on {', '.join(args.csv)}", "",
             f"{len(blocks)} (shape, tile) blocks; {len(bad)} failed or incorrect points.", "",
             "## Fitted constants", "", "```json", json.dumps(c, indent=1), "```", ""]
    lines += ["## Regret (time / block's best): geomean / median / 90th percentile / worst", "",
              "| set | rule | dp | best owner | best rs |", "|---|---|---|---|---|"]

    def fam(b, prefix):
        v = [x for k, x in b.items() if k.startswith(prefix) and not k.startswith("_")]
        return min(v) if v else None

    for name, sel in (("all", lambda i: True), ("below one wave", lambda i: i["T"] < P),
                      ("one wave or more", lambda i: i["T"] >= P)):
        keys = [k for k in blocks if sel(blocks[k]["_info"])]
        rr, dd, oo, ss = [], [], [], []
        for key in keys:
            b = blocks[key]
            best = b[best_kind(b)]
            kind = rule(b["_info"], c)
            if kind in b:
                rr.append(b[kind] / best)
            dd.append(b["dp"] / best)
            if fam(b, "ow"):
                oo.append(fam(b, "ow") / best)
            if fam(b, "rs"):
                ss.append(fam(b, "rs") / best)
        lines.append(f"| {name} ({len(keys)}) | {summary(rr)} | {summary(dd)} | {summary(oo)} | {summary(ss)} |")
    gm, worst, lose, missing = score(blocks, c)
    lines += ["", f"Rule slower than dp beyond {NOISE:.2f}x on {lose} blocks; rule's choice not timed on {missing}.", ""]
    if not args.no_cv:
        lines += ["## Cross-validation (refit without the held-out bin)", ""]
        for by in ("k", "S"):
            lines.append(f"- leave one {by} out: {summary(cross_validate(blocks, by))}")
    winners = collections.Counter(best_kind(b).rstrip("0123456789") for b in blocks.values())
    lines += ["", f"Best family per block: {dict(winners)}", ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        open(args.out, "w").write(text + "\n")
    if args.rule_out:
        json.dump(c, open(args.rule_out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
