#!/usr/bin/env python3
"""Merge full-sweep rounds and write points.csv, best.csv and summary.md."""

import argparse
import collections
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
A16 = os.path.dirname(HERE)
AIS = os.path.abspath(os.path.join(A16, "..", "..", "..", "..", "experiments", "aiter_lixun_shapes"))
DEFAULT_SOURCE = os.path.join(HERE, "results", "full_points.csv")
DEFAULT_RUN = "/home/jbowley/repos/aiter/streamk_study/2026-10-22_full_v28/results"


def as_bool(value):
    return str(value).lower() == "true"


def load_rounds(path):
    by_point = collections.defaultdict(list)
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            by_point[row["point"]].append(row)
    return by_point


def merge_one(source, rounds, canonical):
    row = dict(source)
    target = canonical.get(source["alias"], source["point"])
    timed = rounds.get(target, [])
    by_round = {int(r["round"]): r for r in timed}
    for number in (1, 2):
        got = by_round.get(number, {})
        row[f"round{number}_us"] = got.get("median_us", "")
        row[f"round{number}_p10_us"] = got.get("p10_us", "")
        row[f"round{number}_p90_us"] = got.get("p90_us", "")
    medians = [float(r["median_us"]) for r in timed if r.get("median_us") and not r.get("error")]
    row["median_us"] = f"{statistics.median(medians):.3f}" if medians else ""
    row["correct"] = bool(timed) and all(as_bool(r["correct"]) for r in timed)
    row["mean_ratio"] = (
        f"{max(float(r['mean_ratio']) for r in timed if r.get('mean_ratio')):.6g}"
        if any(r.get("mean_ratio") for r in timed)
        else ""
    )
    row["error"] = "; ".join(dict.fromkeys(r["error"] for r in timed if r.get("error")))
    row["rounds"] = len(medians)
    if source["reject_reason"]:
        row["error"] = source["reject_reason"]
    return row


def load_picks(path):
    rows = collections.defaultdict(list)
    if os.path.exists(path):
        for row in csv.DictReader(open(path)):
            rows[(row["M"], row["N"], row["K"])].append(row)
    return rows


def markdown_counter(title, counter):
    lines = [f"### {title}", "", "| value | shapes |", "|---|---:|"]
    lines += [f"| `{key}` | {count} |" for key, count in counter.most_common()]
    return lines


def write_report(points, best_rows, picks, path):
    shapes = {(r["M"], r["N"], r["K"]) for r in points}
    timed = [r for r in points if r["median_us"]]
    rejects = sum(bool(r["reject_reason"]) for r in points)
    aliases = sum(bool(r["alias"]) for r in points)
    errors = sum(bool(r["error"]) and not r["reject_reason"] for r in points)
    incorrect = sum(bool(r["median_us"]) and not r["correct"] for r in points)
    winners = [r for r in best_rows if r["rank"] == 1]
    kind_counts = collections.Counter(r["kind"] for r in winners)
    tile_counts = collections.Counter(r["tile"] for r in winners)
    near = collections.Counter()
    for r in best_rows:
        near["1%"] += r["within_1pct"]
        near["2%"] += r["within_2pct"]
        near["5%"] += r["within_5pct"]

    tile_agree = exact_origami = refit_agree = comparable = 0
    for winner in winners:
        shape = (winner["M"], winner["N"], winner["K"])
        candidates = picks.get(shape, [])
        if not candidates:
            continue
        comparable += 1
        o_tile = candidates[0].get("o_tile", "")
        tile_agree += winner["tile"] == o_tile
        origami = {
            (r["kind"], r["tile"])
            for r in candidates
            if r.get("slot") in {"v9", "pdp", "rs", "ow", "2t", "sp", "rsb", "owb", "rsp", "owp"}
        }
        refit = {(r["kind"], r["tile"]) for r in candidates if r.get("slot") == "refit"}
        exact_origami += (winner["kind"], winner["tile"]) in origami
        refit_agree += (winner["kind"], winner["tile"]) in refit

    lines = [
        "# Full v28 sweep summary",
        "",
        f"- Shapes: {len(shapes):,}",
        f"- Enumerated points: {len(points):,}",
        f"- Points with timing: {len(timed):,}",
        f"- Exact launch aliases: {aliases:,}",
        f"- Host rejects: {rejects:,}",
        f"- Timing errors: {errors:,}",
        f"- Incorrect timed points: {incorrect:,}",
        f"- Best/near-best rows in `best.csv`: {len(best_rows):,}",
        "",
        "## Near-best counts",
        "",
        f"- Within 1% (including each winner): {near['1%']:,}",
        f"- Within 2%: {near['2%']:,}",
        f"- Within 5%: {near['5%']:,}",
        "",
        "## Pick agreement",
        "",
        f"- Shapes comparable with the origami run: {comparable:,}",
        f"- Winning tile equals origami's tile: {tile_agree:,}",
        f"- Winning `(kind, tile)` was an origami slot: {exact_origami:,}",
        f"- Winning `(kind, tile)` equals the REFIT slot: {refit_agree:,}",
        "",
    ]
    lines += markdown_counter("Winning kinds", kind_counts) + [""]
    lines += markdown_counter("Winning tiles", tile_counts) + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_page(path, run_dir, points_count):
    rel = os.path.relpath(run_dir, os.path.dirname(path))
    text = f"""# Full v28 configuration sweep

Every valid v28 configuration at each of the 34 candidate tiles, after the
two-times-padding rule, on all shapes with an origami pick. Exact duplicate
host launches are represented as aliases and are not timed twice.

Timing uses consecutive rotating-buffer graph windows. Round 1 covers every
canonical point; points within 5% of their shape best receive a second round
from freshly allocated buffers. Times are not directly comparable with the
older harness's MALL-warm rows.

- Enumerated points: {points_count:,}
- [`summary.md`]({rel}/summary.md)
- [`points.csv`]({rel}/points.csv): every point, alias and rejection
- [`best.csv`]({rel}/best.csv): winners and points within 5%
- [`manifest.json`]({rel}/manifest.json): frozen source hashes and device choices
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--picks", default=os.path.join(AIS, "picks_origami.csv"))
    ap.add_argument("--page")
    args = ap.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    source = list(csv.DictReader(open(args.source)))
    canonical = {r["point"]: r["point"] for r in source if not r["alias"]}
    rounds = load_rounds(os.path.join(args.run_dir, "rounds.csv"))
    points = [merge_one(r, rounds, canonical) for r in source]
    point_fields = list(points[0]) if points else []
    with open(os.path.join(args.run_dir, "points.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=point_fields)
        writer.writeheader()
        writer.writerows(points)

    by_shape = collections.defaultdict(list)
    for row in points:
        if row["median_us"] and row["correct"] and not row["error"]:
            by_shape[(row["M"], row["N"], row["K"])].append(row)
    best_rows = []
    for rows in by_shape.values():
        rows.sort(key=lambda r: float(r["median_us"]))
        best = float(rows[0]["median_us"])
        for rank, row in enumerate(rows, 1):
            ratio = float(row["median_us"]) / best
            if ratio > 1.05:
                break
            best_rows.append(
                {
                    **row,
                    "rank": rank,
                    "gap_pct": f"{100 * (ratio - 1):.4f}",
                    "within_1pct": ratio <= 1.01,
                    "within_2pct": ratio <= 1.02,
                    "within_5pct": ratio <= 1.05,
                }
            )
    best_fields = list(best_rows[0]) if best_rows else point_fields + [
        "rank", "gap_pct", "within_1pct", "within_2pct", "within_5pct"
    ]
    with open(os.path.join(args.run_dir, "best.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_fields)
        writer.writeheader()
        writer.writerows(best_rows)
    write_report(points, best_rows, load_picks(args.picks), os.path.join(args.run_dir, "summary.md"))
    if args.page:
        write_page(args.page, args.run_dir, len(points))
    print(f"{len(points):,} points, {len(best_rows):,} best/near-best rows -> {args.run_dir}")


if __name__ == "__main__":
    main()
