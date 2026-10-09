"""Summary figure of the stage 3 costs (no GPU, about a second).

    python make_figures.py     # reads results/costs.md (and handshake_release.md), writes images/costs_summary.png
                               # and handoff_costs.png

Left: one 256 KiB partial per reader, 16 writers per XCD at once (a stream-K tail), split into the time
to publish, for the reader to see the flag, and to read. Right: a last-arriver reduction of 16 x 8 KiB
partials (aiter's fused split-K), split into publish, waiting for the last arrival, and the sum. Both
with cold caches (a 1 GiB read before each launch), the closest of the two cache modes to a GEMM's ending.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")

HANDSHAKE = [
    # label, sequence (as in costs.md), placement, recommended
    ("L2 path: default stores, plain flag,\n`.cg` loads (v21 l2)", "default stores, drain, plain flag polled with "
     "`.cg`, `.cg` reads (v21 `l2`)", "same XCD", True),
    ("write-through: `.wt` stores, flag,\n`.cv` loads", "`.wt` stores, drain, atomic flag, `.cv` reads", "same XCD", True),
    ("default stores, flag, acquire,\ndefault loads", "default stores, drain, atomic flag, agent acquire, default reads",
     "same XCD", False),
    ("write-through: `.wt` stores, flag,\n`.cv` loads (v21 wt)", "`.wt` stores, drain, atomic flag, `.cv` reads (v21 "
     "`wt`)", "other XCD", True),
    ("default stores, agent release, flag,\n`.cg` loads", "default stores, agent release, atomic flag, `.cg` reads",
     "other XCD", False),
    ("default stores, release + acquire\nin every wave, flag, default loads", "default stores, agent release, "
     "atomic flag, agent acquire, default reads", "other XCD", False),
]

HANDOFF = [
    # label, results file, sequence (as in that file), placement, recommended
    ("L2 path: default stores, plain flag\n(.cg poll), .cg loads", "costs", "default stores, drain, plain flag polled "
     "with `.cg`, `.cg` reads (v21 `l2`)", "same XCD", True),
    ("write-through: .wt stores,\natomic flag, .cv loads", "costs", "`.wt` stores, drain, atomic flag, `.cv` reads",
     "same XCD", True),
    ("default stores, atomic flag,\nacquire (compiler), default loads", "release", "default stores, drain, atomic flag, "
     "acquire (compiler), default reads", "same XCD", False),
    ("write-through: .wt stores,\natomic flag, .cv loads", "costs", "`.wt` stores, drain, atomic flag, `.cv` reads "
     "(v21 `wt`)", "other XCD", True),
    ("default stores, release flag\n(compiler, one per CU), .cg loads", "release", "default stores, drain, release flag "
     "(compiler), `.cg` reads", "other XCD", False),
    ("default stores, release flag +\nacquire (compiler), default loads", "release", "default stores, drain, release "
     "flag, acquire (compiler), default reads", "other XCD", False),
    ("default stores, buffer_wbl2 sc1 in\nevery wave, atomic flag, .cg loads", "costs", "default stores, agent release, "
     "atomic flag, `.cg` reads", "other XCD", False),
]

ARRIVE = [
    ("aiter: `acq_rel` `cta` arrival\n(not guaranteed)", "aiter: default stores, drain, `acq_rel` `cta` arrival, "
     "`.cv` sum", "one XCD", False),
    ("relaxed `gpu` arrival, `.cv` sum", "default stores, drain, relaxed `gpu` arrival, `.cv` sum", "one XCD", True),
    ("`.wt` stores, relaxed `gpu` arrival", "`.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum", "one XCD", True),
    ("`acq_rel` `gpu` arrival (compiler)", "default stores, drain, `acq_rel` `gpu` arrival, default sum", "one XCD",
     False),
    ("`.wt` stores, relaxed `gpu` arrival", "`.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum", "spread", True),
    ("`acq_rel` `gpu` arrival (compiler)", "default stores, drain, `acq_rel` `gpu` arrival, default sum", "spread",
     False),
]


def tables(path):
    """{section: [row dict]} for every markdown table in a results file."""
    out, section, header = {}, None, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("## "):
            section, header = line[3:].strip(), None
        elif line.startswith("|") and section:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = cells
            elif not set(line) <= set("|-"):
                out.setdefault(section, []).append(dict(zip(header, cells)))
        else:
            header = None
    return out


def find(rows, **want):
    hits = [r for r in rows if all(r[k] == v for k, v in want.items())]
    assert len(hits) == 1, (want, len(hits))
    return hits[0]


def bars(ax, items, title, segs, colors, notes=None):
    """items: (label, phase medians, measured total, group, recommended). Phases are medians on their own, so
    they are scaled to add up to the measured median total."""
    y, yt, labels, prev = 0.8, [], [], None
    for label, values, total, group, rec in items:
        if group != prev:
            if prev is not None:
                y += 0.9
            ax.text(0.15, y - 0.75, group, ha="left", va="center", fontsize=9, fontweight="bold", color="0.25")
            prev = group
        scale = total / sum(values)
        left = 0
        for v, c in zip(values, colors):
            ax.barh(y, v * scale, left=left, color=c, edgecolor="white", height=0.7)
            left += v * scale
        note = "" if notes is None else notes.get(label + group, "")
        ax.text(left + 0.3, y, f"{total:.1f} us{note}", va="center", fontsize=8,
                fontweight="bold" if rec else "normal")
        yt.append(y)
        labels.append(("* " if rec else "") + label.replace("`", ""))
        y += 1
    ax.set_yticks(yt)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(y - 0.4, -0.3)
    ax.set_xlabel("us")
    ax.set_title(title, fontsize=10)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], segs, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=0.3)


def main():
    t = tables(os.path.join(HERE, "results", "costs.md"))
    hs, ar = t["handshake"], t["arrive"]
    h_items, notes = [], {}
    for label, seq, place, rec in HANDSHAKE:
        r = find(hs, sequence=seq, placement=place, **{"writers per XCD": "16", "dirty MiB per XCD": "0"})
        one = find(hs, sequence=seq, placement=place, **{"writers per XCD": "1", "dirty MiB per XCD": "0"})
        h_items.append((label, [float(r["publish us"]), float(r["flag seen after us"]), float(r["read us"])],
                        float(r["total us"]), place, rec))
        notes[label + place] = f"   (1 writer: {float(one['total us']):.1f})"
    a_items = []
    for label, seq, place, rec in ARRIVE:
        r = find(ar, sequence=seq, placement=place, n="16", **{"partial KiB": "8"})
        pub, last, s = float(r["publish us"]), float(r["last arrival us"]), float(r["sum us"])
        a_items.append((label, [pub, last - pub, s], float(r["total us"]), place, rec))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    bars(a0, h_items, "Handoff of a 256 KiB partial, 16 writers per XCD at once", ["publish", "flag seen", "read"],
         ["#4878a8", "#e0a040", "#6aa56a"], notes)
    bars(a1, a_items, "Last-arriver sum of 16 x 8 KiB partials (fused split-K)", ["publish", "wait for last arrival",
                                                                                  "sum"],
         ["#4878a8", "#e0a040", "#6aa56a"])
    a0.set_xlim(0, 32)
    a1.set_xlim(0, 9.5)
    fig.text(0.01, 0.01, "* recommended. Cold caches (1 GiB read before each launch), medians; all sequences shown "
             "returned correct data. Each bar's phases are scaled to its measured median total. "
             "Source: results/costs.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(IMG, exist_ok=True)
    out = os.path.join(IMG, "costs_summary.png")
    fig.savefig(out, dpi=150)
    print(out)

    rows = {"costs": hs, "release": tables(os.path.join(HERE, "results", "handshake_release.md"))["handshake_release"]}
    o_items, o_notes = [], {}
    for label, src, seq, place, rec in HANDOFF:
        want = {"dirty MiB per XCD": "0"} if src == "costs" else {}
        r = find(rows[src], sequence=seq, placement=place, **{"writers per XCD": "16"}, **want)
        one = find(rows[src], sequence=seq, placement=place, **{"writers per XCD": "1"}, **want)
        o_items.append((label, [float(r["publish us"]), float(r["flag seen after us"]), float(r["read us"])],
                        float(r["total us"]), place, rec))
        o_notes[label + place] = f"   (1 writer: {float(one['total us']):.1f})"
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    bars(ax, o_items, "Handoff of a 256 KiB partial, 16 writers per XCD at once", ["publish", "flag seen", "read"],
         ["#4878a8", "#e0a040", "#6aa56a"], o_notes)
    ax.set_xlim(0, 32)
    fig.text(0.01, 0.01, "* recommended. Atomic flag: a relaxed gpu-scope swap, polled with relaxed gpu-scope atomic "
             "loads. Compiler release / acquire: one\nbuffer_wbl2 sc1 / buffer_inv sc1 per CU. The last row issues "
             "buffer_wbl2 sc1 by hand from all 4 waves of every writer (pessimistic). Cold caches\n(1 GiB read before "
             "each launch), medians; all sequences returned correct data. Each bar's phases are scaled to its measured "
             "median\ntotal. Source: results/costs.md, results/handshake_release.md.", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = os.path.join(IMG, "handoff_costs.png")
    fig.savefig(out, dpi=150)
    print(out)
    exchange_figure()


PATTERN_STYLE = {
    "handoff": ("#4878a8", "1:1 handoff (W more CUs each read one partial)"),
    "reduce-scatter": ("#6aa56a", "reduce-scatter (each of n reads 1/n of all n)"),
    "owner": ("#c0504d", "owner reads the other n - 1 (spins on each flag)"),
    "last arriver": ("#e0a040", "last arriver reads the other n - 1"),
}


def exchange_figure():
    """Model lines and measured points for the four ways of exchanging partials (results/exchange.md)."""
    from exchange import model_us  # imports triton and torch, but needs no GPU

    allrows = tables(os.path.join(HERE, "results", "exchange.md"))
    rows = [r for r in allrows["exchange"] if r["sync"] == "wt" and r["placement"] == "one XCD"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, pk in zip(axes[:2], (128, 256)):
        for pattern, (color, label) in PATTERN_STYLE.items():
            got = sorted((int(r["W"]), float(r["measured us"])) for r in rows
                         if r["pattern"] == pattern and r["P KiB"] == str(pk))
            ws = [w for w, _ in got]
            ax.plot(ws, [model_us(pattern, w, pk * 1024)[0] for w in ws], "--", color=color, lw=1.2)
            ax.plot(ws, [m for _, m in got], "o-", color=color, lw=2, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([16, 32, 64, 128, 256])
        ax.set_xticklabels(["16", "32", "64", "128", "256"])
        ax.set_yticks([3, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["3", "5", "10", "20", "50", "100"])
        ax.set_xlabel("writers W (W / 8 per XCD)")
        ax.set_ylabel("us, all exchanges done")
        what = "256x256 fp32" if pk == 256 else "128x256 fp32 or 256x256 bf16"
        ax.set_title(f"{pk} KiB partials ({what}), groups of 16", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="center left")
    axes[1].text(0.98, 0.03, "solid: measured; dashed: model (slowest of the\nCU, XCD-link and chip limits per phase)",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8, color="0.3")
    ax = axes[2]
    for pattern, (color, label) in PATTERN_STYLE.items():
        got = sorted((int(r["W"]), float(r["achieved TB/s"])) for r in rows
                     if r["pattern"] == pattern and r["P KiB"] == "256")
        ax.plot([w for w, _ in got], [b for _, b in got], "o-", color=color, lw=2)
    ax.axhline(7.5, ls=":", color="0.4")
    ax.text(17, 7.6, "chip limit, reads from the MALL (7.5 TB/s); writes 7.4", fontsize=8, color="0.3", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.set_ylim(0, 8.5)
    ax.set_xlabel("writers W")
    ax.set_ylabel("TB/s (bytes written + read / time)")
    ax.set_title("Bandwidth used, 256 KiB partials", fontsize=10)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Every group on one XCD; .wt stores, drain, relaxed gpu flag or counter, .cv loads (16 in "
             "flight per CU); cold caches, "
             "medians of 30 launches, all sums correct. Handoff and reduce-scatter share one model line, owner and "
             "last arriver another. Source: results/exchange.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(IMG, "exchange_scaling.png")
    fig.savefig(out, dpi=150)
    print(out)
    methods_figure(allrows)


SYNC_STYLE = {
    "wt": ("#4878a8", "write-through: .wt stores, relaxed sync, .cv loads (correct for any placement)"),
    "acq_rel": ("#c0504d", "compiler: default stores, gpu release / acquire, default loads (any placement)"),
    "l2": ("#6aa56a", "L2 path: default stores, relaxed sync, .cg loads (needs the group on one XCD)"),
}


def methods_figure(allrows):
    """Each exchange pattern with each synchronisation method, and the split count."""
    rows = [r for r in allrows["exchange"] if r["placement"] == "one XCD"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.15, 1, 0.9]})
    for ax, W in zip(axes[:2], (128, 256)):
        pats = [p for p in PATTERN_STYLE if any(r["pattern"] == p and r["W"] == str(W) for r in rows)]
        width = 0.27
        for k, (sync, (color, _)) in enumerate(SYNC_STYLE.items()):
            for i, p in enumerate(pats):
                r = find(rows, pattern=p, sync=sync, W=str(W), **{"P KiB": "256"})
                tot, pub = float(r["measured us"]), float(r["publish us"])
                x = i + (k - 1) * width
                ax.bar(x, pub, width, color=color, alpha=1.0, edgecolor="white")
                ax.bar(x, tot - pub, width, bottom=pub, color=color, alpha=0.45, edgecolor="white")
                ax.text(x, tot + 1, f"{tot:.0f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(pats)))
        ax.set_xticklabels([p.replace("reduce-scatter", "reduce-\nscatter").replace("last arriver", "last\narriver")
                            .replace("handoff", "1:1\nhandoff") for p in pats], fontsize=9)
        ax.set_ylabel("us, all exchanges done")
        ax.set_title(f"{W} writers, 256 KiB partials, groups of 16", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 98)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in SYNC_STYLE.values()],
               [lab for _, lab in SYNC_STYLE.values()], fontsize=8, loc="lower left", bbox_to_anchor=(0.03, 0.04),
               ncol=3, frameon=False)
    axes[1].text(0.02, 0.97, "dark: publish (stores + drain);\nlight: sync and reads", transform=axes[1].transAxes,
                 fontsize=8, va="top", color="0.3")
    ax = axes[2]
    split = [r for r in allrows["split"] if r["setup"] == "CUs 256" and r["placement"] == "one XCD"
             and r.get("partials", "fp32") == "fp32"]
    for p in ("reduce-scatter", "owner", "last arriver"):
        color = PATTERN_STYLE[p][0]
        got = sorted((int(r["n"]), float(r["measured us"]), float(r["model us"])) for r in split if r["pattern"] == p)
        ax.plot([g[0] for g in got], [g[1] for g in got], "o-", color=color, lw=2, label=p)
        ax.plot([g[0] for g in got], [g[2] for g in got], "--", color=color, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16])
    ax.set_xticklabels(["2", "4", "8", "16"])
    ax.set_xlabel("splits per tile n (all 256 CUs, 256 / n tiles)")
    ax.set_ylabel("us")
    ax.set_title("Split count, 256 KiB partials, write-through", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "Every group on one XCD (as stream-K places a tile); cold caches, medians of 30 launches, "
             "all sums correct; dashed: model. Source: results/exchange.md.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out = os.path.join(IMG, "exchange_methods.png")
    fig.savefig(out, dpi=150)
    print(out)
    phases_figure(allrows)


PHASE_COLORS = ["#4878a8", "#e0a040", "#6aa56a", "#8064a2"]
ZERO_COLOR = "0.6"
PHASE_PATTERNS = (("reduce-scatter", "reduce-scatter"), ("owner", "owner"), ("last arriver", "last arriver"),
                  ("atomic bf16", "atomic bf16 pk_add\ninto C"), ("atomic fp32", "atomic fp32\n(fp32 output)"),
                  ("atomic fp32 + finalize", "atomic fp32, then last\narriver stores bf16 C"))


def phases_figure(allrows):
    """Publish / sync / read / C store on the critical path against the split count n, measured and modelled,
    for the three exchanges and the three atomic endings; the atomic endings' separate zeroing launch on top."""
    split = [r for r in allrows["split"] if r["placement"] == "one XCD" and r.get("partials", "fp32") in ("fp32", "-")]
    fig, axes = plt.subplots(2, len(PHASE_PATTERNS), figsize=(22, 9), sharey="row")
    setups = (("CUs 256", "all 256 CUs, 256 / n tiles"), ("tiles 16", "16 tiles, W = 16 n CUs"))
    for row, (setup, sdesc) in enumerate(setups):
        for col, (p, pdesc) in enumerate(PHASE_PATTERNS):
            ax = axes[row, col]
            rs = sorted((r for r in split if r["setup"] == setup and r["pattern"] == p), key=lambda r: int(r["n"]))
            for i, r in enumerate(rs):
                meas = [float(r["publish us"]), float(r["sync us"]), float(r["read us"]), float(r["C store us"])]
                mod = [float(r["model publish"]), float(r["model sync"]), float(r["model read"])]
                zero = float(r["zero us"])
                for x, vals, kw in ((i - 0.19, meas, {}), (i + 0.19, mod, {"hatch": "//", "alpha": 0.45})):
                    bottom = 0
                    for v, c in zip(vals, PHASE_COLORS):
                        ax.bar(x, v, 0.36, bottom=bottom, color=c, edgecolor="white", **kw)
                        bottom += v
                    label = f"{bottom:.0f}"
                    if zero and not kw:
                        ax.bar(x, zero, 0.36, bottom=bottom, color=ZERO_COLOR, edgecolor="white", hatch="..")
                        label = f"{bottom:.0f}\n+{zero:.0f}"
                        bottom += zero
                    ax.text(x, bottom + 1, label, ha="center", fontsize=7)
            ax.set_xticks(range(len(rs)))
            ax.set_xticklabels([f"n = {r['n']}" for r in rs])
            ax.set_title(f"{pdesc}\n{sdesc}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel("us on the critical path")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS] +
               [plt.Rectangle((0, 0), 1, 1, facecolor=ZERO_COLOR, edgecolor="white", hatch=".."),
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.4", hatch="//")],
               ["publish (until the group's last writer has published; atomics: until its adds complete)",
                "sync (flag, counter or arrival)", "read (the consumer that finishes last)",
                "bf16 C store (that consumer)", "zeroing the atomics' tiles (a launch of its own)",
                "right-hand bar of each pair: model (no C store, no zeroing)"],
               fontsize=8, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.01, 0.005, "256 KiB fp32 partials, every group on one XCD (as stream-K places a tile), cold caches. "
             "Exchanges: write-through sync, then the consumers store bf16 C. Atomic endings: coalesced, no "
             "convert_layout. Each phase is the median over groups and 30 launches; all sums correct. "
             "Source: results/exchange.md (split).", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = os.path.join(IMG, "exchange_phases.png")
    fig.savefig(out, dpi=150)
    print(out)
    totals_figure(allrows)


ATOMIC_STYLE = {"atomic bf16": ("#8064a2", "atomic bf16 pk_add into C"),
                "atomic fp32": ("#7f5f3f", "atomic fp32 (fp32 output)")}


def totals_figure(allrows):
    """Measured totals against n: the exchanges with fp32 and bf16 partials, and the atomic endings with and
    without their zeroing launch."""
    split = [r for r in allrows["split"] if r["placement"] == "one XCD"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (setup, sdesc) in zip(axes, (("CUs 256", "all 256 CUs, 256 / n tiles"),
                                         ("tiles 16", "16 tiles, W = 16 n CUs"))):
        for p in ("reduce-scatter", "owner", "last arriver"):
            color = PATTERN_STYLE[p][0]
            for pdt, ls, mk in (("fp32", "-", "o"), ("bf16", "--", "s")):
                got = sorted((int(r["n"]), float(r["measured us"])) for r in split
                             if r["setup"] == setup and r["pattern"] == p and r["partials"] == pdt)
                ax.plot([g[0] for g in got], [g[1] for g in got], ls=ls, marker=mk, color=color, lw=2 if pdt ==
                        "fp32" else 1.6, label=f"{p}, {pdt} partials")
        for p, (color, label) in ATOMIC_STYLE.items():
            got = sorted((int(r["n"]), float(r["measured us"]), float(r["zero us"])) for r in split
                         if r["setup"] == setup and r["pattern"] == p)
            ax.plot([g[0] for g in got], [g[1] for g in got], "-", marker="D", color=color, lw=2, label=label)
            ax.plot([g[0] for g in got], [g[1] + g[2] for g in got], ":", marker="D", mfc="white", color=color,
                    lw=1.2, label=label + " + zeroing launch")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2, 4, 8, 16])
        ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_yticks([5, 10, 20, 50, 100])
        ax.set_yticklabels(["5", "10", "20", "50", "100"])
        ax.set_xlabel("splits per tile n")
        ax.set_ylabel("us, all endings done (measured)")
        ax.set_title(sdesc, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Tile endings, 256x256 tiles: exchanges with fp32 or bf16 partials against atomic adds into C",
                 fontsize=12)
    fig.text(0.01, 0.01, "Every group on one XCD, cold caches, medians of 30 launches, all sums correct. Exchanges: "
             "write-through stores, relaxed flags or counters, .cv loads, then the bf16 C store. Precision: fp32 "
             "partials round once; bf16 partials round each partial\nonce and the fp32 sum once; atomic bf16 rounds the "
             "running sum at every add, in no fixed order. The bf16 owner and last-arriver reads keep about 8 loads "
             "in flight instead of 12 (register pressure from the conversion),\nso they are slower per byte than the "
             "fp32 ones. Source: results/exchange.md (split).", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    out = os.path.join(IMG, "exchange_totals.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
