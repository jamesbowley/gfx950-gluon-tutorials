"""Bar charts of the timing results (no GPU).

    python make_figure.py

Reads the three results/timing*.md files and writes images/latency.png,
images/cache_policy.png and the combined images/summary.png.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {  # case in results/timing.md -> label in the figure
    "plain miss": "plain miss",
    "L2 hit (primed, not written)": "L2 hit",
    "written into L2": "written into L2",
    "primed, then written": "primed, then written",
}


def rows(path):
    lines = [l for l in open(path) if l.startswith("|")]
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in l.strip().strip("|").split("|")))) for l in lines[2:]]


def latency_figure():
    data = {r["case"]: r for r in rows(os.path.join(HERE, "results", "timing.md"))}
    cases = list(LABELS)[::-1]  # top to bottom in the order of the README
    first = [float(data[c]["first chase ns per load"]) for c in cases]
    hit, miss = (float(data[c]["first chase ns per load"]) for c in ("L2 hit (primed, not written)", "plain miss"))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = range(len(cases))
    from_hbm = [f > (hit + miss) / 2 for f in first]
    ax.barh(list(y), first, 0.6, color=["#e45756" if m else "#4c78a8" for m in from_hbm])
    for i in y:
        ax.text(first[i] + 4, i, f"{first[i]:.0f} ns", va="center", fontsize=9, fontweight="bold")
    ax.legend(handles=[Patch(color="#e45756", label="from HBM"), Patch(color="#4c78a8", label="from L2")],
              loc="lower right", fontsize=9, frameon=False)
    ax.set_yticks(list(y), [LABELS[c] for c in cases])
    ax.set_xlabel("first read, ns per dependent load")
    ax.set_xlim(0, miss * 1.18)
    ax.set_ylim(-0.6, len(cases) - 0.4)
    ax.set_title("MI355X: the first read of a line written into L2 costs an HBM trip", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "images", "latency.png"), dpi=150)
    plt.close(fig)


def cache_policy_figure():
    files = [
        (".cg (sc0 nt)", "Hit Stream", "timing.md"),
        ("default", "Hit LRU", "timing_default_loads.md"),
        (".cv (sc0 sc1)", "observed Hit LRU", "timing_cv_loads.md"),
    ]
    first, second = [], []
    for _, _, filename in files:
        data = {r["case"]: r for r in rows(os.path.join(HERE, "results", filename))}
        plain = data["plain miss"]
        first.append(float(plain["first chase ns per load"]))
        second.append(float(plain["second chase ns per load"]))

    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    x = list(range(len(files)))
    width = 0.34
    first_bars = ax.bar([i - width / 2 for i in x], first, width, color="#e45756",
                        label="first read after flush")
    second_bars = ax.bar([i + width / 2 for i in x], second, width, color="#4c78a8",
                         label="second read of the same lines")
    ax.bar_label(first_bars, labels=[f"{v:.0f} ns" for v in first], padding=3, fontsize=9)
    ax.bar_label(second_bars, labels=[f"{v:.0f} ns" for v in second], padding=3, fontsize=9)
    ax.set_xticks(x, [f"{name}\n{policy}" for name, policy, _ in files])
    ax.set_ylabel("ns per dependent load")
    ax.set_ylim(0, max(first) * 1.35)
    ax.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MI355X: whether a missed line remains in L2 depends on the load policy", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "images", "cache_policy.png"), dpi=150)
    plt.close(fig)


def summary_figure():
    policies = [
        (".cg (sc0 nt)", "Hit Stream", "timing.md"),
        ("default", "Hit LRU", "timing_default_loads.md"),
        (".cv (sc0 sc1)", "observed Hit LRU", "timing_cv_loads.md"),
    ]
    by_policy = []
    for name, policy, filename in policies:
        data = {r["case"]: r for r in rows(os.path.join(HERE, "results", filename))}
        by_policy.append((name, policy, data))

    core = by_policy[0][2]
    cases = list(LABELS)[::-1]
    core_first = [float(core[c]["first chase ns per load"]) for c in cases]
    hit = float(core["L2 hit (primed, not written)"]["first chase ns per load"])
    miss = float(core["plain miss"]["first chase ns per load"])
    midpoint = (hit + miss) / 2

    policy_first = [float(data["plain miss"]["first chase ns per load"]) for _, _, data in by_policy]
    policy_second = [float(data["plain miss"]["second chase ns per load"]) for _, _, data in by_policy]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

    y = list(range(len(cases)))
    left.barh(y, core_first, 0.6, color=["#e45756" if v > midpoint else "#4c78a8" for v in core_first])
    for i, value in enumerate(core_first):
        left.text(value + 4, i, f"{value:.0f} ns", va="center", fontsize=9, fontweight="bold")
    left.set_yticks(y, [LABELS[c] for c in cases])
    left.set_xlabel("first timed read, ns per dependent load")
    left.set_xlim(0, miss * 1.18)
    left.set_title("(a) First read after each setup", fontsize=10)
    left.spines[["top", "right"]].set_visible(False)

    x = list(range(len(policies)))
    width = 0.34
    first_bars = right.bar(
        [i - width / 2 for i in x],
        policy_first,
        width,
        color=["#e45756" if v > midpoint else "#4c78a8" for v in policy_first],
    )
    second_bars = right.bar(
        [i + width / 2 for i in x],
        policy_second,
        width,
        color=["#e45756" if v > midpoint else "#4c78a8" for v in policy_second],
    )
    right.bar_label(first_bars, labels=[f"1st\n{v:.0f}" for v in policy_first], padding=3, fontsize=8)
    right.bar_label(second_bars, labels=[f"2nd\n{v:.0f}" for v in policy_second], padding=3, fontsize=8)
    right.set_xticks(x, [f"{name}\n{policy}" for name, policy, _ in policies])
    right.set_ylabel("ns per dependent load")
    right.set_ylim(0, miss * 1.28)
    right.set_title("(b) Plain miss: is the line retained?", fontsize=10)
    right.spines[["top", "right"]].set_visible(False)

    fig.legend(
        handles=[Patch(color="#e45756", label="from HBM"), Patch(color="#4c78a8", label="from L2")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    fig.suptitle("MI355X: L2 write-allocate read-back and load-policy behaviour", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "images", "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    latency_figure()
    cache_policy_figure()
    summary_figure()


if __name__ == "__main__":
    main()
