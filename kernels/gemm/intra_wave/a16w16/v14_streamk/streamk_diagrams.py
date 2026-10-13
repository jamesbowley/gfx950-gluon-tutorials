"""Schedule model and diagrams for the stream-K tail of a "data-parallel + one-tile stream-K" GEMM.

The partition, the per-program segment loop and the owner's peer walk mirror
TensorAtlas/kernels/gemm/streamk_matmul.py (lines 176-182, 185-187 and 262-362). The
"reversed" variant is the proposed alternative: the program holding a tile's last k-step
reduces it, and every program processes its segments last-to-first.

Time is measured in k-steps (one BLOCK_M x BLOCK_N x BLOCK_K MAC iteration) since the
stream-K phase started, assuming every program retires one k-step per unit of time. The
store / read cost of a partial is a parameter; the policies table also has a column with the
costs measured in experiments/streamk_costs (MEASURED).

    python streamk_diagrams.py            # self-checks, tables, images/*.png
"""

import argparse
import bisect
import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ORIGINAL = "original"
REVERSED = "reversed"

TILE_COLOURS = plt.cm.Set2.colors
XCD_COLOURS = plt.cm.tab10.colors


@dataclass(frozen=True)
class Problem:
    name: str
    num_programs: int
    streamk_tiles: int
    iters_per_tile: int
    full_tiles: int = 0
    num_pid_m: int = 0
    num_pid_n: int = 0
    group_size_m: int = 4
    pids_per_xcd: int = 0  # 0: no XCD information

    @staticmethod
    def from_shape(M, N, K, BM=256, BN=256, BK=64, num_programs=256, num_xcds=8, group_size_m=4):
        num_pid_m = -(-M // BM)
        num_pid_n = -(-N // BN)
        total = num_pid_m * num_pid_n
        sk = total % num_programs
        return Problem(
            name=f"{M}x{N}x{K}",
            num_programs=num_programs,
            streamk_tiles=sk,
            iters_per_tile=-(-K // BK),
            full_tiles=total - sk,
            num_pid_m=num_pid_m,
            num_pid_n=num_pid_n,
            group_size_m=group_size_m,
            pids_per_xcd=num_programs // num_xcds,
        )

    @property
    def total_iters(self):
        return self.streamk_tiles * self.iters_per_tile

    @property
    def L(self):
        return self.total_iters // self.num_programs

    @property
    def R(self):
        return self.total_iters % self.num_programs

    def length(self, pid):
        """Iterations owned by pid: streamk_iters_pcu + (pid < streamk_remainder_iters)."""
        return self.L + (pid < self.R)

    def start(self, pid):
        return pid * self.L + min(pid, self.R)

    def pid_of_iter(self, i):
        """Closed-form inverse of start(): the program whose range contains iteration i."""
        head = self.R * (self.L + 1)
        if i < head:
            return i // (self.L + 1)
        # With L == 0 only the first R programs have work, and they cover every iteration.
        assert self.L > 0, f"iteration {i} is past the end of the stream-K iterations"
        return self.R + (i - head) // self.L

    def xcd(self, pid):
        # v14's get_logical_chiplet_mapped_pids: logical pids are contiguous per XCD.
        return pid // self.pids_per_xcd if self.pids_per_xcd else 0

    def tile_mn(self, tile):
        """GROUP_SIZE_M swizzle of the absolute tile id (TensorAtlas order)."""
        tile_id = self.full_tiles + tile
        npg = self.group_size_m * self.num_pid_n
        first = (tile_id // npg) * self.group_size_m
        gs = min(self.num_pid_m - first, self.group_size_m)
        return first + (tile_id % npg) % gs, (tile_id % npg) // gs


@dataclass
class Segment:
    pid: int
    begin: int  # stream-K-relative iteration index
    end: int
    ipt: int
    role: str = ""  # "reduce", "partial" or "complete"
    peers: list = field(default_factory=list)

    @property
    def tile(self):
        return self.begin // self.ipt

    @property
    def k0(self):
        return self.begin - self.tile * self.ipt

    @property
    def k1(self):
        return self.end - self.tile * self.ipt

    @property
    def tile_begin(self):
        return self.tile * self.ipt

    @property
    def tile_end(self):
        return self.tile_begin + self.ipt


@dataclass
class Event:
    kind: str  # "mac", "store", "wait", "read"
    t0: float
    t1: float
    seg: Segment = None
    peer: int = -1


def split_segments(pr, pid):
    """The kernel's `while start_iter < last_iter` loop."""
    start, last = pr.start(pid), pr.start(pid) + pr.length(pid)
    ipt = pr.iters_per_tile
    segs = []
    while start < last:
        remainder = start % ipt
        end_iter = min(start + (ipt - remainder), last)
        segs.append(Segment(pid, start, end_iter, ipt))
        start = end_iter
    return segs


def forward_peers(pr, pid, end_iter, tile_iter_end):
    """The kernel's fixup walk: `end += streamk_iters_pcu + (next_pid < remainder)`."""
    peers, end, next_pid = [], end_iter, pid + 1
    while end < tile_iter_end and next_pid < pr.num_programs:
        peers.append(next_pid)
        end += pr.length(next_pid)
        next_pid += 1
    assert end >= tile_iter_end, "walk ran out of programs before reaching the tile end"
    return peers


def backward_peers(pr, pid, begin, tile_iter):
    """Mirror of forward_peers for the reversed variant: walk down until the tile start."""
    peers, start, prev_pid = [], begin, pid - 1
    while start > tile_iter and prev_pid >= 0:
        peers.append(prev_pid)
        start -= pr.length(prev_pid)
        prev_pid -= 1
    assert start <= tile_iter, "walk ran out of programs before reaching the tile start"
    return peers


@dataclass
class Schedule:
    pr: Problem
    variant: str
    segs: dict  # pid -> segments in iteration order
    events: dict  # pid -> events in time order
    publish: dict  # pid -> time its partial becomes visible


def build_schedule(pr, variant, store_cost=0.5, read_cost=0.5):
    segs = {p: split_segments(pr, p) for p in range(pr.num_programs)}
    for p, ss in segs.items():
        for s in ss:
            if s.begin == s.tile_begin and s.end == s.tile_end:
                s.role = "complete"
            elif variant == ORIGINAL:
                if s.begin == s.tile_begin:
                    s.role, s.peers = "reduce", forward_peers(pr, p, s.end, s.tile_end)
                else:
                    s.role = "partial"
            else:
                if s.end == s.tile_end:
                    s.role, s.peers = "reduce", backward_peers(pr, p, s.begin, s.tile_begin)
                else:
                    s.role = "partial"

    order = {p: (ss if variant == ORIGINAL else ss[::-1]) for p, ss in segs.items()}

    # A partial is always the first segment a program processes, so publish times do not
    # depend on anybody's fixup.
    publish = {}
    for p, ss in order.items():
        for i, s in enumerate(ss):
            if s.role == "partial":
                assert i == 0, f"p{p}: partial is not the first processed segment"
                publish[p] = s.end - s.begin + store_cost

    events = {}
    for p, ss in order.items():
        t, ev = 0.0, []
        for s in ss:
            n = s.end - s.begin
            ev.append(Event("mac", t, t + n, s))
            t += n
            if s.role == "partial":
                ev.append(Event("store", t, t + store_cost, s))
                t += store_cost
            elif s.role == "reduce":
                for q in s.peers:
                    if publish[q] > t:
                        ev.append(Event("wait", t, publish[q], s, q))
                        t = publish[q]
                    ev.append(Event("read", t, t + read_cost, s, q))
                    t += read_cost
        events[p] = ev
    return Schedule(pr, variant, segs, events, publish)


def self_check(sched):
    pr = sched.pr
    covered = [0] * pr.total_iters
    by_tile = {}
    for p, ss in sched.segs.items():
        assert sum(s.role == "partial" for s in ss) <= 1, f"p{p} needs more than one P slot"
        for s in ss:
            for i in range(s.begin, s.end):
                covered[i] += 1
            by_tile.setdefault(s.tile, []).append(s)
        if pr.length(p):
            assert pr.pid_of_iter(pr.start(p)) == p
            assert pr.pid_of_iter(pr.start(p) + pr.length(p) - 1) == p
    assert all(c == 1 for c in covered), "iterations not covered exactly once"

    for tile, ss in by_tile.items():
        pids = {s.pid for s in ss}
        reducers = [s for s in ss if s.role in ("reduce", "complete")]
        assert len(reducers) == 1, f"tile {tile}: {len(reducers)} reducers"
        r = reducers[0]
        assert set(r.peers) == pids - {r.pid}, f"tile {tile}: walk found {r.peers}, covering {pids}"
        if sched.variant == ORIGINAL:
            assert all(q > r.pid for q in r.peers)
            last = pr.pid_of_iter(r.tile_end - 1)
            assert len(r.peers) == last - r.pid, "closed-form peer count disagrees with the walk"
        else:
            assert all(q < r.pid for q in r.peers)
            first = pr.pid_of_iter(r.tile_begin)
            assert len(r.peers) == r.pid - first


def reducers(sched):
    return sorted(
        (s for ss in sched.segs.values() for s in ss if s.role in ("reduce", "complete")),
        key=lambda s: s.tile,
    )


def finish_time(sched):
    return max(ev[-1].t1 for ev in sched.events.values() if ev)


# ----------------------------------------------------------------------------- tables


def k_range(s):
    return f"k{s.k0}" if s.k1 - s.k0 == 1 else f"k{s.k0}-{s.k1 - 1}"


def program_table(sched):
    pr = sched.pr
    readers = {q: s.pid for s in reducers(sched) for q in s.peers}
    rows = [
        "| pid | iterations | count | segments (tile: k-steps, role) | fixup |",
        "|---|---|---|---|---|",
    ]
    for p in range(pr.num_programs):
        ss = sched.segs[p]
        desc = ", ".join(f"T{s.tile} {k_range(s)} {s.role}" for s in ss)
        fix = []
        for s in ss:
            if s.role == "reduce":
                fix.append(f"reads P[{', '.join(str(q) for q in s.peers)}] for T{s.tile}")
        if p in readers:
            fix.append(f"read by p{readers[p]}")
        rows.append(
            f"| p{p} | [{pr.start(p)}, {pr.start(p) + pr.length(p)}) | {pr.length(p)} | {desc} | "
            f"{'; '.join(fix)} |"
        )
    return "\n".join(rows)


def tile_table(sched):
    rows = ["| tile | reducer | peers (walk order) |", "|---|---|---|"]
    for s in reducers(sched):
        rows.append(f"| T{s.tile} | p{s.pid} | {', '.join(f'p{q}' for q in s.peers) or '-'} |")
    return "\n".join(rows)


def coincident_loads(sched, window):
    """Fraction of A / B k-slice loads that another program on the same XCD also loads
    within `window` k-steps of the same time. A lockstep proxy for L2 reuse."""
    pr = sched.pr
    keys = {"A": {}, "B": {}}
    for p, ev in sched.events.items():
        for e in ev:
            if e.kind != "mac":
                continue
            m, n = pr.tile_mn(e.seg.tile)
            for i in range(e.seg.end - e.seg.begin):
                t, k = e.t0 + i, e.seg.k0 + i
                keys["A"].setdefault((pr.xcd(p), m, k), []).append((t, p))
                keys["B"].setdefault((pr.xcd(p), n, k), []).append((t, p))
    out = {}
    for op, table in keys.items():
        shared = total = 0
        for loads in table.values():
            loads.sort()
            times = [t for t, _ in loads]
            for t, p in loads:
                lo = bisect.bisect_left(times, t - window)
                hi = bisect.bisect_right(times, t + window)
                total += 1
                shared += any(loads[j][1] != p for j in range(lo, hi))
        out[op] = shared / total
    return out


# ----------------------------------------------------------------------------- split policies
#
# Tile-aligned policies give every program one chunk inside one tile, so a policy is just the
# list of chunk lengths per tile. `c` is the cost, in k-steps, of storing one whole partial;
# `read` (default: c) is the cost of reading one.

# Measured on MI355X in experiments/streamk_costs (lane-contiguous partials, k-steps): storing
# a partial while every contributor publishes at once, storing one when publishes are spread
# out (the two-tile hybrid), and an owner reading one partial.
MEASURED = dict(burst_store=6.0, spread_store=1.5, read=2.0)

# The data-parallel last wave measured on v13 (v15_streamk_onetile/WORKLOG.md): the kernel
# time minus one whole wave's, in 1.21 us k-steps. Its few tiles run on otherwise idle CUs
# at about 0.76 us per k-step, so they cost far less than ipt contended k-steps.
MEASURED_DP_TAIL = {"4352x4096x8192": 80.0, "4352x4352x8192": 83.5}


def even_chunks(ipt, n):
    """Split ipt k-steps into n chunks whose lengths differ by at most one (longest first)."""
    return [ipt // n + (j < ipt % n) for j in range(n)]


def aligned_split(S, P, ipt):
    """(a) Tile-aligned: every tile gets P // S programs, the first P % S tiles one more."""
    return [even_chunks(ipt, min(P // S + (t < P % S), ipt)) for t in range(S)]


def uniform_split(S, s, ipt):
    """(b) Every tile split s ways; S * s programs used, the rest idle."""
    return [even_chunks(ipt, s) for _ in range(S)]


def serial_tile(chunks, c, read=None):
    """Critical path of one tile when chunks[0]'s program (the owner) reads the others in turn."""
    read = c if read is None else read
    t = chunks[0]
    for n in chunks[1:]:
        t = max(t, n + c) + read
    return t


def scatter_tile(chunks, c, read=None):
    """Critical path of one tile under reduce-scatter: every program stores its partial, then
    each sums 1/n of the tile from the other n - 1 partials."""
    read = c if read is None else read
    n = len(chunks)
    return chunks[0] if n == 1 else max(chunks) + c + (n - 1) / n * read


def best_uniform(S, P, ipt, c, read=None):
    """(b) The split count s <= P // S with the shortest serial critical path."""
    return min(
        (max(serial_tile(ch, c, read) for ch in uniform_split(S, s, ipt)), s)
        for s in range(1, min(P // S, ipt) + 1)
    )


def two_tile(pr):
    """The same GEMM with one full wave moved into stream-K (paper Fig. 3c)."""
    assert pr.full_tiles >= pr.num_programs, "two-tile needs at least one full wave to borrow"
    return Problem(
        pr.name + " two-tile", pr.num_programs, pr.streamk_tiles + pr.num_programs,
        pr.iters_per_tile, pr.full_tiles - pr.num_programs, pr.num_pid_m, pr.num_pid_n,
        pr.group_size_m, pr.pids_per_xcd,
    )


def longest_range(pr, unit):
    """Longest program range, in k-steps, when the partition hands out `unit` k-steps at a time."""
    units = pr.total_iters // unit
    return unit * (units // pr.num_programs + (units % pr.num_programs > 0))


def cost_columns():
    """(label, store cost when every contributor publishes at once, store cost when publishes
    are spread out, read cost) per column of the policies table."""
    m = MEASURED
    return [(f"c = {c}", c, c, c) for c in (1, 2, 4)] + [
        ("measured", m["burst_store"], m["spread_store"], m["read"])]


def policies_table(pr):
    """Critical path, in k-steps, measured from the start of the last full wave, so one-tile
    policies (last full wave + stream-K tail) and two-tile (both in stream-K) compare directly.

    All contributors of the one-tile policies publish at the same moment, so they pay the burst
    store cost; two-tile's publishes are spread over the range and pay the spread cost."""
    S, P, ipt = pr.streamk_tiles, pr.num_programs, pr.iters_per_tile
    costs = cost_columns()
    head = "| policy | programs | longest chunk | peers per tile | crossing programs | partials read | "
    head += " | ".join(label for label, *_ in costs) + " |"
    rows = [head, "|---" * (6 + len(costs)) + "|"]

    def row(name, programs, longest, peers, crossing, reads, paths):
        rows.append(f"| {name} | {programs} | {longest} | {peers} | {crossing} | {reads} | "
                    + " | ".join(paths) + " |")

    dp = [str(2 * ipt) for _ in costs]
    if pr.name in MEASURED_DP_TAIL:
        dp[-1] = f"{ipt + MEASURED_DP_TAIL[pr.name]:.1f}"
    rows.append(f"| data-parallel last wave | {S} | {ipt} | 0 | 0 | 0 | " + " | ".join(dp) + " |")
    for label, q, base, burst in (("one-tile stream-K (today)", pr, ipt, True),
                                  ("two-tile stream-K", None, 0, False)):
        if q is None:
            if pr.full_tiles < P:
                continue
            q = two_tile(pr)
        s0 = build_schedule(q, ORIGINAL)
        peers = [len(s.peers) for s in reducers(s0)]
        busy = sum(q.length(p) > 0 for p in range(P))
        paths = [f"{base + finish_time(build_schedule(q, ORIGINAL, sb if burst else ss, r)):.1f}"
                 for _, sb, ss, r in costs]
        row(label, busy, q.L + (q.R > 0), f"{min(peers)}-{max(peers)}",
            sum(len(ss) > 1 for ss in s0.segs.values()), sum(peers), paths)
    al = aligned_split(S, P, ipt)
    al_peers = sorted({len(ch) - 1 for ch in al})
    al_reads = sum(len(ch) - 1 for ch in al)
    al_longest = max(max(ch) for ch in al)
    row("(a) tile-aligned", sum(len(ch) for ch in al), al_longest,
        "-".join(map(str, (al_peers[0], al_peers[-1]))), 0, al_reads,
        [f"{ipt + max(serial_tile(ch, sb, r) for ch in al):.1f}" for _, sb, _, r in costs])
    best = [best_uniform(S, P, ipt, sb, r) for _, sb, _, r in costs]
    row("(b) fewer programs (best s)", "S x s", "ipt / s", "s - 1", 0, "S x (s - 1)",
        [f"{ipt + t:.1f} (s = {s}, {S * s} programs)" for t, s in best])
    row("(c) tile-aligned + reduce-scatter", sum(len(ch) for ch in al), al_longest,
        "-".join(map(str, (al_peers[0], al_peers[-1]))), 0, f"{al_reads} (as slices)",
        [f"{ipt + max(scatter_tile(ch, sb, r) for ch in al):.1f}" for _, sb, _, r in costs])
    return "\n".join(rows)


# ----------------------------------------------------------------------------- figures


def draw_schedule(ax, sched, title):
    pr = sched.pr
    h = 0.62
    for p, ev in sched.events.items():
        for e in ev:
            y = p - h / 2
            w = e.t1 - e.t0
            if e.kind == "mac":
                s = e.seg
                ax.add_patch(
                    Rectangle(
                        (e.t0, y), w, h, facecolor=TILE_COLOURS[s.tile % 8], edgecolor="black",
                        hatch="///" if s.role == "reduce" else None, lw=1.0,
                    )
                )
                ax.text(
                    e.t0 + w / 2, p, f"T{s.tile} {k_range(s)}", ha="center", va="center",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8),
                )
            elif e.kind == "store":
                ax.add_patch(Rectangle((e.t0, y), w, h, facecolor="#d95f02", edgecolor="black"))
                ax.text(e.t0 + w / 2, p, "st", ha="center", va="center", fontsize=7, color="white")
            elif e.kind == "wait":
                ax.add_patch(
                    Rectangle((e.t0, y), w, h, facecolor="#dddddd", edgecolor="grey", hatch="..")
                )
            elif e.kind == "read":
                ax.add_patch(Rectangle((e.t0, y), w, h, facecolor="#555555", edgecolor="black"))
                ax.text(
                    e.t0 + w / 2, p, f"+p{e.peer}", ha="center", va="center", fontsize=7, color="white"
                )
                up = e.peer > p
                ax.annotate(
                    "", xy=(e.t0 + 0.08, p + (h / 2 if up else -h / 2)),
                    xytext=(sched.publish[e.peer], e.peer),
                    arrowprops=dict(arrowstyle="->", color="#7a0177", lw=1.1, shrinkA=0, shrinkB=1,
                                    connectionstyle=f"arc3,rad={-0.15 if up else 0.15}"),
                )
    ax.set_xlim(0, finish_time(sched) + 0.25)
    ax.set_ylim(pr.num_programs - 0.4, -0.6)
    ax.set_yticks(range(pr.num_programs), [f"p{p}" for p in range(pr.num_programs)])
    ax.set_xlabel("time since the stream-K phase started (k-steps)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)


def schedule_legend(fig, sk_tiles):
    handles = [Patch(facecolor=TILE_COLOURS[t % 8], edgecolor="black", label=f"tile T{t}")
               for t in range(sk_tiles)]
    handles += [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="reducer's own segment"),
        Patch(facecolor="#d95f02", edgecolor="black", label="store partial to P[pid], set lock"),
        Patch(facecolor="#dddddd", edgecolor="grey", hatch="..", label="spin on peer's lock"),
        Patch(facecolor="#555555", edgecolor="black", label="read peer's partial (+pX)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)


def fig_iteration_space(pr, sched, path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ipt = pr.iters_per_tile
    for t in range(pr.streamk_tiles):
        ax.add_patch(Rectangle((t * ipt, 1.2), ipt, 0.8, facecolor=TILE_COLOURS[t % 8],
                               edgecolor="black", lw=1.5))
        ax.text(t * ipt + ipt / 2, 1.6, f"tile T{t}: k-steps 0-{ipt - 1}", ha="center",
                va="center", fontsize=10)
    for p, ss in sched.segs.items():
        for s in ss:
            ax.add_patch(Rectangle((s.begin, 0.1), s.end - s.begin, 0.8,
                                   facecolor=TILE_COLOURS[s.tile % 8], edgecolor="black", lw=0.8,
                                   hatch="///" if s.role == "reduce" else None))
        b, e = pr.start(p), pr.start(p) + pr.length(p)
        ax.add_patch(Rectangle((b, 0.1), e - b, 0.8, facecolor="none", edgecolor="black", lw=2.2))
        ax.text((b + e) / 2, 0.5, f"p{p}", ha="center", va="center", fontsize=11, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.text((b + e) / 2, -0.15, f"{pr.length(p)} steps", ha="center", va="center", fontsize=8)
    for t in range(pr.streamk_tiles + 1):
        ax.axvline(t * ipt, color="black", lw=1.5, ls="--", zorder=0)
    ax.set_xlim(-0.2, pr.total_iters + 0.2)
    ax.set_ylim(-0.4, 2.1)
    ax.set_xticks(range(pr.total_iters + 1))
    ax.set_yticks([0.5, 1.6], ["programs", "tiles"])
    ax.set_xlabel("stream-K iteration index  (tile x iters_per_tile + k-step)")
    ax.set_title(
        f"Toy: {pr.num_programs} programs, {pr.streamk_tiles} stream-K tiles x {ipt} k-steps = "
        f"{pr.total_iters} iterations; L = {pr.L}, R = {pr.R} (p0-p{pr.R - 1} get {pr.L + 1}). "
        "Hatched = the segment that starts at k-step 0 (the tile's owner).", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_schedule(sched, path, store_cost, read_cost):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    draw_schedule(ax, sched, f"Original schedule (k-step-0 program reduces). Store and read boxes "
                             f"use an illustrative cost of {store_cost} / {read_cost} k-steps.")
    schedule_legend(fig, sched.pr.streamk_tiles)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_alt(orig, rev, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True)
    draw_schedule(axes[0], orig, "Original: segments in increasing k; the program holding k-step 0 "
                                 "of a tile reduces it and walks peers upward (p+1, p+2, ...)")
    draw_schedule(axes[1], rev, "Reversed: segments processed last-to-first; the program holding the "
                                "last k-step reduces and walks peers downward (p-1, p-2, ...)")
    end = max(finish_time(orig), finish_time(rev)) + 0.25
    for ax in axes:
        ax.set_xlim(0, end)
    axes[0].set_xlabel("")
    schedule_legend(fig, orig.pr.streamk_tiles)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_tile_map(sched, path):
    pr = sched.pr
    ipt = pr.iters_per_tile
    rows = pr.streamk_tiles
    fig, ax = plt.subplots(figsize=(16, 1.0 + 0.36 * rows))
    xcds = set()
    for p, ss in sched.segs.items():
        for s in ss:
            w = s.end - s.begin
            xcds.add(pr.xcd(p))
            ax.add_patch(Rectangle((s.k0, s.tile - 0.42), w, 0.84,
                                   facecolor=XCD_COLOURS[pr.xcd(p) % 10], alpha=0.75,
                                   edgecolor="black", lw=1.6 if s.role == "reduce" else 0.6,
                                   hatch="///" if s.role == "reduce" else None))
            if w >= 5:
                ax.text(s.k0 + w / 2, s.tile, f"p{p}", ha="center", va="center", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    for s in reducers(sched):
        ax.text(ipt + 1.5, s.tile, f"p{s.pid} reads {len(s.peers)} partials", va="center", fontsize=8)
    labels = []
    for t in range(rows):
        m, n = pr.tile_mn(t)
        labels.append(f"tile {pr.full_tiles + t} (m{m}, n{n})")
    ax.set_yticks(range(rows), labels, fontsize=8)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlim(0, ipt + 22)
    ax.set_xticks(range(0, ipt + 1, 8))
    ax.set_xlabel("k-step within the tile")
    ax.set_title(
        f"{pr.name}: {pr.num_programs} programs, {rows} stream-K tiles x {ipt} k-steps; "
        f"L = {pr.L}, R = {pr.R}. Colour = XCD of the program ({pr.pids_per_xcd} logical pids per "
        "XCD); hatched = owner (k-step 0).", fontsize=10)
    handles = [Patch(facecolor=XCD_COLOURS[x % 10], alpha=0.75, label=f"XCD {x}") for x in sorted(xcds)]
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=len(handles), frameon=False,
              bbox_to_anchor=(1.0, -0.4 / max(rows, 1) * 4))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


BAR_STYLE = {
    "store": dict(facecolor="#d95f02", edgecolor="black"),
    "wait": dict(facecolor="#dddddd", edgecolor="grey", hatch=".."),
    "read": dict(facecolor="#555555", edgecolor="black"),
    "slice": dict(facecolor="#1b7837", edgecolor="black"),
}


def schedule_bars(sched):
    """(pid, t0, t1, kind, tile, label, is_reducer) bars and (pid, tile, k0, k1) strip pieces."""
    bars, strip = [], []
    for p, ev in sched.events.items():
        for e in ev:
            s = e.seg
            if e.kind == "mac":
                bars.append((p, e.t0, e.t1, "mac", s.tile, f"T{s.tile} {k_range(s)}", s.role == "reduce"))
                strip.append((p, s.tile, s.k0, s.k1))
            else:
                bars.append((p, e.t0, e.t1, e.kind, s.tile, f"+p{e.peer}" if e.kind == "read" else "", False))
    return bars, strip


def split_bars(tiles_chunks, c, scatter):
    """Bars for a tile-aligned policy with a serial owner fixup or a reduce-scatter fixup."""
    bars, strip, pid = [], [], 0
    for t, chunks in enumerate(tiles_chunks):
        pids = list(range(pid, pid + len(chunks)))
        pid += len(chunks)
        k = 0
        for q, n in zip(pids, chunks):
            bars.append((q, 0, n, "mac", t, f"T{t} k{k}-{k + n - 1}", q == pids[0] and not scatter))
            strip.append((q, t, k, k + n))
            k += n
        if len(chunks) == 1:
            continue
        if scatter:
            ready = max(chunks) + c
            for q, n in zip(pids, chunks):
                bars.append((q, n, n + c, "store", t, "", False))
                if ready > n + c:
                    bars.append((q, n + c, ready, "wait", t, "", False))
                bars.append((q, ready, ready + (len(chunks) - 1) / len(chunks) * c, "slice", t,
                             f"1/{len(chunks)}", False))
        else:
            t_owner = chunks[0]
            for q, n in zip(pids[1:], chunks[1:]):
                bars.append((q, n, n + c, "store", t, "", False))
                if n + c > t_owner:
                    bars.append((pids[0], t_owner, n + c, "wait", t, "", False))
                    t_owner = n + c
                bars.append((pids[0], t_owner, t_owner + c, "read", t, f"+p{q}", False))
                t_owner += c
    return bars, strip


def draw_bars(ax, bars, num_programs, xmax):
    h = 0.62
    for p, t0, t1, kind, tile, label, reducer in bars:
        w = t1 - t0
        if kind == "mac":
            ax.add_patch(Rectangle((t0, p - h / 2), w, h, facecolor=TILE_COLOURS[tile % 8],
                                   edgecolor="black", hatch="///" if reducer else None))
            ax.text(t0 + w / 2, p, label, ha="center", va="center", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        else:
            ax.add_patch(Rectangle((t0, p - h / 2), w, h, **BAR_STYLE[kind]))
            if label:
                ax.text(t0 + w / 2, p, label, ha="center", va="center", fontsize=6, color="white")
    ax.set_xlim(0, xmax)
    ax.set_ylim(num_programs - 0.4, -0.6)
    ax.set_yticks(range(num_programs), [f"p{p}" for p in range(num_programs)], fontsize=7)
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)


def draw_strip(ax, strip, S, ipt, num_programs):
    for p, tile, k0, k1 in strip:
        x = tile * ipt + k0
        ax.add_patch(Rectangle((x, 0), k1 - k0, 1, facecolor=TILE_COLOURS[tile % 8],
                               edgecolor="black", lw=1.2))
        ax.text(x + (k1 - k0) / 2, 0.5, f"p{p}", ha="center", va="center", fontsize=7 if k1 - k0 > 1 else 5.5)
    for t in range(S + 1):
        ax.axvline(t * ipt, color="black", lw=2, ls="--")
    for t in range(S):
        ax.text(t * ipt + ipt / 2, 1.25, f"tile T{t}", ha="center", fontsize=8)
    used = {p for p, *_ in strip}
    idle = [p for p in range(num_programs) if p not in used]
    if idle:
        ax.text(S * ipt / 2, -0.45, "idle: " + ", ".join(f"p{p}" for p in idle), ha="center",
                fontsize=8, color="#b2182b")
    ax.set_xlim(-0.3, S * ipt + 0.3)
    ax.set_ylim(-0.8, 1.6)
    ax.set_yticks([])
    ax.set_xticks(range(0, S * ipt + 1, 4))
    ax.tick_params(labelsize=7)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)


def fig_split_options(pr, c, path):
    S, P, ipt = pr.streamk_tiles, pr.num_programs, pr.iters_per_tile
    cur = build_schedule(pr, ORIGINAL, c, c)
    _, s = best_uniform(S, P, ipt, c)
    rows = [
        ("today: seams anywhere, owner collects serially", *schedule_bars(cur)),
        ("(a) tile-aligned: cut only inside tiles", *split_bars(aligned_split(S, P, ipt), c, False)),
        (f"(b) fewer programs: every tile cut {s} ways", *split_bars(uniform_split(S, s, ipt), c, False)),
        ("(c) tile-aligned + reduce-scatter: everyone collects a slice",
         *split_bars(aligned_split(S, P, ipt), c, True)),
    ]
    xmax = max(t1 for _, bars, _ in rows for _, _, t1, *_ in bars) + 0.5
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 3.1 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1, 1.35]))
    for (title, bars, strip), (ax_s, ax_t) in zip(rows, axes):
        crit = max(t1 for _, _, t1, *_ in bars)
        draw_strip(ax_s, strip, S, ipt, P)
        ax_s.set_title(f"{title}\nwhere the cuts go", fontsize=9, loc="left")
        draw_bars(ax_t, bars, P, xmax)
        ax_t.axvline(crit, color="#b2182b", lw=1.2, ls=":")
        ax_t.set_title(f"timeline: tail finishes at {crit:.1f} k-steps "
                       f"(data-parallel last wave: {ipt})", fontsize=9, loc="left")
    axes[-1][0].set_xlabel("stream-K iteration (tile x k-steps per tile + k-step)", fontsize=8)
    axes[-1][1].set_xlabel("time since the stream-K phase started (k-steps)", fontsize=8)
    handles = [Patch(facecolor=TILE_COLOURS[t], edgecolor="black", label=f"tile T{t}") for t in range(S)]
    handles += [
        Patch(facecolor="white", edgecolor="black", hatch="///", label="owner's own chunk"),
        Patch(**BAR_STYLE["store"], label="store partial"),
        Patch(**BAR_STYLE["wait"], label="wait for partials"),
        Patch(**BAR_STYLE["read"], label="owner reads a whole partial (+pX)"),
        Patch(**BAR_STYLE["slice"], label="sum 1/n of the tile from all partials"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"Three ways to split the stream-K tail: {P} programs, {S} tiles x {ipt} k-steps, "
                 f"storing or reading one partial costs {c} k-step(s)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_reduce_scatter(path, s=4):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    slice_colours = plt.cm.Dark2.colors
    for ax in (ax_l, ax_r):
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    # Serial: the owner reads each peer's whole partial in turn.
    ax_l.set_title("Today: the owner reads every partial, one after another", fontsize=10)
    ax_l.add_patch(Rectangle((6.5, 3.5), 2.5, 3, facecolor=TILE_COLOURS[0], edgecolor="black", hatch="///"))
    ax_l.text(7.75, 5, "p0 (owner)\nacc += P[1]\nacc += P[2]\nacc += P[3]\nwrite C", ha="center",
              va="center", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85))
    for i in range(1, s):
        y = 8.2 - (i - 1) * 2.8
        ax_l.add_patch(Rectangle((1, y - 1), 2, 1.8, facecolor="#fdd0a2", edgecolor="black"))
        ax_l.text(2, y - 0.1, f"P[{i}]\n256 KiB", ha="center", va="center", fontsize=8)
        ax_l.annotate("", xy=(6.5, 5), xytext=(3, y - 0.1),
                      arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555"))
        ax_l.text(4.6, (y - 0.1 + 5) / 2 + 0.25, f"step {i}", fontsize=8, color="#555555")
    ax_l.text(5, 0.6, f"{s - 1} x 256 KiB, serial, on one program", ha="center", fontsize=9)

    # Reduce-scatter: program j sums slice j of every partial.
    ax_r.set_title(f"Reduce-scatter: program j sums slice j of every partial, all {s} at once",
                   fontsize=10)
    h = 7.2 / s
    for b in range(s):
        x = 0.4 + b * 1.35
        ax_r.text(x + 0.5, 9.1, f"P[{b}]", ha="center", fontsize=9)
        for j in range(s):
            y = 8.6 - (j + 1) * h
            ax_r.add_patch(Rectangle((x, y), 1.0, h, facecolor=slice_colours[j], alpha=0.35 + 0.15 * (b == j),
                                     edgecolor="black"))
            ax_r.text(x + 0.5, y + h / 2, f"slice {j}", ha="center", va="center", fontsize=7)
            if b != j:
                ax_r.annotate("", xy=(7.0, 8.6 - (j + 0.5) * h), xytext=(x + 1.0, y + h / 2),
                              arrowprops=dict(arrowstyle="->", lw=0.9, color=slice_colours[j], alpha=0.8))
    for j in range(s):
        y = 8.6 - (j + 1) * h
        ax_r.add_patch(Rectangle((7.0, y + 0.1), 2.6, h - 0.2, facecolor=slice_colours[j], alpha=0.55,
                                 edgecolor="black"))
        ax_r.text(8.3, y + h / 2, f"p{j}: own slice {j}\n+ {s - 1} peers' slice {j}\n-> C rows of slice {j}",
                  ha="center", va="center", fontsize=7)
    ax_r.text(5, 0.6, f"each program reads {s - 1} slices of {256 // s} KiB ({s - 1}/{s} of a partial); "
                      "all programs work in parallel", ha="center", fontsize=9)
    fig.suptitle(f"One tile split {s} ways: collecting the partials", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_release_mechanism(path):
    import random

    rng = random.Random(3)
    waves, lines = 4, 7
    issue = {w: [0.15 * w + 0.26 * i for i in range(lines)] for w in range(waves)}
    arrive = {w: [t + rng.uniform(0.8, 3.6) for t in issue[w]] for w in range(waves)}
    last = max(max(a) for a in arrive.values())
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), sharex=True)
    rows = {**{f"wave {w}": w for w in range(waves)}, "memory side\n(arrivals)": waves + 0.3, "owner": waves + 1.5}

    def base(ax, title):
        ax.set_yticks(list(rows.values()), list(rows.keys()), fontsize=8)
        ax.set_ylim(waves + 2.2, -0.7)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", color="#eeeeee")
        for w in range(waves):
            ax.add_patch(Rectangle((issue[w][0], w - 0.3), issue[w][-1] - issue[w][0] + 0.1, 0.6,
                                   facecolor="#9ecae1", edgecolor="black"))
            ax.text(issue[w][0] + 0.05, w, "issue P stores", va="center", fontsize=7)
            for a in arrive[w]:
                ax.plot([a, a], [waves + 0.05, waves + 0.55], color=plt.cm.tab10.colors[w], lw=2)

    # Issue order only.
    ax = axes[0]
    base(ax, "Issue order only (P stores, barrier, flag store): the flag can overtake the partial")
    barrier = max(i[-1] for i in issue.values()) + 0.2
    ax.axvline(barrier, ymin=0.55, color="black", ls=":", lw=1.2)
    ax.text(barrier + 0.05, -0.45, "barrier", fontsize=7)
    flag_issue, flag_arrive = barrier + 0.1, barrier + 0.8
    ax.plot(flag_issue, 0, marker=">", color="#b2182b", ms=9)
    ax.text(flag_issue + 0.12, 0.3, "flag store", fontsize=7, color="#b2182b")
    ax.plot(flag_arrive, waves + 0.3, marker="*", color="#b2182b", ms=14)
    seen = flag_arrive + 0.2
    ax.add_patch(Rectangle((0, waves + 1.2), seen, 0.6, **BAR_STYLE["wait"]))
    ax.text(0.1, waves + 1.5, "spin on flag", va="center", fontsize=7)
    ax.add_patch(Rectangle((seen, waves + 1.2), 1.6, 0.6, facecolor="#555555", edgecolor="black"))
    ax.text(seen + 0.8, waves + 1.5, "read P", va="center", ha="center", fontsize=7, color="white")
    ax.add_patch(Rectangle((seen, waves - 0.05), last - seen + 0.05, 0.65, facecolor="#b2182b", alpha=0.15))
    ax.text((seen + last) / 2, waves + 0.95, "P lines still in flight when the owner reads: stale data",
            ha="center", fontsize=8, color="#b2182b")

    # Release.
    ax = axes[1]
    base(ax, "Release (each wave waits for its own stores to be acknowledged, then barrier, then flag)")
    for w in range(waves):
        ax.add_patch(Rectangle((issue[w][-1] + 0.1, w - 0.3), max(arrive[w]) - issue[w][-1] - 0.1, 0.6,
                               **BAR_STYLE["wait"]))
        ax.text(max(arrive[w]) - 0.05, w, "wait vmcnt(0)", ha="right", va="center", fontsize=7)
    barrier = last + 0.1
    ax.axvline(barrier, ymin=0.55, color="black", ls=":", lw=1.2)
    ax.text(barrier + 0.05, -0.45, "barrier", fontsize=7)
    flag_issue, flag_arrive = barrier + 0.1, barrier + 0.8
    ax.plot(flag_issue, 0, marker=">", color="#1b7837", ms=9)
    ax.text(flag_issue + 0.12, 0.3, "flag store (release)", fontsize=7, color="#1b7837")
    ax.plot(flag_arrive, waves + 0.3, marker="*", color="#1b7837", ms=14)
    seen = flag_arrive + 0.2
    ax.add_patch(Rectangle((0, waves + 1.2), seen, 0.6, **BAR_STYLE["wait"]))
    ax.text(0.1, waves + 1.5, "spin on flag (acquire)", va="center", fontsize=7)
    ax.add_patch(Rectangle((seen, waves + 1.2), 1.6, 0.6, facecolor="#555555", edgecolor="black"))
    ax.text(seen + 0.8, waves + 1.5, "read P", va="center", ha="center", fontsize=7, color="white")
    ax.text((last + seen) / 2, waves + 0.95, "every P line has landed before the flag", ha="center",
            fontsize=8, color="#1b7837")
    ax.set_xlabel("time (schematic)")
    ax.set_xlim(0, seen + 2)
    fig.suptitle("Contributor workgroup (4 waves) publishing its partial to an owner on another CU. "
                 "Ticks: when each P line (colour = wave) and the flag (star) reach the memory side.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_release_crossing(path, c=2.0, refill=1.0):
    tail, head, chunk, read = 6.0, 5.0, 11.0, 2.0
    fig, ax = plt.subplots(figsize=(13, 4.6))
    h = 0.55

    def box(y, t0, t1, label, fc, hatch=None, colour="black", **kw):
        ax.add_patch(Rectangle((t0, y - h / 2), t1 - t0, h, facecolor=fc, edgecolor="black", hatch=hatch, **kw))
        ax.text((t0 + t1) / 2, y, label, ha="center", va="center", fontsize=7, color=colour,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75) if colour == "black" else None)

    def flag(y, t, label):
        ax.plot(t, y - h / 2 - 0.08, marker="v", color="#b2182b", ms=9)
        ax.text(t, y - h / 2 - 0.22, label, ha="center", fontsize=7, color="#b2182b")

    # Non-crossing contributor.
    y = 0
    box(y, 0, chunk, "compute its only chunk (T0)", TILE_COLOURS[0])
    box(y, chunk, chunk + c, "store P, wait acks", "#fdd0a2")
    flag(y, chunk + c, "flag")
    ax.text(chunk + c + 0.2, y, "exit: the wait cost nothing", va="center", fontsize=8)

    # Crossing, simple release.
    y = 1.4
    box(y, 0, tail, "tail of T0 (partial)", TILE_COLOURS[0])
    ax.plot(tail - 0.4, y + h / 2 + 0.08, marker="^", color="#2166ac", ms=8)
    ax.text(tail - 0.4, y + h / 2 + 0.3, "prefetch T1 k0-1", ha="center", fontsize=7, color="#2166ac")
    box(y, tail, tail + c, "store P + vmcnt(0) stall", "#dddddd", hatch="..")
    flag(y, tail + c, "flag")
    box(y, tail + c, tail + c + refill, "refill", "#f7f7f7", hatch="xx")
    t = tail + c + refill
    box(y, t, t + head, "head of T1 (owner)", TILE_COLOURS[1], hatch="///")
    box(y, t + head, t + head + read, "+peer", "#555555", colour="white")
    simple_end = t + head + read

    # Crossing, targeted ordering.
    y = 2.8
    box(y, 0, tail, "tail of T0 (partial)", TILE_COLOURS[0])
    box(y, tail, tail + 0.3, "", "#fdd0a2")
    ax.text(tail + 0.15, y + h / 2 + 0.3, "stores, then prefetch", ha="center", fontsize=7, color="#2166ac")
    box(y, tail + 0.3, tail + 0.3 + head, "head of T1 (owner)", TILE_COLOURS[1], hatch="///")
    flag(y, tail + c, "vmcnt(N), flag")
    box(y, tail + 0.3 + head, tail + 0.3 + head + read, "+peer", "#555555", colour="white")
    targeted_end = tail + 0.3 + head + read

    ax.annotate("", xy=(simple_end, 3.35), xytext=(targeted_end, 3.35),
                arrowprops=dict(arrowstyle="<->", color="#b2182b"))
    ax.text((simple_end + targeted_end) / 2, 3.6, f"T1 finishes ~{simple_end - targeted_end:.1f} later\n"
            "(on T1's critical path)", ha="center", fontsize=8, color="#b2182b")
    ax.set_yticks([0, 1.4, 2.8], ["non-crossing\ncontributor", "crossing program,\nsimple release",
                                  "crossing program,\nstores before prefetch"], fontsize=8)
    ax.set_ylim(4.0, -0.8)
    ax.set_xlim(0, simple_end + 1)
    ax.set_xlabel("time (k-steps, schematic)")
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_title(f"Where the release cost lands. Illustrative costs: store and acknowledge a partial = {c:g}, "
                 f"pipeline refill = {refill:g}, read one partial = {read:g} k-steps", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

TOY = Problem("toy", num_programs=7, streamk_tiles=3, iters_per_tile=8)
SPLIT_TOY = Problem("split toy", num_programs=11, streamk_tiles=2, iters_per_tile=16)
SINGLE = Problem("257 tiles (256 + 1)", num_programs=256, streamk_tiles=1, iters_per_tile=128,
                 full_tiles=256, num_pid_m=257, num_pid_n=1, pids_per_xcd=32)
REAL = [Problem.from_shape(4352, 4096, 8192), Problem.from_shape(4352, 4352, 8192)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
    ap.add_argument("--store-cost", type=float, default=0.5, help="k-steps to store one partial")
    ap.add_argument("--read-cost", type=float, default=0.5, help="k-steps to read one partial")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for pr in [TOY, SPLIT_TOY, SINGLE] + REAL + [two_tile(q) for q in [SINGLE] + REAL]:
        for variant in (ORIGINAL, REVERSED):
            self_check(build_schedule(pr, variant, args.store_cost, args.read_cost))
    print("self-checks passed\n")

    toy = build_schedule(TOY, ORIGINAL, args.store_cost, args.read_cost)
    toy_rev = build_schedule(TOY, REVERSED, args.store_cost, args.read_cost)
    print("## Toy, original\n")
    print(program_table(toy), "\n")
    print(tile_table(toy), "\n")
    print("## Toy, reversed\n")
    print(program_table(toy_rev), "\n")
    print(tile_table(toy_rev), "\n")

    fig_iteration_space(TOY, toy, os.path.join(args.out, "iteration_space_toy.png"))
    fig_schedule(toy, os.path.join(args.out, "schedule_toy.png"), args.store_cost, args.read_cost)
    fig_alt(toy, toy_rev, os.path.join(args.out, "alt_schedule_toy.png"))

    print("## Real shapes (256x256x64 tiles, 256 programs)\n")
    print("| shape | tiles | STREAMK_TILES | L | R | peers per owner | crossing programs "
          "| coincident A / B loads, original | reversed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for pr in REAL:
        orig = build_schedule(pr, ORIGINAL, args.store_cost, args.read_cost)
        rev = build_schedule(pr, REVERSED, args.store_cost, args.read_cost)
        fig_tile_map(orig, os.path.join(args.out, f"schedule_{pr.name}.png"))
        peers = sorted({len(s.peers) for s in reducers(orig)})
        crossing = sum(len(ss) > 1 for ss in orig.segs.values())
        cells = []
        for s in (orig, rev):
            c = [coincident_loads(s, w) for w in (0, 4)]
            cells.append(" ; ".join(f"w={w}: {x['A']:.0%} / {x['B']:.0%}" for w, x in zip((0, 4), c)))
        print(f"| {pr.name} | {pr.full_tiles + pr.streamk_tiles} | {pr.streamk_tiles} | {pr.L} | "
              f"{pr.R} | {peers[0]}-{peers[-1]} | {crossing} | {cells[0]} | {cells[1]} |")

    print("\n## Split policies: critical path from the start of the last full wave (k-steps)\n")
    for pr in REAL + [SINGLE]:
        plural = "s" if pr.streamk_tiles > 1 else ""
        print(f"### {pr.name}: {pr.streamk_tiles} leftover tile{plural} on {pr.num_programs} programs\n")
        print(policies_table(pr), "\n")

    print("## Partition granularity: longest program range (k-steps)\n")
    print("| shape | policy | 1-k-step units | 2-k-step units |")
    print("|---|---|---|---|")
    for pr in REAL:
        for label, q in (("one-tile", pr), ("two-tile", two_tile(pr))):
            print(f"| {pr.name} | {label} | {longest_range(q, 1)} | {longest_range(q, 2)} |")

    fig_split_options(SPLIT_TOY, 1.0, os.path.join(args.out, "split_options_toy.png"))
    fig_reduce_scatter(os.path.join(args.out, "reduce_scatter.png"))
    fig_release_mechanism(os.path.join(args.out, "release_mechanism.png"))
    fig_release_crossing(os.path.join(args.out, "release_crossing.png"))
    print(f"\nimages written to {args.out}")


if __name__ == "__main__":
    main()
