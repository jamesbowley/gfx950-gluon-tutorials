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
"""Figures for PARTITION_CAMPING.md: a toy channel model and the measured placement data.

    python make_figures.py        # writes images/*.png from results/*.txt

The toy model is the textbook one: NCH memory channels, consecutive GRANULE-byte blocks of
the address space dealt round-robin to the channels, channel = (addr // GRANULE) % NCH. Real
GPUs also hash higher address bits into the channel; the measured panels show what that does.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
IMG = os.path.join(HERE, "images")
NCH, GRANULE, ROWS = 16, 256, 256
ROW = 16384  # B's row: 8192 bf16 elements


def channel(addr):
    return (addr // GRANULE) % NCH


def loads(stride, rows=ROWS):
    counts = [0] * NCH
    for r in range(rows):
        counts[channel(r * stride)] += 1
    return counts


def rows_of(path, cols):
    out = []
    for line in open(os.path.join(RES, path)):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > max(cols) and parts[1].replace(".", "").isdigit():
            out.append([float(parts[c]) for c in cols])
    return out


def fig_toy_histograms():
    cases = [(ROW, "16384 B (no pad)"), (ROW + 2048, "16384 + 2048 B"), (ROW + 512, "16384 + 512 B"),
             (ROW + 256, "16384 + 256 B")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (s, title) in zip(axes, cases):
        c = loads(s)
        used = sum(x > 0 for x in c)
        ax.bar(range(NCH), c, color="tab:red" if max(c) > ROWS // NCH else "tab:green")
        ax.axhline(ROWS / NCH, color="k", ls="--", lw=0.8)
        ax.set_title(f"row stride {title}\n{used} of {NCH} channels, busiest {max(c)} requests", fontsize=9)
        ax.set_xlabel("channel")
    axes[0].set_ylabel(f"requests (of {ROWS})")
    fig.suptitle(f"Toy model: {ROWS} rows read at the same column, {NCH} channels x {GRANULE} B interleave "
                 f"(dashed: even share)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "toy_channel_histograms.png"), dpi=130)
    plt.close(fig)


def fig_pad_sweep():
    pads_b = list(range(0, 2048 + 1, 32))
    busiest = [max(loads(ROW + p)) for p in pads_b]
    meas = []
    for line in open(os.path.join(RES, "e8_bpad_sweep.txt")):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) > 5 and parts[0].isdigit():
            meas.append((int(parts[0]), float(parts[1]), float(parts[4])))
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax0.plot(pads_b, busiest, "o-", ms=3, color="tab:red")
    ax0.axhline(ROWS / NCH, color="k", ls="--", lw=0.8)
    ax0.set_ylabel(f"toy: busiest channel\n(requests of {ROWS}; even = {ROWS // NCH})")
    ax0.set_yscale("log", base=2)
    ax0.set_title("Toy model against measurement: padding B's 16 KB rows", fontsize=10)
    for off, colour in ((64, "tab:red"), (128, "tab:orange"), (232, "tab:green")):
        pts = sorted((p * 2, us) for p, o, us in meas if o == off)
        ax1.plot([x for x, _ in pts], [y for _, y in pts], "o-", ms=4, color=colour,
                 label=f"B at {off} MB of the buffer")
    ax1.set_xlabel("padding per row of B (bytes)")
    ax1.set_ylabel("measured v13, 256x32768x8192 (us)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax0.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "pad_sweep.png"), dpi=130)
    plt.close(fig)


def fig_b_offset():
    unpadded = rows_of("e8_bsweep_v13.txt", [1, 4])
    padded = {}
    for line in open(os.path.join(RES, "e8_bpad.txt")):
        parts = [p.strip() for p in line.split("|")]
        if line.startswith("pad 256"):
            padded[float(parts[1])] = float(parts[4])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot([x for x, _ in unpadded], [y for _, y in unpadded], "o-", ms=4, color="tab:red",
            label="B rows 16384 B (unpadded)")
    ax.plot(sorted(padded), [padded[k] for k in sorted(padded)], "s-", ms=5, color="tab:green",
            label="B rows 16384 + 512 B (padded)")
    for x in (64, 128, 192, 256):
        ax.axvline(x, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("B's offset in a 1.5 GB buffer (MB)")
    ax.set_ylabel("v13 time per GEMM (us)")
    ax.set_title("256x32768x8192: where B lands decides the time, unless its rows are padded", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "b_offset_skinny.png"), dpi=130)
    plt.close(fig)


def fig_c_offset():
    sweep = rows_of("e2_c_sweep.txt", [2, 4])
    padded = {}
    for line in open(os.path.join(RES, "e8_cpad_4096.txt")):
        parts = [p.strip() for p in line.split("|")]
        if line.startswith("c-pad 256"):
            padded[float(parts[2])] = float(parts[4])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k in range(2):
        ax.axvspan(256 * k + 96, 256 * k + 256, color="tab:red", alpha=0.08)
    ax.plot([x for x, _ in sweep], [y for _, y in sweep], ".-", ms=4, color="tab:red", label="C rows 8192 B (unpadded)")
    ax.plot(sorted(padded), [padded[k] for k in sorted(padded)], "s", ms=6, color="tab:green",
            label="C rows 8192 + 512 B (padded)")
    ax.set_xlabel("C's offset in a 1 GB buffer (MB); shaded: the 32 MB C overlaps the second 128 MB of a 256 MB period")
    ax.set_ylabel("v9 time per GEMM (us)")
    ax.set_title("4096x4096x1024, back-to-back: C's offset gives two speeds, padding gives one", fontsize=10)
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "c_offset_4096.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(IMG, exist_ok=True)
    fig_toy_histograms()
    fig_pad_sweep()
    fig_b_offset()
    fig_c_offset()
    for s, label in ((ROW, "16384"), (ROW + 128, "16384+128"), (ROW + 256, "16384+256"), (ROW + 512, "16384+512"),
                     (ROW + 2048, "16384+2048")):
        c = loads(s)
        print(f"stride {label} B: channels used {sum(x > 0 for x in c)}, busiest {max(c)}")


if __name__ == "__main__":
    main()
