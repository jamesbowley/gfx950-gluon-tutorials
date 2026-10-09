"""Sweep of slice_kernel's variants (called from bench.py --sweep slice)."""

import re
import statistics

import slice_kernel as SK  # imports triton before torch
import torch


class Bufs:

    def __init__(self, device="cuda"):
        self.p = torch.zeros(SK.NUM_PROGRAMS * SK.SLOT, device=device, dtype=torch.float32)
        self.c = torch.zeros((4096, SK.C_N), device=device, dtype=torch.bfloat16)
        self.cnt = torch.zeros(SK.NUM_PROGRAMS * 32, device=device, dtype=torch.int32)
        self.gcnt = torch.zeros(1, device=device, dtype=torch.int32)
        self.rec = torch.zeros(SK.NUM_PROGRAMS * SK.REC, device=device, dtype=torch.int64)
        self.flush = torch.ones(1024 * 2**20 // 4, device=device, dtype=torch.float32)


def launch(b, W, n, variant):
    return SK.rs_kernel[(SK.NUM_PROGRAMS, )](b.p, b.c, b.cnt, b.gcnt, b.rec, W, N=n, VARIANT=SK.VARIANTS[variant],
                                            num_warps=4)


def wrong(b, W, n):
    want = SK.expected_tile(n).to(b.c.device)
    bad = 0
    for g in range(SK.groups(W, n)):
        t = b.c[(g // 16) * 256:(g // 16 + 1) * 256, (g % 16) * 256:(g % 16 + 1) * 256]
        bad += int((t != want).sum())
    return bad


def run(b, W, n, variant, launches=30):
    span, pub, syn, rd, cst = [], [], [], [], []
    bad = 0
    for i in range(launches + 1):
        b.c.zero_()
        b.flush.sum()
        b.cnt.zero_()
        b.gcnt.zero_()
        launch(b, W, n, variant)
        torch.cuda.synchronize()
        if i == 0:
            continue
        bad += wrong(b, W, n)
        r = b.rec.view(SK.NUM_PROGRAMS, SK.REC).cpu()
        act = r[:, 5] != 0
        t0 = r[act, 0].min()
        span.append(float(r[act, 4].max() - t0) / 100)
        for gv in r[act, 6].unique().tolist():
            mem = r[act & (r[:, 6] == gv)]
            crit = mem[mem[:, 4].argmax()]
            pe = float(mem[:, 1].max() - t0) / 100
            t2, t3, t4 = (float(crit[k] - t0) / 100 for k in (2, 3, 4))
            pub.append(pe)
            syn.append(max(0.0, t2 - pe))
            rd.append(t3 - max(t2, pe))
            cst.append(t4 - t3)
    med = statistics.median
    return med(span), med(pub), med(syn), med(rd), med(cst), bad


def asm_stats(k):
    asm = k.asm["amdgcn"]
    ops = re.findall(r"^\s*((?:buffer|global)_(?:load|store)_\w+|ds_(?:read|write)\w*|scratch_\w+)", asm, re.M)
    vgpr = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm)
    barriers = len(re.findall(r"^\s*s_barrier", asm, re.M))
    return {m: ops.count(m) for m in sorted(set(ops))}, barriers, int(vgpr.group(1)), int(spill.group(1)) if spill \
        else 0


def sweep_slice():
    b = Bufs()
    print("## slice\n")
    print("Synthetic one-tile reduce-scatter (slice_kernel.py): W programs, n per 256x256 tile, every tile on one "
          "XCD; `.wt` fp32 partials, drain, relaxed gpu counter, `.cv` loads, bf16 C. Variants: v20 (L3 slice, "
          "8 B per lane C store), v20_old_cvt (L3, then convert to columns-first L3: v21's dropped attempt), "
          "v20_store2d (L3, reshaped to the slice's band and converted to gStoreLayoutC), mfma_read (slice loaded "
          "in the MFMA layout, owner-style convert to gStoreLayoutC), writer_cvt (the writer converts to 4 fp32 "
          "per lane and stores row-major; the reader loads 8 per lane and stores C with no conversion), "
          "writer_cvt8 (the writer converts to gStoreLayoutC itself: 32 bytes per lane, so each `.wt` store "
          "instruction leaves 16-byte holes). Medians over 30 cold "
          "launches (1 GiB read before each) and over groups of each group's critical path; total is the "
          "makespan. Every C element checked.\n")
    print("| variant | setup | n | W | total us | publish us | sync us | read us | C store us | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for setup in ("tiles 16", "CUs 256"):
        for n in (2, 4, 8, 16):
            W = 16 * n if setup == "tiles 16" else 256
            for variant in SK.VARIANTS:
                tot, pub, syn, rd, cst, bad = run(b, W, n, variant)
                print(f"| {variant} | {setup} | {n} | {W} | {tot:.2f} | {pub:.2f} | {syn:.2f} | {rd:.2f} | "
                      f"{cst:.2f} | {bad} |", flush=True)
                if (variant, n) not in stats:
                    stats[(variant, n)] = asm_stats(launch(b, W, n, variant))
    print("\n### Compiled code\n")
    print("Whole kernel, per variant and n: memory and LDS instructions by width, s_barrier count (including "
          "the stamps' and the drain's), VGPRs and spills.\n")
    print("| variant | n | instructions | s_barrier | VGPRs | spills |")
    print("|---|---|---|---|---|---|")
    for (variant, n), (ops, bars, vgpr, spill) in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        txt = ", ".join(f"{v} {k}" for k, v in ops.items())
        print(f"| {variant} | {n} | {txt} | {bars} | {vgpr} | {spill} |")
