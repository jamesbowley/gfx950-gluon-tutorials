"""Sweeps of split_kernel (called from bench.py --sweep splits bf16)."""

import re
import statistics

import split_kernel as SP  # imports triton before torch
import torch


class Bufs:

    def __init__(self, device="cuda"):
        self.p = torch.zeros(SP.NUM_PROGRAMS * SP.SLOT, device=device, dtype=torch.float32)
        self.p16 = self.p.view(torch.bfloat16)
        self.c = torch.zeros((4096, SP.C_N), device=device, dtype=torch.bfloat16)
        self.cnt = torch.zeros(SP.NUM_PROGRAMS * 32, device=device, dtype=torch.int32)
        self.gcnt = torch.zeros(1, device=device, dtype=torch.int32)
        self.rec = torch.zeros(SP.NUM_PROGRAMS * SP.REC, device=device, dtype=torch.int64)
        self.flush = torch.ones(1024 * 2**20 // 4, device=device, dtype=torch.float32)


def launch(b, W, n, r, store, part):
    return SP.split_kernel[(SP.NUM_PROGRAMS, )](b.p, b.p16, b.c, b.cnt, b.gcnt, b.rec, W, N=n, R=r, G=SP.group_size(r),
                                                STORE=SP.STORES[store], PART=SP.PARTS[part], num_warps=4)


def wrong(b, W, n):
    want = SP.expected_tile(n).to(b.c.device)
    bad = 0
    for g in range(SP.groups(W, n)):
        t = b.c[(g // 16) * 256:(g // 16 + 1) * 256, (g % 16) * 256:(g % 16 + 1) * 256]
        bad += int((t != want).sum())
    return bad


def run(b, W, n, r, store, part, launches=30):
    span, pub, syn, sl = [], [], [], []
    bad = 0
    for i in range(launches + 1):
        b.c.zero_()
        b.flush.sum()
        b.cnt.zero_()
        b.gcnt.zero_()
        launch(b, W, n, r, store, part)
        torch.cuda.synchronize()
        if i == 0:
            continue
        bad += wrong(b, W, n)
        rec = b.rec.view(SP.NUM_PROGRAMS, SP.REC).cpu()
        act = (rec[:, 4] & 1) != 0
        t0 = rec[act, 0].min()
        span.append(float(rec[act, 3].max() - t0) / 100)
        for gv in rec[act, 5].unique().tolist():
            mem = rec[act & (rec[:, 5] == gv)]
            rd = mem[(mem[:, 4] & 2) != 0]
            crit = rd[rd[:, 3].argmax()]
            pe = float(mem[:, 1].max() - t0) / 100
            t2, t3 = float(crit[2] - t0) / 100, float(crit[3] - t0) / 100
            pub.append(pe)
            syn.append(max(0.0, t2 - pe))
            sl.append(t3 - max(t2, pe))
    med = statistics.median
    return med(span), med(pub), med(syn), med(sl), bad


def asm_ops(k, pattern):
    return len(re.findall(pattern, k.asm["amdgcn"], re.M))


def asm_summary(k):
    asm = k.asm["amdgcn"]
    ops = re.findall(r"^\s*(buffer_(?:load|store)_\w+)", asm, re.M)
    vgpr = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm)
    return (", ".join(f"{ops.count(m)} {m}" for m in sorted(set(ops))), int(vgpr.group(1)),
            int(spill.group(1)) if spill else 0)


def pow2_floor(n):
    return 1 << (n.bit_length() - 1)


def sweep_splits():
    b = Bufs()
    print("## splits\n")
    print("Split counts that do not divide 64, and 32 / 64. N programs per tile publish fp32 partials; the first "
          "R read: reader j sums blocks [64 j / R, 64 (j + 1) / R) of all N partials in groups of G blocks, then "
          "stores C. 'plain' stores from v20's L3 layout (8 B per lane, 4 rows per 4-lane group); 'block' "
          "converts each group to 8 bf16 per lane along a block row; 'band' (power-of-two R only) reshapes the "
          "group to its 2-D band and converts to a row layout of 16 B per lane (8 B for a 32 x 32 block). "
          "N <= 32 on one XCD per tile, N = 64 spread. 'slice' is the critical reader's read + C store. Medians "
          "over 30 cold launches and over groups; every C element checked.\n")
    print("| setup | N | W | R | G | store | total us | publish us | sync us | slice us | wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    cases = []
    for n in (3, 5, 6, 10, 12):
        r2 = pow2_floor(n)
        cases += [(n, n, "plain"), (n, n, "block"), (n, r2, "plain"), (n, r2, "band")]
    for n in (16, 32, 64):
        cases += [(n, n, "plain"), (n, n, "band")]
    for setup in ("tiles 16", "CUs 256"):
        for n, r, store in cases:
            W = SP.setup_W(setup, n)
            if W is None:
                continue
            tot, pub, syn, sl, bad = run(b, W, n, r, store, "fp32")
            print(f"| {setup} | {n} | {W} | {r} | {SP.group_size(r)} | {store} | {tot:.2f} | {pub:.2f} | {syn:.2f} | "
                  f"{sl:.2f} | {bad} |", flush=True)


def sweep_bf16():
    b = Bufs()
    print("\n## bf16\n")
    print("Partial type and map, power-of-two n (R = n), the slice loaded in the MFMA layout and stored like v20's "
          "owner (STORE 'mfma'). fp32: v20's lane-contiguous map (16 B per lane). bf16: the same map with bf16 "
          "values (8 B per lane). bf16_pairs: a lane's first 8 registers (two blocks 32 columns apart) as 16 "
          "contiguous bytes. Medians over 30 cold launches; every C element checked.\n")
    print("| setup | n | W | partials | total us | publish us | sync us | slice us | wrong |")
    print("|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for setup in ("tiles 16", "CUs 256"):
        for n in (2, 4, 8, 16):
            W = SP.setup_W(setup, n)
            for part in ("fp32", "bf16", "bf16_pairs"):
                tot, pub, syn, sl, bad = run(b, W, n, n, "mfma", part)
                print(f"| {setup} | {n} | {W} | {part} | {tot:.2f} | {pub:.2f} | {syn:.2f} | {sl:.2f} | {bad} |",
                      flush=True)
                if (part, n) not in stats:
                    stats[(part, n)] = asm_summary(launch(b, W, n, n, "mfma", part))
    print("\n### Compiled code (bf16)\n")
    print("| partials | n | buffer loads and stores | VGPRs | spills |")
    print("|---|---|---|---|---|")
    for (part, n), (txt, vgpr, spill) in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"| {part} | {n} | {txt} | {vgpr} | {spill} |")
