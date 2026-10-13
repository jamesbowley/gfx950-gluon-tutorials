"""Layout conversions of v20's reduce-scatter slice, and reduce-scatter slice variants (see README.md).

    HIP_VISIBLE_DEVICES=1 python bench.py --sweep probe > results/probe.md
    HIP_VISIBLE_DEVICES=1 python bench.py --sweep slice > results/slice.md
"""

import argparse
import os
import re
import statistics
import time

import kernel as K  # imports triton before torch
import torch


def asm_stats(k, lo="s_memrealtime", hi="s_memrealtime"):
    """LDS instructions, barriers and permutes between the first and the last `lo` / `hi` marker (the
    timed loop), and the kernel's VGPRs, spills and LDS bytes."""
    asm = k.asm["amdgcn"]
    i, j = asm.find(lo), asm.rfind(hi)
    body = asm[i:j] if 0 <= i < j else asm
    ops = re.findall(r"^\s*(ds_\w+|v_perm\w*|v_permlane\w*|scratch_\w+)", body, re.M)
    vgpr = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    spill = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm)
    return {
        "ops": {m: ops.count(m) for m in sorted(set(ops))},
        "barriers": max(len(re.findall(r"^\s*s_barrier", body, re.M)) - 1, 0),
        "vgpr": int(vgpr.group(1)) if vgpr else -1,
        "spill": int(spill.group(1)) if spill else -1,
        "lds": k.metadata.shared,
    }


def fmt_ops(ops):
    return ", ".join(f"{v} {k}" for k, v in ops.items()) or "none"


def sweep_probe(reps=1000, launches=20):
    out = torch.zeros(128 * K.C_N, device="cuda", dtype=torch.bfloat16)
    rec = torch.zeros(2, device="cuda", dtype=torch.int64)
    print("## probe\n")
    print("One 128x128 quadrant of bf16 values in a source layout, converted REPS = "
          f"{reps} times in a loop on one CU (4 warps); 'ns per conversion' is the loop time per iteration "
          f"(median of {launches} launches) minus the loop with no conversion in the same source layout. "
          "Instruction counts are the loop body's; 'LDS rounds' is its s_barrier count / 2. 'checked': the "
          "converted values stored row-major equal every (row, col). Each loop iteration also does an add, a "
          "bf16 convert and a select over the whole quadrant to keep every conversion live; that is the "
          "no-conversion baseline.\n")
    print("| case | conversion | trivial | instructions in the loop | LDS rounds | LDS bytes | ns per iteration | "
          "ns per conversion | VGPRs | spills | checked |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    base = {}
    for name, (case, desc) in K.CASES.items():
        try:
            k = K.probe_kernel[(1, )](out, rec, reps, CASE=case, num_warps=4)
        except Exception as e:  # assert_trivial fails at compile time
            print(f"| {name} | {desc} | compile error: {str(e).splitlines()[-1][:120]} | | | | | | | | |")
            continue
        torch.cuda.synchronize()
        ts = []
        for _ in range(launches):
            K.probe_kernel[(1, )](out, rec, reps, CASE=case, num_warps=4)
            torch.cuda.synchronize()
            r = rec.cpu()
            ts.append(float(r[1] - r[0]) * 10 / reps)
        t = statistics.median(ts)
        out.zero_()
        K.probe_kernel[(1, )](out, rec, 1, CASE=case, num_warps=4)
        torch.cuda.synchronize()
        ok = torch.equal(out.view(128, K.C_N)[:, :128].cpu(), K.expected_quadrant())
        s = asm_stats(k)
        if name in ("mfma_add", "l3_add"):
            base["2d" if name == "mfma_add" else "3d"] = t
        b = base["2d"] if case <= 1 else base.get("3d", 0.0)
        triv = "yes" if not any(m.startswith("ds_") for m in s["ops"]) else "no"
        print(f"| {name} | {desc} | {triv} | {fmt_ops(s['ops'])} | {s['barriers'] / 2:g} | {s['lds']} | {t:.1f} | "
              f"{max(t - b, 0.0):.1f} | {s['vgpr']} | {s['spill']} | {'ok' if ok else 'WRONG'} |", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", nargs="+", default=["probe"], choices=["probe", "slice", "splits", "bf16"])
    args = p.parse_args()
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}, {torch.cuda.get_device_name()}, "
          f"{time.strftime('%Y-%m-%d %H:%M')}\n")
    if "probe" in args.sweep:
        sweep_probe()
    if "slice" in args.sweep:
        import slice_bench
        slice_bench.sweep_slice()
    if "splits" in args.sweep:
        import split_bench
        split_bench.sweep_splits()
    if "bf16" in args.sweep:
        import split_bench
        split_bench.sweep_bf16()


if __name__ == "__main__":
    main()
