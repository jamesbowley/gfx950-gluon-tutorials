"""Inputs for AITER's a16w16 tuner on the 380 AITER shapes, FlyDSL only.

    source kernels/gemm/intra_wave/a16w16/v31_streamk_aiter/env.sh; cd $REPO/experiments/tune_flydsl
    python make_inputs.py              # everything (about 4 minutes of catalog generation)
    python make_inputs.py --pilot      # pilot.csv only, from catalog_sizes.csv

Writes untuned.csv (every shape, bf16 in and out, no bias), pilot.csv (see write_pilot),
catalog_sizes.csv (FlyDSL candidates per shape from get_flydsl_a16w16_configs) and
aot_names.csv (every unique FlyDSL kernel name, in the tuned-CSV columns that
``python -m aiter.aot.flydsl.gemm --csv`` reads).
"""

import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tune_aiter_shapes"))
import shapes  # noqa: E402
from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_configs  # noqa: E402
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm_kernel_name  # noqa: E402

UNTUNED_FIELDS = ("M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle")
TUNED_FIELDS = ("gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle",
                "libtype", "solidx", "splitK", "us", "kernelName", "err_ratio", "tflops", "bw")
BF16 = "torch.bfloat16"


def row(M, N, K):
    return dict(M=M, N=N, K=K, bias=False, dtype=BF16, outdtype=BF16, scaleAB=False, bpreshuffle=False)


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    sizes, names = {}, {}
    for M, N, K in shapes.SHAPES:
        configs = get_flydsl_a16w16_configs(M, N, K, torch.bfloat16, torch.bfloat16, False)
        sizes[(M, N, K)] = len(configs)
        for c in configs:
            name = flydsl_hgemm_kernel_name(dtype=torch.bfloat16, out_dtype=torch.bfloat16, config=c,
                                            has_bias=False)
            names.setdefault(name, (M, N, K, c))
    write(os.path.join(HERE, "untuned.csv"), UNTUNED_FIELDS, [row(*s) for s in shapes.SHAPES])
    write(os.path.join(HERE, "catalog_sizes.csv"), ("M", "N", "K", "candidates"),
          [dict(M=s[0], N=s[1], K=s[2], candidates=n) for s, n in sizes.items()])
    write_pilot(sizes)
    aot = []
    for name, (M, N, K, c) in sorted(names.items()):
        aot.append(dict(gfx="gfx950", cu_num=256, M=M, N=N, K=K, bias=False, dtype=BF16, outdtype=BF16,
                        scaleAB=False, bpreshuffle=False, libtype="flydsl", solidx=0, splitK=c["split_k"],
                        us=0, kernelName=name, err_ratio=0, tflops=0, bw=0))
    write(os.path.join(HERE, "aot_names.csv"), TUNED_FIELDS, aot)
    print(f"{len(shapes.SHAPES)} shapes, {sum(sizes.values())} candidates, {len(names)} unique kernels, "
          f"{sum(1 for n in sizes.values() if n == 0)} shapes with none")


def write_pilot(sizes):
    """4 shapes that sample the catalog sizes without the 3,555-candidate extreme: the
    one nearest 600 candidates, the median K=35840 shape, 2561x6288x7168 and the median."""
    ranked = sorted((s for s in sizes if sizes[s]), key=sizes.get)
    big = min(ranked, key=lambda s: abs(sizes[s] - 600))
    k35840 = [s for s in ranked if s[2] == 35840]
    pilot = list(dict.fromkeys([big, k35840[len(k35840) // 2], (2561, 6288, 7168), ranked[len(ranked) // 2]]))
    write(os.path.join(HERE, "pilot.csv"), UNTUNED_FIELDS, [row(*s) for s in pilot])
    print("pilot:", [(s, sizes[s]) for s in pilot], "total", sum(sizes[s] for s in pilot))


if __name__ == "__main__":
    if sys.argv[1:] == ["--pilot"]:
        write_pilot({(int(r["M"]), int(r["N"]), int(r["K"])): int(r["candidates"])
                     for r in csv.DictReader(open(os.path.join(HERE, "catalog_sizes.csv")))})
    else:
        main()
