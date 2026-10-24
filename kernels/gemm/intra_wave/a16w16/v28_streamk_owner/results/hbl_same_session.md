# v28 against hipBLASLt in the same session: not run yet (2026-10-19)

**Status: ready, not run.** The doc's stack is not in this container: `/home/jbowley/rocm724_env` (torch 2.10.0+rocm7.2.4,
hipBLASLt 1.2.2) and `/opt/rocm-7.2.4` do not exist, so `run_rocm724_v20.sh`'s hipBLASLt worker fails at `import torch`.
The only torch here is `/opt/venv` (torch 2.13.0+rocm7.14.0). Its hipBLASLt does not run the stream-K owner kernel on
this shape: it runs `Cijk_..._MT128x256x64_MI32x32x1_..._GSU` plus `Cijk_SB_BiasS_HAS_ScaleAlphaVec_PostGSU8_VW4`
(split-K with a separate reduction), about 436 TFLOPS warm on HIP 1. So it is no owner-ending reference, and the
comparison waits for the 1.2.2 environment.

## What is ready

- **Harness** ([run_v20_vs_aiter.py](../../../../../../experiments/aiter_lixun_shapes/run_v20_vs_aiter.py)):
  `--kernel v28` runs as `--kernel v27` does, with v28's live `matmul_kernel.py` as the snapshot; HIP 1
  (amd-smi GPU 0, PCI 0x05) is in `GPUS`; `STREAMK_OWNER_OPT` is in `PICK_OPTION_KEYS`, so a pick's variant is cleared
  after it. The "new" worker launches and expects `KERNELS[kernel][1]` (`v28_streamk_owner`). Checked statically (the
  module resolves the kernel, bus and pick keys); not run end to end.
- **Picks** ([v28_sk1024_picks.csv](v28_sk1024_picks.csv)), all at 128x128x128 with fp32 partials on 1024x1024x7168:
  `best` rsall, `fp32` owall, `tile2` owall+b, `tile3` ow2, `x5` ow2+b, `x6` owall+s, `x7` ow2+s.

## Commands (in the environment that has rocm724_env and /opt/rocm-7.2.4)

From `experiments/aiter_lixun_shapes/`, on physical GPU 0 (HIP 1). Check the HIP numbering there first: `GPUS` maps HIP
1 to PCI bus 0x05, and the harness asserts it.

```bash
P=../../kernels/gemm/intra_wave/a16w16/v28_streamk_owner/results/v28_sk1024_picks.csv
run() { tag=$1; shift; V20CMP_GPU=1 ./run_rocm724_v20.sh "$@" python run_v20_vs_aiter.py sweep --headline capped \
    --realloc --rounds 5 --v20-sched aiter --kernel v28 --jit-tag main --tag $tag --rotating-mb 2048 \
    --gpu-guard --v27-picks $P --only 405; }
run sk1024_v28_dry env --rounds 1    # dry run first (a later --rounds wins); then delete its results and snapshot
run sk1024_v28_def1 env; run sk1024_v28_def2 env
run sk1024_v28_g128 env TENSILE_STREAMK_FIXED_GRID=128
```

Index 405 is 1024x1024x7168 bf16. The default grid is hipBLASLt's SK3 on 4 workgroups per tile (compare owall,
owall+b, owall+s); `TENSILE_STREAMK_FIXED_GRID=128` is 2 per tile (compare ow2, ow2+b, ow2+s). rsall is the control
against the doc's 541.3. Results land in `results/details_sk1024_v28_*.csv`.

## Until then: the cross-session estimate

From [WORKLOG](../WORKLOG.md) section 4 (rocprof back-to-back, HIP 1) and the doc's session (cold harness, physical GPU
1). On v27's code, this run read higher than the doc's: owall 496.4 against 484.2 (x1.025), ow3 473.6 against 469.5
(x1.009), ow2 388.8 against 384.4 (x1.011), rsall 550.2 against 541.3 (x1.016). Scaling b by the same kernel's offset:

| point | this run | scaled to the doc's session | hipBLASLt (doc) | gap |
|---|---|---|---|---|
| owall+b (4 per tile) | 538.0 | about 525 | 546.0 | about -3.9% (v27: -11.3%) |
| ow2+b (2 per tile) | 398.1 | about 394 | 407.0 | about -3.3% (v27: -5.6%) |

An estimate only: it assumes b has the same offset between the two setups as v27's code at the same split.
