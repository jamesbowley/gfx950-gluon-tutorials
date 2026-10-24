# Load latency and loads in flight per CU on MI355X

[docs/tcp_bound_tile_model.md](../../docs/tcp_bound_tile_model.md) caps a CU's loads at `C / L` with
`C = 44 KB` (the 32 KB TCP plus a ~12-entry VMEM queue) and `L = 1000` cycles. Neither number is measured
in the tree: `L` cites the [v8 README §4.5](../../kernels/gemm/intra_wave/a16w16/v8_sliceMN/README.md),
where ~1000 cycles is the time v7 takes to *fill* the TCP and queue (so an upper bound on the round trip
at K = 8192, a lower bound at K = 16384), and the queue depth is quoted as "approximately 12". The
v9_tile WORKLOG on `wip_streamk` uses a different pair, 32 KB over ~550 ns, inferred from measured
bandwidth. This experiment measures the round trip directly, separates capacity from latency, and reads
both off ATT traces.

## TL;DR

- **Capacity: 44 KB per CU, counted from when a wave issues a load, is right.** In ATT traces where a CU's
  loads wait for slots, a stalled `buffer_load` leaves the wave when the load issued **44-45 loads earlier on
  the same CU** returns. That holds at 64 and 128 KB nominally in flight, from HBM and the MALL, on one XCD
  and on the whole chip. So the TCP's 32 slots plus ~12 queue entries both hold an issued load.
- **From HBM, the loads past the TCP's 32 add latency, not bandwidth.** One CU's HBM bandwidth stops rising
  at 32 KB nominally in flight (25.0 B/clk; 27.0 at 128 KB). A loaded CU delivers 44 KB over the round trip
  *measured from issue*, which is the same thing as 32 KB over the TCP-side round trip: the two pairs are
  equivalent only if L is defined to match. (Nominal in flight is an upper bound, between 1 and 2 x unroll
  per wave, so the sweep alone can't place the knee between 32 and 44 KB; MALL bandwidth still rises to
  ~48-64 KB nominal.)
- **L = 1000 cycles is the unloaded HBM round trip at 2.4 GHz, not the one that goes with 44 KB.** One
  1 KB load on an idle GPU takes 128 cycles from L1, ~226 from L2, ~558 from the MALL and **~985 from
  HBM** (410 ns). But with 44 KB in flight the issue-to-return time is much longer: ~1720 cycles on one CU
  reading HBM (44 KB / 26.2 B/clk), ~2600 with all 32 CUs of an XCD reading, ~3800 with all 256.
  (44 KB, 1000 cycles) is 45 B/clk, 108 GB/s per CU at 2.4 GHz: one CU gets that from the MALL
  (49 B/clk) or L2 (64 B/clk, the TCP's processing rate), never from HBM (26 B/clk).
- **Our WORKLOG's pair (32 KB, ~550 ns) is the HBM-bound single CU.** One CU reading HBM tops out at
  63 GB/s, which is 32 KB over 520 ns, or 44 KB over 715 ns from issue.
- **L grows with chip load, as the branch says, but the per-CU cap binds first.** The probe's HBM round trip
  stays at 910-1000 cycles up to 3.2 TB/s of background traffic, then rises to 1174 cycles at 5.1 TB/s and
  1662 at 6.1 TB/s. L2 round trips double only near the 28 TB/s L2 peak.
- **Cycles or ns: inconclusive.** L1 is fixed in cycles. Beyond L1, latency moves less in ns than in
  cycles (HBM: 976 cycles / 407 ns at 2.40 GHz, 914 cycles / 427 ns at 2.14 GHz), but MFMA-only
  co-runners only push the clock down 11%, and they slow the fabric too (MALL latency rises in both units).
- **The GEMM hot loops can't calibrate either number.** In the v7/v8 traces the loads that stall are paced
  by the TCP's processing rate, 1 KB per 16 cycles (D_k = 16 k exactly up to k ≈ 32-36), not by free slots,
  and the loops hide the round trip: almost no `s_waitcnt vmcnt` stalls in the README's build (4 in v7, 30
  in v8 at K = 8192).

## Method

### Probe ([kernel.py](kernel.py) `probe_kernel`, [bench.py](bench.py) sweeps `lat`, `loaded`, `clock`)

One wave walks a region of 1 KB tiles one `buffer_load_dwordx4` at a time. Around each load:

```
s_waitcnt vmcnt(0); s_memrealtime; s_memtime      <- sample, no lgkmcnt wait
buffer_load_dwordx4 ...
s_waitcnt vmcnt(0); s_memtime; s_waitcnt lgkmcnt(0)
```

The first timestamps' registers are tied into the second asm block, so nothing waits on them before the load
issues (gfx950 has no `HW_REG_SHADER_CYCLES`). The ISA was checked: nothing sits between the sample and the
load but the descriptor's `s_and`. The L1-resident run (constant 128-132 cycles) is the floor and includes
this overhead. Per-load `s_memrealtime` gives the local clock, so ns are at the clock of each load.

Tiers come from passes over a region after a 1 GB flush: pass 0 is a first touch (HBM); passes 1-3 of a
1 MB region hit L2 (larger than the 32 KB L1, inside the 4 MB L2); pass 1 of a 16 MB region hits the MALL.
Under load (`loaded`), workgroup 0 probes while workgroups 1..W (one per CU) stream 64 KB in flight each from
HBM (a 1 GB buffer) or from L2 (64 KB per CU), until the probe sets a flag. `clock` uses MFMA-only
co-runners with hashed bf16 operands (constant operands draw too little power to move the clock).

### Loads in flight ([bench.py](bench.py) sweep `inflight`)

`bw_kernel` (ported from `experiments/hbm_bandwidth_xcd` on `wip_streamk`) streams reads double-buffered in
registers, so each wave keeps up to `2 x unroll` 1 KB loads in flight. HWSEL placement puts `wpc`
workgroups on one chosen CU (`1cu`) or on every CU (`256cu`); nominal KB in flight per CU =
`2 x num_warps x unroll x wpc`. L2 and MALL points are steady state (P against 4P passes); HBM is one cold
pass. Dropped from the figure: points whose workgroups did not all run concurrently, and `wpc = 5`
(its fifth workgroup doesn't fit the LDS and lands on another CU with the same HW_ID key: one "CU" then
reads 80 B/clk from L2). `num_warps = 1` is not used: HWSEL makes every candidate workgroup active there.

### ATT ([att_latency.py](att_latency.py), [collect_att.sh](collect_att.sh), [collect_att_stream.sh](collect_att_stream.sh))

Per wave, `vmcnt` counts VMEM operations in order, so the `s_waitcnt vmcnt(N)` that needs operation i is
satisfied when i returns; if it stalled at least 16 cycles, its end is that return
("exact" L, measured from when the load left the wave). Per CU (the traced waves of one SE), a
`buffer_load` that stalls before leaving the wave is waiting for a slot. If the CU holds k loads, it leaves
when the load k earlier returns, so `D_k = issue(n) - issue(n - k)` equals L at the true k.

- **Stream traces:** `att_stream_kernel`, which is `bw_kernel`'s loop without placement logic (the decoder loses its
  place in HWSEL's branches, "Stitch Incomplete"). It runs on the 32 CUs of XCD 0 or on all 256, cold HBM or warm MALL.
  ATT traces CU 0 of SE0 and SE2 only: SE1 and SE3 of XCD 0 have CU 0 harvested on this GPU.
- **GEMM traces:** a16w16 v7 and v8, fp16, M = N = 4096, K = 8192 and 16384, dispatch 15, under the
  LLIR scheduler alone. On the installed Triton, `llir+amdgcnas` fails `bench.py`'s check against torch, and
  this build reaches ~1100-1200 TFLOPS, against the README's 1583-1588. The September traces in
  `kernels/gemm/intra_wave/a16w16/tmp/llir+amdgcnas/` (K = 8192, the README's build) are reduced too, as
  `*_sep29`.
- `rocprofv3` is called directly: `scripts/run_perf_table.py` deletes `kernels/gemm/intra_wave/a16w16/tmp/`
  before every run.

## Results

Numbers: [results/summary.md](results/summary.md) (from `plot.py`); raw: `results/*.csv`,
`results/*_samples.npz`, `results/att_*.json`. Per-trace timelines and D_k plots: `results/att_*_timeline.png`,
`results/att_*_dk.png`; traces: `tmp/att/*/ui_*`.

### Unloaded round trip

![one load in flight, per tier](images/latency_tiers.png)

| tier | p50 cycles | p50 ns (2.40 GHz) | p10-p90 cycles |
|---|---:|---:|---:|
| L1 | 132 | 55 | 132-132 |
| L2 | 226 | 94 | 212-240 |
| MALL | 558 | 233 | 404-708 |
| HBM | 985 | 410 | 738-1176 |

### Under load

![probe latency against background traffic](images/latency_loaded.png)

The HBM stream evicts the probe's L2 and MALL lines, so under it every tier is an HBM measurement
(`summary.md` marks them). The L2 stream leaves the probe's HBM latency alone up to 18 TB/s; at 28 TB/s
the L2 probe doubles (484 cycles) and HBM rises 5%.

### Clock

![latency against clock](images/latency_clock.png)

### Loads in flight

![bandwidth per CU against KB in flight](images/inflight.png)

| scope | tier | plateau B/clk | GB/s per CU | L from the rising part, cycles |
|---|---|---:|---:|---:|
| one CU | L2 | 63.9 | 153 | 250 |
| one CU | MALL | 49.2 | 118 | 665 |
| one CU | HBM | 26.2 | 63 | 1098 |
| 256 CUs | L2 | 62.1 | 142 | 278 |
| 256 CUs | MALL | 12.7 | 30 | 673 |
| 256 CUs | HBM | 9.0 | 21 | 1113 |

On one CU the rising slopes give round trips close to the probe's (250 against 226 for L2, 1098 against 985
for HBM). The plateaus are the TCP's 64 B/clk from L2 (reached at 24 KB nominal) and the slots from HBM
(32 KB) and the MALL (48-64 KB). On all 256 CUs, the MALL (fabric) and HBM limits bind long before the
per-CU ones.

### ATT

![D_k against k](images/att_dk.png)

| trace | stretched loads | exact L p50, cycles (ns) | D_32 | D_44 | k with D_k = L |
|---|---:|---:|---:|---:|---:|
| stream, 8 waves x u4, HBM, XCD 0 | 47% | 2604 (1088) | 1844 | 2540 | 45 |
| stream, 16 waves x u4, HBM, XCD 0 | 50% | 2652 (1101) | 1868 | 2584 | 45 |
| stream, 16 waves x u4, MALL, XCD 0 | 55% | 2604 (1090) | 1844 | 2548 | 45 |
| stream, 16 waves x u4, HBM, 256 CUs | 46% | 3808 (1589) | 2792 | 3840 | 44 |
| v7 K = 8192 / 16384 (LLIR build) | 58% / 65% | 3676 / 4196 | 512 / 952 | 2624 / 2660 | none |
| v8 K = 8192 / 16384 (LLIR build) | 26% / 23% | 4300 / 4312 | 988 / 992 | 1660 / 1648 | none |

The XCD 0 stream runs are capped by the XCD's link (32 CUs x 41.7 GB/s = 1.33 TB/s), and Little's law closes
on them: 41.7 GB/s x 1.09 us = 45 KB. The stream runs below 25% stretched loads (4 waves x u2/u4) aren't
slot-bound and are left out of the figure.

In the GEMM traces, D_k = 16 k up to k = 32-36: the stalled loads are issued back to back at the TCP's
1 KB per 16 cycles, so they measure the processing rate, not the slots. Their exposed waits (3700-4300 cycles
from issue, 2.3-3 us at the traced 1.4-1.7 GHz) are far above the stream kernel's 1.1-1.6 us. That is
unexplained here. One candidate is that `vmcnt` for `buffer_load ... lds` only drops once the LDS write lands,
behind the loop's `ds_read` traffic, after the TCP slot has been freed.

## What this means for the two models

Both models are Little's law; only the constants differ, and capacity and latency only make sense in pairs:

| pair | B/clk | matches |
|---|---:|---|
| branch: 44 KB / 1000 cycles | 45 | one CU from the MALL (49); not one CU from HBM (26) |
| WORKLOG: 32 KB / 550 ns (at 2.4 GHz: 1320 cycles) | 25 | one CU from HBM (26) |
| measured, HBM: 32 KB / ~1250 TCP-side = 44 KB / ~1720 from issue | 26 | |
| measured, MALL: 44 KB / ~915 from issue | 49 | |

For a GEMM the right L depends on the L2 hit rate and on how many CUs load at once, and neither GEMM
trace set here can calibrate it. The branch's 1000 cycles is the unloaded HBM latency at boost clock. With
C = 44 KB it needs the round trip from issue at saturation, which is longer.

## Running

Needs the deepbind shim from `wip_streamk` (`scripts/triton_deepbind_shim/sitecustomize.py`; this branch
only has its `__pycache__`), or Triton crashes on import with the pip-installed ROCm.

```bash
export SHIM=<dir with sitecustomize.py>
PYTHONPATH=$SHIM HIP_VISIBLE_DEVICES=2 python bench.py --sweep lat loaded clock inflight --reps 3
GPU=2 SHIM=$SHIM bash collect_att_stream.sh     # stream ATT, ~30 s
GPU=5 SHIM=$SHIM bash collect_att.sh            # v7 / v8 ATT at K = 8192, 16384
python att_latency.py <ui_dir> --name <name> --plot
python plot.py
```

Measured on MI355X GPU 2 (GEMM ATT on GPU 5), Triton main 872c90a, 2026-10-13.
