# Hot-loop latency: 256x256x64 against 256x128x64 on 4096x4096x8192 (2026-10-14)

Why the two latency models disagree: the canvas (`tile-tcp-intuition`: Q = 32 KB over L = 550 ns, fixed in ns)
and `docs/tcp_bound_tile_model.md` on `tcp-bound-tile-model` (C = 44 KB over L = 1000 cycles, fixed in cycles).

Setup: GPU 6, v3.0 stack with the MFMA scheduler alone (`GPU=6 source v9_tile/env.sh; unset
TRITON_AMDGCNAS_PLUGIN`), bf16, cold inputs (`--rotating-mb 2048`). 4096x4096x8192 is 256 tiles of 256x256
(one per CU) and 512 of 256x128 (one workgroup per CU, two rounds), so every CU is busy in both runs. ATT
traces CU 0 of SE0-3; only SE0 and SE2's CU 0 ran waves (8 waves for 256x256, 16 for 256x128 over its two
rounds). Two launches per tile were traced: the 10th (`--att-launch 10`, the default) and the 1st
(`--att-launch 1`, before the power cap pulls the clock down).

```
python v9_tile/sweep.py att [--att-launch 1] --points 256x256:4096x4096x8192,256x128:4096x4096x8192 \
    --rotating-mb 2048 --tmp v9_tile/tmp/tcpL[_launch1] --out v9_tile/results/tcpL_att[_launch1].csv
python ../../../../experiments/tcp_latency/att_latency.py <ui_dir> --loop --plot --exact-stall 64 --out v9_tile/results/tcpL
python v9_tile/att_stalls.py v9_tile/tmp/tcpL[_launch1]/sweep_att/<point>
python v9_tile/sweep.py counters --sets 0,1,5,7 --points ... --rotating-mb 2048 --tmp v9_tile/tmp/tcpL --out v9_tile/results/tcpL_counters.csv
```

## Results

Hot loop, from ATT (clock from the trace's own `realtime.json`):

| tile | launch | clock GHz | loop MFMA eff | cycles / iteration (2 k-steps) | CU load rate in the loop | `buffer_load` issue stall / iter / wave | `vmcnt` stall / iter / wave | in-loop round trip, ATT |
|---|---|---|---|---|---|---|---|---|
| 256x256 | 10 | 1.36 | 95.6% | 4286 | 30.6 B/clk (41.7 B/ns) | 161 | 32 | <= 2844 cyc (2090 ns), no wait stalls |
| 256x256 | 1 | 1.95 | 96.0% | 4265 | 30.8 B/clk (60.0 B/ns) | 144 | 36 | <= 2836 cyc (1454 ns), no wait stalls |
| 256x128 | 10 | 1.42 | 81.2% | 2524 | 39.0 B/clk (55.6 B/ns) | 88 | 48 | <= 1632 cyc (1146 ns), no wait stalls |
| 256x128 | 1 | 1.96 | 84.1% | 2434 | 40.5 B/clk (79.4 B/ns) | 16 | 48 | <= 1628 cyc (829 ns), no wait stalls |

Whole kernel, from counters (one counter pass per set; the dispatches run serialised, at their own clock):

| tile | clock GHz | MFMA busy | L2 hit | L_issue (`VmemLatency`) | L_tcp (TCP to L2) | KB in flight per CU, from issue / TCP-side | B/ns per CU (TCP side) |
|---|---|---|---|---|---|---|---|
| 256x256 | 1.70-1.83 | 80.5% | 80.3% | 636 cyc, 406 ns | 388 cyc, 211 ns | 16.1 / 9.6 | 46.8 |
| 256x128 | 1.86-1.99 | 64.6% | 82.7% | 1008 cyc, 550 ns | 470 cyc, 239 ns | 31.5 / 13.8 | 58.8 |

Models at the trace clocks (window ceiling; the canvas's pipeline and TCP-rate ceilings are 100% for both tiles):

| tile | clock GHz | canvas: 32 KB / 550 ns | branch: 44 KB / 1000 cyc | branch at measured L_issue | measured |
|---|---|---|---|---|---|
| 256x256 | 1.36 | 100% | 100% | 100% | 95.6% |
| 256x256 | 1.95 | 95% | 100% | 100% | 96.0% |
| 256x128 | 1.42 | 87% | 94% | 93% | 81.2% |
| 256x128 | 1.96 | 63% | 94% | 93% | 84.1% |

## What the traces say

- **ATT cannot time an in-loop load on either tile: no `vmcnt` wait in either hot loop waits for data.** Every
  wait in both loops is `s_waitcnt vmcnt(N) lgkmcnt(0)`. The only one that stalls is 256x128's loop-top
  `vmcnt(14)`, and it stalls exactly 20 cycles on 992 of 1008 executions, about 208 cycles after the last
  `ds_read_b128`. That is a fixed cost at the top of the loop, not a load arriving, so `--exact-stall` is 64 here.
  (At the default of 16 those waits gave a constant "exact" L of 1608 cycles, which was the schedule's lead
  time, not a latency.) What ATT gives is an upper bound: each load is back by the time its wait issues, about
  1630 cycles after issue on 256x128 and 2840 on 256x256. The only waits that did stall at 64 cycles or more are
  256x256 launch 1's fourth wait of the first iteration, once per wave (~4100 cycles): start-up, not the loop.
- **The measured round trip from issue is the counters': 1008 cycles = 550 ns on 256x128, 636 cycles = 406 ns
  on 256x256.** Both are well inside the ATT bounds, consistent with the absent `vmcnt` stalls. For 256x128,
  1008 cycles at 1.99 GHz is both models' L at once (550 ns is about 1000 cycles at ~1.9 GHz). At this clock the
  models do not disagree on L. They disagree on what happens when the clock changes, and on the capacity.
- **The hot loop is fixed in cycles, not ns.** From the 1.36-1.42 GHz launch to the 1.95-1.96 GHz one, the
  cycles per iteration fall by 0.5% (256x256) and 3.6% (256x128). The CU's load rate in the loop stays at
  30.6-30.8 and 39.0-40.5 B/clk. In ns it rises 1.43x to 60 and 79 B/ns per CU, past the canvas's
  32 KB / 550 ns = 59.6 B/ns. A window fixed in ns would have cost 256x128 a quarter of its loop at the higher clock
  (87% -> 63%); it went from 81% to 84%.
- **256x128's lost 16-19% is not memory, in these launches.** At 1.96 GHz a wave stalls 16 cycles per iteration on
  `buffer_load` issue and 48 on `vmcnt` (of which 20 is the loop-top fixed cost). The rest of the gap is the
  non-MFMA instructions (`ds_read` 354, address and other 398, `buffer_load` 112, `s_barrier` 98 cycles of issue
  per iteration) not fitting behind 128 MFMAs. 256x256 issues the same per k-step against twice the MFMAs and
  hides them ([att_stalls.md](tcpL/att_stalls.md), [att_stalls_launch1.md](tcpL/att_stalls_launch1.md)).
- **In flight: about 40 KB per CU on 256x128, 19 KB on 256x256**: the ATT loop rate times the counters' L_issue,
  at 40.5 x 1008 and 30.8 x 636. The counters' own whole-kernel averages are 31.5 and 16.1 KB. 256x128 is near,
  but not at, the 44 KB cap, and the TCP side holds 14 KB, under 32. Neither model's capacity binds here.
- **The bandwidth the canvas was fitted on is a whole-kernel number, and it runs 25-30% below the loop's.** The
  counter run moves 46.8 and 58.8 B/ns per CU over the whole kernel (the canvas's ~60 B/ns), while CU 0's hot
  loop at a similar clock moves 60 and 79. The whole-kernel figure includes the prologue, epilogue, the two
  rounds' tails and the CUs that finish last. In cycles the whole kernel is 11-12% (256x256) and 18-22% (256x128)
  longer than CU 0's waves (prologue + loop + epilogue, times two rounds for 256x128). With two CUs traced, it cannot be told here whether other CUs run a slower loop.

The clock comparison above (launch 1 against launch 10, unpinned) changed the power state and the chip's load
along with the clock. The pinned sweep below replaces it: the round trip is close to fixed in ns, and in steady
state (launch 10) the canvas's ~60 B/ns per CU describes the 256x128 loop better than the branch's 1000 cycles.

## Pinned clocks (2026-10-14)

The shader clock pinned from the host with `rocm-smi -d 6 --setperfdeterminism 1400` and `1900` (fclk 1250 MHz
and mclk 2000 MHz unchanged); the kernels ran at 1.20-1.22 and 1.58-1.65 GHz. Everything else as above,
by [run_clock_pin.sh](../run_clock_pin.sh); results in [clockpin/](clockpin/).

Round trip, counters (whole kernel, serialised dispatches):

| tile | clock GHz | L_issue | L_tcp | KB in flight per CU from issue |
|---|---|---|---|---|
| 256x256 | 1.22 | 490 cyc, 401 ns | 242 cyc, 199 ns | 12.8 |
| 256x256 | 1.64 | 682 cyc, 415 ns | 353 cyc, 217 ns | 17.7 |
| 256x128 | 1.22 | 594 cyc, 488 ns | 283 cyc, 231 ns | 20.7 |
| 256x128 | 1.67 | 890 cyc, 542 ns | 409 cyc, 243 ns | 28.7 |

Hot loop, ATT (cycles per iteration of 2 k-steps; stalled = `buffer_load`s stalled at issue, per CU of 12096 or 8064):

| tile | launch | 1.2 GHz: eff, cycles, B/ns per CU, stalled | 1.6 GHz: eff, cycles, B/ns per CU, stalled |
|---|---|---|---|
| 256x256 | 1 | 96.0%, 4265, 37.0, 0 | 96.1%, 4264, 49.0, 0 |
| 256x256 | 10 | 96.2%, 4260, 37.5, 0 | 94.8%, 4322, 49.7, 106 |
| 256x128 | 1 | 83.7%, 2448, 48.4, 4 | 83.9%, 2440, 63.7, 16 |
| 256x128 | 10 | 82.5%, 2482, 48.2, 172 | 77.5%, 2642, 61.3, 636 |

Window ceilings for 256x128 (48 B/clk at 100%): canvas (32 KB / 550 ns) 100% at 1.22 GHz, 75% at 1.645 GHz;
branch (44 KB / 1000 cycles) 94% at both.

- **The round trip is close to fixed in ns.** For a 1.35x clock, L_issue grows 3% (256x256) and 11% (256x128) in
  ns, and 39% and 50% in cycles. L_tcp grows 9% and 5% in ns. The branch's "1000 cycles" is not a constant: it is
  ~400-550 ns, 490-890 cycles over this range.
- **In steady state, the 256x128 loop starts losing to the window at the higher clock.** On launch 10 it slows from
  2482 to 2642 cycles per iteration (82.5% to 77.5%), and its stalled loads go from 172 to 636 per CU. Its loop moves
  61.3 B/ns per CU at 1.645 GHz, at the canvas's 59.6. The canvas's 75% ceiling there is 2.5 points under the
  measured 77.5%; the branch's 94% is 16.5 over. At 1.22 GHz neither window binds, and the loop sits at the ~83%
  that the instruction overhead allows: the efficiency is about min(overhead ceiling, window ceiling).
- **256x256 is just reaching it at 1.64 GHz on launch 10** (106 stalled loads, 96.2% to 94.8%), where it needs
  52.6 B/ns against the window's 59.6.
- **Launch 1 sees no window at either clock**: cycle-invariant, 0-16 stalled loads, and 63.7 B/ns at 1.58 GHz
  (79.4 unpinned at 1.96), above 59.6. The first launch after the host's set-up gap runs with the memory system
  otherwise idle; launch 10 follows nine back-to-back launches. The earlier conclusion that the loop is fixed in
  cycles came from launch 1 and does not hold in steady state.
- **D_k does not pin the capacity down.** On launch 10 at 1.645 GHz, D_44 is 1328 cycles (807 ns): 44 KB over
  807 ns is 56 B/ns, close to the 61 measured, but the counters' mean L_issue is 542 ns, so the stalled loads see
  longer round trips than the average (bursts). The earlier D_44 = 996 cycles (unpinned launch 1) rested on 16
  stalled loads per CU.

So: the round trip behaves as the canvas assumes, fixed in ns, and in steady state the canvas's window
(32 KB over 550 ns, ~60 B/ns per CU) predicts 256x128's loop at 1.6 GHz within a few points. Its (32 KB, 550 ns)
pair is not physical (the CU holds more than 32 KB from issue, over a shorter mean round trip), but the bandwidth
it implies matches. The canvas needs the instruction-overhead ceiling added, which is what binds below ~1.4 GHz.

## Caveats

- ATT is one launch on two CUs. Launches 1 and 10 agree to 0.4 points on 256x256 and 3 points on 256x128, with a
  1.4x clock difference between them.
- Neither ATT launch is at the power-capped back-to-back clock (1.71 / 1.92 GHz, WORKLOG section 4). Launch 10
  is below it, launch 1 at or above it.
- These loop efficiencies (81-84% for 256x128) are higher than WORKLOG sections 4 and 16 report for 256x128
  (59-73%), which ran 4096x2048 (one tile per CU) and 2048x4096, partly on an older stack and on other GPUs. Not
  investigated here.
- The counters cover the whole kernel; the hot loop is ~94% of every wave's time at K = 8192.

## Files

Traces (decoded `ui_*` directories with `code.json`, `realtime.json` and one `se*_sm*_sl*_wv*.json` per wave;
the raw per-SE `.att` files are in the sibling `smci355-ccs-aus-m01-29/` directory):

- `v9_tile/tmp/tcpL/sweep_att/256x256_4096x4096x8192/ui_output_agent_6903_dispatch_58` (launch 10)
- `v9_tile/tmp/tcpL/sweep_att/256x128_4096x4096x8192/ui_output_agent_58856_dispatch_58` (launch 10)
- `v9_tile/tmp/tcpL_launch1/sweep_att/256x256_4096x4096x8192/ui_output_agent_57856_dispatch_5` (launch 1)
- `v9_tile/tmp/tcpL_launch1/sweep_att/256x128_4096x4096x8192/ui_output_agent_13283_dispatch_5` (launch 1)

Results: [tcpL_att.csv](tcpL_att.csv), [tcpL_att_launch1.csv](tcpL_att_launch1.csv),
[tcpL_counters.csv](tcpL_counters.csv), `tcpL/att_<tile>_loop[_launch1].json` (+ `_timeline.png`, `_dk.png`),
[tcpL/att_stalls.md](tcpL/att_stalls.md), [tcpL/att_stalls_launch1.md](tcpL/att_stalls_launch1.md).
