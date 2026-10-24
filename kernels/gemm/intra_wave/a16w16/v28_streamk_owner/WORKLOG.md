# v28_streamk_owner work log

v28 is v27 (`v27_streamk_tile`, snapshot sha256 `1175b73525d3cc3f`, unchanged) with a faster owner ending on the
quadrant tiles (section 10). Sections 1-9 tried the owner-ending changes of
[v27's OWNER_ENDING_OPTIMIZATIONS.md](../v27_streamk_tile/OWNER_ENDING_OPTIMIZATIONS.md), and a few more, as opt-in
compile-time variants (`STREAMK_OWNER_OPT`); that kernel is archived as
[experiments/matmul_kernel_variants.py](experiments/matmul_kernel_variants.py). Everything here ran on HIP 1 only (`amd-smi` GPU 0, PCI 0000:05:00.0; [env.sh](env.sh)), on v27's stack,
on 2026-10-19.

## 1. The build (2026-10-19)

- **Files:** [matmul_kernel.py](matmul_kernel.py), [sweep.py](sweep.py), [check_streamk.py](check_streamk.py),
  [check_grid.py](check_grid.py) and [env.sh](env.sh) are v27's, with the kernel renamed `v28_streamk_owner` (and
  `LAST_LAUNCH["kernel"]`, the imports and the rocprof filter). The GPU lock files keep v27's name
  (`/tmp/v27_check_gpu<N>.lock`), so v27 and v28 runs never share a GPU. New: [asm_owner.py](asm_owner.py),
  [run_checks.sh](run_checks.sh), [owner_opt.py](owner_opt.py).
- **`STREAMK_OWNER_OPT`**, a set of letters, read by the host (`owner_options`) and passed as constexprs to the
  quadrant body only. Empty (the default) passes nothing, so the call is v27's.
  - **b, change 1** (`OWNER_QPB`): the owner reads QPB quadrants of a peer's partial per batch, sized to 16 fp32
    loads per wave: 4 at 128x128, 2 at 256x128 and 128x256, 1 at 256x256 (which keeps v27's code). New
    `owner_fixup_tile`: contributors store their four quadrants; the owner sums each batch's peers into VGPR tensors
    (`peer_sums`), adds them to the accumulator once outside the loop, and stores that batch of C before the next
    batch's reads.
  - **p, change 2** (`OWNER_POLL`): the poll is `gl.atomic_poll(flag, 1, relaxed, gpu)` instead of the polled
    `atomic_add(flag, 0)`. This changes 256x256 builds too.
  - **w, change 3** (`OWNER_WAIT`): the owner waits for each peer just before that peer's first batch, instead of for
    all peers first.
  - **s, change 4** (`OWNER_STATIC`): in the tile-aligned one-tile split, the peer loop is a `static_range` over
    the most peers a tile has (`N_HI` if `EXTRA > 0`, else `N_LO`, minus one), the loads of peers past `n_peers`
    masked off; only up to 8 programs per tile (7 peers), otherwise v27's run-time loop. The end-to-end splits (2t,
    sp) keep the run-time loop, so s leaves them unchanged.
- **Two fixes found on the way:**
  - **At 128x128x64 the batch of 16 loads was not kept.** With each quadrant's load followed by its add, LLVM's
    scheduler reused one quadrant's 16 registers for the next and waited for each in turn (`vmcnt(3..0)` four times),
    at 192 VGPRs; 128x128x128 (one workgroup per CU) kept all 16 in flight. Sections 3-6 timed b with an empty inline
    asm that took and returned a peer's quadrants (`together2`, `together4`), forcing every load of the batch to issue
    before any was used: 16 loads per wait, 228 VGPRs. Section 7 found that writing the loads before the adds is
    enough on its own, and the asm is gone.
  - **Summing all four quadrants before storing C spilled at 256x256** (89-718 VGPRs for bpw and s: four 64-VGPR sums
    beside the 256-AGPR accumulator). Storing each batch's C before the next batch's reads, as v27 does, removed every
    one-tile spill.

## 2. Code checks

- **The base is v27** ([asm_owner_base.md](results/asm_owner_base.md), [asm_owner_variants.md](results/asm_owner_variants.md)):
  every owner build of the five large tiles (one-tile n = 2 and all at S = 64; n = 2, 4 and all at S = 32 after one
  wave; 2t; sp), fp32, each in a fresh process with an empty cache, kernel name normalised: 32-33 of 35 identical to
  v27. The others are spread builds, and are v27's own nondeterminism (v27 WORKLOG §3, §13): repeated fresh builds of
  each version give the same set of forms (at 256x256 a `v_bitop3_b32` operand swap with the truth table to match, in
  both versions; at 128x128x64 v27 has a rare second form and v28 always gives v27's usual one).
- **ISA per variant** (fp32; [asm_owner_variants.md](results/asm_owner_variants.md); bf16 and bf16-pair builds of b in
  [asm_owner_b_bf16.md](results/asm_owner_b_bf16.md), [asm_owner_b_bf16_pairs.md](results/asm_owner_b_bf16_pairs.md),
  no new spills):

  | variant | partial loads per vmcnt(0) | poll | VGPRs / spills (one-tile; 2t, sp) |
  |---|---|---|---|
  | base (v27) | 4 at 128x128, 8 at 256x128, 16 at 256x256 | `global_atomic_or` + LDS broadcast | 196-256 at 128x128, 336-340 rect, 492 at 256x256; 2t/sp 256x256 5 spills |
  | b | 16 at every tile (1, 2, 4 batches per peer) | as base | 228-256, 336-340, 492 (= base); 2t/sp 304-441, unchanged at 256x256 |
  | bp | as b | `global_load_dword ... sc1` spin, no `ds_write`/`ds_read` | as b; 2t/sp 256x256 3 spills |
  | bpw | as b, the poll inside the first batch loop | as bp | one-tile as b; **2t/sp 256x256 34 spills** |
  | s, ps | 4 x peers at 128x128 (up to 28), 8 x peers rect (up to 56), 16 x peers at 256x256 (up to 112) | base / p | 192-256, 332-432, 488-512; no one-tile spills; 2t/sp as base / bp |

- **Stress** ([run_checks.sh](run_checks.sh), `results/checks/`, summary in
  [summary.txt](results/checks/summary.txt)): `check_streamk.py --fixups owner --exchanges wt` with the variant set,
  on all five tiles plus `--no-full` on 128x128x128 and 256x256x64: one-tile (both placements, n = 4, an uneven n and
  all programs), two-tile and spread, with and without bias, 10 launches each, C in a sentinel window.
  - b and s (which restructure the reads): fp32, bf16 and bf16-pair partials, 1,680 configurations each.
  - bp, bpw and ps (which only move the poll and the wait): fp32, 560 each.
  - **All 5,040 PASS:** no wrong launch, no flag left up, no hang.

## 3. The sweep ([owner_opt.py](owner_opt.py), [report.md](results/owner_opt/report.md))

- **Shapes (112):** per tile (256x256x64, 256x128x64, 128x256x64, 128x128x64, 128x128x128), exactly `waves * 256 + S`
  tiles in the most nearly square M x N, for full waves 0 (S = 16, 64, 128, 192), 1 (S = 4, 32, 128, 224) and 4
  (S = 8, 32, 128), at K = 2048 and 8192; plus 1024x1024x7168 at 128x128x128 and 128x128x64. The S values are where
  v27's split tables have the owner family within 10% of the best (S >= 64 without a full wave, every S with one).
- **Points:** ow2, ow3, ow4, owall, 2t and sp where valid, and cap where the capped policy runs the owner ending or
  spread, each in base, b, bp, bpw, s and ps (2t, sp and an end-to-end cap without s and ps); cap otherwise once; rsall
  and rsall+bpw as a null control. 2,667 points.
- **Timing:** `sweep.py`'s rocprof worker (graph of 100 launches over 2 GB of rotating inputs, the median of the last
  100), one shuffled block per shape in one profiler process, `cap:256x256x64:4096^3` before and after each chunk.
  Rounds 1 and 2 timed everything (15 minutes each), round 3 the 647 points whose variant / base ratio differed
  between rounds by more than sigma (3 minutes); warm compile 1.5 minutes (2,668 builds). Every point matched torch;
  the reference stayed at 92.4-95.7 us; no foreign process.
- **Noise:** sigma (90th percentile of |rel_1 / rel_2 - 1|) 1.12%, median 0.36%. Null control rsall+bpw / rsall:
  geomean 0.9999, 0.985-1.014, 3 of 92 outside sigma.

## 4. Results

A ratio is a variant's time over base's in the same block (median over rounds); below 1 is faster. Geomean [best,
worst] over every owner point of the tile (one-tile, 2t, sp, cap):

| tile | b | bp | bpw | s | ps |
|---|---|---|---|---|---|
| 128x128x128 | **0.971** [0.883, 1.015] | 0.971 [0.788, 1.015] | 0.977 [0.880, 1.016] | 0.976 [0.884, 1.009] | 0.977 [0.892, 1.024] |
| 128x128x64 | **0.961** [0.819, 1.006] | 0.954 [0.701, 1.007] | 0.967 [0.838, 1.005] | 0.975 [0.880, 1.006] | 0.970 [0.851, 1.021] |
| 256x128x64 | 0.988 [0.922, 1.009] | 0.984 [0.834, 1.019] | 0.990 [0.913, 1.027] | **0.981** [0.924, 1.006] | 0.981 [0.908, 1.006] |
| 128x256x64 | 0.986 [0.914, 1.009] | 0.983 [0.834, 1.018] | 0.988 [0.898, 1.022] | **0.980** [0.915, 1.003] | 0.980 [0.895, 1.016] |
| 256x256x64 | 1.000 [0.989, 1.015] | 0.997 [0.947, 1.034] | 1.020 [0.967, **1.321**] | **0.991** [0.956, 1.012] | 0.991 [0.948, 1.022] |

- **Change 1 (b) is the win, and it is where the doc expected.** One-tile owner on the 128x128 tiles: 0.967 / 0.956
  (BK 128 / 64), up to 12-18% faster; 2t and sp 0.966-0.975; 1.4-1.6% on the rectangular tiles; 256x256x64 compiles
  to v27's code and times level (0.999-1.000). The gain falls with full waves (0 waves 0.933-0.944, 4 waves
  0.984-0.991 on one-tile owner at 128x128), since the ending is a smaller share of a longer run.
- **The anchor, 1024x1024x7168 at 128x128x128 (TFLOPS):** owall (4 per tile) 496.4 -> **538.0**, ow3 473.6 -> 498.8,
  ow2 388.8 -> 398.1; rsall 550.2. owall on 4 is now within 2.3% of our reduce-scatter ending (it was 10.8%).
  - **hipBLASLt was not timed in this run.** Its 546.0 (4 per tile) and 407.0 (2 per tile) are the doc's, from another
    session, GPU (physical 1) and harness (cold). On v27's code this run reads 1.1-2.5% higher than that session, by a
    different amount per kernel (section 6), so the raw ratios (-1.5%, -2.2%) overstate how close b is.
  - Scaled by the same kernel's offset, b is about 525 against 546 on 4 (about -3.9%; v27 -11.3%) and about 394
    against 407 on 2 (about -3.3%; v27 -5.6%).
  - **Same session and GPU against the installed hipBLASLt 1.3.2** (section 6): b is 3.9% behind on 4 programs per
    tile (v27's ending 11.2%) and 1.3% behind on 2 (3.4%).
- **Change 2 (p) depends on the number of peers** (bp / b, geomean by programs per tile, report "By programs per
  tile"): 0.87-0.96 at 16 and 64 programs per tile (0.96-1.00 at 32; 1-8 points each), 0.99-1.00 at 8, and level to
  3% slower (0.999-1.030) at 2-4, on every tile class. With many peers, the owners' polling atomics on the shared flag line delay the peers' flag sets (the doc's
  side effect); with few, the single-lane load spin plus barrier is slightly slower than v27's poll. On average it is
  a wash (bp / b 0.994-1.002 on one-tile owner), and at 256x256 it has points both 5% faster and 3.4% slower, which
  fails the regression rule.
- **Change 3 (w) does not pay.** On top of bp it is neutral to worse on one-tile (1.005-1.012 at 128x128), and it
  costs the 256x256 end-to-end splits 34 spills: 2t / sp up to 1.39x slower (geomean 1.069). The doc's hope was that
  reading early peers overlaps later peers' stores; here the peers finish together. That fits the doc's note that it
  was tried before and was slower.
- **Change 4 (s) is the only change that helps 256x256.** Where it applies (up to 8 programs per tile) it beats base
  by 1.4-1.6% at 256x256 (ow2, ow3) and 1.5-5.5% on the rectangular tiles, and is 0.3-2% ahead of b on those tiles; at
  128x128 it is 0.3-1.3% behind b (more peers' loads, but each batch only one quadrant). Above 8 programs per tile it
  is v27's code. ps adds p's pattern on top (no gain at 2-4 programs, where s applies).

**Verdict** (report "Verdict per variant": 128x128 geomean below 1; no 256x256x64 point worse than base by more than
2 sigma, 1.0224; 256x256x64 geomean at most 0.5% above base. The plan said "within 0.5%"; read as the regression guard
it was meant as, so a faster 256x256 passes):

| variant | 128x128 | 256x256x64 geomean / worst | rect | accept |
|---|---|---|---|---|
| b | 0.966 | 1.000 / 1.015 | 0.987 | yes |
| bp | 0.962 | 0.997 / 1.034 | 0.983 | no (256x256 worst) |
| bpw | 0.972 | 1.020 / 1.321 | 0.989 | no |
| s | 0.975 | 0.991 / 1.012 | 0.981 | yes |
| ps | 0.974 | 0.991 / 1.022 | 0.981 | yes (borderline worst) |

## 5. What to do with it

- **Take b, in its asm-free form (section 7).** It is the doc's main change, it gives what the doc predicted at
  128x128 (two thirds of the owner ending's deficit to hipBLASLt 1.3.2 removed at both 4 and 2 programs per tile, same
  session; section 6), helps the rectangular tiles, and leaves 256x256 untouched. Without the inline-asm tie it keeps
  all but 0-0.6% of that.
- **s alone does not replace b** (section 7): at 128x128 it is level to 1.3% behind b at 2-8 programs per tile, 2-22% behind
  beyond 8 (where it is v27's code), and does nothing for two-tile and spread. It is level with or slightly ahead of b on
  the rectangular tiles, and the only change that helps 256x256.
- **Better still: b and s together (bs, section 8).** b's batches inside s's compile-time peer loop are the fastest
  form on every tile timed: 1-2% ahead of b at 2-4 programs per tile, level with it beyond, no spills; at 256x256 bs is
  s. Same session against hipBLASLt 1.3.2, owall on 4 programs per tile is 3.8% behind (b 4.7%) and ow2 on 2 is 0.8%
  behind (b 2.0%).
- **p only with many peers** (16 or more programs per tile, which capped's owner choices never use: it picks the owner
  ending only at small n). So p does not matter for the deployed rules; drop it, or gate it on n >= 16.
- **Drop w.**
- **The capped crossover** (owner against reduce-scatter by n; the doc's "wider check") is not re-fitted here. owall
  on 4 at the anchor is now 538 against rsall's 550 (it was 496), so the gap that made capped prefer reduce-scatter
  from 4 programs per tile shrinks from 10.8% to 2.3%.

## 6. Against hipBLASLt in the same session (2026-10-19)

### The installed hipBLASLt 1.3.2 ([hbl_current.md](results/hbl_current.md), [hbl_current.py](hbl_current.py))

1024x1024x7168 on HIP 1, `/opt/venv` (torch 2.13.0+rocm7.14.0, hipBLASLt 1.3.2), every method timed the same way in one
process (CUDA graphs of 100 launches over 2 GB of rotating inputs, 5 shuffled rounds), five sessions: three at the
default grid, `TENSILE_STREAMK_FIXED_GRID` = 128 and 64. hipBLASLt runs one `MT128x128x128 ... SK5 ... SKXCCM8` kernel
on 256, 128 or 64 workgroups (a newer stream-K variant than 1.2.2's SK3, same tile and XCD mapping). All correct; the
v28 points agree across sessions to 0.6%.

| programs per tile | hipBLASLt | v27's ending | b | s | reduce-scatter |
|---|---|---|---|---|---|
| 4 (default grid) | 561.0 TFLOPS | 0.888 (+3.37 us) | **0.961 (+1.09 us)** | 0.952 (+1.35 us) | 0.992 (+0.23 us) |
| 2 (grid 128) | 403.1 | 0.966 (+1.32 us) | **0.987 (+0.49 us)** | 0.981 (+0.72 us) | |
| 1 (grid 64, no fix-up) | 254.0 | dp 1.031 (-1.75 us) | | | |

- b removes two thirds of the owner ending's deficit at both splits; s a little less.
- Our main loop is 3.0% faster at one program per tile. Allowing for that (about 0.45 us per program on 4, 0.9 us on
  2, if it scales as 1/n), hipBLASLt's ending is still about 1.4-1.5 us cheaper than b's at both splits: a fixed cost
  per tile, not one per peer.
- v28 as deployed (auto: 64x128x128, owner on 2) is 12.9% behind (10.9% with b) on this shape, mostly the tile choice.

### hipBLASLt 1.2.2 (the doc's version): prepared, not run

The like-for-like check against the doc's numbers (hipBLASLt 1.2.2's SK3 kernel in the doc's cold harness, physical
GPU 0) is prepared but could not run here; details and commands in [hbl_same_session.md](results/hbl_same_session.md).
- **Why not run:** this container has neither the doc's environment (`/home/jbowley/rocm724_env`: torch 2.10.0+rocm7.2.4,
  hipBLASLt 1.2.2) nor `/opt/rocm-7.2.4`. Its only torch, `/opt/venv` (2.13.0+rocm7.14.0), runs this shape as split-K
  (`MT128x256x64 ... GSU` and a `PostGSU8` reduction kernel, about 436 TFLOPS warm), not as the stream-K owner kernel,
  so it is no reference for the owner ending.
- **Ready:** [run_v20_vs_aiter.py](../../../../../experiments/aiter_lixun_shapes/run_v20_vs_aiter.py) takes `--kernel v28`
  (as v27, with v28's live kernel as the snapshot), HIP 1 (PCI 0x05) and `STREAMK_OWNER_OPT` in a pick; picks
  [v28_sk1024_picks.csv](results/v28_sk1024_picks.csv) (rsall, owall, owall+b, ow2, ow2+b, owall+s, ow2+s). Checked
  statically, not run end to end.
- **The offset between the two setups, on v27's code** (this run over the doc's session): owall x1.025, ow3 x1.009, ow2
  x1.011, rsall x1.016. It differs per kernel, which is why section 4's estimate scales each point by its own kernel's
  offset and is only an estimate.

## 7. b without inline asm, and s against b (2026-10-20)

**Sections 3-6 timed b with an inline-asm tie** (`together2` / `together4`: an empty asm that took and returned a
batch's quadrants). That form is gone. b is now the batch alone, its loads written before its adds, with no asm.

- **The problem.** Change 1 needs a peer's 16 loads per wave (128x128) in flight before the first wait. LLVM's
  scheduler decides that, not the source: it reorders loads and adds within a basic block, and 16 loads in flight need
  about 64 more VGPRs.
  - With the source written load, add, load, add (the first build, section 1), 128x128x128 and the rectangular tiles
    batched anyway, but 128x128x64 reused one quadrant's registers for the next and waited for each (4 loads per wait,
    as v27). Likely because 128x128x64's LDS (74 KB) leaves room for two workgroups per CU, so LLVM budgets for two
    waves per SIMD; 128x128x128's (140 KB) allows one.
  - The tie forced the batch. Writing the four loads before the four adds, which went in together with the tie, had
    never been built without it.
- **Loads first is enough** ([asm_owner_noasm.md](results/asm_owner_noasm.md)): b without the tie issues 16 partial
  loads per `vmcnt(0)` at every tile below 256x256, 228 VGPRs at 128x128x64 (as with the tie), no spills, 256x256 still
  v27's code. The schedules differ from the tied build's elsewhere (two-tile and spread batch 32 loads per wait).
  `waves_per_eu = 1`, the planned fallback, was not needed. Stress: fp32 owner checks on 128x128x64, 128x128x128,
  256x128x64 and no-full 128x128x128 / 256x256x64, 384 configurations, all PASS (`results/checks_noasm/`). After
  removing the tie code, every build's registers and batches are unchanged and the default build is still v27's.
- **Timing** ([owner_opt_noasm/report.md](results/owner_opt_noasm/report.md); `owner_opt.py --tiles
  128x128x64,128x128x128,256x128x64 --variants ba,b,s`, where `ba` was the tied b): 68 shapes, 2,583 timings in three
  rounds, all correct, sigma 1.12%, null control 1.0000.

  | tile | tied b / base | b / base | s / base | b / tied b | s / b |
  |---|---|---|---|---|---|
  | 128x128x64 one-tile owner | 0.957 | 0.963 | 0.974 | 1.006 | 1.011 |
  | 128x128x64 2t / sp | 0.966 | 0.972 | | 1.006 | |
  | 128x128x128 one-tile owner | 0.966 | 0.968 | 0.973 | 1.002 | 1.006 |
  | 128x128x128 2t / sp | 0.977 | 0.975 | | 0.998 | |
  | 256x128x64 one-tile owner | 0.985 | 0.981 | 0.981 | 0.996 | 1.001 |
  | 256x128x64 2t / sp | 0.990 | 0.991 | | 1.000 | |

  - **Without the tie b keeps nearly all of its gain:** 0.6% behind the tied form at 128x128x64 (1.2% on the shapes
    without a full wave: 0.945 against 0.934 of base), level at 128x128x128, 0.4% ahead at 256x128x64. Anchor (1024x1024x7168, owall at 128x128x128): 533.4 tied, 531.7 untied
    TFLOPS (base 493.5). Against section 6's hipBLASLt 1.3.2 that puts untied b about 4.2% behind on 4 programs per
    tile (tied: 3.9%), by the same-block ratio.
  - **s against b, by programs per tile** (report "By programs per tile"): at 128x128, s is level with b to 0.6% behind at 2-8
    programs per tile (0.3-1.3% behind in section 4's sweep) and 2.3-22% behind at 16-64 (s is v27's code there); on the rectangular tile it is 0.1-1.1% ahead
    at 2-4 and 0.7-2.7% behind at 8-16. So s alone does not match b at 128x128, and the case for s is 256x256 and the
    rectangular tiles at few programs per tile (section 5).

## 8. b and s together: bs (2026-10-20)

`STREAMK_OWNER_OPT=bs`: b's batch of QPB quadrants per peer (16 fp32 loads per wave) inside s's compile-time peer loop
(up to 7 peers, those past `n_peers` masked off), so several peers' whole partials can be in flight before a wait. No
code change: `owner_fixup_tile` already passes `MAX_PEERS` to the QPB-quadrant `peer_sums` calls. Above 8 programs per
tile it is b's run-time loop, and at 256x256 (QPB = 1) it is s.

- **Why more can help even past the in-flight limit.** One CU keeps about 48 KB of loads outstanding (the litmus
  write-up's 71 GB/s per CU with 16 loads in flight, at about 0.7 us latency). v27 issues 16 KB per CU and then
  `vmcnt(0)`: the queue drains to empty and every 16 KB pays a full latency. A longer batch issues the excess at the
  queue, keeps it at the limit, and starts the first quadrant's adds while the rest arrive; past the limit the only
  further gain is fewer drains between batches.
- **ISA** ([asm_owner_bs.md](results/asm_owner_bs.md)), loads per `vmcnt(0)`: 128x128x128 16 / 48 / 112 at 2 / 4 / 8
  programs per tile, one wait each, at 256 / 284 / 424 VGPRs; 128x128x64 16 / 48 / 76 (two waits at 8) at 192 / 264 /
  418 VGPRs; rectangular tiles 32 / 48 / 112. No spills anywhere. At 128x128x64 more than 256 VGPRs means one
  workgroup per CU, which stream-K runs anyway.
- **Stress** (`results/checks_bs/`): owner ending, fp32, bf16 and bf16-pair partials, on 128x128x64, 128x128x128,
  256x128x64 and no-full 128x128x128 / 256x256x64: 1,152 configurations, all PASS.
- **Timing** ([owner_opt_bs/report.md](results/owner_opt_bs/report.md); `--variants b,s,bs`, the same 68 shapes, three
  rounds, 2,349 timings, all correct, sigma 1.17%, null control 1.0007):

  | tile, one-tile owner | b / base | s / base | bs / base | bs / b |
  |---|---|---|---|---|
  | 128x128x64 | 0.963 | 0.973 | **0.953** | 0.989 |
  | 128x128x128 | 0.969 | 0.975 | **0.960** | 0.991 |
  | 256x128x64 | 0.980 | 0.981 | **0.973** | 0.992 |

  By programs per tile, bs / b: 0.981-0.990 at 2-4 on the 128x128 tiles and 0.985-0.993 on the rectangular one,
  0.998 at 8, level above 8 (the same code). Two-tile and spread have no compile-time peer count, so bs is b there.
  Even with one peer (2 programs per tile, 16 loads per wait for both) bs is 1.1-1.2% ahead of b: the unrolled form
  has no peer loop, and at 128x128x64 it uses fewer VGPRs (192 against 228).
- **Same session against hipBLASLt 1.3.2** ([hbl_current.md](results/hbl_current.md), sessions `bs_def1`, `bs_g128`;
  b here is the asm-free form):

  | programs per tile | hipBLASLt | v27's ending | b | bs |
  |---|---|---|---|---|
  | 4 (default grid) | 560.6 TFLOPS | 0.885 | 0.953 | **0.962** (+1.07 us) |
  | 2 (grid 128) | 404.2 | 0.961 | 0.980 | **0.992** (+0.29 us) |
  | 3 (ow3; default grid, no hipBLASLt counterpart) | | 472.8 | 495.2 | **505.1** |

  At 2 programs per tile bs is within 0.8% of hipBLASLt, and our main loop is 3% faster at one program per tile, so
  its ending is now at most a fraction of a microsecond more than hipBLASLt's there. On 4 the gap is 1.07 us.

## 9. A register prefetch in the run-time peer loop: f (2026-10-20)

b's run-time loop counts down to `vmcnt(0)` in every peer's trip: each trip's adds need all of its own 16 loads, and
the next peer's loads are only issued after the branch back. The main loop avoids this with async copies into LDS
(`wait_group(N)` is an explicit `vmcnt(N)`); for loads into registers the waits are LLVM's, set by which registers each
instruction reads. `STREAMK_OWNER_OPT=bf` (b only, so not at 256x256) loads the next peer's batch before adding the
current one, so a trip needs only the loads of the trip before.

- **Three things in the first forms drained the queue anyway** (the asm, not the timing, showed it):
  - `x = y` at the end of the trip: the loop carried the prefetch in other registers, and 64 `v_mov`s copying it back
    had to wait for its loads (`vmcnt(15..0)`). Fix: two batches in turn, two peers per trip, each reloaded right
    after its last use.
  - `other=0.0` on the masked loads: every register went through a `v_cndmask` select that waited for its load.
    A masked-off buffer load already returns 0, so the loads have no `other`.
  - Masks inside the loop: the masked offsets were recomputed every trip into registers that the allocator had given
    the previous batch, so each trip began with `vmcnt(0)`. Fix: the loop runs only the trips whose two peers exist,
    unmasked; the first peer and the last odd one are loaded, masked, outside it.
- **The final loop** (`peer_batch`, `peer_sums`): no copies, and no wait inside it below `vmcnt(8)` (at least 8 loads
  per wave in flight across every trip; mostly 16 or more). 300-420 VGPRs, no spills
  ([asm_owner_pf.md](results/asm_owner_pf.md); every default build v27's). Stress (`results/checks_pf/`): bf and bsf,
  all partial types, 128x128x64, 128x128x128, 256x128x64 and no-full 128x128x128 / 256x256x64, 2,304 configurations,
  all PASS.
- **Timing** ([owner_opt_pf/report.md](results/owner_opt_pf/report.md); `--variants b,bf,bs,bsf`, the same 68 shapes,
  three rounds, 2,960 timings, all correct, sigma 1.14%, null control 1.0003): bsf is bs up to 8 programs per tile
  and bf above.

  | | 128x128 one-tile, 2-8 per tile | 128x128 one-tile, 16-64 | 128x128 two-tile / spread | rectangular one-tile | rectangular two-tile / spread |
  |---|---|---|---|---|---|
  | bf / b | 1.000-1.006 | 0.997-1.002 | 1.013-1.018 | 1.004-1.026 | 1.017-1.018 |

  **The prefetch gains nothing and costs up to 2.6%.** Even above 8 programs per tile, with up to 63 peers, removing
  the drain between peers is level with b; on the end-to-end splits and the rectangular tile it is slower (more VGPRs,
  more address arithmetic). bsf against bs is 0.997-1.004 where it is the same code and level above 8.
- **Why: the drain between peers is not where the time goes once a batch is a whole partial.** A 128x128 peer is 64 KB
  per CU, past the ~48 KB a CU keeps in flight, so within a batch the queue is already full; the drain at its end costs
  only the tail of one batch. That also says where bs's 1-2% over b came from: not drains (bs / b is 0.989 with one
  peer, where both have one batch and one wait, and 0.998 at 8 peers) but the unrolled form itself (no loop, fewer
  VGPRs at 128x128x64).
- **Not adopted.** bs stays the proposal (section 8).

## 10. v28 is bs (2026-10-20)

v28's kernel now has one owner ending, bs, with no switch; the other variants (p, w, f, and the switch itself) are
gone from it.
- **The archive:** [experiments/matmul_kernel_variants.py](experiments/matmul_kernel_variants.py) is the kernel of
  sections 1-9, with every `STREAMK_OWNER_OPT` letter (sha256 `f62cfb4ee693d6de`). The tools load it with
  `V28_MATMUL_MODULE=v28_streamk_owner.experiments.matmul_kernel_variants` (sweep.py and so owner_opt.py and
  hbl_current.py, check_streamk.py, check_grid.py; `run_checks.sh VARIANTS` sets it for any variant but `v28`;
  `asm_owner.py --archive / --compare`). owner_opt.py refuses variants without it, and v28's host refuses
  `STREAMK_OWNER_OPT`.
- **The code** ([matmul_kernel.py](matmul_kernel.py)):
  - host `owner_options`: `OWNER_QPB = min(4, 16 // (BM BN / 4096))` quadrants per batch (4 at 128x128, 2 at
    256x128 / 128x256), nothing passed at 256x256 (1);
  - kernel, tile-aligned split: `OWNER_MAX_PEERS = N_MAX - 1` if `N_MAX <= 8`, else 0 (`N_MAX`: the most programs a
    tile has);
  - owner branch: `owner_fixup_tile` (contributor stores, `peer_sums` batches, a batch's C stored before the next)
    when `OWNER_QPB > 1` or `OWNER_MAX_PEERS > 0`, v27's four `fixup_quadrant` calls otherwise (256x256 above 8
    programs per tile and end to end). v27's poll, flag set and re-arm are unchanged.
- **It is the code that was timed** ([asm_owner_v28.md](results/asm_owner_v28.md), `asm_owner.py --compare bs`):
  34 of the 35 owner builds of the five large tiles are instruction-identical to the archive's bs; the other (256x256
  spread) gives the same two forms as the archive's in repeated fresh builds (v27's `v_bitop3` swap).
- **Stress** (`results/checks_v28/`): the owner ending on all five tiles with fp32, bf16 and bf16-pair partials, and
  no-full 128x128x128 / 256x256x64, 1,680 configurations; every ending, exchange and placement (the full default
  check) on 128x128x128 and 256x256x64, 1,808. All PASS.
- **Same session against hipBLASLt 1.3.2** (hbl_current.py `v28_def`, `v28_g128`), 1024x1024x7168 at 128x128x128:

  | programs per tile | hipBLASLt | v28 | v27's ending (section 8's sessions) |
  |---|---|---|---|
  | 4 (default grid) | 560.7 TFLOPS | **542.3 (0.967)** | 496.0 (0.885) |
  | 2 (grid 128) | 401.7 | **400.9 (0.998)** | 388.3 (0.961) |
  | 3 (ow3) | | 505.3 | 472.8 |

  v28 as deployed (auto: 64x128x128, owner on 2) runs 502.1 here (v27 486.4).

## Files

| Path | What |
|---|---|
| `results/hbl_current.md`, `results/hbl_current/*.json` | v28 against the installed hipBLASLt 1.3.2, same session (section 6) |
| `results/hbl_same_session.md`, `results/v28_sk1024_picks.csv` | the prepared run against hipBLASLt 1.2.2 (section 6) |
| `results/owner_opt_noasm/`, `results/asm_owner_noasm.md`, `results/checks_noasm/` | b without the asm tie, against the tied b and s (section 7) |
| `results/owner_opt_bs/`, `results/asm_owner_bs.md`, `results/checks_bs/`, `results/hbl_current/bs_*.json` | b and s together (section 8) |
| `results/owner_opt_pf/`, `results/asm_owner_pf.md`, `results/checks_pf/` | the register prefetch, f (section 9) |
| `experiments/matmul_kernel_variants.py` | the archived experiment kernel (sections 1-9) |
| `results/asm_owner_v28.md`, `results/checks_v28/`, `results/hbl_current/v28_*.json` | v28 as bs (section 10) |
| `results/asm_owner_*.md` | the ISA checks (asm_owner.py); before section 7, "b" is the tied form |
| `results/checks/` | the stress checks per variant and tile, `summary.txt` |
| `results/owner_opt/r1.csv`, `r2.csv`, `r3.csv` | every timing, by round (block = shape) |
| `results/owner_opt/report.md` | the sweep's report (owner_opt.py report) |
| `results/owner_opt/smoke_*.md` | the per-change smoke timings (3 rounds, before the sweep) |
