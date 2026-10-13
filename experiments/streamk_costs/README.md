# What stream-K's data movement costs on MI355X

The stream-K design notes ([STREAMK_EXPLAINED.md](../../kernels/gemm/intra_wave/a16w16/v14_streamk/STREAMK_EXPLAINED.md))
estimated three costs:

- storing or reading one 256x256 fp32 partial (256 KiB): 1-2 k-steps;
- a stream-K tile switch: 2-3 k-steps.

This experiment measures all of them inside v14's pipeline. The time unit is v14's own k-step.

```bash
export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so
unset TRITON_AMDGCNAS_PLUGIN             # see "Tooling" below
HIP_VISIBLE_DEVICES=2 python bench.py    # checks + all sweeps, about 40 s
```

Machine: MI355X (gfx950), 256 CUs, Triton v2.2 tutorial stack. Raw output of two full runs:
[results/run1.md](results/run1.md) and [results/run2.md](results/run2.md).

## TL;DR

One k-step is **1.27 us** in these builds (1.21 us for v14 with the amdgcnas peephole). All
numbers below are the mean of the two runs, in k-steps. They use lane-contiguous partials
(see below) unless noted.

| cost | alone (16 programs) | all at once (240-256 programs) | row-major partials, alone |
|---|---|---|---|
| store one partial | **1.5** | **4-11** for the first partial | 4.1 |
| owner reads one partial, in the fixup | **1.8** | n/a | 4.9 |
| owner reads one partial, from a workspace filled by the host | 2.8 | 1.8 | 5.7 |
| tile switch that stores a partial (fp32 + release flag) | **3.4** | 13-16 | 6.0 |
| tile switch that stores bf16 C (v14's persistent switch) | 2.8 | 4.6 | |
| segment boundary with no store (pipeline restart only) | | 1.2 | |

- **The estimates hold for a good partial layout and uncontended traffic.**
  - Storing a partial costs about 1.5 k-steps and reading one about 1.8-2.8.
  - A switch costs about 3.4, of which about 1.2 is restarting the pipeline and about 0.5 is
    the release drain.
- **Layout matters by 2-3x.** Writing the partial row-major (addressed from the MFMA layout)
  makes every 16-byte store instruction touch 16 half cache lines. With a lane-contiguous
  layout each wave writes 1 KiB contiguously. It is private workspace, so the port can choose
  the layout freely. That choice turns 4.1 k-steps into 1.5 for a store, 4.9 into 1.8 for a
  read, and 6.0 into 3.4 for a switch.
- **Concurrency matters as much as layout.**
  - In the one-tile regime, 240 contributors publish at the same moment. That is 60 MiB,
    and the first partial then costs 4-11 k-steps instead of 1.5.
  - The two-tile hybrid spreads its publishes over the range, so it pays the uncontended
    cost.
  - With the measured costs (burst store 6, spread store 1.5, read 2), the schedule model now
    ranks two-tile slightly ahead of tile-aligned reduce-scatter
    (`streamk_diagrams.py`, "measured" column).
- **A whole one-tile fixup matches the model.**
  - Groups of 16 (15 contributors publish, the owner reads 15 partials) add 45 us, or 35.6
    k-steps, on top of the owner's own work.
  - The model gives 6 + 15 x 2 = 36.
  - On 4352x4096x8192 that makes the tail 8 + 36 = 44 k-steps, against 128 for the
    data-parallel last wave.
- **Things that do not matter:**
  - `.wt` / `.cv` against default caching. The release and acquire carry the ordering, so both
    are correct, and they cost the same.
  - Flag polling, as long as the owner polls with relaxed atomics.
- **Things that matter a lot:**
  - Polling with acquire atomics. Every GPU-scope acquire invalidates the XCD's L2 under the
    neighbours' K loops. My first version did this, and it doubled the fixup intercept
    (42 us against 20 us for one peer).
  - Register pressure (see Caveats).

## Method

**Shape.** 4096x4096xK, bf16. That is 256 tiles of 256x256, one per program, so
`STREAMK_TILES = 0`. Every run is data-parallel GEMM plus a controlled amount of stream-K data
movement, and each cost is the **slope** over how many times that movement happens. Fitting
over 4-7 points averages out run-to-run noise.

**Timing.** Each point is the median per-launch time of 100 back-to-back launches, queued
behind a GPU sleep so that the host is ahead of the GPU.

- `triton.testing.do_bench` is not usable at this size. It zeroes a 256 MB buffer before every
  launch, and together with host launch overhead (about 150 us per launch through the Python
  wrapper) that puts a floor of about 105-150 us under a 170 us kernel.
- For example, v14 at K=4096 (both with the peephole) read 105-117 us with `do_bench` and 92
  us here.
- The `regress.py` / `bench.py` TFLOPS numbers for small shapes therefore carry about 5-10 us
  of that artifact.

**Kernel** ([kernel.py](kernel.py)). A copy of v14 with two generalisations:

- **Segments.** A tile runs as `SEGMENTS` K segments. Each segment is v14's tile body starting
  at an arbitrary even k-step, and its epilogue prefetches the next segment's first two
  k-steps. The accumulator restarts in every segment, as it does in stream-K.
- **Endings.** The end of every segment is picked by a constexpr `MODE`, with per-program run-time
  roles (selected by `spid % sel_mod`, where `spid` is the XCD-mapped pid that stream-K uses):
  - `STORE`: v14's interleaved C stores, then up to 4 partial stores and a release flag.
  - `FIXUP`: a contributor stores its partial and a release flag. The owner polls relaxed,
    acquires once, sums its peers' partials in VGPRs, adds them into the accumulator and stores
    C after the last MFMA.
  - `SWITCH`: at each segment boundary, store nothing, store bf16 C, or store a partial with a
    release (or relaxed) flag.

Partials are four 128x128 fp32 quadrants in the accumulator's own MFMA layout. They are
stored straight from the AGPRs, with no `convert_layout`. Two workspace layouts:

- **row-major**: offset `row * 128 + col`.
- **lane-contiguous**: `[16 register vectors][4 warps][64 lanes][4 fp32]`. This is a bit
  permutation of (row, col) read off the layout's bases (`gl.to_linear_layout`):
  - registers: col 1, 2, 32, 64 and row 32, 64;
  - lanes: row 1, 2, 4, 8 and col 4, 8;
  - warps: col 16, row 16.

  The column term has to be written as `c + (terms constant over 4 columns)`. Written with
  shifts and masks, or as `c % 4 + ...`, axis analysis loses the 4-element contiguity, the
  accesses compile to single dwords, and the build spills heavily.

**Checks** (`--sweep check`, run before every sweep):

- the baseline C matches `A @ B`;
- a 2-segment run produces the second K half, for every switch mode;
- the stored partials match the fp32 reference tiles in both layouts;
- the fixup owner's C matches the sum of its group's tiles, and the flags are re-armed.

The fixup check is run for s = 2, 4, 8 and 16, in both layouts, with `.wt` / `.cv` and with
default caching.

## Results

### One k-step

| K | 1024 | 2048 | 4096 | 6144 | 8192 | 10240 | 12288 |
|---|---|---|---|---|---|---|---|
| us (run 1) | 33.1 | 56.4 | 103.4 | 141.7 | 179.5 | 216.9 | 260.0 |

The fit is 1.27 us per k-step (1.269 and 1.256 in the two runs) plus about 17 us of launch,
prologue and epilogue. With the amdgcnas peephole, stock v14 fits 1.21 us.

- The slope rises beyond K=12288: K=16384 costs 1.42 us per k-step, as the A/B working set
  outgrows the MALL and the CUs drift.
- The costs below are all measured at K=8192.

### Storing a partial

Each storer writes r partials to distinct slots, then drains and sets a release flag.

| storers | layout, store cache | first partial (r = 0 to 1), run 1 / 2 | slope over r = 0-4 |
|---|---|---|---|
| 16 | lane, `.wt` | 0.7 / 1.7 | 1.3 / 1.7 |
| 16 | lane, default | 3.3 / 6.1 | 1.3 / 1.7 |
| 16 | row-major, `.wt` | 2.3 / 2.5 | 4.1 / 4.0 |
| 240 | lane, `.wt` | 4.1 / 7.8 | 11.5 / 10.1 |
| 240 | row-major, `.wt` | 6.1 / 7.9 | 17.2 / 13.5 |
| 256 | lane, `.wt` | 7.5 / 10.7 | 11.8 / 10.8 |

- With 16 storers the per-CU store rate is about 120-150 GB/s with the lane layout and about
  50 GB/s with row-major.
- With 240-256 storers the first partial (60-64 MiB in total) is limited by chip-wide write
  bandwidth.
- The slope over r = 4 is superlinear: 256 storers x 4 partials is 256 MiB, which no longer
  fits the MALL. Stream-K stores at most one partial per program, so the first-partial
  column is the one that applies.
- The first-partial column is a single difference, so it is noisy (±3 us). The slopes
  reproduce.

### Reading partials

- **Read-only rows:** the host fills the workspace and presets the flags, so the owners only
  read.
- **Fixup rows:** contributors publish first, so the partials the owner reads were written a
  few microseconds earlier, as in real stream-K.

Slopes are fitted over n >= 1 (the n = 0 point skips the owner's per-quadrant setup).

| case | layout | slope per partial, run 1 / 2 |
|---|---|---|
| fixup, groups of s, x = s - 1 peers | lane | 1.78 / 1.87 |
| fixup, groups of s | lane, default caching | 1.81 / 1.75 |
| fixup, groups of s | row-major | 4.90 / 4.95 |
| read-only, 16 owners | lane | 2.71 / 2.80 |
| read-only, 16 owners, no flag polling | lane | 2.72 / 2.74 |
| read-only, 16 owners | lane, default caching | 2.85 / 2.86 |
| read-only, 16 owners | row-major | 5.63 / 5.72 |
| read-only, 256 owners | lane | 1.75 / 1.78 |
| read-only, 256 owners | row-major | 4.95 / 4.86 |

- **Fixup:** the lane-layout owner reads at about 110 GB/s per CU.
- **Read-only, 16 owners:** the curve is not linear.
  - Up to 8 partials each costs about 1-1.5 k-steps.
  - From 8 to 15 each costs about 4 k-steps.
  - The host-filled workspace (256 MiB) is not guaranteed to be in the MALL, which is the
    likely cause.
- **Read-only, 256 owners:** every slot is read by n owners, most of them on the same XCD, so
  repeats can hit L2. It is not a clean contended-read measurement.
- **Fixup intercept:** the jump from s = 1 to s = 2 is 9-11 k-steps (lane) or 11-14
  (row-major). That is 128 contributors publishing at once, the owner's wait and setup, and
  its first read.
- **Why reads are latency-bound:** the peer loop has one partial quadrant (64 KiB per CU) in
  flight. Keeping two in flight (`READ_DEPTH=2`, still in the kernel) needs 64 more VGPRs, and
  that build spills 211 VGPRs, so it is not measured.

### Tile switches

Each tile runs as S = 1, 2, 4 or 8 segments of 8192 / S. The switches happen at every segment
boundary of the selected programs, and every other program just continues its pipeline.

| switch | switchers | slope per switch, run 1 / 2 |
|---|---|---|
| none (pipeline restart only) | 256 | 1.13 / 1.21 |
| bf16 C store | 256 | 4.78 / 4.52 |
| bf16 C store | 16 | 2.80 / 2.73 |
| partial, lane, `.wt`, release | 16 | 3.28 / 3.44 |
| partial, lane, relaxed flag (no drain) | 16 | 3.08 / 2.61 |
| partial, lane, default cache, release | 16 | 2.85 / 3.33 |
| partial, row-major, `.wt`, release | 16 | 5.96 / 6.01 |
| partial, lane, `.wt`, release | 256 | 15.99 / 15.51 |
| partial, lane, relaxed flag | 256 | 10.45 / 9.91 |

- **The pipeline restart alone costs about 1.2 k-steps**, even though the next segment's
  first two k-steps are prefetched in the epilogue, as in v14's persistent loop. That is the
  cost of a segment boundary with no store at all.
- **A stream-K switch in the one-tile regime** (16 crossing programs) costs about 3.4 k-steps
  with lane-contiguous partials, or about 6 with row-major.
- **The release drain** is the `vmcnt(0)`, which also waits for the prefetch already in
  flight. It adds only about 0.5 k-steps.
- **When all 256 programs switch together**, each switch writes 64 MiB, and seven switches
  cycle through 256 MiB of slots. That is bandwidth- and MALL-bound. In the two-tile regime the
  switches happen at different times, so this row is an upper bound, not a prediction.

## What this means for the port

1. **Write partials lane-contiguous, not row-major.** The address is a bit permutation of the
   accumulator layout, it needs 16 offset VGPRs like row-major does, and it halves every cost
   above.
2. **Poll flags relaxed, then acquire once.** Acquire in the spin loop invalidates L2 under
   the rest of the XCD.
3. **`.wt` / `.cv` or default caching:** either works with a release / acquire pair, and they
   perform the same.
4. **Contention is the policy question.**
   - One-tile stream-K makes 240 contributors publish in the same few microseconds.
   - Two-tile spreads its publishes over the range, and reduce-scatter adds a concurrent read
     burst on top.
   - With the measured costs, the schedule model gives:

     | shape | one-tile (today) | tile-aligned reduce-scatter | two-tile |
     |---|---|---|---|
     | 4352x4096x8192 | 172 | 144 | 139.5 |
     | 4352x4352x8192 | 167 | 155 | 148.5 |

     These are critical paths in k-steps, against 256 for data-parallel.
5. **Register pressure is the practical limit for the fixup.**
   - A run-time loop, or a branch, that carries the AGPR accumulator spills heavily, so the
     peers are summed in VGPRs and added into the accumulator once.
   - A run-time loop around stores of the accumulator spills too, so the stores are unrolled
     and guarded.
   - The owner's C store has to come after the peer sum. Keeping v14's interleaved C store on
     one path and a late C store on the other costs 70-80 spilled VGPRs.

## Caveats

- **Spills outside the K loop.**
  - The `STORE`, `FIXUP` and `SWITCH` builds are at 512 VGPRs, with 16, 26 and 9 spilled VGPRs
    respectively and no scratch access in the K loop. The `BASE` build is v14 (488 VGPRs, no
    spills).
  - I checked the scratch ops' placement: none sit between partial stores or inside the peer
    loop. They sit in the prologue and between quadrants, so every point of a sweep (one binary)
    pays them equally, and they cancel in the slopes.
  - A real port with fewer knobs should have less pressure.
- **Default caching is checked by a single launch per configuration.** It relies on the release
  (`buffer_wbl2`) and the acquire (`buffer_inv`) for cross-XCD visibility. Stress-test it before
  relying on it.
- **Scope of the shape.** Everything is measured at 4096x4096x8192 with one tile per program.
  In a real stream-K tail, the neighbours' traffic and the MALL contents differ, and so do the
  exact absolute numbers.

## Tooling

- **amdgcnas emitted invalid buffer ops.** The peephole
  ([plugins/amdgcnas/amdgcnas_ext.py](../../plugins/amdgcnas/amdgcnas_ext.py)) re-emitted
  cache-policy modifiers on buffer instructions with a comma (`offen, sc0, sc1`), which the
  assembler rejects. That breaks any buffer op with `.cv`, `.wt` or `nt`. It is fixed: `buffer_`
  is now handled like `global_`, `flat_` and `scratch_`.
- **amdgcnas faults on these builds.** Even after that fix, the builds with spills fault on the
  GPU (memory access fault) when the peephole is enabled, and run correctly without it. I have
  not isolated which rewrite is responsible; `bench.py` refuses to run with
  `TRITON_AMDGCNAS_PLUGIN` set.
