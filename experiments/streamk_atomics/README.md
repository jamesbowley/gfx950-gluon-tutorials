# Atomic stream-K tile endings on MI355X

Can a stream-K tile with only a few fixups (1-3 peers) end with an atomic add instead of a partial
store and an owner read? The design notes
([STREAMK_EXPLAINED.md](../../kernels/gemm/intra_wave/a16w16/v14_streamk/STREAMK_EXPLAINED.md))
measured today's exchange at about 1.5 k-steps to store a 256x256 fp32 partial and 1.8 to read it.
The only earlier atomic number, the split-K proxy in
[hbm_bandwidth_xcd](../hbm_bandwidth_xcd/README.md) (sweep F), gave about 330 GB/s of fp32
atomics chip-wide. But it used 4 fp32 per lane and 64-256 programs at once, and it never measured
what one CU can do on its own.

```bash
HIP_VISIBLE_DEVICES=0 python bench.py                          # check, asm, warm sweeps (about 5 s)
HIP_VISIBLE_DEVICES=0 python bench.py --sweep main pair --cold # evict L2 and the MALL before every launch
```

Machine: MI355X (gfx950), 256 CUs, GPU 0, Triton 3.8 Gluon, no LLIR or amdgcnas plugins. Raw
output: [results/run1.md](results/run1.md) and [results/run2.md](results/run2.md) (warm), and
[results/cold.md](results/cold.md). One k-step is 1.21 us, v14's.

## TL;DR

- **The plan's decision rule fails, so there is no v19 prototype.**
  - A 128 KiB bf16 `pk_add` from one CU costs **4.8 us (3.9 k-steps)**. The bar was about 2 us.
  - The best fp32 atomic add of a 256 KiB tile costs **7.2 us (5.9 k-steps)**. The bar was 1.5 us.
- **fp32 atomics lose outright.** One CU atomically adds a 256 KiB tile in 7.2 us at best, 4x a
  plain store (1.9 us). That needs a layout where a wave's 64 lanes hit 64 consecutive dwords.
  - In the accumulator's natural 16-bytes-per-lane map it takes 13 us.
  - Chip-wide, fp32 atomics saturate at about 1.35 TB/s, coalesced, or 320 GB/s in the
    16-byte-per-lane map. Stores reach 8.2 TB/s. The 320 GB/s reproduces the old split-K
    proxy, which used the same access pattern.
- **The threshold in that rule was too strict, though.** It compared `pk_add` with the owner's
  partial read alone, but `pk_add` straight into C also replaces the owner's C store.
  - On the whole one-peer handshake, bf16 into C wins by about 1 k-step per tile at S <= 8, and
    by 2-4 k-steps at S = 16-32.
  - In that design the contributor stores its partial as bf16 into C and raises a flag. The
    owner then does `pk_add` of its accumulator into C.
  - For a reversed two-tile kernel (v19, S <= 16) that is about 1% of the kernel, paid for with an
    extra bf16 rounding of the partial. See the decision section.
- **Several writers on one tile cost nothing extra.** With 1, 2 or 4 writers per tile, on
  different XCDs, the time per program is the same. The limits are the issuing CU and the
  chip-wide atomic rate, not serialisation on an address.
- **Device-scope atomics are correct across XCDs** on ordinary `torch` (coarse-grained)
  allocations. Writers of one tile ran on 2-4 different XCDs, and every sum was exact.

## Method

**Kernel** ([kernel.py](kernel.py)). No GEMM. Every active program loads a synthetic 256x256
fp32 "accumulator" in v14's MFMA layout: four 128x128 quadrants, 4 warps, 64 fp32 per thread
per quadrant. Then it runs one ending, picked by a constexpr `MODE`:

| mode | ending | bytes per program |
|---|---|---|
| `store` | fp32 partial store, lane-contiguous (v16+) | 256 KiB |
| `load` | fp32 partial read of a host-filled slot, lane-contiguous | 256 KiB |
| `owner` | read one fp32 partial, add, convert, store bf16 C (v14's C layout) | 256 + 128 KiB |
| `c_store` | bf16 C store, v14's layout (convert to 8 bf16 per lane) | 128 KiB |
| `atom_lane` | fp32 `buffer_atomic_add` in the lane-contiguous map: each lane adds 4 consecutive dwords, so one instruction spans 1 KiB with 16-byte gaps | 256 KiB |
| `atom_coal` | fp32 `buffer_atomic_add` in a coalesced map: for each register, 64 lanes on 64 consecutive dwords (256 B), a bit permutation of the MFMA layout | 256 KiB |
| `c_atom_acc` | bf16 `pk_add` into row-major C, straight from the MFMA layout | 128 KiB |
| `c_atom_coal` | bf16 `pk_add` into row-major C after `convert_layout` to 2 bf16 per lane, 64 lanes along a row | 128 KiB |
| `pair_today` | writer 0: `store`, drain, release flag. Writer 1: poll relaxed, acquire, then `owner` on writer 0's partial | |
| `pair` | writer 0: `c_store`, drain, release flag. Writer 1: poll, acquire, then `c_atom_coal` | |

- **Timing.** `s_memrealtime` (100 MHz) stamps before and after the ending, each after
  `s_waitcnt vmcnt(0)` and a workgroup barrier. The interval covers issue and completion of
  every store or atomic.
  - The pair modes stamp a third time, between the flag and the owner's ending.
  - The accumulator's loads are drained before the first stamp.
  - A `"; pin"` asm on the loaded partial keeps `load` from being sunk into a dead branch. The
    first version of this mode measured 0.2 us because of exactly that.
- **Placement.** Programs `pid < active` take part, so round-robin dispatch spreads them over
  the XCDs. Active program i adds into tile `i // writers`, so the writers of a tile sit on
  different XCDs (the check verifies it from `HW_REG_XCC_ID`).
  - All active programs pass a grid barrier before the first stamp, so they really run their
    endings at the same time. The start skew is at most 0.3 us.
- **Statistics.** 20 launches per point after a warm-up. Each value is the median over all
  active programs and launches.
- **Caches.** Warm: back-to-back launches, so the atomics' and stores' targets are the previous
  launch's lines. Cold: a 1 GB read before every launch evicts L2 and the 256 MB MALL.
- **Atomics.** `sem="relaxed"`, `scope="gpu"`, result unused. The assembly has no `sc0` or `sc1`
  on any atomic, so they are the no-return, device-scope form. Per thread:

  | mode | atomics | stores |
  |---|---|---|
  | `store` | | 64 `buffer_store_dwordx4` |
  | `atom_lane`, `atom_coal` | 256 `buffer_atomic_add_f32` | |
  | `c_store` | | 32 `buffer_store_dwordx4`, plus 33 + 33 LDS ops for the convert |
  | `c_atom_acc` | 128 `buffer_atomic_pk_add_bf16` | |
  | `c_atom_coal` | 128 `buffer_atomic_pk_add_bf16`, plus 33 + 33 LDS ops | |

  There is no wider fp32 atomic. Triton forces `vec = 1` for fp32 buffer atomics and `vec = 2`
  for bf16 (`LoadStoreOpToLLVM.cpp`). No build spills (264-348 VGPRs).

**Checks** (`--sweep check`). Inputs are small integers, so every sum is exact in fp32 and bf16.
- Each mode's output equals the reference tile or the sum of its writers' tiles.
- Atomics are checked with 1, 2 and 4 writers per tile, 3 launches each.
- The writers of every tile ran on distinct XCDs.

## Results

### One ending per program (warm)

us per program, k-steps in brackets; GB/s is chip-wide at 256 active programs.

| ending | KiB | 1 active | 16 active | 64 active | 256 active | GB/s at 256 |
|---|---|---|---|---|---|---|
| `store` fp32 | 256 | 1.88 (1.55) | 1.92 (1.59) | 2.40 (1.98) | 6.64 (5.49) | 8189 |
| `load` fp32 | 256 | 1.88 (1.55) | 1.84 (1.52) | 2.04 (1.69) | 8.36 (6.91) | 7154 |
| `c_store` bf16 | 128 | 2.64 (2.18) | 2.60 (2.15) | 2.68 (2.21) | 3.56 (2.94) | 8056 |
| `owner` (read + add + C store) | 384 | 5.00 (4.13) | 5.04 (4.17) | 6.16 (5.09) | 13.24 (10.94) | 6850 |
| `atom_lane` fp32 | 256 | 13.14 (10.86) | 16.76 (13.85) | 49.92 (41.26) | 191.76 (158.48) | 318 |
| `atom_coal` fp32 | 256 | 7.24 (5.98) | 7.20 (5.95) | 12.64 (10.45) | 47.76 (39.47) | 1348 |
| `c_atom_acc` bf16 | 128 | 15.16 (12.53) | 14.66 (12.12) | 24.74 (20.45) | 96.96 (80.13) | 337 |
| `c_atom_coal` bf16 | 128 | 4.80 (3.97) | 4.76 (3.93) | 6.72 (5.55) | 23.88 (19.74) | 1320 |

- **Store and read match the earlier measurement**: 1.55 k-steps each, against 1.5 and 1.8 in
  [streamk_costs](../streamk_costs/README.md).
- **Coalescing decides the atomic rate.**
  - When a wave's lanes hit consecutive dwords, an fp32 atomic instruction touches 2 cache lines.
  - In the accumulator's natural map it touches 8, each a quarter used. That map is 1.8x slower
    for one program and 4x slower chip-wide.
  - bf16 straight from the MFMA layout (4 bytes per lane spread over 16 rows) is 3x slower than
    after a `convert_layout`.
- **A single CU's atomic rate**, coalesced, is about 36 GB/s for fp32 and 27 GB/s for bf16
  including the convert. A store reaches about 140 GB/s. So the atomic version of a tile costs
  4x the store for fp32 and 1.8x the C store for bf16.
- **Atomics stop scaling at about 16-64 programs.** Stores keep scaling to 256.
- **Writers per tile do not matter.** At 16 active programs, `atom_coal` takes 7.20 us with 1, 2
  or 4 writers per tile, and `c_atom_coal` takes 4.76-4.80 us.

**Cold caches** ([results/cold.md](results/cold.md)) change the atomics little:
- `atom_coal`: 7.4 us for one program;
- `c_atom_coal`: 5.0 us;
- `atom_lane`: 16-19 us.

They hurt the read: a partial that has left the MALL takes 4.7 us (3.9 k-steps), and `owner`
10.4 us. A freshly published partial is not in that state, which is why the pair sweep below
is the more realistic comparison.

### One-peer handshake: today vs bf16 into C

S tiles, each with a contributor and an owner on different XCDs, both with their accumulator
ready. Columns:
- "publish": the contributor's store, drain and release flag;
- "wait": the owner's wait for the flag;
- "owner ending": everything the owner does after the acquire;
- "whole": the owner's end minus the pair's first start.

All values in k-steps, warm run 1; run 2 agrees within 0.2.

| S | today: publish | today: owner ending | today: whole | atomic: publish | atomic: owner ending | atomic: whole | saved |
|---|---|---|---|---|---|---|---|
| 1 | 2.68 | 5.22 | 8.33 | 3.04 | 3.88 | 7.38 | 0.95 |
| 4 | 2.64 | 5.06 | 8.20 | 2.98 | 3.93 | 7.43 | 0.77 |
| 8 | 3.11 | 5.07 | 8.69 | 3.11 | 3.97 | 7.58 | 1.11 |
| 16 | 4.10 | 5.09 | 9.79 | 3.37 | 3.97 | 7.97 | 1.82 |
| 32 | 6.58 | 5.22 | 12.46 | 3.93 | 3.97 | 8.66 | 3.80 |
| 64 | 5.39 | 5.98 | 12.37 | 4.96 | 5.26 | 11.27 | 1.10 |
| 128 | 9.62 | 7.67 | 18.38 | 6.78 | 9.55 | 17.72 | 0.66 |

With cold caches the "whole" columns are 7.93 against 7.37 at S = 1, 8.79 against 7.95 at S = 16,
and 10.04 against 8.56 at S = 32.

- **Today's owner ending costs 5.1-5.2 k-steps in the handshake**, 1 more than `owner` with a
  host-filled partial (4.1). The partial was just written from another XCD, so it does not come
  out of this XCD's L2.
- **The atomic owner ending is 3.9-4.0 k-steps up to S = 32.** That is about 1.2 k-steps cheaper,
  because the `pk_add` replaces both the 256 KiB read and the 128 KiB C store.
- **The atomic contributor publishes 0.3-0.4 k-steps slower at small S.** Its bf16 store needs a
  `convert_layout` through LDS.
- **At S = 16-32, today's publish grows** (2.7 to 6.6 k-steps warm, 2.3 to 4.0 cold). The
  publishes are several 256 KiB fp32 stores per XCD followed by GPU-scope releases. The bf16 one
  moves half the bytes and grows less. I have not isolated the cause; the release's L2 writeback
  (`buffer_wbl2`) is the likely suspect, since it is the part that depends on other programs'
  dirty lines.
- **From S = 64 the atomic owner ending degrades** (9.6 k-steps at S = 128) as the chip-wide atomic
  rate is reached. The two designs end up level.

## Decision

The plan's rule was:
1. prototype the one-peer bf16-into-C ending in v19 if a 128 KiB bf16 `pk_add` from one CU costs
   below about 2 us, the owner read it would replace;
2. consider fp32 atomics for v20's reduce-scatter slice below about 1.5 us per tile.

Measured: 4.8 us and 7.2 us. **Both fail, so per the plan I have not prototyped anything.**

The first threshold was too strict, and the handshake table shows why: the `pk_add` replaces the
owner's read *and* its C store (5.1-5.2 k-steps together in the real handoff), not the read alone.
Against that, the atomic ending wins.

- **Owner's ending:** about 1.2 k-steps cheaper.
- **Whole one-peer fixup:** 0.8-1.1 k-steps faster at S <= 8, 1.8 at S = 16, 3.8 at S = 32.
- **What that buys v19** (reversed two-tile, S <= 16): every program publishes first and owns a
  tile last, so it saves roughly 0.8-1.8 k-steps on a kernel of about 140-150 k-steps at
  K = 8192, **about 1%**.
- **What it costs:**
  - The contributor's partial is rounded to bf16 before the add. That means two roundings
    instead of one, though the result is deterministic: the order is fixed by the flag.
  - The owner needs a `convert_layout` through LDS (the A/B buffers are free by then).
  - It only works when C is bf16 or fp16; fp32 C would need fp32 atomics, which lose.
  - It only works with exactly one peer: a second contributor could not plain-store into C.
    With more peers, every writer would `pk_add` into a zeroed C, which needs the zeroing and
    rounds n times.

**fp32 atomics are not an option** for any stream-K ending on this hardware. A coalesced fp32
atomic add of one 256 KiB partial (5.9 k-steps) costs more than storing it and reading it back
(3.1 k-steps). The many-peer case saturates at 1.35 TB/s chip-wide, against more than 7 TB/s for
the reduce-scatter slice's plain loads.

## Caveats

- **Synthetic endings.** The real kernels interleave the C store with the last MFMAs, sit at
  about 500 VGPRs, and have neighbours whose K loops compete for the memory system. The absolute
  costs will differ; the ratios between endings should hold.
- **One chip, one allocation type.** Cross-XCD correctness of device-scope atomics was checked on
  `torch` allocations (coarse-grained) on GPU 0 only.
- **The cause of the S = 16-32 publish growth is unconfirmed** (see above).
