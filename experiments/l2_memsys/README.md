# How the gfx950 L2 handles written data

Three questions from v21 ([WORKLOG](../../kernels/gemm/intra_wave/a16w16/v21_streamk_l2_partials/WORKLOG.md))
and the [streamk_l2_partials](../streamk_l2_partials/README.md) probe:
1. Do written lines stay in L2, and does the store's cache policy (default LRU, or streaming
   `nt`) change how long?
2. When a line written into L2 is read back, does the read go to memory, and is it served from L2,
   the last-level cache (LLC, 256 MB, memory-side) or HBM?
3. What does writing the dirty L2 lines back at the end of the kernel cost?

```bash
HIP_VISIBLE_DEVICES=1 python bench.py --sweep m1 m2 m3 m4 m5   # about 2 minutes
```

Machine: MI355X (gfx950), 256 CUs, GPU 1 for every measurement, Triton `gfx950-tutorial-v2.2` +
deepbind shim, no plugins. Raw output: [results/m1.md](results/m1.md) (and
[results/m1_forward.md](results/m1_forward.md)), [results/m2.md](results/m2.md),
[results/m3.md](results/m3.md), [results/m4.md](results/m4.md), [results/m5.md](results/m5.md).

## TL;DR

1. **Written lines stay in L2 under LRU, until about 3-4 MiB per XCD of newer traffic pushes them
   out.**
   - Streaming stores (`.cs`, `sc0 nt`) to lines that are not in L2 are written back and dropped
     almost at once: 47% survive at 1 MiB and none once any other data arrives.
   - Write-through stores (`.wt`) never leave a readable copy in L2.
   - A streaming store to a line that is already in L2 keeps it.
   - An `nt` read does not demote a line.
2. **The first read of a line written into L2 always fetches the whole line from memory**, and
   from wherever the line's old contents live: HBM (369 ns per dependent load) after a cache
   flush, the LLC (225 ns) if it is there.
   - It does not matter how much of the line was written (whole lines, 64-byte halves, one
     dword, or written twice), nor whether the read covers only written bytes (M5).
   - It is a fill: one 128-byte memory read per line, no writes. The L2 merges its written bytes
     over the fetched copy, and the line is then valid, so later reads are L2 hits.
   - A line that was valid in L2 before the write (read first, or left by the previous launch)
     is served from L2 (85 ns).
   - That fetch is what made v21's slice no faster.
3. **The kernel-end write-back costs about 1-3 us on the kernel's last instruction, 0.6-1.3 us
   net.**
   - At most about 2 MiB per XCD is still dirty at kernel end; the L2 writes back the rest
     during the kernel.
   - A kernel whose stores stay in L2 ends 1-3 us later after its last instruction than one that
     writes through.
   - Its stores complete sooner, so the whole kernel is 0.6-1.3 us slower than the write-through
     version for 4-128 MiB written.
   - Nothing is left over for the next kernel.

## What the documentation says

The CDNA4 ISA guide (section 9.1.10.2, Tables 49 and 50) gives the policy for every combination of
a load's or store's cache bits. With more than one L2 (MI355X in single-partition mode):

| bits (Triton `cache=`) | L1 | L2 | LLC |
|---|---|---|---|
| none (default) | load Hit LRU, store Miss LRU | Hit LRU | Hit LRU |
| `sc0 nt` (`.cg` load, `.cs` load or store) | Miss Evict | Hit Stream | Hit Evict |
| `sc1` (device scope) | Miss Evict | Coherent Cache Bypass | Hit LRU |
| `sc0 sc1` (`.cv` load, `.wt` store) | Miss Evict | Coherent Cache Bypass | Hit LRU |

*AMD Instinct MI300 GPU Memory Bandwidth* (X. Lu, 2024, section 3.5.1) defines the L2 policies:
- **LRU:** a hit resets the line's age.
- **Streaming:** "identical behavior as LRU, except that the cache line age is not updated", so the
  line "aged out quickly".
- **Bypass:** a hit line is written back if dirty and invalidated, the data is fetched from memory,
  and the line is invalidated as soon as the data is returned.

It also describes the L2 as "writeback and write-allocate", with 16 channels of 128 sets x 16 ways,
128-byte lines, 64-byte write requests from L1, and memory reads of 32, 64 or 128 bytes.

These explain the policy results below: the streaming results of M1, and the `.wt` lines that M2 and
M3 never find in L2. Neither document describes the state of a line that stores have allocated,
or why its first read fetches it (M5); that is established only by the measurements.

## Method

[kernel.py](kernel.py) has three kernels. All timestamps are `s_memrealtime` (100 MHz, one clock
for the device), taken after `s_waitcnt vmcnt(0)` and a workgroup barrier.

- **`memsys_kernel` (M1, M2).** The active programs (here XCD 0's 32, `pid % 8 == 0`, checked with
  `HW_REG_XCC_ID`) run a fixed sequence of phases. Each phase is switched on by a constexpr and
  ends with a drain and a grid sync:
  1. pre-read: one dword of every line of the program's own region;
  2. write the region (cache policy, pattern, once or twice);
  3. first read;
  4. pollute: read E MiB of other data;
  5. second read.

  The reads cover every byte of the neighbouring program's region, so another CU reads what a CU
  wrote.
- **`chase_kernel` (M3).** One wave follows a chain of 4096 dependent loads, one per 128-byte
  line, over a random cycle through a 1 MiB buffer. Before it starts, helper programs set the
  chain up in the same launch: XCD 0's other 31 CUs write it, or XCD 1 reads or writes it.
- **`dirty_kernel` and `follow_kernel` (M4).** All 256 programs write D MiB in total, stamping
  their first and last instruction. A one-instruction follower runs straight after.

**Counters.** rocprofv3, one process per counter group:
- the configurations run in order, 6 launches each, a 1 GiB read before every launch;
- the medians of launches 2-6;
- a phase's counters are the difference between two configurations that differ only in that
  phase.

**rocprof's counter collection disturbs the cache state between kernels.** Without the 1 GiB read
between launches, the counters still show the read-back fetching every line from memory, while
in-process timing (M2) and the chase (M3) show L2 hits. So every counter configuration flushes
first, and conclusions about state carried between kernels come from in-process timing only.

## 1. Retention by cache policy (M1)

XCD 0 writes F MiB (its L2 is 4 MiB), reads E MiB of other data, then reads the F MiB back,
newest lines first (front to back thrashes LRU and shows only all or nothing:
[results/m1_forward.md](results/m1_forward.md)). Read-back hit %, from
[results/m1.md](results/m1.md) (the counter also sees a few hundred lines outside the region, so
the raw values reach 101-106%; they are shown as 100):

| written | F MiB | E = 0 | E = 1 | E = 2 | E = 3 | E = 4 |
|---|---|---|---|---|---|---|
| default | 1 | 100 | 100 | 100 | 100 | 2 |
| default | 2 | 100 | 100 | 100 | 22 | 1 |
| default | 4 | 100 | 48 | 24 | 0 | 0 |
| default | 8 | 23 | 10 | 0 | 0 | 0 |
| `.cs` (`sc0 nt`) | 1 | 47 | 3 | 3 | 3 | 2 |
| `.cs` | 2 | 13 | 2 | 2 | 1 | 1 |
| `.cs` | 4 | 2 | 1 | 1 | 1 | 1 |
| `.cs` after a pre-read | 2 | 100 | 100 | 100 | 34 | 1 |
| `.cs` after a pre-read | 4 | 100 | 64 | 35 | 7 | 0 |
| default after a pre-read | 4 | 100 | 46 | 23 | 1 | 0 |
| `.wt` (`sc0 sc1`) | any | 0-5 | 0-2 | 0-2 | 0-2 | 0-2 |

- **Default stores are LRU.** Lines survive until the written data plus the pollution passes about
  3-4 MiB, then the oldest go first. E = 4 MiB alone flushes everything.
- **Streaming stores to absent lines are not kept.** Most are written back during the kernel
  (`TCC_NORMAL_WRITEBACK`: 5.1K of 8.2K lines at F = 1, E = 0) and evicted first.
- **Streaming stores to lines already in L2 keep them**, a little better than default stores
  (64% against 46% at F = 4, E = 1). This fits the documented streaming policy, which never
  updates a line's age: a line allocated by a streaming store starts old and is the next victim
  in its set (the set is always full: the 1 GiB read leaves every way valid), while a streaming
  store that hits keeps the age the earlier read gave the line.
- **Write-through stores leave no readable copy** (M2: 1.0 miss per line).
- **An `nt` read does not demote a line.** Reading 2 MiB once with `.cg` (`sc0 nt`) or with the
  default policy, then E MiB of pollution, leaves the same share for a second read: 100% at
  E = 1-2, about 20% at E = 3, about 1% at E = 4.
- **Dirty data is capped at about 2 MiB per XCD.** Writing 4 MiB on one XCD with nothing else
  writes back half of it during the kernel (16.5K lines), yet all 4 MiB stay readable: the L2
  cleans lines early but keeps them.

## 2. Reading back a line written into L2 (M2, M3)

**Counters** ([results/m2.md](results/m2.md)), XCD 0, 2 MiB (16384 lines), per line:

| write | memory reads on read-back | read-back hits | read-back us | write us |
|---|---|---|---|---|
| full lines (as the v21 partials) | 1.00 | 1.03 | 5.98 | 1.80 |
| full lines, read by the writing CU | 1.00 | 1.03 | 5.98 | 1.80 |
| 64-byte halves | 1.00 | 1.03 | 6.04 | 1.20 |
| one dword per line | 1.00 | 1.03 | 6.14 | 1.24 |
| full lines written twice | 1.00 | 1.03 | 6.26 | 3.10 |
| full lines after a pre-read | 0.00 | 1.01 | 2.32 | 1.72 |
| `.cs` (`sc0 nt`) | 1.00 | 0.10 | 6.24 | 2.08 |
| `.wt` (`sc0 sc1`) | 1.00 | 0.03 (misses 1.00) | 4.60 | 2.64 |
| full lines, no 1 GiB read between launches | (see above) | | 2.14 | 1.40 |

- A read of a line that was written into L2, but not valid there before, is a hit that still
  reads the whole line from memory: one 128-byte read per line, whatever was written. The stores
  themselves fetch nothing.
- The read-back takes 6 us for 2 MiB when that fetch goes to HBM, 2.3 us without it.
- Without the 1 GiB read the previous launch leaves the lines valid in L2, and the read-back takes
  2.1 us (in-process timing; under counter collection rocprof's own flush hides this).

**Where the read is served from** ([results/m3.md](results/m3.md)), ns per dependent load:

| case | ns per load |
|---|---|
| reference: L2 (the chaser read the chain first) | 87 |
| reference: LLC (XCD 1 read the chain first) | 225 |
| reference: HBM (1 GiB read first) | 369 |
| written by XCD 0's other CUs, no cache bits | 368 |
| the same, chased with default loads | 369 |
| the same, without the 1 GiB read (previous launch's copy still in L2) | 85 |
| written with `.cs` (`sc0 nt`) | 370 |
| written after a pre-read | 85 |
| written with `.wt` on the same XCD | 223 |
| written with `.wt` on XCD 1, read with `.cv` (v20's path) | 223 |

So the first read of a freshly written line takes as long as reading the line from wherever its
old contents are: here HBM, because the 1 GiB read evicted it from the LLC. Write-through data
lands in the LLC and comes back from there, whether it is read on the same XCD or across XCDs.

### What exactly the read fetches (M5)

M5 tests each part of that description ([results/m5.md](results/m5.md)). Counters of the
read-back alone, per line of a 2 MiB region on XCD 0, after a 1 GiB read:

| case | memory reads | of which 128-byte | memory writes | L2 write-backs | read-back us |
|---|---|---|---|---|---|
| full lines written, every byte read | 1.00 | 1.00 | 0.00 | 0.01 | 6.86 |
| full lines written, one dword per line read | 1.00 | 1.00 | 0.00 | 0.01 | 6.18 |
| first 64 B written, only those 64 B read | 1.00 | 1.00 | 0.00 | 0.01 | 6.64 |
| first 64 B written, only the other 64 B read | 1.00 | 1.00 | 0.00 | 0.02 | 6.70 |
| full lines written, read twice: the second read | 0.00 | 0.00 | 0.00 | 0.01 | 2.24 |
| control: pre-read, full lines written, every byte read | 0.00 | 0.00 | 0.00 | 0.01 | 2.32 |
| control: nothing written, every byte read (plain miss) | 1.00 | 1.00 | 0.00 | 0.00 | 6.94 |

And the chase, ns per dependent load:

| case | ns per load |
|---|---|
| written by XCD 0's other CUs, old copy in HBM | 371 |
| written by XCD 0's other CUs, old copy in the LLC (XCD 1 read it first, then XCD 0 wrote it) | 224 |
| written after a pre-read | 88 |

What each result rules in or out:
- **The whole line is fetched.** One 128-byte memory read per line, never a 32- or 64-byte one,
  even when every byte of the line was written.
- **It is a fill, not a write-back and re-read.** No memory writes and no L2 write-backs happen
  during the read-back. The values returned are the new ones, so the L2 lays its dirty bytes over
  the fetched copy.
- **The dirty mask is not consulted for reads.** A read of only the bytes that were written, or of
  a single written dword, fetches the line just like a read of unwritten bytes.
- **The read waits for the old copy, wherever it is.** The same written line takes 371 ns with
  the old copy in HBM and 224 ns with it in the LLC, exactly the reference latencies. A read of a
  written-but-unfilled line costs as much as a plain miss (6.9 us for 2 MiB either way).
- **One fill makes the line valid.** The second read fetches nothing and runs at L2 speed, like
  the pre-read control.

So, as far as can be seen from outside, the L2:
1. allocates a line on a write without fetching it, recording which bytes were written;
2. treats the line as readable only after a fill;
3. on any read of an unfilled line, fetches the whole line from the next level and merges.

Nothing documents the internals (LLVM's AMDGPUUsage describes the coherence operations, not this),
so the "why" is inference:
- Not fetching on a write miss saves a read for every line of a write-only output, which is most
  GPU stores.
- The fill-and-merge path is needed anyway for partly written lines. Sending every read of an
  unfilled line down that path is simpler than checking, per request, whether the written bytes
  cover it, or promoting a fully written line to valid.
- The case it penalises, data written and then read once inside the same kernel without having
  been read before, is rare in typical workloads. Stream-K partials are exactly that case.

## 3. The kernel-end write-back (M4)

All 256 programs write D MiB in total ([results/m4.md](results/m4.md)), us:

| D MiB | default: kernel | default: after the last stamp | `.wt`: kernel | `.wt`: after the last stamp | default - `.wt` kernel | lines written back at kernel end |
|---|---|---|---|---|---|---|
| 0 | 4.6 | 3.7 | 4.6 | 3.7 | 0.0 | 0 |
| 4 | 5.0 | 3.7 | 4.4 | 2.6 | +0.6 | 33K (all) |
| 8 | 5.4 | 3.8 | 4.5 | 2.2 | +0.9 | 66K (all) |
| 16 | 5.7 | 3.3 | 4.5 | 1.2 | +1.2 | 66K (half) |
| 32 | 7.4 | 3.4 | 6.2 | 0.3 | +1.3 | 131K |
| 64 | 12.2 | 3.2 | 11.1 | 0.6 | +1.0 | 131K |
| 128 | 21.4 | 2.8 | 20.8 | 1.0 | +0.6 | 131K |

- **The write-back is at most about 16 MiB (2 MiB per XCD)**, the dirty cap. Above that the L2
  wrote the rest back during the kernel (393K lines at D = 64).
- **It costs 1-3 us after the last instruction.** With default stores the time from the last
  stamp to the kernel's end stays at 2.8-3.8 us, the same as an empty kernel's. With write-through
  stores it falls to 0.3-2.6 us, because their stores were already complete in memory.
  - The difference, 1.1-3.1 us for 4-32 MiB, is the write-back.
  - At D = 32, 16 MiB in 3.1 us is about 5.4 TB/s.
- **Net of the faster stores, the kernel is 0.6-1.3 us slower** than with write-through.
- **Nothing is deferred.** The follower starts immediately, takes 4.3-4.4 us at every D, and
  writes nothing to memory.
- **Streaming stores (`.cs`)** leave less dirty data (10-34K lines), but are slower overall: up to
  3.4 us behind write-through at 128 MiB.

## What this means for v21's `l2` mode

- **The slice read was always going to fetch from memory.** Each partial slot was written without
  being read first, so its first read fetches the line's old contents from the LLC or HBM, as
  slowly as v20's `.cv` read. Only lines valid before the write avoid it: a pre-read, which the
  GEMM could not hide, or the previous launch's lines, which the data-parallel phase evicts first.
- **Streaming stores (`.cs`) would make it worse**: the partials would not stay in L2.
- **The 4 MiB-per-XCD footprint is at the edge of what L2 keeps** with any other traffic, and
  8 MiB is past it, as v21 measured.
- **The kernel-end write-back of the partials costs at most about 1-3 us**: 2 MiB per XCD is
  left dirty, whatever the footprint. That matches the probe's 1.2-2.1 us.
- **The gains v21 kept come from the store's completion signal** (L2 instead of the memory side)
  and from the flag barrier. Neither depends on the read.
