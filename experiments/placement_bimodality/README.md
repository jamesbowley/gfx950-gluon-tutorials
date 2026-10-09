# Fast and slow buffer placements

The same GEMM, with the same inputs, runs fast or slow depending on where its buffers sit in
memory. At 4096x4096x1024 (v9, back-to-back in a CUDA graph) a C allocation gives either about
28.4 us or about 29.5 us, about 4%. The skinny shapes in
[STREAMK.md](../../kernels/gemm/intra_wave/a16w16/STREAMK.md) (section 9 and its open
questions) swing by 13-30% between processes. These experiments find what decides it.

```bash
export PYTHONPATH=<repo>/scripts/triton_deepbind_shim:<triton_gfx950-tutorial-v2.2>/python
export LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so TRITON_AMDGCNAS_PLUGIN=1
export HIP_VISIBLE_DEVICES=2
python place.py e1                                             # E1: reuse or buffer property
python place.py sweep --what c --start 0 --stop 512 --step 2   # E2: C's offset in one buffer
python place.py e3 --path hip                                  # E3: allocation paths
python e4_channels.py                                          # E4: per-channel L2 traffic
bash e5_clock_lock.sh 1800                                     # E5: needs root (blocked here)
python place.py e7                                             # E7: idle gaps, other writes
python place.py layout --version 13 --shape 256x32768x8192 --calls 100 \
    --b-offs 8,64,96,128,232 --c-offs 1400 [--b-pad 256]       # E8: skinny shape, padding
```

Machine: MI355X (gfx950), GPU 2, Triton `gfx950-tutorial-v2.2` + deepbind shim, LLIR scheduler
and amdgcnas plugins, bf16. `place.py` times a graph of `--calls` GEMMs (A, B fixed, C from a
pattern of buffers) replayed `--replays` times and reports the median per GEMM, with hwmon sclk
and power sampled while it replays. Results: [results/](results/).

[PARTITION_CAMPING.md](PARTITION_CAMPING.md) explains the mechanism with a numerical example
and figures, lists libraries that pad leading dimensions, and argues for padded strides in
benchmarks.

## TL;DR

- **It is the physical address of one operand, not the allocator or the kernel.**
  - At 4096x4096x1024 it is C. Inside one contiguous buffer the time repeats every 256 MB of
    C's offset: fast only if all of C lies in the first 128 MB of a 256 MB period, slow if even
    2 MB of it falls in the second half ([E2](results/e2_offset.md)). A and B can be anywhere.
  - Every allocation path is bimodal in the same proportion (caching allocator, `hipMalloc`,
    fine- and coarse-grained, caching off), and the virtual address predicts nothing
    ([E3](results/e3_alloc_path.md)).
  - At 256x32768x8192 it is B, the 512 MB matrix streamed from HBM: v13 runs at 650-789 TFLOPS
    depending on B's offset, the same offset gives the same time in every process, and C does
    not matter ([E8](results/e8_skinny.md)). That is the skinny-shape process variation.
- **Padding the rows removes it.** Both operands have power-of-two row strides (C: 8 KB, B:
  16 KB, so B's 4 MB panels start exactly 4 MB apart). Padding each row by 256-1024 elements
  makes the time the same at every offset: B at 166-176 us (784-830 TFLOPS, as good as the best
  unpadded placement or better; 256 bytes is fastest), C at 28.9-29.1 us (between the unpadded
  halves) ([E8](results/e8_skinny.md)). Pads below 256 bytes make it worse.
  So the many CUs that stream such an operand at once produce addresses that differ only in
  high bits, and how the memory-side address hash spreads them depends on where the buffer sits.
- **Fast is a property of the buffer, and slow is sticky.** Two fast C buffers alternating stay
  fast; a fast and a slow one alternating are as slow as the slow one alone (29.40 against
  29.45 us), so the slow state carries over to the next kernels ([E1](results/e1_reuse.md)).
  That is why rotating several random copies always measured slow. It is the GEMM's own
  writes: another kernel writing into the slow half between GEMMs does not trigger it
  ([E7](results/e7_time_constant.md)).
- **Not visible at the L2.** Per-channel and per-XCD write and read requests are identical for
  fast and slow C, evenly spread over all 16 channels and 8 XCDs; write latency differs by 1-2%
  ([E4](results/e4_channels.md)). The bit that matters is decoded beyond the L2.
- **Part clock, part cycles, and only back-to-back.** Slow runs at a lower clock (20-57 MHz) and
  also costs cycles; the cost is inside each kernel's duration, not between kernels
  ([E6](results/e6_backtoback_counters.md)). With 100 us or more of idle time after each GEMM,
  both halves run at the same clock and differ by 0-1% ([E7](results/e7_time_constant.md)), which
  is why serialised profiling runs (counter collection, Python-paced launches) do not show it.
  Locking the clock to split the two exactly needs root ([E5](results/e5_clock.md)).
- **What did not work.** SPM in this rocprofv3 serialises every dispatch (4570 us per GEMM) and
  cannot run with `--pmc`, so no counters exist for back-to-back execution
  ([E6](results/e6_backtoback_counters.md)). The clock could not be locked from this container.

## Benchmarking so placement cannot skew a comparison

The recommended default is padded strides (256 bytes or more per row, on every streamed
operand); [PARTITION_CAMPING.md](PARTITION_CAMPING.md) section 5 gives the reasons and the
costs. Without padding:

- Run every version on the same buffers in one process, as `place.py` does, or use several
  placements and report the spread. `v20_streamk_reduce_scatter/rocprof_check.py` runs each
  version in its own `bench.py` process, so each version gets its own placement.
- For a single number that does not depend on placement, pad the row stride of the streamed
  operand (B on skinny shapes, C on output-heavy shapes) by 256 elements or more, or rotate over
  enough copies that the mix of placements averages out. Note the second only helps if the
  effect averages: with C, one slow copy makes the whole rotation slow.
- Unrotated (warm) back-to-back numbers on shapes like 4096x4096x1024 carry about a 4% placement
  term; cold, Python-paced runs carry almost none.

## Open questions

- Which address bits the memory-side hash uses. The 4096 case behaves as if one physical bit
  (128 MB halves, so presumably bit 27) decides it, while B's pattern on the skinny shape has
  peaks at multiples of 64 MB and no 256 MB period.
- The clock and cycle split, with the clock locked (`e5_clock_lock.sh` from a root shell).
- 256x49152x8192 (the other skinny shape in STREAMK.md), and whether padding helps the kernels
  in general: it costs a strided view or a copy, and B's padding at 1024 elements was 2% faster
  than the best placement.
