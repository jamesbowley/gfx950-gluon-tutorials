# a16w16 beyond v9: versions and `bench_ext.py`

[README.md](README.md) and `bench.py` cover the tutorial, v0-v9, and stay as on main. The
versions below, and the benchmark options they need, live alongside them.

## 1. Versions

```
a16w16/
├── bench_ext.py                    # bench.py plus everything below (section 2)
├── v9_tile/                        # v9 with the tile size as a parameter (256x128, 128x256, 128x128); tile-size study
├── v9_ragged/                      # v9 for M, N, K not multiples of the tile (clamped loads, masked stores, masked last pair)
├── v10_persistant/                 # v9 as a persistent kernel (one program per CU loops over tiles)
├── v11_persistant_overlap_global/  # + next tile's first global->LDS loads in the epilogue
├── v12_persistant_overlap_lds/     # + next tile's first LDS reads in the epilogue
├── v13_persistant_peel_acc/        # + peeled first K-steps with an inline-0 accumulator
├── v14_streamk/                    # v13 with the last wave peeled out; stream-K notes and models
├── v15_streamk_onetile/            # + one-tile stream-K tail (row-major partials)
├── v16_streamk_lane_partials/      # + lane-contiguous partials
├── v17_streamk_tile_aligned/       # + tile-aligned cuts
├── v18_streamk_chunk_major/        # + chunk-major placement, rotating owner
├── v19_streamk_two_tile_reversed/  # + two-tile stream-K, segments in reversed order
├── v20_streamk_reduce_scatter/     # + reduce-scatter ending, host policy rule, spread
├── v21_streamk_l2_partials/        # + partials and flags through one XCD's L2 (STREAMK_PARTIALS)
├── v22_streamk_atomic_c/           # v20 + an atomic ending: bf16 pk_add into C (STREAMK_FIXUP=atomic)
├── v23_streamk_rs_band/            # v20 + band C store, power-of-two readers, opt-in bf16 partials
├── v24_streamk_combined/           # v23 + v22's atomic_iz, v21's L2 exchange, v18's chunk-major placement (switches)
├── v25_streamk_ragged_mn/          # v24 for M, N not multiples of the tile (clamped loads, masked stores)
└── v26_streamk_ragged_k/           # v25 for K not a multiple of 128 (the last, partial pair first, masked)
```

v14-v24 balance a partly empty last wave with stream-K. [STREAMK.md](STREAMK.md) summarizes
what each version changes and what it measured.

## 2. How to Run

From the `a16w16` directory:

```bash
python bench_ext.py --version 20 --K 8192 --dtype bf16 --M 8192 --N 8192
```

`bench_ext.py` takes everything `bench.py` does, plus:

| Option | Default | |
|---|---|---|
| `--version` | `9` | 0-26, `9t` (v9_tile) or `9r` (v9_ragged) |
| `--M`, `--N` | `4096` | Problem size; an explicit `--K` outside the default sweep runs as given |
| `--tile` | `256x256x64` | v9_tile only: BMxBN or BMxBNxBK (sets `V9_TILE`) |
| `--bias` | off | Add a `bias[N]` (versions whose `matmul()` takes `bias`) |
| `--schedule-hint` | the kernel's | As `bench.py`: `""` turns the MFMA scheduler off |
| `--pad` | off | Pad B and C rows (below); `--no-pad` is the default, contiguous as in `bench.py` |

Scripts under the version directories and `experiments/` import it as `import bench_ext as bench`
for its plugin hooks, the path to `kernels/gemm/utils/`, and the tensor helpers (`make_b`,
`make_c`, `gen_rotating_tensors`).

Like main's v5-v9, v9_tile, v9_ragged and v10 on pass `schedule_hint="mfma-schedule"` to their
GEMM launch (`matmul()` in each `matmul_kernel.py`), the MFMA scheduler that was the `llir` plugin
before `gfx950-tutorial-v3.0`; the stream-K versions forward it to their `v9_ragged` fallback.
`scripts/benchmark_prepared_ext.py` compiles with the same default (`--schedule-hint` there too).

### Padded B and C rows (`--pad`)

With `--pad`, `bench_ext.py` pads a row of B (stored (N, K)) or C (M, N) by 256 bytes when the
row is a multiple of 512 bytes, and runs on the (N, K) / (M, N) view, as AMD's MI300 / MI350 tuning guide
and vLLM's ROCm weight padding do ([`../../utils/row_padding.py`](../../utils/row_padding.py)).
Without it, power-of-two row strides camp on a few memory channels, and how badly depends on where
the buffers land in physical memory: the same kernel moves by 4% (4096×4096×1024, warm) to 20%
(256×32768×8192) between allocations. Why, and how it was measured:
[PARTITION_CAMPING.md](../../../../experiments/placement_bimodality/PARTITION_CAMPING.md).
By default B and C are contiguous, as in `bench.py`; results recorded between 2026-10-06 and
2026-10-13 used padded rows, when padding was the default. A stays contiguous because the kernels
require it. `v20_streamk_reduce_scatter/rocprof_check.py` and `v9_tile/check_tiles.py` take the
same `--pad` (rocprof_check also `--warm`, for one copy instead of the rotating 512 MB).

v1, v3, v4 and v6 give wrong results under `TRITON_AMDGCNAS_PLUGIN=1` (padded or not): the
peephole assumes AGPR-pinned accumulators, which start at v7. Run them without the plugins.

### Bias (v9_tile, v9_ragged, v10 on)

These versions take an optional `bias[N]`: `matmul(a, b, bias=bias)`, or `--bias` on
`bench_ext.py`. `v9_beyond_hotloop` stays as on main, without one; `v9_ragged` is the same kernel
with a bias (and compiles to the same code without edges), and the stream-K versions fall back to
it.
The bias is not added before the C stores: the epilogue already holds the register peak, and
adding it there pushes the persistent kernels into spills. Instead `init_acc` in
[`../../utils/bias.py`](../../utils/bias.py) starts the four fp32 accumulators from the
broadcast bias, loaded before the prologue loads, so the K loop and epilogue carry no extra
registers. Without bias the accumulators stay a constant zero and the code is unchanged. The
measurements behind this choice are in
[`experiments/aiter_bias_spills/results.md`](../../../../experiments/aiter_bias_spills/results.md).

4096×4096×8192 FP16, `llir+amdgcnas`, Triton `gfx950-tutorial-v2.2`, rocprof timing (1000
dispatches, last-100 average), mean of two rounds on one MI355X. The bias cost is relative to the
same kernel without bias on the same GPU; absolute TFLOPS on that GPU sit ~17% below the
well-performing part the other tables use, so they are not quoted. The v9_beyond_hotloop row was
measured with the bias since moved to v9_ragged.

| Version | VGPRs / spills | VGPRs / spills, bias | MFMA Eff. | MFMA Eff., bias | Bias cost |
|---------|----------------|----------------------|-----------|-----------------|-----------|
| v9_beyond_hotloop             | 448 / 0 | 448 / 0 | 98.08% | 97.96% | -0.7% |
| v10_persistant                | 480 / 0 | 488 / 0 | 97.72% | 97.47% | -0.5% |
| v11_persistant_overlap_global | 452 / 0 | 464 / 0 | 98.06% | 97.50% | -0.3% |
| v12_persistant_overlap_lds    | 440 / 0 | 444 / 0 | 97.63% | 97.35% | -0.2% |
| v13_persistant_peel_acc       | 480 / 0 | 488 / 0 | 97.78% | 98.12% | +0.2% |

Run-to-run noise on these rows is about ±1% TFLOPS and ±0.9pp MFMA efficiency. On shapes where
the epilogue weighs more (fewer tiles than CUs: 16384×512×8192, 2048×2304×16384; short K:
4096×8192×1024; BF16) the bias costs 0.4–2.5%, with no version consistently worse.
