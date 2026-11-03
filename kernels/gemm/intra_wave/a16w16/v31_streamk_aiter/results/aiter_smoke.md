# v31 inside AITER (llirSched_integration, 66e96218c)

Scratch worktree of `upstream/llirSched_integration`, not committed: v31's `matmul_kernel.py`
as `aiter/ops/triton/_gluon_kernels/gfx950/gemm/basic/gemm_a16w16_streamk.py`, and a pytest
`conftest.py` that routes the branch's gfx950 entry points (`_KERNEL_MAP["compute_bound"]`,
`unsupported_reason`, `_gemm_a16w16_compute_bound_kernel`) to v31, with the branch's default
tile (`pick_gemm_tile`) and v31's default plan (a tile-only config). rocm-smi GPU 1 under the
guard.

## The branch's gfx950 gluon tests (`test_gemm_a16w16.py -k "gluon and compute_bound"`)

89 passed, 6 failed, 179 skipped (shapes and features the kernel does not take).

- **4 failures, fp16 with bias at 4096x4096x8192 and 8192x8192x8192:** the reference is
  wrong, not the kernel. Against an fp32 reference, torch's `F.linear` (fp16, bias) is off by
  up to 4.0, v31 and the branch's v9 kernel by 0.126 (fp16 rounding); v9 fails the same test.
  bf16 passes.
- **2 failures, the 64x64 tile at 1024x1024x512 (bias on and off):** the result is right,
  but the compiled kernel has no `sched_barrier`. K is a compile-time constant in v31 (v9
  passes it at run time), so at 8 K-steps LLVM unrolls the whole K loop (64 MFMAs inline)
  and the MFMA scheduler's per-iteration schedule is gone. At K = 4096 the barriers are
  there (64 in the 64x64x64 build). See WORKLOG section 5.

## Direct calls

Tile-only configs and complete ones through `_KERNEL_MAP["compute_bound"]`: correct at
every tile the tests cover; the default plan is applied when the config names only the tile.

## Compile time (cold Triton cache, 256x256x64)

| launch | plan | first call | second call |
|---|---|---|---|
| 4096x4096x8192 | dp, one wave (no loop; before the non-persistent path was removed) | 1.12 s | 0.56 ms |
| 4352x4096x512 | dp, tail (persistent) | 0.71 s | 0.35 ms |
| 4352x4096x8192 | rs, S = 16, n = 12 | 1.28 s | 0.54 ms |
| 4096x6144x8192 | owner, S = 128, n = 2 | 1.28 s | 0.61 ms |
| 4608x4096x8192 | rs, S = 32, n = 8 | 1.22 s | 0.55 ms |
| 4352x4096x6144 | rs, another K | 1.29 s | 0.49 ms |
| v9, 4352x4096x8192 | | 0.56 s | |

Every new K, and every new S on a Stream-K path, is a new compile (K, STREAMK_TILES,
SPLIT_N and the reader counts are constexprs), at about 1.3 s each.
