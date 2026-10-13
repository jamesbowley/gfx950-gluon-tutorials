##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""A plain Triton tl.dot GEMM, the reference for v9_tile_small: the compiler picks the instruction
(16x16x32 MFMA, the 4x64 / 64x4 MFMA for a side of 4-15, or v_dot2 below 4), the warp layout, the
LDS layout and the pipeline (num_stages = 2; async copies on gfx950).

A spec is "triBMxBNxBK[wW][eE]": W warps (default 4), launched with waves_per_eu = E. The pids use
v9_tile's XCD remap and GROUP_SIZE_M rule, so only the code inside a tile differs.

    python v9_tile/triton_ref.py tri4x256x128,tri1x256x128,tri256x4x128,tri16x16x128w1   # inspect

prints, per spec, the dot's layout and LDS encodings from the TTGIR and the instruction counts in
the assembly. Compile it stock (no LLVM_PASS_PLUGIN_PATH, no TRITON_AMDGCNAS_PLUGIN): it is what
the compiler produces on its own.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "..", "..", "utils"), HERE]
import bench_ext as bench  # noqa: E402,F401  (first: loads Triton the way the rest of the repo does)
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

SPEC_RE = re.compile(r"^tri(\d+)x(\d+)x(\d+)(?:w([1248]))?(?:e(\d))?$")
LAST_LAUNCH = {}


@triton.jit
def triton_ref_matmul(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                      stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GRID_MN: tl.constexpr, NUM_XCDS: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                      EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def parse(spec):
    m = SPEC_RE.match(spec.lower())
    assert m, f"spec {spec!r}: expected triBMxBNxBK[wW][eE]"
    bm, bn, bk, w, e = m.groups()
    return int(bm), int(bn), int(bk), int(w or 4), int(e or 0)


def matmul(a, b, c, tile):
    import matmul_kernel as mk  # the GROUP_SIZE_M rule

    BM, BN, BK, W, E = parse(tile)
    M, K = a.shape
    N = b.shape[1]
    assert K % BK == 0, f"K={K} must be a multiple of {BK}"
    grid_m, grid_n = triton.cdiv(M, BM), triton.cdiv(N, BN)
    GRID_MN = grid_m * grid_n
    gm = mk.group_size_m(BM, BN, grid_m, GRID_MN, 8, tiles_per_cu=max(1, 4 // W))
    k = triton_ref_matmul[(GRID_MN, 1)](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GRID_MN=GRID_MN, NUM_XCDS=8, GROUP_SIZE_M=gm,
        EVEN_M=M % BM == 0, EVEN_N=N % BN == 0, num_warps=W, num_stages=2, waves_per_eu=E)
    LAST_LAUNCH.update(kernel="triton_ref_matmul", compiled=k)
    return c


def inspect(k):
    """The dot's encodings and the loop's instruction mix of a compiled triton_ref_matmul."""
    ttgir, asm = k.asm["ttgir"], k.asm["amdgcn"]
    mma = re.search(r"#ttg\.amd_mfma<\{[^}]*\}>", ttgir)
    shared = sorted(set(re.findall(r"#ttg\.(padded_shared|swizzled_shared|shared_linear)<", ttgir)))
    count = lambda p: len(re.findall(p, asm))  # noqa: E731
    mfmas = sorted(set(re.findall(r"v_mfma_\w+", asm)))
    return {
        "mma": mma.group(0) if mma else "none (FMA)",
        "shared": ",".join(shared) or "none",
        "mfma": ",".join(mfmas) or "-",
        "n_mfma": count(r"\bv_mfma_"), "n_dot2": count(r"\bv_dot2\w*"), "n_fma": count(r"\bv_(?:pk_)?fma\w*"),
        "bpermute": count(r"\bds_bpermute"), "async_lds": count(r"buffer_load_dword\w*[^\n]*\blds\b"),
        "global_load": count(r"\bglobal_load"), "vgpr": int(re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", asm).group(1)),
        "lds": k.metadata.shared,
    }


def main():
    M, N, K = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "2048x4096x8192").split("x"))
    print("| spec | shape | correct | dot layout (TTGIR) | LDS encodings | MFMA | MFMAs | v_dot2 | v_fma | "
          "ds_bpermute | async LDS loads | global loads | VGPRs | LDS B |")
    print("|" + "---|" * 14)
    for spec in sys.argv[1].split(","):
        bm, bn, _, _, _ = parse(spec)
        m_, n_ = (M, 256) if bn < bm else (M, N)
        a = torch.rand((m_, K), device="cuda", dtype=torch.bfloat16) - 0.5
        b = bench.make_b(n_, K, torch.bfloat16, False, lambda t: t.uniform_(-0.5, 0.5))
        c = bench.make_c(m_, n_, torch.bfloat16, False)
        matmul(a, b, c, spec)
        ok = torch.allclose(c.float(), a.float() @ b.float(), atol=1e-1, rtol=1e-2)
        r = inspect(LAST_LAUNCH["compiled"])
        mma = re.sub(r"\s+", " ", r["mma"]).replace("|", "/")
        print(f"| {spec} | {m_}x{n_}x{K} | {ok} | `{mma}` | {r['shared']} | {r['mfma']} | {r['n_mfma']} | {r['n_dot2']} "
              f"| {r['n_fma']} | {r['bpermute']} | {r['async_lds']} | {r['global_load']} | {r['vgpr']} | {r['lds']} |",
              flush=True)


if __name__ == "__main__":
    main()
