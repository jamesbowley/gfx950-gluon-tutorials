"""F.linear TFLOPS for shapes where torch and standalone hipBLASLt disagree, at the current HIPBLASLT_WORKSPACE_SIZE."""
import os, torch, triton
import torch.nn.functional as F
assert torch.cuda.device_count() == 1
for M, N, K in ((2048, 2304, 16384), (1024, 4608, 16384), (1024, 4608, 8192), (32768, 8192, 1024)):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ms = min(triton.testing.do_bench_cudagraph(lambda: F.linear(x, w)) for _ in range(2))
    print(f"ws={os.environ.get('HIPBLASLT_WORKSPACE_SIZE', 'default')} {M}x{N}x{K}: {2*M*N*K/ms*1e-9:.0f} TFLOPS", flush=True)
