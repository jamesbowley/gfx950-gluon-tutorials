# E3: allocation path, and physical against virtual address

`python place.py e3 --path P --n 16 --seed {0,1}`: v9, 4096x4096x1024, bf16, GPU 2. Each run
allocates 16 C buffers (random 1-64 MB pads in between, all kept alive) and times each one
alone back-to-back; fast is below 29.0 us. `torch_nocache` is the caching allocator path with
`PYTORCH_NO_HIP_MEMORY_CACHING=1` (one run). `slice` takes 32 random 4 KB-aligned offsets in
one 1 GB `hipMalloc`. Raw: `e3_raw.jsonl` ((virtual address MB, us) pairs).

| path | fast |
|---|---|
| torch caching allocator | 13 / 32 |
| `hipMalloc` | 13 / 32 |
| `hipExtMallocWithFlags`, fine-grained | 16 / 32 |
| `hipExtMallocWithFlags`, coarse-grained | 15 / 32 |
| torch, caching off | 8 / 16 |
| slices of one 1 GB buffer | 8 / 32 |

Virtual addresses do not predict the mode. Taking E2's rule on the virtual address (slow if any
of C falls in the second 128 MB of a 256 MB period):

| C's virtual range touches the second half | fast | slow |
|---|---|---|
| no | 25 | 36 |
| yes | 48 | 67 |

**Conclusion.** Every allocation path is bimodal in about the same proportion (roughly 40-50%
fast for separate allocations), so no allocator setting avoids it, and the virtual address
carries no information. Since a contiguous buffer shows a clean pattern fixed in the address
(E2), the mode is set by C's physical address. Random slices of one 1 GB buffer are fast less
often (8 / 32), as E2 predicts: a 32 MB C must avoid every second 128 MB, which only (128 - 32)
/ 256 = 37.5% of offsets do, and the fast half has a few slower spots.
