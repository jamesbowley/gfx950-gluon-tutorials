# Compact launch against the full grid: 256x16x128

Kernel BLOCK 16x256x128, 4 waves, 256 programs, swapped (C^T = B^T A^T). 3 launches per mode plus 2 recorded. grid: full / compact; programs: the stream-K programs the host passes (the request capped at one per pair, S x pairs), all working; C: bit-identical (or both correct, for atomic_iz and bf16 partials); XCC rotations r (workgroup p on XCC (p + r) % 8); CUs: distinct CUs / working programs per XCD, compact launch; at most per CU.

| shape | configuration | STREAMK_NUM_PROGRAMS | bias | grid | programs | C | wrong | flags up | rotations | CUs (compact) | per CU | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 3 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 128 | 0 | 256 / 256 | 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 128 | 1 | 256 / 256 | 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 144 | 0 | 256 / 256 | 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 144 | 1 | 256 / 256 | 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 165 | 0 | 256 / 256 | 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 165 | 1 | 256 / 256 | 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 200 | 0 | 256 / 256 | 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | 200 | 1 | 256 / 256 | 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | default | 0 | 256 / 256 | 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=1 | default | 1 | 256 / 256 | 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 1 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=fp32, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 7 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=tile, EXCHANGE=l2, XCD_LOCAL=0 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=rs, PARTIALS=bf16, RS_READERS=all, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=fp32, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 5 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=tile, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=one_tile, FIXUP=owner, PARTIALS=bf16, RS_READERS=pow2, PLACEMENT=chunk, EXCHANGE=wt | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 3 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=fp32 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 128 | 0 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 128 | 1 | 256 / 64 | 128 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 144 | 0 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 144 | 1 | 256 / 64 | 144 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 165 | 0 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 165 | 1 | 256 / 64 | 165 → 64 | identical | 0 | 0 | 4 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 200 | 0 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 0 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | 200 | 1 | 256 / 64 | 200 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | default | 0 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 2 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=spread, PARTIALS=bf16 | default | 1 | 256 / 64 | 256 → 64 | identical | 0 | 0 | 6 | 8/8 8/8 8/8 8/8 8/8 8/8 8/8 8/8 | 1 | ok |
| 8192x1x512 | POLICY=capped | default | 0 | None / None |  | identical | 0 | 0 |  |  |  | ok |
| 8192x1x512 | POLICY=capped | default | 1 | None / None |  | identical | 0 | 0 |  |  |  | ok |

192 configurations: 192 ok, 0 failed, 0 errors, 0 skipped (host rejects); compact launch in 180; the request capped at one program per pair in 180. XCC rotations seen (launches): r=0: 180, r=1: 3, r=2: 181, r=3: 8, r=4: 204, r=5: 4, r=6: 176, r=7: 4. Compact grids not a multiple of 8: .

PASS
