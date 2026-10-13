# E8: the skinny-shape process variation, and row padding

256x32768x8192, bf16, GPU 2, LLIR + amdgcnas. At this shape B is 512 MB, so the "rotating
512 MB" rocprof method of STREAMK.md held a single copy: those were fixed buffers launched
back-to-back. `place.py layout` puts A (4 MB) at 0, B and C at chosen offsets of one 1.5 GB
`hipMalloc` and times each layout as a graph of 100 GEMMs, median of 10 replays. Raw:
`e8_layout_v13_p1.txt`, `e8_bsweep_v13.txt`, `e8_repeat.txt`, `e8_bpad.txt`,
`e8_cpad_4096.txt`.

## C does not matter here, B does (v13)

| B offset (MB) | C offsets (MB) | us | TFLOPS |
|---|---|---|---|
| 64 | 600, 664, 700, 800, 860, 900 | 210.3 - 211.4 | 650 - 654 |
| 96 | 664 - 900 | 177.9 - 179.3 | 767 - 773 |
| 128 | 664 - 900 | 188.2 - 188.7 | 728 - 730 |
| 192 | 800 - 900 | 187.7 - 188.4 | 729 - 732 |

C's offsets cover both 128 MB halves (mod 256: 32 - 188) with no effect; B's offset moves the
time by 18%.

## B's offset, 8 MB steps (C at 1400 MB)

| B offset (MB) | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 120 | 128 | 136 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| us | 195.5 | 191.4 | 188.6 | 190.2 | 191.3 | 192.5 | 193.9 | 210.6 | 187.6 | 181.8 | 180.2 | 178.8 | 180.4 | 180.5 | 182.4 | 189.1 | 187.0 |

| B offset (MB) | 144 | 152 | 160 | 168 | 176 | 184 | 192 | 200 | 208 | 216 | 224 | 232 | 240 | 248 | 256 | 264 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| us | 183.9 | 183.2 | 181.5 | 182.6 | 182.8 | 184.1 | 187.9 | 179.6 | 177.9 | 174.9 | 174.8 | 174.2 | 174.7 | 176.1 | 178.1 | 175.8 |

174 - 211 us (650 - 789 TFLOPS), with local peaks where B starts on a multiple of 64 MB. Unlike
C at 4096x4096x1024, it does not repeat every 256 MB (8 MB: 195.5 us, 264 MB: 175.8 us).

## The same offsets in new processes, and v9

| B offset (MB) | 8 | 64 | 96 | 128 | 232 |
|---|---|---|---|---|---|
| v13, process 1 | 196.0 | 209.3 | 178.8 | 188.8 | 174.4 |
| v13, process 2 | 198.3 | 211.1 | 179.3 | 188.8 | 174.5 |
| v9 | 201.3 | 212.8 | 179.6 | 189.7 | 175.5 |

The buffers started at different virtual addresses (112 and 208 MB mod 256) yet gave the same
times, so a large allocation's offset is a stable stand-in for its physical layout, and the
effect is not specific to one version.

## Padding B's rows (`--b-pad`, B is (N, K + pad) viewed as (N, K))

| pad (elements) | B at 8 | 64 | 96 | 128 | 192 | 232 MB | spread |
|---|---|---|---|---|---|---|---|
| 0 | 194.9 | 210.0 | 177.6 | 188.5 | 188.1 | 174.7 | 35.3 us (20%) |
| 64 (128 B) | 197.5 | 199.3 | 193.6 | 196.4 | 199.3 | 195.0 | 5.7 us |
| 256 (512 B) | 175.7 | 175.4 | 175.1 | 175.0 | 174.6 | 175.6 | 1.1 us |
| 1024 (2 KB) | 170.0 | 172.1 | 172.0 | 171.6 | 171.4 | 171.6 | 2.1 us |

With 256 or 1024 elements of padding the offset no longer matters, and 1024 is faster than the
best unpadded placement (170 - 172 against 174.2 us). The padded results match torch.

## Padding C's rows at 4096x4096x1024 (v9, `--c-pad`)

| pad (elements) | C at 32 | 64 | 160 | 200 | 288 | 416 MB |
|---|---|---|---|---|---|---|
| 0 | 28.3 | 28.3 | 29.4 | 29.5 | 28.4 | 29.4 |
| 64 | 28.7 | 28.9 | 29.0 | 29.0 | 28.5 | 28.9 |
| 256 | 28.9 | 28.9 | 28.9 | 29.0 | 29.0 | 29.0 |
| 1024 | 29.0 | 29.1 | 29.0 | 28.9 | 29.0 | 29.0 |

Padding C removes the fast/slow halves too: every offset lands at 28.9 - 29.1 us, between the
unpadded fast half (28.3) and slow half (29.4).

## Follow-ups

- `e8_bpad_sweep.txt`: v13 at B offsets 64, 128 and 232 MB with 0-1024 elements of padding
  (table and figure in [PARTITION_CAMPING.md](../PARTITION_CAMPING.md)). Pads below 128
  elements (256 bytes) are worse than no padding at a good offset; from 128 elements up every
  offset is within 165.6-175.1 us, best at 128 elements. 8 elements does not compile.
- `padded_all_versions.txt`: v9-v21 at 4096x4096x1024 with 128 elements of padding on B and
  C match torch at both C offsets, and v13 and v20 match at 4352x4096x8192. The 4096 times in
  that file come from 50-call graphs and are about 2 us high; do not compare them with the
  400-call numbers.
- `pad_4096_split.txt`: B and C padded separately at 4096x4096x1024 (400-call graphs). B's
  padding changes nothing (B stays in the MALL); C's padding gives 28.8-29.2 us (v9) and
  29.5-30.0 us (v13) at both offsets.

**Conclusion.** The skinny shapes' 13 - 30% process-to-process swing is the same kind of
effect with a different victim: there it is where B, the 512 MB matrix streamed from HBM,
lands; C does not matter. In both cases the operand has a power-of-two row stride (B: 16 KB
rows, panels exactly 4 MB apart; C: 8 KB rows), so the many CUs that stream it at once
generate addresses that differ only in high bits, and how the memory-side address hash spreads
them depends on where the buffer sits. Breaking the power-of-two stride with 256 - 1024
elements of row padding makes the time independent of placement; for B it is also faster than
any unpadded placement, for C it settles between the two halves.
