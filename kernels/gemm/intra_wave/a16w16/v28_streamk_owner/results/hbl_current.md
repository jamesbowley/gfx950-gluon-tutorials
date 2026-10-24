# v28 against the installed hipBLASLt, same process and GPU (2026-10-19)

1024x1024x7168 bf16 on HIP 1 (`amd-smi` GPU 0, PCI 0x05), `/opt/venv`: torch 2.13.0+rocm7.14.0, hipBLASLt 1.3.2.
[hbl_current.py](../hbl_current.py): every method timed the same way in one process (CUDA graphs of 100 launches over
2 GB of rotating inputs, so cold; 10 replays per round, 5 rounds in a shuffled order, the median), after a check
against torch (all correct). B is the (K, N) view of a contiguous (N, K) tensor for every method. Five sessions: three
at hipBLASLt's default grid, and one each with `TENSILE_STREAMK_FIXED_GRID` = 128 and 64. Data: `hbl_current/*.json`.

**hipBLASLt's kernel** in every session (torch.profiler, rocprof): one launch of
`Cijk_Alik_Bljk_..._MT128x128x128_MI16x16x1_..._SK5_SKFTR0_SKFDPO0_SKXCCM8_..._WG32_8_1`, on 256, 128 or 64 workgroups
(default, 128, 64). This is a newer stream-K variant (SK5) than the doc's 1.2.2 SK3, with the same tile and XCD mapping.

## TFLOPS per session

| method | def1 | def2 | def3 | grid 128 | grid 64 |
|---|---|---|---|---|---|
| **hipBLASLt** | **563.2** | **560.6** | **561.0** | **403.1** | **254.0** |
| dp (v9_tile, 1 per tile) | | | 261.8 | | 261.7 |
| rsall (4 per tile) | 556.6 | 556.2 | 556.2 | 555.8 | 554.2 |
| owall (4 per tile), v27's ending | 497.0 | 498.5 | 498.3 | 498.4 | 497.4 |
| owall+b | 539.1 | 539.1 | 539.0 | 539.4 | 539.3 |
| owall+s | 532.0 | 534.1 | 535.0 | 535.3 | 533.0 |
| ow3 | 474.0 | 474.2 | 474.3 | 474.2 | 474.5 |
| ow3+b | 499.5 | 499.2 | 499.7 | 499.2 | 499.8 |
| ow2 (2 per tile), v27's ending | 389.4 | 389.2 | 389.4 | 389.3 | 389.6 |
| ow2+b | 397.9 | 398.0 | 397.7 | 398.0 | 397.7 |
| ow2+s | 395.6 | 395.2 | 395.5 | 395.0 | 395.8 |
| auto (64x128x128, owner on 2) | 488.8 | 488.6 | 488.6 | 488.4 | 488.4 |
| auto+b | 500.1 | 499.6 | 499.4 | 499.6 | 499.4 |

Round spreads 0.0-1.5%; the v28 points agree across sessions to 0.6% (FIXED_GRID only reaches hipBLASLt).

## Against hipBLASLt at the same split

| programs per tile | hipBLASLt | v27's ending | b | s | reduce-scatter |
|---|---|---|---|---|---|
| 4 (default grid) | 561.0 (26.80 us) | 0.888 (+3.37 us) | **0.961 (+1.09 us)** | 0.952 (+1.35 us) | 0.992 (+0.23 us) |
| 2 (grid 128) | 403.1 (37.29 us) | 0.966 (+1.32 us) | **0.987 (+0.49 us)** | 0.981 (+0.72 us) | |
| 1 (grid 64; no fix-up) | 254.0 (59.21 us) | dp **1.031 (-1.75 us)** | | | |

- **Change 1 removes two thirds of the owner ending's deficit at both splits:** 3.37 to 1.09 us on 4 programs per
  tile, 1.32 to 0.49 us on 2. Change 4 removes a little less (to 1.35 and 0.72 us).
- **The main loops favour us:** one program per tile, our data-parallel kernel is 3.0% faster (1.75 us). Split n ways,
  that is roughly 0.45 us per program on 4 and 0.9 us on 2 (as the doc's model, assuming the loop scales as 1/n), so
  hipBLASLt's ending is still about 1.4-1.5 us cheaper than b's at both splits: a fixed cost, not one per peer.
- **Reduce-scatter on 4 is 0.9% behind** (the doc: 0.9% behind 1.2.2), and owall+b is 3.1% behind it.
- **v28 as deployed (auto: 64x128x128, owner on 2)** is 12.9% behind hipBLASLt, 10.9% with b: the tile rule's choice,
  not the ending, is most of that gap on this shape.
- **b and s together (bs), two more sessions on 2026-10-20** (`bs_def1`, `bs_g128`; b there is the asm-free form,
  WORKLOG section 7): on 4 programs per tile hipBLASLt 560.6, v27's ending 496.0 (0.885), b 534.2 (0.953), **bs 539.0
  (0.962)**; on 2 hipBLASLt 404.2, v27's ending 388.3 (0.961), b 396.3 (0.980), **bs 401.1 (0.992)**. ow3: 472.8, b
  495.2, bs 505.1.
- **Against the cross-session estimate** (WORKLOG section 4, from the doc's 1.2.2 session): -3.9% at 4 per tile (here
  -3.9%) and -3.3% at 2 (here -1.3%). hipBLASLt 1.3.2 is not 1.2.2, so these agree only where the two versions do.
