# Tile + split selection rules (offline-fitted, power-of-two tiles)

Both rules choose a tile. The split kind always comes from the §17 split table at that
tile (looked up by wave class, leftover tiles S and k-steps), or `cap` if the table has no entry.

## Compute-bound (M ≥ 512 and N ≥ 512)

Tiles: BM, BN ∈ {64, 128, 256}, BK ∈ {64, 128}; 256 programs.

1. If 256x256 makes ≥ 256 tiles (a full wave): use **256x256x64**.
2. Otherwise, for each tile:
   - `T = ceil(M/BM) * ceil(N/BN)`
   - `s = min(256 // T, K // 2048)` (largest split, at least 2048 of K per split; 1 if T ≥ 256)
   - `fill = min(1, T*s / 256)`
3. Pick the **largest-area tile with fill ≥ 0.625** (ties: higher fill, then BK 128).
   If none reaches it, pick the best fill.
4. Split: table lookup at that tile.

Regret = time over the best timed point (1.05 = 5% slower):

| | Now (auto + cap) | Rule |
|---|---:|---:|
| REFIT (fitting set) | 1.053 | 1.031 |
| WORST10 (held out) | 1.179 | 1.055 |
| WORST10 losers vs best library | -14.0% | -3.4% |

## Memory-bound (M < 512 or N < 512): time model

Tiles: all 26 REFIT tiles. Programs P = 1024 for 16x16 / 16x32 / 16x64 / 32x32 / 1x16,
512 for 32x128 / 16x128, 256 otherwise.

For each tile and each split `s ≤ min(P // T, K // 2048)`:

```
traffic = 2*M*K*ceil(N/BN) + 2*N*K*ceil(M/BM) + 2*M*N + 8*(s-1)*T*BM*BN   # bytes
padded  = 2 * Mpad * Npad * K                                           # MFMA work incl. padding
busy    = min(256, T*s) / 256                                           # fraction of CUs working
time    = (traffic / 5e12 + padded / 2e15) / busy  (+ small cost if s > 1)
```

Pick the **lowest time** (within 2%: least padding, then fewer splits). Split: table lookup.

| | Now (256x256 + cap) | Time model |
|---|---:|---:|
| REFIT memory-bound, out of sample | 2.38 | 1.075 |
| tilesel small-M (held out) | 1.87 | 1.060 |
| tilesel small-N (held out) | 1.46 | 1.068 |

### Simpler memory-bound alternative

1. BM = smallest side ≥ min(M, N).
2. Long side from 256 down: the first that reaches 128 busy CUs, splitting only if
   each split keeps ≥ 4096 of K.
3. If B (2·N·K bytes) ≤ 16 MB, BM may be smaller than M (re-reads stay in cache).

REFIT 1.095 (out of sample), tilesel small-M 1.110, small-N 1.058.

## Caveats

- Scored offline from existing timings; not yet run on the GPU.
- Power-of-two tiles only (no 80/96/160/192 tiles like hipBLASLt).
- No data at M = 257–511 or N > 32768.

## Sources

Evaluation scripts and reports (aiter repo, `streamk_study/`):

- `repro/eval_joint_rule.py` → `2026-10-20_all_tuned/results/joint_rule_eval.md` (compute-bound)
- `repro/eval_mb_time.py` → `2026-10-20_all_tuned/results/mb_time_eval.md` (time model)
- `repro/eval_mb_cover.py`, `repro/eval_mb_cover_cache.py` → `2026-10-20_all_tuned/results/mb_cover_eval.txt` (simpler alternative)
