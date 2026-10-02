# GPU Roofline and Arithmetic Intensity — Conversation Summary

## The conversation in one arc

The thread that ran through everything: **arithmetic intensity is not a property of a GEMM.** It is a property of a GEMM *and a memory boundary*, and the same 4096³ fp16 matmul has AI of 0.5, 64, or 1365 FLOP/byte depending on which boundary you measure at. Almost every confusion in roofline reasoning comes from comparing numbers taken at different boundaries as though they were rival estimates of one quantity.

The opening question had the direction of the standard calculation inverted, and that turned out to be the productive thing to fix. `gemm_bytes` counts `MK + KN + MN` — each element exactly once — which is only achievable if every reuse is a cache hit. It is the *compulsory* traffic, so it is an **upper** bound on AI and a **best** case. A genuinely cacheless machine gives `1/elem_bytes`, a flat 0.5 FLOP/byte independent of shape, where nothing is ever compute-bound.

## The infinite-cache question

Separating "size" from "speed" matters.

**Capacity** is the assumption the formula actually encodes, and "large enough" is sharper than "infinite": you need to hold roughly one full matrix, so `Z ~ n²` — 33.5 MB for our case, which exceeds MI300X's ~32 MB of aggregate L2 but sits inside the 256 MB Infinity Cache. That matches the Hong–Kung bound `Q = Ω(n³/√Z)`, which only descends to `Θ(n²)` exactly when `Z = Θ(n²)`. Both routes say achievable AI scales as `√Z / e`, which is why quadrupling fast memory only doubles intensity, and why LDS alone can never get there.

**Speed** is different: the single-ceiling roofline does not assume intermediate levels are infinitely fast, it structurally cannot express them being finite. And that omission bites — 128×128 fp16 tiles need 20.4 TB/s to reach compute peak, which exceeds Infinity Cache's ~17 TB/s, so even perfect MALL residency caps you near 84% of peak and only L2 can close it.

The logical point was right and is the whole value of the bound. Memory-bound under the compulsory model is a **certificate**: actual bytes ≥ compulsory bytes, so no schedule escapes it. Compute-bound is only a **permit**. Two things legitimately escape the certificate:

- You can change `Q_comp` itself (quantization, fusion — which is why weight-only int4 exists).
- Cross-invocation residency, where weights already sitting in MALL from a previous launch make real traffic fall *below* "compulsory." That last one is a live benchmarking hazard for a framework like TensorAtlas.

## The tiled formula

Each workgroup reads a full `B_M × K` strip of A and `K × B_N` strip of B, so `BLOCK_K` cancels — it affects LDS footprint and pipelining but not byte count. Summing over `(M/B_M)(N/B_N)` workgroups and dividing by `MNK` makes the problem size cancel too:

```
AI_tiled ≈ 2 / (e * (1/B_M + 1/B_N)) = H(B_M, B_N) / elem_bytes
```

Valid when `K >> (e_c/e) * H/2` (about `K >> 64` at 128×128 fp16). It checks out at both limits: `B_M = B_N = 1` reproduces the cacheless 0.5, and tile-equals-matrix reproduces the compulsory formula.

The harmonic mean showed up because traffic is a *sum of reciprocals* of the tile dimensions — the same reason it governs parallel resistors. Its key property is being dominated by its smaller argument, capped at `2 * min(B_M, B_N)`. Two consequences did real work later:

- For equal dtypes, square tiles are optimal at fixed footprint.
- For mixed precision, the optimum is `B_N/B_M = e_a/e_b` — skew *wide in N* when A is the wider operand.

For skinny GEMMs, `AI_comp ≈ 2M/e`, giving a compute-bound threshold of `M > 246` rows on MI300X that is **invariant to weight precision**, since int8 doubles both the peak and the intensity.

## Q and the bridge between the bounds

`Q` is the I/O-complexity literature's symbol for traffic volume across a named boundary.

- `Q_comp` is the theoretical floor at HBM.
- `Q_tiled` is what workgroups request below LDS.
- `Q_HBM` is what actually happens and the only one you can measure.

First touches must come from HBM; re-requests hit or miss:

```
Q_HBM = Q_comp + (1 - h) * (Q_tiled - Q_comp)
```

`h` is the hit rate on redundant traffic specifically, not the overall rate a counter reports (78% versus 74% in our case). The endpoints reproduce 63 and 1365, so it genuinely bridges the two brackets. Solving for the ridge gives `h* ≈ 78%` at 4096³ with 128×128 — and the curve is convex, so 78→95% nearly triples AI while 0→50% barely doubles it. Scheduling gets *more* valuable the better it already is.

Comparing both bracket ends to the ridge gives three cases, and case 1 (compute-bound regardless of caching) turns out to be **unreachable on MI300X** — it needs `H > 492`, past any feasible tile. So every compute-bound GEMM on that hardware is compute-bound because of L2 and Infinity Cache.

## Smaller shapes, and 256×64

Yes: `AI_tiled` is shape-independent while the ceiling `AI_comp = n/3` descends, so the bracket narrows from above and `h*` climbs — 78% at n=4096, 92% at 1024, 99% at 768, impossible below 738. But feasibility rises in step, since a 1024³ working set is 6.3 MB and fits one XCD's L2. The genuinely painful regime is **skinny, not small**: big weight footprint with tiny `AI_comp`.

For 256×64 the answer was **82.5%** at 4096³ fp16, versus 78.0% for 128×128, because `H(256,64) = 102.4` generates 2617 MB of redundancy against the same fixed 458 MB allowance. On the a8w4 shape, it is worse — 112 versus 165 FLOP/byte — because a8w4 wants `B_N/B_M ≈ 1.9`, the opposite skew. The counter-pressure keeping small `BLOCK_SIZE_M` in the tuning space is ragged per-expert occupancy, which the traffic model cannot see.

## The resolution to the last challenge

Classification without `h` is a weak predictor. But `h` is not a free parameter. Redo the derivation for a window of `a × b` concurrently-resident blocks and you get the *identical* formula with enlarged dimensions:

```
AI = H(a * B_M, b * B_N) / e
```

The hierarchy introduces no new mechanism — it multiplies your effective tile. So `h` is a reparameterization of `(a, b)`, governed by concurrency (CUs × occupancy), per-XCD L2 capacity, and the window's *shape*. Since `H` is dominated by its smaller argument, a row-major sweep with `a=2, b=64` is far worse than a compact `a=b=32` at identical concurrency — which is precisely and quantitatively what `GROUP_SIZE_M` and XCD-aware remapping buy.

And you do not predict `h`, you measure `Q_HBM` and derive it. The bracket's real value is making that measurement interpretable:

- At the wall — stop tuning this shape.
- Getting no cache help — scheduling problem: swizzle, group size, XCD mapping.
- **Below your own tile's intrinsic AI** — the problem is not memory at all but occupancy, bank conflicts, or MFMA utilization.

That third diagnosis is invisible without the tile-level reference line. Absent any `h`, two things still survive:

- Case 3 tells you the whole config space is irrelevant and only bytes matter.
- `min(peak, AI_comp × BW)` gives you an absolute yardstick and a stopping criterion.

## Reference numbers (MI300X, fp16)

| Model | AI (FLOP/byte) | Ceiling at 5.325 TB/s | BW needed at 1307 TFLOPS |
|---|---|---|---|
| No reuse | 0.5 | 2.7 TFLOPS (0.2% of peak) | 2600 TB/s |
| 128×128 tiled | 64 | 341 TFLOPS (26%) | 20.4 TB/s |
| Perfect cache (n=4096) | 1365 | 7.3 PFLOPS (5.6× peak) | 0.96 TB/s |

Ridge point: 1307.4 TFLOPS / 5325 GB/s ≈ 246 FLOP/byte.