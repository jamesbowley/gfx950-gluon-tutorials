HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 13:20

## bits

256 KiB per CU with inline-asm `global_store_dwordx2` (8 bytes per lane), then a drain. One CU per XCD, or 16 per XCD at once. us from the start stamp to the drain, medians over launches x CUs.

| bits | CUs per XCD | us | GB/s per CU |
|---|---|---|---|
| none | 1 | 3.56 | 74 |
| `sc0 nt` | 1 | 3.56 | 74 |
| `sc1` | 1 | 3.72 | 70 |
| `sc0 sc1` | 1 | 3.72 | 70 |
| none | 16 | 3.92 | 67 |
| `sc0 nt` | 16 | 4.36 | 60 |
| `sc1` | 16 | 4.92 | 53 |
| `sc0 sc1` | 16 | 4.88 | 54 |

## ops

Every XCD at once: its 31 helper CUs first write (dirty) or twice read (valid) the given MiB in total, then slot 0 (or all 32 programs) issues the operation and `s_waitcnt vmcnt(0)`, then the helpers read their data again. us, medians over 20 launches x 8 XCDs; a 1 GiB read before each launch.

| XCD's lines | MiB per XCD | operation | issued by | op us | helpers' re-read us | re-read vs none |
|---|---|---|---|---|---|---|
| dirty | 0.12 | none | one CU | 0.04 | 0.16 | +0.00 |
| dirty | 0.12 | `buffer_wbl2 sc1` | one CU | 0.84 | 0.16 | +0.00 |
| dirty | 0.12 | `buffer_wbl2 sc1` | all 32 CUs | 6.72 | 0.16 | +0.00 |
| dirty | 0.12 | `buffer_inv sc1` | one CU | 0.48 | 0.16 | +0.00 |
| dirty | 0.12 | `buffer_inv sc1` | all 32 CUs | 6.44 | 0.64 | +0.48 |
| dirty | 0.97 | none | one CU | 0.04 | 3.52 | +0.00 |
| dirty | 0.97 | `buffer_wbl2 sc1` | one CU | 1.68 | 2.08 | -1.44 |
| dirty | 0.97 | `buffer_wbl2 sc1` | all 32 CUs | 7.56 | 2.08 | -1.44 |
| dirty | 0.97 | `buffer_inv sc1` | one CU | 0.48 | 3.52 | +0.00 |
| dirty | 0.97 | `buffer_inv sc1` | all 32 CUs | 6.40 | 3.64 | +0.12 |
| dirty | 1.94 | none | one CU | 0.04 | 4.60 | +0.00 |
| dirty | 1.94 | `buffer_wbl2 sc1` | one CU | 2.04 | 4.24 | -0.36 |
| dirty | 1.94 | `buffer_wbl2 sc1` | all 32 CUs | 7.92 | 4.20 | -0.40 |
| dirty | 1.94 | `buffer_inv sc1` | one CU | 0.48 | 4.60 | +0.00 |
| dirty | 1.94 | `buffer_inv sc1` | all 32 CUs | 6.40 | 4.72 | +0.12 |
| dirty | 4.00 | none | one CU | 0.04 | 9.12 | +0.00 |
| dirty | 4.00 | `buffer_wbl2 sc1` | one CU | 3.04 | 8.52 | -0.60 |
| dirty | 4.00 | `buffer_wbl2 sc1` | all 32 CUs | 8.84 | 8.52 | -0.60 |
| dirty | 4.00 | `buffer_inv sc1` | one CU | 0.48 | 9.08 | -0.04 |
| dirty | 4.00 | `buffer_inv sc1` | all 32 CUs | 6.36 | 9.16 | +0.04 |
| valid | 0.12 | none | one CU | 0.04 | 0.16 | +0.00 |
| valid | 0.12 | `buffer_wbl2 sc1` | one CU | 0.48 | 0.16 | +0.00 |
| valid | 0.12 | `buffer_wbl2 sc1` | all 32 CUs | 6.40 | 0.16 | +0.00 |
| valid | 0.12 | `buffer_inv sc1` | one CU | 0.48 | 0.12 | -0.04 |
| valid | 0.12 | `buffer_inv sc1` | all 32 CUs | 6.40 | 0.20 | +0.04 |
| valid | 0.97 | none | one CU | 0.04 | 0.96 | +0.00 |
| valid | 0.97 | `buffer_wbl2 sc1` | one CU | 0.48 | 1.00 | +0.04 |
| valid | 0.97 | `buffer_wbl2 sc1` | all 32 CUs | 6.40 | 0.96 | +0.00 |
| valid | 0.97 | `buffer_inv sc1` | one CU | 0.48 | 1.00 | +0.04 |
| valid | 0.97 | `buffer_inv sc1` | all 32 CUs | 6.44 | 1.00 | +0.04 |
| valid | 1.94 | none | one CU | 0.04 | 1.92 | +0.00 |
| valid | 1.94 | `buffer_wbl2 sc1` | one CU | 0.48 | 2.00 | +0.08 |
| valid | 1.94 | `buffer_wbl2 sc1` | all 32 CUs | 6.44 | 1.92 | +0.00 |
| valid | 1.94 | `buffer_inv sc1` | one CU | 0.48 | 2.00 | +0.08 |
| valid | 1.94 | `buffer_inv sc1` | all 32 CUs | 6.40 | 1.88 | -0.04 |
| valid | 4.00 | none | one CU | 0.04 | 5.28 | +0.00 |
| valid | 4.00 | `buffer_wbl2 sc1` | one CU | 0.48 | 5.44 | +0.16 |
| valid | 4.00 | `buffer_wbl2 sc1` | all 32 CUs | 6.44 | 5.32 | +0.04 |
| valid | 4.00 | `buffer_inv sc1` | one CU | 0.48 | 5.44 | +0.16 |
| valid | 4.00 | `buffer_inv sc1` | all 32 CUs | 6.44 | 5.32 | +0.04 |

## handshake

256 KiB partials published by `writers` CUs of every XCD at once and each consumed by one CU of the next XCD (other XCD) or of the same XCD. Writers and readers start at the same stamp. us, medians over 30 launches x all pairs; a 1 GiB read before each launch (cold). 'dirty' = 16 other CUs of every XCD wrote that many MiB in total (default stores) just before. 'bad' counts wrong values the readers saw (all launches). The atomic flag is a relaxed `gpu`-scope swap polled with relaxed `gpu`-scope atomic loads.

| sequence | placement | writers per XCD | dirty MiB per XCD | publish us | flag seen after us | read us | total us | bad |
|---|---|---|---|---|---|---|---|---|
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 1 | 0 | 3.10 | 0.59 | 4.44 | 8.21 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 1 | 2 | 3.08 | 0.61 | 4.44 | 8.17 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 16 | 0 | 4.52 | 0.53 | 4.92 | 9.97 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 1 | 0 | 3.08 | 0.61 | 4.44 | 8.15 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 1 | 2 | 3.08 | 0.61 | 4.44 | 8.14 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 16 | 0 | 4.52 | 0.53 | 5.00 | 10.11 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 1 | 0 | 3.12 | 0.47 | 4.44 | 8.09 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 1 | 2 | 3.08 | 0.49 | 4.44 | 8.05 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 16 | 0 | 4.52 | 0.41 | 4.96 | 9.89 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 1 | 0 | 3.08 | 0.61 | 4.44 | 8.17 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 1 | 2 | 4.14 | 0.61 | 4.44 | 9.21 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 16 | 0 | 13.16 | 0.87 | 6.24 | 22.25 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 1 | 0 | 3.08 | 0.61 | 4.76 | 8.45 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 1 | 2 | 4.12 | 0.61 | 4.76 | 9.53 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 16 | 0 | 13.04 | 0.83 | 8.04 | 22.85 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 1 | 0 | 2.32 | 0.48 | 6.52 | 9.32 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 1 | 2 | 2.32 | 0.48 | 6.50 | 9.30 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 16 | 0 | 3.52 | 0.44 | 6.64 | 10.68 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 1 | 0 | 2.32 | 0.20 | 6.52 | 9.04 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 1 | 2 | 2.32 | 0.20 | 6.48 | 9.00 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 16 | 0 | 3.52 | 0.28 | 6.60 | 10.56 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 1 | 0 | 2.32 | 0.48 | 6.84 | 9.64 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 1 | 2 | 2.32 | 0.48 | 6.72 | 9.52 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 16 | 0 | 3.52 | 0.44 | 9.56 | 13.60 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 1 | 0 | 3.12 | 0.48 | 4.48 | 8.04 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 1 | 2 | 3.08 | 0.48 | 4.44 | 8.00 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 16 | 0 | 4.52 | 0.44 | 4.92 | 9.88 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 1 | 0 | 3.12 | 0.40 | 4.44 | 7.92 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 1 | 2 | 3.08 | 0.40 | 4.44 | 7.92 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 16 | 0 | 4.52 | 0.32 | 4.92 | 9.80 | 0 |

## arrive

Last-arriver reductions over all 256 CUs: groups of n programs, each storing a partial of the given size, draining, meeting its workgroup at `s_barrier` and adding 1 to the group's counter; the program that sees n - 1 sums the n partials. 'One XCD': each group on one XCD (as aiter's `_tile_on_xcd`); 'spread': n consecutive pids on min(n, 8) XCDs. us, medians over 30 launches x groups; a 1 GiB read before each launch (cold). 'last arrival' is from the common start stamp to the last member's atomic returning; 'sum' is the last member reading and adding the n partials. 'wrong' counts groups whose sum was wrong (all launches).

| sequence | placement | status | n | partial KiB | publish us | last arrival us | sum us | total us | wrong / groups |
|---|---|---|---|---|---|---|---|---|---|
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 8 | 0.32 | 0.68 | 4.16 | 4.88 | 0 / 1920 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 256 | 7.52 | 8.08 | 70.94 | 78.52 | 0 / 1920 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 8 | 0.32 | 0.88 | 3.64 | 4.48 | 0 / 480 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 256 | 7.52 | 8.16 | 57.08 | 65.36 | 0 / 480 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 8 | 0.32 | 0.68 | 4.12 | 4.84 | 0 / 1920 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 256 | 7.52 | 8.04 | 70.88 | 78.60 | 0 / 1920 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 8 | 0.32 | 0.84 | 3.60 | 4.48 | 0 / 480 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 256 | 7.52 | 8.14 | 57.16 | 65.44 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 4 | 8 | 0.32 | 0.70 | 4.12 | 4.84 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 4 | 256 | 7.52 | 8.04 | 70.88 | 78.56 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 16 | 8 | 0.32 | 0.84 | 3.68 | 4.52 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 16 | 256 | 7.48 | 8.12 | 57.26 | 65.46 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 4 | 8 | 0.32 | 0.68 | 4.04 | 4.76 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 4 | 256 | 7.48 | 8.04 | 81.40 | 89.12 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 16 | 8 | 0.32 | 0.84 | 3.64 | 4.48 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 16 | 256 | 7.52 | 8.16 | 65.98 | 74.00 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 4 | 8 | 0.56 | 1.00 | 2.64 | 3.64 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 4 | 256 | 8.84 | 9.34 | 70.72 | 79.80 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 16 | 8 | 0.56 | 1.06 | 2.52 | 3.60 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 16 | 256 | 8.78 | 9.46 | 56.90 | 66.56 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 4 | 8 | 0.32 | 2.88 | 4.16 | 7.04 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 4 | 256 | 9.00 | 13.82 | 69.56 | 83.08 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 16 | 8 | 0.32 | 3.00 | 3.92 | 6.92 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 16 | 256 | 9.16 | 14.36 | 56.96 | 71.32 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 4 | 8 | 0.56 | 1.16 | 2.60 | 3.76 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 4 | 256 | 9.36 | 10.37 | 67.88 | 78.27 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 16 | 8 | 0.56 | 1.23 | 2.88 | 4.11 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 16 | 256 | 9.36 | 10.59 | 57.48 | 68.12 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 4 | 8 | 0.32 | 3.05 | 4.12 | 7.17 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 4 | 256 | 9.00 | 14.92 | 69.08 | 84.28 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 16 | 8 | 0.32 | 3.11 | 4.24 | 7.35 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 16 | 256 | 9.24 | 15.71 | 56.20 | 71.98 | 0 / 480 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 4 | 8 | 0.56 | 1.16 | 2.60 | 3.76 | 0 / 1920 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 4 | 256 | 9.40 | 10.43 | 67.88 | 78.31 | 0 / 1920 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 16 | 8 | 0.56 | 1.23 | 2.88 | 4.12 | 0 / 480 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 16 | 256 | 9.36 | 10.64 | 57.44 | 68.07 | 0 / 480 |
| aiter's sequence with the group spread over XCDs | spread | negative | 4 | 8 | 0.32 | 0.87 | 3.64 | 4.50 | 1907 / 1920 |
| aiter's sequence with the group spread over XCDs | spread | negative | 4 | 256 | 7.92 | 8.82 | 68.36 | 77.22 | 235 / 1920 |
| aiter's sequence with the group spread over XCDs | spread | negative | 16 | 8 | 0.32 | 0.96 | 3.60 | 4.55 | 476 / 480 |
| aiter's sequence with the group spread over XCDs | spread | negative | 16 | 256 | 7.88 | 8.95 | 58.78 | 67.69 | 0 / 480 |
