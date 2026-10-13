HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 13:20

## handshake

256 KiB partials published by `writers` CUs of every XCD at once and each consumed by one CU of the next XCD (other XCD) or of the same XCD. Writers and readers start at the same stamp. us, medians over 30 launches x all pairs; back to back, no flush between launches (warm). 'dirty' = 16 other CUs of every XCD wrote that many MiB in total (default stores) just before. 'bad' counts wrong values the readers saw (all launches). The atomic flag is a relaxed `gpu`-scope swap polled with relaxed `gpu`-scope atomic loads.

| sequence | placement | writers per XCD | dirty MiB per XCD | publish us | flag seen after us | read us | total us | bad |
|---|---|---|---|---|---|---|---|---|
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 1 | 0 | 4.44 | 0.60 | 4.44 | 9.29 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 1 | 2 | 4.40 | 0.60 | 4.44 | 9.32 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads (v21 `wt`) | other XCD | 16 | 0 | 15.24 | 0.74 | 4.92 | 23.09 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 1 | 0 | 4.44 | 0.61 | 4.44 | 9.29 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 1 | 2 | 4.40 | 0.61 | 4.44 | 9.27 | 0 |
| `.wt` stores, drain, atomic flag, `.cg` reads | other XCD | 16 | 0 | 15.20 | 0.74 | 6.00 | 27.57 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 1 | 0 | 4.44 | 0.49 | 4.44 | 9.18 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 1 | 2 | 4.40 | 0.49 | 4.48 | 9.21 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads (reference kernel) | other XCD | 16 | 0 | 15.20 | 0.63 | 4.88 | 21.38 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 1 | 0 | 3.80 | 0.61 | 4.44 | 8.82 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 1 | 2 | 4.48 | 0.60 | 4.44 | 9.54 | 0 |
| default stores, agent release, atomic flag, `.cg` reads | other XCD | 16 | 0 | 27.76 | 0.85 | 6.56 | 36.96 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 1 | 0 | 3.80 | 0.61 | 4.76 | 9.17 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 1 | 2 | 4.52 | 0.61 | 4.78 | 9.90 | 0 |
| default stores, agent release, atomic flag, agent acquire, default reads | other XCD | 16 | 0 | 27.52 | 0.85 | 8.24 | 37.34 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 1 | 0 | 2.32 | 0.52 | 3.16 | 6.04 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 1 | 2 | 2.32 | 0.52 | 3.16 | 6.00 | 0 |
| default stores, drain, atomic flag, `.cg` reads | same XCD | 16 | 0 | 3.56 | 0.56 | 3.60 | 7.76 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 1 | 0 | 2.32 | 0.20 | 3.16 | 5.72 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 1 | 2 | 2.32 | 0.20 | 3.16 | 5.68 | 0 |
| default stores, drain, plain flag polled with `.cg`, `.cg` reads (v21 `l2`) | same XCD | 16 | 0 | 3.56 | 0.52 | 3.56 | 7.72 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 1 | 0 | 2.32 | 0.56 | 3.46 | 6.32 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 1 | 2 | 2.32 | 0.52 | 3.48 | 6.32 | 0 |
| default stores, drain, atomic flag, agent acquire, default reads | same XCD | 16 | 0 | 3.56 | 0.56 | 6.88 | 11.08 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 1 | 0 | 3.12 | 0.48 | 4.48 | 8.04 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 1 | 2 | 3.10 | 0.48 | 4.48 | 8.02 | 0 |
| `.wt` stores, drain, atomic flag, `.cv` reads | same XCD | 16 | 0 | 4.52 | 0.52 | 4.92 | 10.00 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 1 | 0 | 3.12 | 0.40 | 4.46 | 7.96 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 1 | 2 | 3.08 | 0.40 | 4.44 | 7.94 | 0 |
| `.wt` stores, drain, `.wt` flag polled with `.cv`, `.cv` reads | same XCD | 16 | 0 | 4.52 | 0.52 | 4.88 | 9.92 | 0 |

## arrive

Last-arriver reductions over all 256 CUs: groups of n programs, each storing a partial of the given size, draining, meeting its workgroup at `s_barrier` and adding 1 to the group's counter; the program that sees n - 1 sums the n partials. 'One XCD': each group on one XCD (as aiter's `_tile_on_xcd`); 'spread': n consecutive pids on min(n, 8) XCDs. us, medians over 30 launches x groups; back to back, no flush between launches (warm). 'last arrival' is from the common start stamp to the last member's atomic returning; 'sum' is the last member reading and adding the n partials. 'wrong' counts groups whose sum was wrong (all launches).

| sequence | placement | status | n | partial KiB | publish us | last arrival us | sum us | total us | wrong / groups |
|---|---|---|---|---|---|---|---|---|---|
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 8 | 0.32 | 0.80 | 1.56 | 2.36 | 0 / 1920 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 256 | 7.00 | 7.56 | 71.04 | 78.40 | 0 / 1920 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 8 | 0.32 | 0.88 | 1.80 | 2.72 | 0 / 480 |
| aiter: default stores, drain, `acq_rel` `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 256 | 6.96 | 7.62 | 57.36 | 65.30 | 0 / 480 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 8 | 0.32 | 0.80 | 1.56 | 2.36 | 0 / 1920 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 4 | 256 | 6.92 | 7.52 | 70.84 | 78.20 | 0 / 1920 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 8 | 0.32 | 0.88 | 1.80 | 2.68 | 0 / 480 |
| default stores, drain, relaxed `cta` arrival, `.cv` sum | one XCD | not guaranteed | 16 | 256 | 6.96 | 7.58 | 57.32 | 65.28 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 4 | 8 | 0.32 | 0.80 | 1.56 | 2.36 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 4 | 256 | 6.94 | 7.54 | 70.96 | 78.24 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 16 | 8 | 0.32 | 0.88 | 1.80 | 2.68 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | one XCD, checked placement | 16 | 256 | 6.98 | 7.62 | 57.46 | 65.32 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 4 | 8 | 0.32 | 0.80 | 1.56 | 2.36 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 4 | 256 | 7.56 | 8.08 | 72.44 | 80.28 | 0 / 1920 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 16 | 8 | 0.32 | 0.88 | 1.92 | 2.76 | 0 / 480 |
| default stores, drain, relaxed `gpu` arrival, `.cg` sum | one XCD | one XCD, checked placement | 16 | 256 | 7.64 | 8.14 | 61.28 | 69.40 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 4 | 8 | 0.60 | 1.08 | 2.64 | 3.72 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 4 | 256 | 8.72 | 9.24 | 70.72 | 79.44 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 16 | 8 | 0.60 | 1.08 | 2.56 | 3.64 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | one XCD | any placement | 16 | 256 | 8.76 | 9.36 | 56.96 | 66.32 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 4 | 8 | 0.32 | 3.00 | 3.00 | 5.96 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 4 | 256 | 8.32 | 13.44 | 69.56 | 82.84 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 16 | 8 | 0.32 | 3.08 | 3.20 | 6.28 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | one XCD | documented | 16 | 256 | 8.52 | 13.90 | 56.96 | 70.92 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 4 | 8 | 0.80 | 1.64 | 2.56 | 4.20 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 4 | 256 | 10.64 | 11.93 | 67.96 | 79.96 | 0 / 1920 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 16 | 8 | 1.40 | 2.61 | 2.52 | 5.13 | 0 / 480 |
| `.wt` stores, drain, relaxed `gpu` arrival, `.cv` sum | spread | any placement | 16 | 256 | 10.48 | 12.93 | 56.60 | 69.72 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 4 | 8 | 0.32 | 3.69 | 3.68 | 7.39 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 4 | 256 | 11.40 | 18.64 | 69.40 | 88.21 | 0 / 1920 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 16 | 8 | 0.32 | 4.77 | 3.96 | 8.60 | 0 / 480 |
| default stores, drain, `acq_rel` `gpu` arrival, default sum | spread | documented | 16 | 256 | 10.12 | 19.18 | 56.52 | 75.67 | 0 / 480 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 4 | 8 | 0.80 | 1.67 | 2.56 | 4.25 | 0 / 1920 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 4 | 256 | 10.56 | 11.98 | 67.88 | 79.89 | 0 / 1920 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 16 | 8 | 1.20 | 2.58 | 2.52 | 5.10 | 0 / 480 |
| `.wt` stores, drain, `acq_rel` `cta` arrival, `.cv` sum | spread | not guaranteed | 16 | 256 | 10.56 | 12.84 | 56.92 | 69.97 | 0 / 480 |
| aiter's sequence with the group spread over XCDs | spread | negative | 4 | 8 | 0.32 | 0.91 | 2.44 | 3.35 | 1898 / 1920 |
| aiter's sequence with the group spread over XCDs | spread | negative | 4 | 256 | 7.16 | 8.42 | 69.44 | 77.92 | 282 / 1920 |
| aiter's sequence with the group spread over XCDs | spread | negative | 16 | 8 | 0.32 | 1.05 | 3.38 | 4.44 | 478 / 480 |
| aiter's sequence with the group spread over XCDs | spread | negative | 16 | 256 | 7.14 | 8.50 | 60.12 | 68.63 | 0 / 480 |
