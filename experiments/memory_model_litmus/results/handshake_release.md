HIP_VISIBLE_DEVICES=3, AMD Instinct MI355X, triton 3.8.0, 2026-10-07 12:09

## handshake_release

As the handshake sweep, with the release and acquire the compiler emits for `sem="release"` on the flag swap and `sem="acquire"` on one atomic load after the spin, instead of `buffer_wbl2 sc1` / `buffer_inv sc1` written in every wave. Every wave still drains and meets the others at `s_barrier` before the flag. us, medians over 30 launches x all pairs; a 1 GiB read before each launch (cold). 'fences in the code' counts the `buffer_wbl2` / `buffer_inv` instructions in the compiled kernel. The compiler's sit in the block that only the program's first lane executes, so they run once per CU; the hand-written one runs in all 4 waves.

| sequence | placement | writers per XCD | publish us | flag seen after us | read us | total us | bad | fences in the code |
|---|---|---|---|---|---|---|---|---|
| `.wt` stores, drain, atomic flag, `.cv` reads | other XCD | 1 | 3.12 | 0.62 | 4.46 | 8.16 | 0 | 0 `buffer_wbl2`, 0 `buffer_inv` |
| `.wt` stores, drain, atomic flag, `.cv` reads | other XCD | 16 | 4.56 | 0.71 | 4.88 | 10.22 | 0 | 0 `buffer_wbl2`, 0 `buffer_inv` |
| default stores, drain, release flag (compiler), `.cg` reads | other XCD | 1 | 2.32 | 1.22 | 4.48 | 8.00 | 0 | 1 `buffer_wbl2`, 0 `buffer_inv` |
| default stores, drain, release flag (compiler), `.cg` reads | other XCD | 16 | 3.64 | 4.07 | 5.04 | 13.84 | 0 | 1 `buffer_wbl2`, 0 `buffer_inv` |
| default stores, drain, release flag, acquire (compiler), default reads | other XCD | 1 | 2.32 | 1.19 | 4.60 | 8.08 | 0 | 1 `buffer_wbl2`, 1 `buffer_inv` |
| default stores, drain, release flag, acquire (compiler), default reads | other XCD | 16 | 3.60 | 4.12 | 5.48 | 14.24 | 0 | 1 `buffer_wbl2`, 1 `buffer_inv` |
| default stores, `buffer_wbl2 sc1` in every wave, atomic flag, `.cg` reads | other XCD | 1 | 3.08 | 0.62 | 4.46 | 8.11 | 0 | 1 `buffer_wbl2`, 0 `buffer_inv` |
| default stores, `buffer_wbl2 sc1` in every wave, atomic flag, `.cg` reads | other XCD | 16 | 13.20 | 0.99 | 5.92 | 21.95 | 0 | 1 `buffer_wbl2`, 0 `buffer_inv` |
| default stores, drain, atomic flag, acquire (compiler), default reads | same XCD | 1 | 2.32 | 0.48 | 6.68 | 9.48 | 0 | 0 `buffer_wbl2`, 1 `buffer_inv` |
| default stores, drain, atomic flag, acquire (compiler), default reads | same XCD | 16 | 3.56 | 0.60 | 7.20 | 11.40 | 0 | 0 `buffer_wbl2`, 1 `buffer_inv` |
