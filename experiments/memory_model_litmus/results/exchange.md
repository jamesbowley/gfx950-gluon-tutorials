HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-06 18:48

## exchange

W writers exchange partials of P KiB in groups of n = 16, with one of four synchronisation methods (sync): `wt` = `.wt` stores, relaxed sync, `.cv` loads (correct for any placement); `acq_rel` = default stores, `gpu` release / acquire, default loads (correct for any placement); `l2` = default stores, relaxed sync, `.cg` loads (correct only with the group on one XCD); `wave_release` = as `acq_rel`, but the release is `buffer_wbl2 sc1` in every wave before the barrier and the flag / counter is relaxed. Placement: 'one XCD' puts each group on one XCD, as stream-K's chiplet swizzle places a tile's programs (groups fill an XCD, then the next); 'spread' (a reference, write-through only) uses consecutive hardware pids, one member per XCD. Loads with 16 in flight per CU, a 1 GiB read before each launch. Measured: medians over 30 launches of the makespan (first stamp to the last program's end), the writers' publish time and the consumers' read time. Model (`wt`): the slowest of the CU, XCD-link and chip limits per phase, bytes counted per XCD for the placement (exchange.py, model_us). MB moved is writes + reads.

| pattern | sync | placement | W | P KiB | MB written | MB read | measured us | model us | write limit | read limit | publish us | read us | achieved TB/s | wrong | timeouts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| handoff | wt | one XCD | 16 | 128 | 2.1 | 2.1 | 5.08 | 4.03 | XCD link | CU | 2.16 | 2.20 | 0.83 | 0 | 0 |
| handoff | wt | one XCD | 32 | 128 | 4.2 | 4.2 | 5.32 | 4.69 | XCD link | CU | 2.36 | 2.20 | 1.58 | 0 | 0 |
| handoff | wt | one XCD | 64 | 128 | 8.4 | 8.4 | 5.68 | 4.69 | XCD link | CU | 2.40 | 2.52 | 2.95 | 0 | 0 |
| handoff | wt | one XCD | 128 | 128 | 16.8 | 16.8 | 5.89 | 5.10 | chip | chip | 2.40 | 2.60 | 5.69 | 0 | 0 |
| handoff | wt | one XCD | 16 | 256 | 4.2 | 4.2 | 8.94 | 7.47 | XCD link | CU | 3.92 | 4.24 | 0.94 | 0 | 0 |
| handoff | wt | one XCD | 32 | 256 | 8.4 | 8.4 | 9.32 | 8.78 | XCD link | CU | 4.28 | 4.16 | 1.80 | 0 | 0 |
| handoff | wt | one XCD | 64 | 256 | 16.8 | 16.8 | 10.00 | 8.78 | XCD link | CU | 4.36 | 4.72 | 3.36 | 0 | 0 |
| handoff | wt | one XCD | 128 | 256 | 33.6 | 33.6 | 10.71 | 9.61 | chip | chip | 4.84 | 4.88 | 6.26 | 0 | 0 |
| handoff | acq_rel | one XCD | 16 | 128 | 2.1 | 2.1 | 8.07 | - | - | - | 1.60 | 3.00 | 0.52 | 0 | 0 |
| handoff | acq_rel | one XCD | 32 | 128 | 4.2 | 4.2 | 8.53 | - | - | - | 1.60 | 2.92 | 0.98 | 0 | 0 |
| handoff | acq_rel | one XCD | 64 | 128 | 8.4 | 8.4 | 8.68 | - | - | - | 1.60 | 3.08 | 1.93 | 0 | 0 |
| handoff | acq_rel | one XCD | 128 | 128 | 16.8 | 16.8 | 8.98 | - | - | - | 1.60 | 3.24 | 3.74 | 0 | 0 |
| handoff | acq_rel | one XCD | 16 | 256 | 4.2 | 4.2 | 11.96 | - | - | - | 3.00 | 4.96 | 0.70 | 0 | 0 |
| handoff | acq_rel | one XCD | 32 | 256 | 8.4 | 8.4 | 12.63 | - | - | - | 3.24 | 4.84 | 1.33 | 0 | 0 |
| handoff | acq_rel | one XCD | 64 | 256 | 16.8 | 16.8 | 12.86 | - | - | - | 3.22 | 5.08 | 2.61 | 0 | 0 |
| handoff | acq_rel | one XCD | 128 | 256 | 33.6 | 33.6 | 13.23 | - | - | - | 3.36 | 5.20 | 5.07 | 0 | 0 |
| handoff | l2 | one XCD | 16 | 128 | 2.1 | 2.1 | 5.52 | - | - | - | 1.60 | 3.16 | 0.76 | 0 | 0 |
| handoff | l2 | one XCD | 32 | 128 | 4.2 | 4.2 | 5.66 | - | - | - | 1.60 | 3.16 | 1.48 | 0 | 0 |
| handoff | l2 | one XCD | 64 | 128 | 8.4 | 8.4 | 5.88 | - | - | - | 1.60 | 3.32 | 2.85 | 0 | 0 |
| handoff | l2 | one XCD | 128 | 128 | 16.8 | 16.8 | 6.20 | - | - | - | 1.60 | 3.52 | 5.41 | 0 | 0 |
| handoff | l2 | one XCD | 16 | 256 | 4.2 | 4.2 | 9.22 | - | - | - | 3.00 | 5.32 | 0.91 | 0 | 0 |
| handoff | l2 | one XCD | 32 | 256 | 8.4 | 8.4 | 9.57 | - | - | - | 3.22 | 5.32 | 1.75 | 0 | 0 |
| handoff | l2 | one XCD | 64 | 256 | 16.8 | 16.8 | 9.92 | - | - | - | 3.24 | 5.64 | 3.38 | 0 | 0 |
| handoff | l2 | one XCD | 128 | 256 | 33.6 | 33.6 | 11.15 | - | - | - | 3.32 | 6.28 | 6.02 | 0 | 0 |
| handoff | wave_release | one XCD | 16 | 128 | 2.1 | 2.1 | 11.43 | - | - | - | 7.84 | 2.36 | 0.37 | 0 | 0 |
| handoff | wave_release | one XCD | 32 | 128 | 4.2 | 4.2 | 12.88 | - | - | - | 8.72 | 2.32 | 0.65 | 0 | 0 |
| handoff | wave_release | one XCD | 64 | 128 | 8.4 | 8.4 | 13.09 | - | - | - | 8.80 | 2.40 | 1.28 | 0 | 0 |
| handoff | wave_release | one XCD | 128 | 128 | 16.8 | 16.8 | 13.48 | - | - | - | 8.80 | 2.44 | 2.49 | 0 | 0 |
| handoff | wave_release | one XCD | 16 | 256 | 4.2 | 4.2 | 16.19 | - | - | - | 10.86 | 4.20 | 0.52 | 0 | 0 |
| handoff | wave_release | one XCD | 32 | 256 | 8.4 | 8.4 | 17.33 | - | - | - | 11.40 | 4.24 | 0.97 | 0 | 0 |
| handoff | wave_release | one XCD | 64 | 256 | 16.8 | 16.8 | 18.00 | - | - | - | 11.44 | 4.48 | 1.86 | 0 | 0 |
| handoff | wave_release | one XCD | 128 | 256 | 33.6 | 33.6 | 18.64 | - | - | - | 11.52 | 4.60 | 3.60 | 0 | 0 |
| handoff | wt | spread | 16 | 128 | 2.1 | 2.1 | 4.56 | 3.96 | CU | CU | 1.64 | 1.92 | 0.92 | 0 | 0 |
| handoff | wt | spread | 32 | 128 | 4.2 | 4.2 | 4.66 | 3.96 | CU | CU | 1.72 | 1.92 | 1.80 | 0 | 0 |
| handoff | wt | spread | 64 | 128 | 8.4 | 8.4 | 4.94 | 3.96 | CU | CU | 1.92 | 2.00 | 3.40 | 0 | 0 |
| handoff | wt | spread | 128 | 128 | 16.8 | 16.8 | 5.96 | 5.10 | chip | chip | 2.40 | 2.56 | 5.63 | 0 | 0 |
| handoff | wt | spread | 16 | 256 | 4.2 | 4.2 | 7.63 | 7.32 | CU | CU | 3.00 | 3.64 | 1.10 | 0 | 0 |
| handoff | wt | spread | 32 | 256 | 8.4 | 8.4 | 7.86 | 7.32 | CU | CU | 3.16 | 3.64 | 2.13 | 0 | 0 |
| handoff | wt | spread | 64 | 256 | 16.8 | 16.8 | 8.46 | 7.32 | CU | CU | 3.52 | 3.80 | 3.97 | 0 | 0 |
| handoff | wt | spread | 128 | 256 | 33.6 | 33.6 | 10.58 | 9.61 | chip | chip | 4.84 | 4.64 | 6.34 | 0 | 0 |
| reduce-scatter | wt | one XCD | 16 | 128 | 2.1 | 2.1 | 5.56 | 4.03 | XCD link | CU | 2.16 | 2.10 | 0.75 | 0 | 0 |
| reduce-scatter | wt | one XCD | 32 | 128 | 4.2 | 4.2 | 5.84 | 4.69 | XCD link | CU | 2.32 | 2.10 | 1.44 | 0 | 0 |
| reduce-scatter | wt | one XCD | 64 | 128 | 8.4 | 8.4 | 6.59 | 4.69 | XCD link | CU | 2.36 | 2.32 | 2.55 | 0 | 0 |
| reduce-scatter | wt | one XCD | 128 | 128 | 16.8 | 16.8 | 6.80 | 5.10 | chip | chip | 2.40 | 2.64 | 4.93 | 0 | 0 |
| reduce-scatter | wt | one XCD | 256 | 128 | 33.6 | 33.6 | 11.09 | 9.61 | chip | chip | 4.40 | 4.32 | 6.05 | 0 | 0 |
| reduce-scatter | wt | one XCD | 16 | 256 | 4.2 | 4.2 | 9.28 | 7.47 | XCD link | CU | 3.88 | 4.24 | 0.90 | 0 | 0 |
| reduce-scatter | wt | one XCD | 32 | 256 | 8.4 | 8.4 | 9.87 | 8.78 | XCD link | CU | 4.28 | 4.36 | 1.70 | 0 | 0 |
| reduce-scatter | wt | one XCD | 64 | 256 | 16.8 | 16.8 | 10.91 | 8.78 | XCD link | CU | 4.36 | 5.12 | 3.08 | 0 | 0 |
| reduce-scatter | wt | one XCD | 128 | 256 | 33.6 | 33.6 | 12.12 | 9.61 | chip | chip | 4.80 | 5.60 | 5.54 | 0 | 0 |
| reduce-scatter | wt | one XCD | 256 | 256 | 67.1 | 67.1 | 21.91 | 18.62 | chip | chip | 8.82 | 9.00 | 6.13 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 16 | 128 | 2.1 | 2.1 | 8.78 | - | - | - | 1.56 | 2.32 | 0.48 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 32 | 128 | 4.2 | 4.2 | 9.16 | - | - | - | 1.56 | 2.28 | 0.92 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 64 | 128 | 8.4 | 8.4 | 9.63 | - | - | - | 1.60 | 2.54 | 1.74 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 128 | 128 | 16.8 | 16.8 | 9.93 | - | - | - | 1.60 | 2.72 | 3.38 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 256 | 128 | 33.6 | 33.6 | 17.14 | - | - | - | 3.12 | 5.04 | 3.91 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 16 | 256 | 4.2 | 4.2 | 12.12 | - | - | - | 2.96 | 4.12 | 0.69 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 32 | 256 | 8.4 | 8.4 | 12.84 | - | - | - | 3.18 | 4.12 | 1.31 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 64 | 256 | 16.8 | 16.8 | 13.55 | - | - | - | 3.26 | 4.80 | 2.48 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 128 | 256 | 33.6 | 33.6 | 14.26 | - | - | - | 3.32 | 5.20 | 4.71 | 0 | 0 |
| reduce-scatter | acq_rel | one XCD | 256 | 256 | 67.1 | 67.1 | 28.22 | - | - | - | 9.00 | 9.48 | 4.76 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 16 | 128 | 2.1 | 2.1 | 6.06 | - | - | - | 1.56 | 3.20 | 0.69 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 32 | 128 | 4.2 | 4.2 | 6.14 | - | - | - | 1.60 | 3.20 | 1.37 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 64 | 128 | 8.4 | 8.4 | 6.46 | - | - | - | 1.60 | 3.36 | 2.60 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 128 | 128 | 16.8 | 16.8 | 6.97 | - | - | - | 1.60 | 3.80 | 4.81 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 256 | 128 | 33.6 | 33.6 | 11.70 | - | - | - | 2.88 | 6.08 | 5.74 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 16 | 256 | 4.2 | 4.2 | 10.26 | - | - | - | 2.96 | 4.44 | 0.82 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 32 | 256 | 8.4 | 8.4 | 10.81 | - | - | - | 3.16 | 4.42 | 1.55 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 64 | 256 | 16.8 | 16.8 | 11.40 | - | - | - | 3.18 | 5.06 | 2.94 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 128 | 256 | 33.6 | 33.6 | 13.43 | - | - | - | 3.24 | 6.08 | 5.00 | 0 | 0 |
| reduce-scatter | l2 | one XCD | 256 | 256 | 67.1 | 67.1 | 24.93 | - | - | - | 7.44 | 10.24 | 5.38 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 16 | 128 | 2.1 | 2.1 | 12.28 | - | - | - | 7.76 | 2.36 | 0.34 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 32 | 128 | 4.2 | 4.2 | 14.04 | - | - | - | 8.68 | 2.36 | 0.60 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 64 | 128 | 8.4 | 8.4 | 14.16 | - | - | - | 8.72 | 2.44 | 1.18 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 128 | 128 | 16.8 | 16.8 | 14.95 | - | - | - | 8.72 | 2.60 | 2.24 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 256 | 128 | 33.6 | 33.6 | 26.89 | - | - | - | 16.72 | 4.80 | 2.50 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 16 | 256 | 4.2 | 4.2 | 17.03 | - | - | - | 10.80 | 4.20 | 0.49 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 32 | 256 | 8.4 | 8.4 | 18.02 | - | - | - | 11.32 | 4.20 | 0.93 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 64 | 256 | 16.8 | 16.8 | 18.74 | - | - | - | 11.32 | 4.68 | 1.79 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 128 | 256 | 33.6 | 33.6 | 20.09 | - | - | - | 11.32 | 5.40 | 3.34 | 0 | 0 |
| reduce-scatter | wave_release | one XCD | 256 | 256 | 67.1 | 67.1 | 37.28 | - | - | - | 21.68 | 9.52 | 3.60 | 0 | 0 |
| reduce-scatter | wt | spread | 16 | 128 | 2.1 | 2.1 | 5.22 | 3.96 | CU | CU | 1.64 | 1.92 | 0.80 | 0 | 0 |
| reduce-scatter | wt | spread | 32 | 128 | 4.2 | 4.2 | 5.30 | 3.96 | CU | CU | 1.72 | 1.92 | 1.58 | 0 | 0 |
| reduce-scatter | wt | spread | 64 | 128 | 8.4 | 8.4 | 5.65 | 3.96 | CU | CU | 1.92 | 1.98 | 2.97 | 0 | 0 |
| reduce-scatter | wt | spread | 128 | 128 | 16.8 | 16.8 | 6.95 | 5.10 | chip | chip | 2.40 | 2.52 | 4.83 | 0 | 0 |
| reduce-scatter | wt | spread | 256 | 128 | 33.6 | 33.6 | 11.89 | 9.61 | chip | chip | 4.60 | 4.56 | 5.64 | 0 | 0 |
| reduce-scatter | wt | spread | 16 | 256 | 4.2 | 4.2 | 8.04 | 7.32 | CU | CU | 3.00 | 3.36 | 1.04 | 0 | 0 |
| reduce-scatter | wt | spread | 32 | 256 | 8.4 | 8.4 | 8.27 | 7.32 | CU | CU | 3.12 | 3.40 | 2.03 | 0 | 0 |
| reduce-scatter | wt | spread | 64 | 256 | 16.8 | 16.8 | 9.15 | 7.32 | CU | CU | 3.52 | 3.68 | 3.67 | 0 | 0 |
| reduce-scatter | wt | spread | 128 | 256 | 33.6 | 33.6 | 12.18 | 9.61 | chip | chip | 4.84 | 5.04 | 5.51 | 0 | 0 |
| reduce-scatter | wt | spread | 256 | 256 | 67.1 | 67.1 | 22.44 | 18.62 | chip | chip | 9.28 | 9.44 | 5.98 | 0 | 0 |
| owner | wt | one XCD | 16 | 128 | 2.0 | 2.0 | 37.17 | 29.45 | CU | CU | 2.12 | 34.48 | 0.11 | 0 | 0 |
| owner | wt | one XCD | 32 | 128 | 3.9 | 3.9 | 37.19 | 30.03 | XCD link | CU | 2.24 | 33.94 | 0.21 | 0 | 0 |
| owner | wt | one XCD | 64 | 128 | 7.9 | 7.9 | 37.50 | 30.03 | XCD link | CU | 2.28 | 33.96 | 0.42 | 0 | 0 |
| owner | wt | one XCD | 128 | 128 | 15.7 | 15.7 | 37.80 | 30.03 | chip | CU | 2.28 | 34.04 | 0.83 | 0 | 0 |
| owner | wt | one XCD | 256 | 128 | 31.5 | 31.5 | 41.07 | 32.16 | chip | CU | 4.04 | 34.56 | 1.53 | 0 | 0 |
| owner | wt | one XCD | 16 | 256 | 3.9 | 3.9 | 64.80 | 58.30 | CU | CU | 3.84 | 60.52 | 0.12 | 0 | 0 |
| owner | wt | one XCD | 32 | 256 | 7.9 | 7.9 | 65.09 | 59.46 | XCD link | CU | 4.04 | 59.38 | 0.24 | 0 | 0 |
| owner | wt | one XCD | 64 | 256 | 15.7 | 15.7 | 65.42 | 59.46 | XCD link | CU | 4.12 | 59.08 | 0.48 | 0 | 0 |
| owner | wt | one XCD | 128 | 256 | 31.5 | 31.5 | 66.13 | 59.46 | chip | CU | 4.56 | 59.34 | 0.95 | 0 | 0 |
| owner | wt | one XCD | 256 | 256 | 62.9 | 62.9 | 71.35 | 63.72 | chip | CU | 8.24 | 59.96 | 1.76 | 0 | 0 |
| owner | acq_rel | one XCD | 16 | 128 | 2.0 | 2.0 | 40.71 | - | - | - | 1.52 | 35.98 | 0.10 | 0 | 0 |
| owner | acq_rel | one XCD | 32 | 128 | 3.9 | 3.9 | 40.82 | - | - | - | 1.56 | 35.46 | 0.19 | 0 | 0 |
| owner | acq_rel | one XCD | 64 | 128 | 7.9 | 7.9 | 40.71 | - | - | - | 1.52 | 35.16 | 0.39 | 0 | 0 |
| owner | acq_rel | one XCD | 128 | 128 | 15.7 | 15.7 | 41.14 | - | - | - | 1.56 | 35.32 | 0.76 | 0 | 0 |
| owner | acq_rel | one XCD | 256 | 128 | 31.5 | 31.5 | 47.19 | - | - | - | 2.84 | 36.20 | 1.33 | 0 | 0 |
| owner | acq_rel | one XCD | 16 | 256 | 3.9 | 3.9 | 68.64 | - | - | - | 2.92 | 61.96 | 0.11 | 0 | 0 |
| owner | acq_rel | one XCD | 32 | 256 | 7.9 | 7.9 | 68.60 | - | - | - | 2.96 | 60.84 | 0.23 | 0 | 0 |
| owner | acq_rel | one XCD | 64 | 256 | 15.7 | 15.7 | 69.07 | - | - | - | 3.00 | 60.62 | 0.46 | 0 | 0 |
| owner | acq_rel | one XCD | 128 | 256 | 31.5 | 31.5 | 69.21 | - | - | - | 3.00 | 60.70 | 0.91 | 0 | 0 |
| owner | acq_rel | one XCD | 256 | 256 | 62.9 | 62.9 | 77.85 | - | - | - | 9.48 | 61.96 | 1.62 | 0 | 0 |
| owner | l2 | one XCD | 16 | 128 | 2.0 | 2.0 | 54.82 | - | - | - | 1.54 | 52.64 | 0.07 | 0 | 0 |
| owner | l2 | one XCD | 32 | 128 | 3.9 | 3.9 | 54.53 | - | - | - | 1.56 | 52.04 | 0.14 | 0 | 0 |
| owner | l2 | one XCD | 64 | 128 | 7.9 | 7.9 | 55.71 | - | - | - | 1.56 | 52.28 | 0.28 | 0 | 0 |
| owner | l2 | one XCD | 128 | 128 | 15.7 | 15.7 | 57.24 | - | - | - | 1.56 | 52.38 | 0.55 | 0 | 0 |
| owner | l2 | one XCD | 256 | 128 | 31.5 | 31.5 | 52.81 | - | - | - | 2.60 | 45.84 | 1.19 | 0 | 0 |
| owner | l2 | one XCD | 16 | 256 | 3.9 | 3.9 | 82.96 | - | - | - | 2.92 | 79.54 | 0.09 | 0 | 0 |
| owner | l2 | one XCD | 32 | 256 | 7.9 | 7.9 | 83.48 | - | - | - | 2.96 | 78.58 | 0.19 | 0 | 0 |
| owner | l2 | one XCD | 64 | 256 | 15.7 | 15.7 | 84.66 | - | - | - | 2.96 | 79.02 | 0.37 | 0 | 0 |
| owner | l2 | one XCD | 128 | 256 | 31.5 | 31.5 | 86.53 | - | - | - | 3.00 | 79.62 | 0.73 | 0 | 0 |
| owner | l2 | one XCD | 256 | 256 | 62.9 | 62.9 | 82.59 | - | - | - | 6.68 | 71.50 | 1.52 | 0 | 0 |
| owner | wave_release | one XCD | 16 | 128 | 2.0 | 2.0 | 46.41 | - | - | - | 8.84 | 36.08 | 0.08 | 0 | 0 |
| owner | wave_release | one XCD | 32 | 128 | 3.9 | 3.9 | 46.30 | - | - | - | 9.44 | 36.40 | 0.17 | 0 | 0 |
| owner | wave_release | one XCD | 64 | 128 | 7.9 | 7.9 | 46.92 | - | - | - | 9.68 | 35.32 | 0.34 | 0 | 0 |
| owner | wave_release | one XCD | 128 | 128 | 15.7 | 15.7 | 47.80 | - | - | - | 9.80 | 35.92 | 0.66 | 0 | 0 |
| owner | wave_release | one XCD | 256 | 128 | 31.5 | 31.5 | 61.31 | - | - | - | 15.84 | 37.12 | 1.03 | 0 | 0 |
| owner | wave_release | one XCD | 16 | 256 | 3.9 | 3.9 | 76.41 | - | - | - | 13.68 | 63.92 | 0.10 | 0 | 0 |
| owner | wave_release | one XCD | 32 | 256 | 7.9 | 7.9 | 76.41 | - | - | - | 13.32 | 60.78 | 0.21 | 0 | 0 |
| owner | wave_release | one XCD | 64 | 256 | 15.7 | 15.7 | 76.59 | - | - | - | 13.52 | 62.26 | 0.41 | 0 | 0 |
| owner | wave_release | one XCD | 128 | 256 | 31.5 | 31.5 | 77.03 | - | - | - | 13.48 | 61.66 | 0.82 | 0 | 0 |
| owner | wave_release | one XCD | 256 | 256 | 62.9 | 62.9 | 92.50 | - | - | - | 20.52 | 63.10 | 1.36 | 0 | 0 |
| owner | wt | spread | 16 | 128 | 2.0 | 2.0 | 36.65 | 29.45 | CU | CU | 1.64 | 34.46 | 0.11 | 0 | 0 |
| owner | wt | spread | 32 | 128 | 3.9 | 3.9 | 36.61 | 29.45 | CU | CU | 1.72 | 33.96 | 0.21 | 0 | 0 |
| owner | wt | spread | 64 | 128 | 7.9 | 7.9 | 37.31 | 29.45 | CU | CU | 1.88 | 34.04 | 0.42 | 0 | 0 |
| owner | wt | spread | 128 | 128 | 15.7 | 15.7 | 37.98 | 30.03 | chip | CU | 2.28 | 34.12 | 0.83 | 0 | 0 |
| owner | wt | spread | 256 | 128 | 31.5 | 31.5 | 40.40 | 32.16 | chip | CU | 4.20 | 34.04 | 1.56 | 0 | 0 |
| owner | wt | spread | 16 | 256 | 3.9 | 3.9 | 64.04 | 58.30 | CU | CU | 3.00 | 60.60 | 0.12 | 0 | 0 |
| owner | wt | spread | 32 | 256 | 7.9 | 7.9 | 64.25 | 58.30 | CU | CU | 3.12 | 59.66 | 0.24 | 0 | 0 |
| owner | wt | spread | 64 | 256 | 15.7 | 15.7 | 65.01 | 58.30 | CU | CU | 3.48 | 59.44 | 0.48 | 0 | 0 |
| owner | wt | spread | 128 | 256 | 31.5 | 31.5 | 66.31 | 59.46 | chip | CU | 4.56 | 59.46 | 0.95 | 0 | 0 |
| owner | wt | spread | 256 | 256 | 62.9 | 62.9 | 71.26 | 63.72 | chip | CU | 8.40 | 59.64 | 1.77 | 0 | 0 |
| last arriver | wt | one XCD | 16 | 128 | 2.1 | 2.0 | 31.66 | 29.52 | XCD link | CU | 2.16 | 28.96 | 0.13 | 0 | 0 |
| last arriver | wt | one XCD | 32 | 128 | 4.2 | 3.9 | 31.70 | 30.17 | XCD link | CU | 2.32 | 28.12 | 0.26 | 0 | 0 |
| last arriver | wt | one XCD | 64 | 128 | 8.4 | 7.9 | 31.88 | 30.17 | XCD link | CU | 2.40 | 28.16 | 0.51 | 0 | 0 |
| last arriver | wt | one XCD | 128 | 128 | 16.8 | 15.7 | 32.23 | 30.17 | chip | CU | 2.40 | 28.30 | 1.01 | 0 | 0 |
| last arriver | wt | one XCD | 256 | 128 | 33.6 | 31.5 | 33.92 | 32.44 | chip | CU | 4.44 | 28.26 | 1.92 | 0 | 0 |
| last arriver | wt | one XCD | 16 | 256 | 4.2 | 3.9 | 59.42 | 58.44 | XCD link | CU | 3.92 | 54.96 | 0.14 | 0 | 0 |
| last arriver | wt | one XCD | 32 | 256 | 8.4 | 7.9 | 59.70 | 59.75 | XCD link | CU | 4.28 | 53.74 | 0.27 | 0 | 0 |
| last arriver | wt | one XCD | 64 | 256 | 16.8 | 15.7 | 59.90 | 59.75 | XCD link | CU | 4.36 | 53.60 | 0.54 | 0 | 0 |
| last arriver | wt | one XCD | 128 | 256 | 33.6 | 31.5 | 60.98 | 59.75 | chip | CU | 4.84 | 53.76 | 1.07 | 0 | 0 |
| last arriver | wt | one XCD | 256 | 256 | 67.1 | 62.9 | 64.41 | 64.28 | chip | CU | 8.76 | 53.72 | 2.02 | 0 | 0 |
| last arriver | acq_rel | one XCD | 16 | 128 | 2.1 | 2.0 | 34.14 | - | - | - | 1.56 | 29.42 | 0.12 | 0 | 0 |
| last arriver | acq_rel | one XCD | 32 | 128 | 4.2 | 3.9 | 34.14 | - | - | - | 1.60 | 28.54 | 0.24 | 0 | 0 |
| last arriver | acq_rel | one XCD | 64 | 128 | 8.4 | 7.9 | 34.28 | - | - | - | 1.60 | 28.44 | 0.47 | 0 | 0 |
| last arriver | acq_rel | one XCD | 128 | 128 | 16.8 | 15.7 | 34.69 | - | - | - | 1.60 | 28.66 | 0.94 | 0 | 0 |
| last arriver | acq_rel | one XCD | 256 | 128 | 33.6 | 31.5 | 38.93 | - | - | - | 3.12 | 29.04 | 1.67 | 0 | 0 |
| last arriver | acq_rel | one XCD | 16 | 256 | 4.2 | 3.9 | 61.66 | - | - | - | 3.00 | 55.02 | 0.13 | 0 | 0 |
| last arriver | acq_rel | one XCD | 32 | 256 | 8.4 | 7.9 | 61.77 | - | - | - | 3.24 | 53.62 | 0.26 | 0 | 0 |
| last arriver | acq_rel | one XCD | 64 | 256 | 16.8 | 15.7 | 61.88 | - | - | - | 3.24 | 53.66 | 0.53 | 0 | 0 |
| last arriver | acq_rel | one XCD | 128 | 256 | 33.6 | 31.5 | 62.42 | - | - | - | 3.32 | 53.84 | 1.04 | 0 | 0 |
| last arriver | acq_rel | one XCD | 256 | 256 | 67.1 | 62.9 | 70.24 | - | - | - | 9.04 | 54.02 | 1.85 | 0 | 0 |
| last arriver | l2 | one XCD | 16 | 128 | 2.1 | 2.0 | 48.94 | - | - | - | 1.56 | 46.74 | 0.08 | 0 | 0 |
| last arriver | l2 | one XCD | 32 | 128 | 4.2 | 3.9 | 49.02 | - | - | - | 1.60 | 46.12 | 0.17 | 0 | 0 |
| last arriver | l2 | one XCD | 64 | 128 | 8.4 | 7.9 | 49.59 | - | - | - | 1.60 | 46.44 | 0.33 | 0 | 0 |
| last arriver | l2 | one XCD | 128 | 128 | 16.8 | 15.7 | 51.16 | - | - | - | 1.60 | 46.86 | 0.64 | 0 | 0 |
| last arriver | l2 | one XCD | 256 | 128 | 33.6 | 31.5 | 44.44 | - | - | - | 2.88 | 37.86 | 1.46 | 0 | 0 |
| last arriver | l2 | one XCD | 16 | 256 | 4.2 | 3.9 | 75.92 | - | - | - | 3.00 | 72.40 | 0.11 | 0 | 0 |
| last arriver | l2 | one XCD | 32 | 256 | 8.4 | 7.9 | 75.93 | - | - | - | 3.20 | 70.74 | 0.21 | 0 | 0 |
| last arriver | l2 | one XCD | 64 | 256 | 16.8 | 15.7 | 76.86 | - | - | - | 3.22 | 71.32 | 0.42 | 0 | 0 |
| last arriver | l2 | one XCD | 128 | 256 | 33.6 | 31.5 | 78.29 | - | - | - | 3.28 | 71.92 | 0.83 | 0 | 0 |
| last arriver | l2 | one XCD | 256 | 256 | 67.1 | 62.9 | 74.91 | - | - | - | 7.52 | 63.68 | 1.74 | 0 | 0 |
| last arriver | wave_release | one XCD | 16 | 128 | 2.1 | 2.0 | 37.52 | - | - | - | 7.84 | 28.74 | 0.11 | 0 | 0 |
| last arriver | wave_release | one XCD | 32 | 128 | 4.2 | 3.9 | 37.76 | - | - | - | 8.64 | 28.00 | 0.22 | 0 | 0 |
| last arriver | wave_release | one XCD | 64 | 128 | 8.4 | 7.9 | 37.92 | - | - | - | 8.64 | 28.04 | 0.43 | 0 | 0 |
| last arriver | wave_release | one XCD | 128 | 128 | 16.8 | 15.7 | 39.41 | - | - | - | 8.76 | 28.02 | 0.82 | 0 | 0 |
| last arriver | wave_release | one XCD | 256 | 128 | 33.6 | 31.5 | 47.92 | - | - | - | 16.60 | 28.12 | 1.36 | 0 | 0 |
| last arriver | wave_release | one XCD | 16 | 256 | 4.2 | 3.9 | 66.53 | - | - | - | 10.88 | 54.58 | 0.12 | 0 | 0 |
| last arriver | wave_release | one XCD | 32 | 256 | 8.4 | 7.9 | 66.69 | - | - | - | 11.28 | 53.20 | 0.24 | 0 | 0 |
| last arriver | wave_release | one XCD | 64 | 256 | 16.8 | 15.7 | 67.04 | - | - | - | 11.40 | 53.30 | 0.48 | 0 | 0 |
| last arriver | wave_release | one XCD | 128 | 256 | 33.6 | 31.5 | 67.60 | - | - | - | 11.40 | 53.36 | 0.96 | 0 | 0 |
| last arriver | wave_release | one XCD | 256 | 256 | 67.1 | 62.9 | 79.38 | - | - | - | 21.84 | 53.56 | 1.64 | 0 | 0 |
| last arriver | wt | spread | 16 | 128 | 2.1 | 2.0 | 31.31 | 29.45 | CU | CU | 1.64 | 28.90 | 0.13 | 0 | 0 |
| last arriver | wt | spread | 32 | 128 | 4.2 | 3.9 | 31.31 | 29.45 | CU | CU | 1.72 | 28.76 | 0.26 | 0 | 0 |
| last arriver | wt | spread | 64 | 128 | 8.4 | 7.9 | 31.63 | 29.45 | CU | CU | 1.92 | 28.84 | 0.51 | 0 | 0 |
| last arriver | wt | spread | 128 | 128 | 16.8 | 15.7 | 32.28 | 30.17 | chip | CU | 2.40 | 28.82 | 1.01 | 0 | 0 |
| last arriver | wt | spread | 256 | 128 | 33.6 | 31.5 | 35.78 | 32.44 | chip | CU | 4.56 | 29.60 | 1.82 | 0 | 0 |
| last arriver | wt | spread | 16 | 256 | 4.2 | 3.9 | 58.39 | 58.30 | CU | CU | 3.00 | 54.68 | 0.14 | 0 | 0 |
| last arriver | wt | spread | 32 | 256 | 8.4 | 7.9 | 58.76 | 58.30 | CU | CU | 3.12 | 54.62 | 0.28 | 0 | 0 |
| last arriver | wt | spread | 64 | 256 | 16.8 | 15.7 | 59.30 | 58.30 | CU | CU | 3.52 | 54.62 | 0.55 | 0 | 0 |
| last arriver | wt | spread | 128 | 256 | 33.6 | 31.5 | 60.74 | 59.75 | chip | CU | 4.88 | 54.56 | 1.07 | 0 | 0 |
| last arriver | wt | spread | 256 | 256 | 67.1 | 62.9 | 66.72 | 64.28 | chip | CU | 9.36 | 55.04 | 1.95 | 0 | 0 |

## split

Rerun separately (HIP_VISIBLE_DEVICES=1, AMD Instinct MI355X, triton 3.8.0, 2026-10-08 06:45; GPU 0 was busy with another job), with the C store, the atomic endings and bf16 partials added.

Split count, 256x256 tiles (256 KiB fp32 or 128 KiB bf16 partials). Two ways to vary n: 'tiles 16' keeps 16 tiles, so W = 16 n writers; 'CUs 256' keeps all 256 CUs, so 256 / n tiles (as in the GEMM tail, where every CU takes part). Placement 'one XCD' (as stream-K places a tile) and, for reference, 'spread'. The exchanges (reduce-scatter, owner, last arriver) use `wt` synchronisation and then store their share of a bf16 C tile (128 KiB per tile). The atomic endings add every partial into a zeroed tile instead: bf16 `pk_add` into C, fp32 into a workspace, or fp32 then a last arriver that reads the sum with `.cv` and stores bf16 C ('+ finalize'); member m adds m + 1 and every tile is checked. The phases are medians over groups and launches of each group's critical path, each phase taken separately: publish (until the group's last writer has published; for the atomic endings, its atomics have completed), sync (until the consumer that finishes last has passed its wait: flag, counter or arrival), read (its reads), C store (its bf16 C store). Zero: the median time of the separate launch that zeroes the atomic endings' tiles (CUDA events, before the cache flush), not part of the measured total. Partials: the exchanges' partial type; bf16 partials (128 KiB, 8 per lane, still `.wt` / `.cv` and 16 bytes per lane) are summed in fp32 and C is the same 128 KiB bf16 tile. Model phases from exchange.model_us (no C store, no zeroing).

| pattern | partials | setup | placement | n | W | measured us | publish us | sync us | read us | C store us | zero us | model publish | model sync | model read | model us | wrong | timeouts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reduce-scatter | fp32 | tiles 16 | one XCD | 2 | 32 | 9.07 | 3.37 | 0.56 | 4.16 | 0.64 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | one XCD | 4 | 64 | 8.99 | 3.74 | 0.60 | 3.84 | 0.44 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | one XCD | 8 | 128 | 11.46 | 5.02 | 0.64 | 5.08 | 0.32 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | one XCD | 16 | 256 | 22.20 | 9.09 | 1.44 | 9.36 | 0.24 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| owner | fp32 | tiles 16 | one XCD | 2 | 32 | 8.68 | 3.13 | 0.48 | 3.64 | 1.04 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| owner | fp32 | tiles 16 | one XCD | 4 | 64 | 17.45 | 3.58 | 0.44 | 11.88 | 1.08 | 0.00 | 3.08 | 0.60 | 10.92 | 14.61 | 0 | 0 |
| owner | fp32 | tiles 16 | one XCD | 8 | 128 | 35.53 | 4.41 | 0.48 | 28.34 | 1.08 | 0.00 | 3.97 | 0.60 | 25.49 | 30.05 | 0 | 0 |
| owner | fp32 | tiles 16 | one XCD | 16 | 256 | 72.92 | 8.63 | 0.40 | 60.58 | 1.08 | 0.00 | 8.50 | 0.60 | 54.61 | 63.72 | 0 | 0 |
| last arriver | fp32 | tiles 16 | one XCD | 2 | 32 | 8.89 | 3.34 | 0.40 | 3.80 | 1.04 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| last arriver | fp32 | tiles 16 | one XCD | 4 | 64 | 16.63 | 3.73 | 0.40 | 10.96 | 1.08 | 0.00 | 3.08 | 0.60 | 10.92 | 14.61 | 0 | 0 |
| last arriver | fp32 | tiles 16 | one XCD | 8 | 128 | 32.70 | 5.01 | 0.40 | 25.16 | 1.08 | 0.00 | 4.53 | 0.60 | 25.49 | 30.62 | 0 | 0 |
| last arriver | fp32 | tiles 16 | one XCD | 16 | 256 | 65.54 | 9.23 | 0.40 | 53.64 | 1.04 | 0.00 | 9.07 | 0.60 | 54.61 | 64.28 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | one XCD | 2 | 32 | 7.01 | 1.87 | 0.60 | 3.56 | 0.64 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | one XCD | 4 | 64 | 6.89 | 2.08 | 0.60 | 3.44 | 0.44 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | one XCD | 8 | 128 | 7.08 | 2.56 | 0.68 | 3.08 | 0.32 | 0.00 | 2.27 | 0.60 | 2.24 | 5.10 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | one XCD | 16 | 256 | 12.35 | 4.70 | 1.16 | 5.20 | 0.24 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| owner | bf16 | tiles 16 | one XCD | 2 | 32 | 6.36 | 1.76 | 0.48 | 2.60 | 1.04 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| owner | bf16 | tiles 16 | one XCD | 4 | 64 | 12.98 | 1.98 | 0.48 | 8.96 | 1.08 | 0.00 | 1.54 | 0.60 | 5.46 | 7.60 | 0 | 0 |
| owner | bf16 | tiles 16 | one XCD | 8 | 128 | 26.27 | 2.37 | 0.56 | 21.56 | 1.08 | 0.00 | 1.98 | 0.60 | 12.74 | 15.33 | 0 | 0 |
| owner | bf16 | tiles 16 | one XCD | 16 | 256 | 53.51 | 4.40 | 0.52 | 46.04 | 1.08 | 0.00 | 4.25 | 0.60 | 27.31 | 32.16 | 0 | 0 |
| last arriver | bf16 | tiles 16 | one XCD | 2 | 32 | 6.54 | 1.85 | 0.36 | 2.80 | 1.04 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| last arriver | bf16 | tiles 16 | one XCD | 4 | 64 | 12.09 | 2.06 | 0.36 | 8.16 | 1.04 | 0.00 | 1.54 | 0.60 | 5.46 | 7.60 | 0 | 0 |
| last arriver | bf16 | tiles 16 | one XCD | 8 | 128 | 23.46 | 2.56 | 0.36 | 18.84 | 1.04 | 0.00 | 2.27 | 0.60 | 12.74 | 15.61 | 0 | 0 |
| last arriver | bf16 | tiles 16 | one XCD | 16 | 256 | 46.34 | 4.67 | 0.40 | 39.76 | 1.04 | 0.00 | 4.53 | 0.60 | 27.31 | 32.44 | 0 | 0 |
| atomic bf16 | - | tiles 16 | one XCD | 2 | 32 | 4.45 | 4.25 | 0.00 | 0.04 | 0.04 | 6.44 | 4.85 | 0.00 | 0.00 | 4.85 | 0 | 0 |
| atomic bf16 | - | tiles 16 | one XCD | 4 | 64 | 7.12 | 6.91 | 0.00 | 0.04 | 0.04 | 6.48 | 6.36 | 0.00 | 0.00 | 6.36 | 0 | 0 |
| atomic bf16 | - | tiles 16 | one XCD | 8 | 128 | 13.27 | 13.03 | 0.00 | 0.04 | 0.04 | 6.44 | 12.71 | 0.00 | 0.00 | 12.71 | 0 | 0 |
| atomic bf16 | - | tiles 16 | one XCD | 16 | 256 | 27.17 | 26.03 | 0.00 | 0.04 | 0.04 | 6.46 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic fp32 | - | tiles 16 | one XCD | 2 | 32 | 7.79 | 7.59 | 0.00 | 0.04 | 0.04 | 6.72 | 7.28 | 0.00 | 0.00 | 7.28 | 0 | 0 |
| atomic fp32 | - | tiles 16 | one XCD | 4 | 64 | 13.15 | 12.94 | 0.00 | 0.04 | 0.04 | 6.70 | 12.43 | 0.00 | 0.00 | 12.43 | 0 | 0 |
| atomic fp32 | - | tiles 16 | one XCD | 8 | 128 | 25.77 | 25.23 | 0.00 | 0.04 | 0.04 | 6.76 | 24.86 | 0.00 | 0.00 | 24.86 | 0 | 0 |
| atomic fp32 | - | tiles 16 | one XCD | 16 | 256 | 51.98 | 49.86 | 0.00 | 0.04 | 0.04 | 6.76 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | one XCD | 2 | 32 | 13.04 | 7.59 | 0.40 | 3.76 | 1.04 | 7.32 | 7.28 | 0.60 | 3.64 | 11.52 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | one XCD | 4 | 64 | 18.33 | 12.93 | 0.40 | 3.72 | 1.04 | 7.32 | 12.43 | 0.60 | 3.64 | 16.67 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | one XCD | 8 | 128 | 30.94 | 25.27 | 0.40 | 3.68 | 1.04 | 7.32 | 24.86 | 0.60 | 3.64 | 29.10 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | one XCD | 16 | 256 | 56.95 | 49.91 | 0.52 | 3.80 | 1.04 | 7.36 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | one XCD | 2 | 256 | 20.46 | 9.04 | 0.68 | 8.00 | 1.04 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | one XCD | 4 | 256 | 20.32 | 9.13 | 0.68 | 8.08 | 0.52 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | one XCD | 8 | 256 | 20.50 | 9.18 | 0.84 | 8.32 | 0.32 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | one XCD | 16 | 256 | 22.29 | 9.20 | 1.40 | 9.40 | 0.24 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| owner | fp32 | CUs 256 | one XCD | 2 | 256 | 12.94 | 4.92 | 0.52 | 4.64 | 1.64 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| owner | fp32 | CUs 256 | one XCD | 4 | 256 | 21.75 | 6.99 | 0.44 | 12.52 | 1.20 | 0.00 | 6.80 | 0.60 | 10.92 | 18.32 | 0 | 0 |
| owner | fp32 | CUs 256 | one XCD | 8 | 256 | 38.99 | 8.10 | 0.44 | 28.52 | 1.08 | 0.00 | 7.94 | 0.60 | 25.49 | 34.02 | 0 | 0 |
| owner | fp32 | CUs 256 | one XCD | 16 | 256 | 72.78 | 8.62 | 0.40 | 60.60 | 1.08 | 0.00 | 8.50 | 0.60 | 54.61 | 63.72 | 0 | 0 |
| last arriver | fp32 | CUs 256 | one XCD | 2 | 256 | 16.88 | 9.22 | 0.44 | 4.72 | 1.52 | 0.00 | 9.07 | 0.60 | 4.47 | 14.14 | 0 | 0 |
| last arriver | fp32 | CUs 256 | one XCD | 4 | 256 | 23.05 | 9.11 | 0.40 | 11.60 | 1.24 | 0.00 | 9.07 | 0.60 | 10.92 | 20.59 | 0 | 0 |
| last arriver | fp32 | CUs 256 | one XCD | 8 | 256 | 36.55 | 9.05 | 0.40 | 25.46 | 1.12 | 0.00 | 9.07 | 0.60 | 25.49 | 35.15 | 0 | 0 |
| last arriver | fp32 | CUs 256 | one XCD | 16 | 256 | 65.52 | 9.26 | 0.40 | 53.54 | 1.04 | 0.00 | 9.07 | 0.60 | 54.61 | 64.28 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | one XCD | 2 | 256 | 11.35 | 4.60 | 0.68 | 4.36 | 1.20 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | one XCD | 4 | 256 | 11.09 | 4.64 | 0.68 | 4.56 | 0.56 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | one XCD | 8 | 256 | 11.42 | 4.68 | 0.84 | 4.68 | 0.32 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | one XCD | 16 | 256 | 12.20 | 4.70 | 1.16 | 5.24 | 0.24 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| owner | bf16 | CUs 256 | one XCD | 2 | 256 | 8.85 | 2.57 | 0.52 | 2.92 | 1.68 | 0.00 | 2.27 | 0.60 | 2.24 | 5.10 | 0 | 0 |
| owner | bf16 | CUs 256 | one XCD | 4 | 256 | 15.07 | 3.52 | 0.52 | 9.04 | 1.20 | 0.00 | 3.40 | 0.60 | 5.46 | 9.46 | 0 | 0 |
| owner | bf16 | CUs 256 | one XCD | 8 | 256 | 27.88 | 4.10 | 0.48 | 21.24 | 1.08 | 0.00 | 3.97 | 0.60 | 12.74 | 17.31 | 0 | 0 |
| owner | bf16 | CUs 256 | one XCD | 16 | 256 | 53.38 | 4.41 | 0.44 | 46.12 | 1.08 | 0.00 | 4.25 | 0.60 | 27.31 | 32.16 | 0 | 0 |
| last arriver | bf16 | CUs 256 | one XCD | 2 | 256 | 10.16 | 4.56 | 0.44 | 2.96 | 1.52 | 0.00 | 4.53 | 0.60 | 2.24 | 7.37 | 0 | 0 |
| last arriver | bf16 | CUs 256 | one XCD | 4 | 256 | 14.94 | 4.62 | 0.36 | 8.24 | 1.24 | 0.00 | 4.53 | 0.60 | 5.46 | 10.60 | 0 | 0 |
| last arriver | bf16 | CUs 256 | one XCD | 8 | 256 | 25.29 | 4.63 | 0.40 | 18.76 | 1.12 | 0.00 | 4.53 | 0.60 | 12.74 | 17.88 | 0 | 0 |
| last arriver | bf16 | CUs 256 | one XCD | 16 | 256 | 46.34 | 4.64 | 0.40 | 39.76 | 1.04 | 0.00 | 4.53 | 0.60 | 27.31 | 32.44 | 0 | 0 |
| atomic bf16 | - | CUs 256 | one XCD | 2 | 256 | 25.48 | 24.73 | 0.00 | 0.04 | 0.04 | 7.96 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | one XCD | 4 | 256 | 25.37 | 24.39 | 0.00 | 0.04 | 0.04 | 7.08 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | one XCD | 8 | 256 | 27.48 | 26.30 | 0.00 | 0.04 | 0.04 | 6.64 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | one XCD | 16 | 256 | 27.18 | 25.85 | 0.00 | 0.04 | 0.04 | 6.44 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic fp32 | - | CUs 256 | one XCD | 2 | 256 | 50.10 | 47.41 | 0.00 | 0.04 | 0.04 | 10.02 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | one XCD | 4 | 256 | 49.98 | 47.69 | 0.00 | 0.04 | 0.04 | 7.96 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | one XCD | 8 | 256 | 49.97 | 47.53 | 0.00 | 0.04 | 0.04 | 8.34 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | one XCD | 16 | 256 | 51.66 | 49.81 | 0.00 | 0.04 | 0.04 | 6.72 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | one XCD | 2 | 256 | 56.65 | 47.30 | 0.72 | 4.68 | 1.52 | 10.00 | 49.71 | 0.60 | 4.47 | 54.78 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | one XCD | 4 | 256 | 55.48 | 47.55 | 0.52 | 4.08 | 1.28 | 7.96 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | one XCD | 8 | 256 | 55.10 | 47.37 | 1.08 | 3.80 | 1.12 | 7.04 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | one XCD | 16 | 256 | 57.05 | 50.19 | 0.56 | 3.80 | 1.04 | 7.28 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | spread | 2 | 32 | 9.12 | 3.33 | 0.67 | 4.24 | 0.64 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | spread | 4 | 64 | 8.82 | 3.77 | 0.72 | 3.68 | 0.44 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | spread | 8 | 128 | 11.93 | 5.22 | 0.95 | 5.24 | 0.32 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | fp32 | tiles 16 | spread | 16 | 256 | 23.14 | 10.27 | 1.07 | 11.24 | 0.24 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| owner | fp32 | tiles 16 | spread | 2 | 32 | 8.71 | 3.19 | 0.59 | 3.60 | 1.08 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| owner | fp32 | tiles 16 | spread | 4 | 64 | 17.73 | 3.63 | 0.51 | 11.88 | 1.08 | 0.00 | 3.08 | 0.60 | 10.92 | 14.61 | 0 | 0 |
| owner | fp32 | tiles 16 | spread | 8 | 128 | 35.88 | 4.69 | 0.46 | 28.16 | 1.08 | 0.00 | 3.97 | 0.60 | 25.49 | 30.05 | 0 | 0 |
| owner | fp32 | tiles 16 | spread | 16 | 256 | 74.88 | 9.52 | 0.03 | 61.70 | 1.08 | 0.00 | 8.50 | 0.60 | 54.61 | 63.72 | 0 | 0 |
| last arriver | fp32 | tiles 16 | spread | 2 | 32 | 8.79 | 3.32 | 0.40 | 3.80 | 1.08 | 0.00 | 3.08 | 0.60 | 3.64 | 7.32 | 0 | 0 |
| last arriver | fp32 | tiles 16 | spread | 4 | 64 | 16.73 | 3.75 | 0.40 | 11.16 | 1.12 | 0.00 | 3.08 | 0.60 | 10.92 | 14.61 | 0 | 0 |
| last arriver | fp32 | tiles 16 | spread | 8 | 128 | 33.30 | 5.20 | 0.41 | 26.14 | 1.24 | 0.00 | 4.53 | 0.60 | 25.49 | 30.62 | 0 | 0 |
| last arriver | fp32 | tiles 16 | spread | 16 | 256 | 68.02 | 10.24 | 0.40 | 55.12 | 1.26 | 0.00 | 9.07 | 0.60 | 54.61 | 64.28 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | spread | 2 | 32 | 7.04 | 1.89 | 0.66 | 3.56 | 0.64 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | spread | 4 | 64 | 6.86 | 2.11 | 0.72 | 3.44 | 0.44 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | spread | 8 | 128 | 7.05 | 2.62 | 0.90 | 2.96 | 0.32 | 0.00 | 2.27 | 0.60 | 2.24 | 5.10 | 0 | 0 |
| reduce-scatter | bf16 | tiles 16 | spread | 16 | 256 | 12.92 | 5.16 | 1.09 | 6.24 | 0.28 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| owner | bf16 | tiles 16 | spread | 2 | 32 | 6.38 | 1.81 | 0.59 | 2.60 | 1.08 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| owner | bf16 | tiles 16 | spread | 4 | 64 | 13.11 | 2.04 | 0.54 | 9.00 | 1.08 | 0.00 | 1.54 | 0.60 | 5.46 | 7.60 | 0 | 0 |
| owner | bf16 | tiles 16 | spread | 8 | 128 | 26.19 | 2.41 | 0.53 | 21.40 | 1.08 | 0.00 | 1.98 | 0.60 | 12.74 | 15.33 | 0 | 0 |
| owner | bf16 | tiles 16 | spread | 16 | 256 | 54.25 | 4.70 | 0.40 | 46.88 | 1.08 | 0.00 | 4.25 | 0.60 | 27.31 | 32.16 | 0 | 0 |
| last arriver | bf16 | tiles 16 | spread | 2 | 32 | 6.43 | 1.89 | 0.36 | 2.80 | 1.08 | 0.00 | 1.54 | 0.60 | 1.82 | 3.96 | 0 | 0 |
| last arriver | bf16 | tiles 16 | spread | 4 | 64 | 12.11 | 2.10 | 0.36 | 8.16 | 1.16 | 0.00 | 1.54 | 0.60 | 5.46 | 7.60 | 0 | 0 |
| last arriver | bf16 | tiles 16 | spread | 8 | 128 | 23.62 | 2.62 | 0.40 | 18.84 | 1.40 | 0.00 | 2.27 | 0.60 | 12.74 | 15.61 | 0 | 0 |
| last arriver | bf16 | tiles 16 | spread | 16 | 256 | 47.48 | 5.14 | 0.40 | 39.92 | 1.40 | 0.00 | 4.53 | 0.60 | 27.31 | 32.44 | 0 | 0 |
| atomic bf16 | - | tiles 16 | spread | 2 | 32 | 4.59 | 4.41 | 0.00 | 0.04 | 0.04 | 6.48 | 4.85 | 0.00 | 0.00 | 4.85 | 0 | 0 |
| atomic bf16 | - | tiles 16 | spread | 4 | 64 | 7.03 | 6.90 | 0.00 | 0.04 | 0.04 | 6.52 | 6.36 | 0.00 | 0.00 | 6.36 | 0 | 0 |
| atomic bf16 | - | tiles 16 | spread | 8 | 128 | 13.16 | 13.02 | 0.00 | 0.04 | 0.04 | 6.46 | 12.71 | 0.00 | 0.00 | 12.71 | 0 | 0 |
| atomic bf16 | - | tiles 16 | spread | 16 | 256 | 26.16 | 26.02 | 0.00 | 0.04 | 0.04 | 6.44 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic fp32 | - | tiles 16 | spread | 2 | 32 | 8.14 | 7.95 | 0.00 | 0.04 | 0.04 | 6.72 | 7.28 | 0.00 | 0.00 | 7.28 | 0 | 0 |
| atomic fp32 | - | tiles 16 | spread | 4 | 64 | 13.04 | 12.89 | 0.00 | 0.04 | 0.04 | 6.72 | 12.43 | 0.00 | 0.00 | 12.43 | 0 | 0 |
| atomic fp32 | - | tiles 16 | spread | 8 | 128 | 25.29 | 25.15 | 0.00 | 0.04 | 0.04 | 6.76 | 24.86 | 0.00 | 0.00 | 24.86 | 0 | 0 |
| atomic fp32 | - | tiles 16 | spread | 16 | 256 | 49.88 | 49.74 | 0.00 | 0.04 | 0.04 | 6.72 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | spread | 2 | 32 | 12.94 | 7.54 | 0.36 | 3.72 | 1.08 | 7.36 | 7.28 | 0.60 | 3.64 | 11.52 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | spread | 4 | 64 | 18.32 | 12.89 | 0.36 | 3.72 | 1.12 | 7.40 | 12.43 | 0.60 | 3.64 | 16.67 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | spread | 8 | 128 | 30.79 | 25.14 | 0.38 | 3.68 | 1.28 | 7.64 | 24.86 | 0.60 | 3.64 | 29.10 | 0 | 0 |
| atomic fp32 + finalize | - | tiles 16 | spread | 16 | 256 | 55.44 | 49.73 | 0.38 | 3.76 | 1.36 | 7.78 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | spread | 2 | 256 | 21.01 | 9.91 | 0.72 | 8.64 | 1.00 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | spread | 4 | 256 | 21.08 | 10.14 | 0.68 | 9.24 | 0.56 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | spread | 8 | 256 | 21.48 | 10.28 | 0.82 | 9.72 | 0.32 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| reduce-scatter | fp32 | CUs 256 | spread | 16 | 256 | 23.10 | 10.30 | 1.03 | 11.20 | 0.24 | 0.00 | 9.07 | 0.60 | 8.95 | 18.62 | 0 | 0 |
| owner | fp32 | CUs 256 | spread | 2 | 256 | 17.22 | 7.33 | 0.61 | 6.36 | 2.32 | 0.00 | 6.45 | 0.60 | 6.21 | 13.27 | 0 | 0 |
| owner | fp32 | CUs 256 | spread | 4 | 256 | 23.98 | 7.85 | 0.53 | 13.00 | 1.40 | 0.00 | 6.80 | 0.60 | 10.92 | 18.32 | 0 | 0 |
| owner | fp32 | CUs 256 | spread | 8 | 256 | 40.36 | 8.51 | 0.00 | 28.14 | 1.12 | 0.00 | 7.94 | 0.60 | 25.49 | 34.02 | 0 | 0 |
| owner | fp32 | CUs 256 | spread | 16 | 256 | 74.98 | 9.60 | 0.00 | 61.26 | 1.08 | 0.00 | 8.50 | 0.60 | 54.61 | 63.72 | 0 | 0 |
| last arriver | fp32 | CUs 256 | spread | 2 | 256 | 19.98 | 9.86 | 0.36 | 6.56 | 2.40 | 0.00 | 9.07 | 0.60 | 4.47 | 14.14 | 0 | 0 |
| last arriver | fp32 | CUs 256 | spread | 4 | 256 | 32.02 | 10.08 | 0.36 | 18.76 | 1.96 | 0.00 | 9.07 | 0.60 | 10.92 | 20.59 | 0 | 0 |
| last arriver | fp32 | CUs 256 | spread | 8 | 256 | 50.73 | 10.22 | 0.40 | 37.20 | 1.68 | 0.00 | 9.07 | 0.60 | 25.49 | 35.15 | 0 | 0 |
| last arriver | fp32 | CUs 256 | spread | 16 | 256 | 67.78 | 10.28 | 0.36 | 54.84 | 1.26 | 0.00 | 9.07 | 0.60 | 54.61 | 64.28 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | spread | 2 | 256 | 11.69 | 4.95 | 0.69 | 4.44 | 1.16 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | spread | 4 | 256 | 11.42 | 5.01 | 0.71 | 4.68 | 0.56 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | spread | 8 | 256 | 11.59 | 5.13 | 0.85 | 4.92 | 0.36 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| reduce-scatter | bf16 | CUs 256 | spread | 16 | 256 | 12.97 | 5.15 | 1.10 | 6.24 | 0.24 | 0.00 | 4.53 | 0.60 | 4.47 | 9.61 | 0 | 0 |
| owner | bf16 | CUs 256 | spread | 2 | 256 | 10.94 | 3.87 | 0.60 | 3.52 | 2.36 | 0.00 | 3.23 | 0.60 | 3.11 | 6.93 | 0 | 0 |
| owner | bf16 | CUs 256 | spread | 4 | 256 | 16.15 | 4.17 | 0.55 | 9.12 | 1.40 | 0.00 | 3.40 | 0.60 | 5.46 | 9.46 | 0 | 0 |
| owner | bf16 | CUs 256 | spread | 8 | 256 | 28.36 | 4.42 | 0.16 | 21.32 | 1.12 | 0.00 | 3.97 | 0.60 | 12.74 | 17.31 | 0 | 0 |
| owner | bf16 | CUs 256 | spread | 16 | 256 | 54.31 | 4.68 | 0.39 | 46.86 | 1.08 | 0.00 | 4.25 | 0.60 | 27.31 | 32.16 | 0 | 0 |
| last arriver | bf16 | CUs 256 | spread | 2 | 256 | 12.21 | 4.94 | 0.36 | 3.80 | 2.44 | 0.00 | 4.53 | 0.60 | 2.24 | 7.37 | 0 | 0 |
| last arriver | bf16 | CUs 256 | spread | 4 | 256 | 18.32 | 5.02 | 0.36 | 9.68 | 1.88 | 0.00 | 4.53 | 0.60 | 5.46 | 10.60 | 0 | 0 |
| last arriver | bf16 | CUs 256 | spread | 8 | 256 | 30.33 | 5.15 | 0.40 | 21.72 | 1.76 | 0.00 | 4.53 | 0.60 | 12.74 | 17.88 | 0 | 0 |
| last arriver | bf16 | CUs 256 | spread | 16 | 256 | 47.46 | 5.16 | 0.40 | 39.96 | 1.44 | 0.00 | 4.53 | 0.60 | 27.31 | 32.44 | 0 | 0 |
| atomic bf16 | - | CUs 256 | spread | 2 | 256 | 25.46 | 25.10 | 0.00 | 0.04 | 0.04 | 8.14 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | spread | 4 | 256 | 25.29 | 25.06 | 0.00 | 0.04 | 0.04 | 7.06 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | spread | 8 | 256 | 25.26 | 25.09 | 0.00 | 0.04 | 0.04 | 6.64 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic bf16 | - | CUs 256 | spread | 16 | 256 | 26.14 | 25.98 | 0.00 | 0.04 | 0.04 | 6.40 | 25.42 | 0.00 | 0.00 | 25.42 | 0 | 0 |
| atomic fp32 | - | CUs 256 | spread | 2 | 256 | 50.30 | 49.85 | 0.00 | 0.04 | 0.04 | 10.00 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | spread | 4 | 256 | 49.89 | 49.64 | 0.00 | 0.04 | 0.04 | 7.96 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | spread | 8 | 256 | 49.90 | 49.73 | 0.00 | 0.04 | 0.04 | 8.32 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 | - | CUs 256 | spread | 16 | 256 | 49.89 | 49.73 | 0.00 | 0.04 | 0.04 | 6.72 | 49.71 | 0.00 | 0.00 | 49.71 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | spread | 2 | 256 | 59.48 | 49.81 | 0.40 | 6.08 | 2.12 | 10.04 | 49.71 | 0.60 | 4.47 | 54.78 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | spread | 4 | 256 | 59.00 | 49.66 | 0.40 | 5.88 | 1.92 | 8.00 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | spread | 8 | 256 | 57.06 | 49.69 | 0.40 | 4.52 | 1.64 | 7.04 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
| atomic fp32 + finalize | - | CUs 256 | spread | 16 | 256 | 55.66 | 49.73 | 0.36 | 3.84 | 1.40 | 7.76 | 49.71 | 0.60 | 3.64 | 53.95 | 0 | 0 |
