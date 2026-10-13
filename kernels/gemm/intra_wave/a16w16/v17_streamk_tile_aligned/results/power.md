# Energy and power, v13 against v17 (HIP_VISIBLE_DEVICES=2, amd-smi GPU 2, bf16, v9 tile order)

`python v17_streamk_tile_aligned/power_check.py MxNxK`: 15 s of back-to-back launches on the
same inputs per run; energy from amd-smi's total-energy counter.

## 4096x7936x8192 (S = 240: stream-K cannot shorten the tail)

| run | launches | wall us per launch | average power (W) | median sampled socket power (W) | energy per launch (mJ) |
|---|---|---|---|---|---|
| idle | 0 | | 255 | 231 | |
| v13 (rep 1) | 46600 | 323.2 | 1410 | 1397 | 455.8 |
| v17 (rep 1) | 46000 | 326.7 | 1415 | 1397 | 462.3 |
| v13 (rep 2) | 46400 | 323.8 | 1414 | 1398 | 457.8 |
| v17 (rep 2) | 46000 | 327.1 | 1408 | 1398 | 460.5 |
| idle | 0 | | 251 | 232 | |

## 4352x4096x8192 (S = 16: stream-K shortens the tail)

| run | launches | wall us per launch | average power (W) | median sampled socket power (W) | energy per launch (mJ) |
|---|---|---|---|---|---|
| idle | 0 | | 256 | 231 | |
| v13 (rep 1) | 55600 | 270.0 | 1152 | 1136 | 311.1 |
| v17 (rep 1) | 67200 | 223.4 | 1364 | 1349 | 304.7 |
| v13 (rep 2) | 56000 | 268.4 | 1162 | 1148 | 311.8 |
| v17 (rep 2) | 67400 | 222.9 | 1375 | 1357 | 306.5 |
| idle | 0 | | 253 | 232 | |
