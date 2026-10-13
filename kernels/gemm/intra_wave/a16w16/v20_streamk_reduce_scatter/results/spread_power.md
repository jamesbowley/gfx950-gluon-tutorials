# Spread (S > 128) experiment: power (power_check.py, HIP_VISIBLE_DEVICES=2, SMI_GPU=2)

shape 4096x7168x32768, amd-smi GPU 2, 15 s per run
| run | launches | wall us per launch | average power (W, energy / time) | median sampled socket power (W) | energy per launch (mJ) |
|---|---|---|---|---|---|
| idle | 0 | 0.0 | 257 | 231 | 0.0 |
| v13 (rep 1) | 11680 | 1284.8 | 1404 | 1393 | 1804.0 |
| v20 policy=spread (rep 1) | 12340 | 1217.0 | 1415 | 1400 | 1721.8 |
| v16 (rep 1) | 11660 | 1286.8 | 1416 | 1400 | 1821.6 |
| v13 (rep 2) | 11660 | 1287.1 | 1405 | 1393 | 1808.7 |
| v20 policy=spread (rep 2) | 12300 | 1219.6 | 1409 | 1400 | 1718.1 |
| v16 (rep 2) | 11720 | 1280.5 | 1415 | 1400 | 1811.5 |
| idle | 0 | 0.0 | 252 | 233 | 0.0 |

shape 4096x7936x32768, amd-smi GPU 2, 15 s per run
| run | launches | wall us per launch | average power (W, energy / time) | median sampled socket power (W) | energy per launch (mJ) |
|---|---|---|---|---|---|
| idle | 0 | 0.0 | 251 | 232 | 0.0 |
| v13 (rep 1) | 11200 | 1339.8 | 1417 | 1398 | 1899.1 |
| v20 policy=spread (rep 1) | 11240 | 1334.7 | 1414 | 1400 | 1887.0 |
| v16 (rep 1) | 8200 | 1832.5 | 1416 | 1399 | 2594.4 |
| v13 (rep 2) | 11180 | 1342.0 | 1408 | 1398 | 1889.0 |
| v20 policy=spread (rep 2) | 11220 | 1338.2 | 1411 | 1400 | 1888.6 |
| v16 (rep 2) | 8200 | 1832.0 | 1410 | 1400 | 2582.3 |
| idle | 0 | 0.0 | 253 | 233 | 0.0 |

