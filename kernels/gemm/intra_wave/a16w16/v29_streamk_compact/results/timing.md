# v29 against v28: 9 rounds of 5 x 100 launches, median us per launch

| point | tile | shape | launched (v29) | v28 us | v29 us | v29 / v28 | spread v28 / v29 |
|---|---|---|---|---|---|---|---|
| anchor owall | 128x128x128 | 1024x1024x7168 | v29_streamk_compact one_tile / owner / 256 | 27.96 | 27.85 | 0.996 | 1.6% / 1.8% |
| anchor ow3 | 128x128x128 | 1024x1024x7168 | v29_streamk_compact one_tile / owner / 192 | 29.95 | 29.84 | 0.996 | 1.5% / 1.5% |
| anchor ow2 | 128x128x128 | 1024x1024x7168 | v29_streamk_compact one_tile / owner / 128 | 37.75 | 37.72 | 0.999 | 0.6% / 0.6% |
| anchor rsall | 128x128x128 | 1024x1024x7168 | v29_streamk_compact one_tile / rs / 256 | 27.29 | 27.23 | 0.998 | 1.1% / 1.3% |
| anchor rsall l2 | 128x128x128 | 1024x1024x7168 | v29_streamk_compact one_tile / rs / 256 | 28.08 | 27.89 | 0.993 | 2.0% / 2.0% |
| 256 ow16 | 256x256x64 | 4096x4352x8192 | v29_streamk_compact one_tile / owner / 256 | 244.33 | 246.21 | 1.008 | 1.6% / 0.6% |
| 256 2t | 256x256x64 | 4096x4608x8192 | v29_streamk_compact two_tile / rs / 256 | 218.47 | 219.13 | 1.003 | 2.7% / 1.0% |
| 256 2t+b | 256x256x64 | 4096x4608x8192 | v29_streamk_compact two_tile / rs / 256 | 223.83 | 223.82 | 1.000 | 1.8% / 3.1% |
| 256 sp | 256x256x64 | 4096x4608x8192 | v29_streamk_compact spread / rs / 256 | 234.19 | 235.54 | 1.006 | 1.2% / 1.3% |
| 256 ragged sp+b | 256x256x64 | 4072x4560x8176 | v29_streamk_compact spread / rs / 256 | 292.04 | 291.84 | 0.999 | 0.6% / 0.7% |
| 128x64 rsall+b | 128x128x64 | 2048x2304x8192 | v29_streamk_compact one_tile / rs / 256 | 86.93 | 86.91 | 1.000 | 0.6% / 0.7% |
| cap 4096^3 | 256x256x64 | 4096x4096x4096 | v9_tile | 94.70 | 94.80 | 1.001 | 2.1% / 2.3% |
