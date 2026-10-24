# Tiered rule fit (REFIT2)

Training: REFIT (350 shapes) only (results/refit2.csv not merged yet); test: REFIT2's split (0 shapes); post hoc: the held-out AITER shapes (617, a reduced factorial of quadrant tiles). Noise sigma 1.89% (REFIT). Extreme: above 1.15x the best timed point.


## compute-bound: 198 training shapes, 0 test

### Split rule (capped_policy's constants per tile class)

| class | pairs (tile within 1.15 of the shape's best) | default: > 1.15, sum log | fitted | changed |
|---|---|---|---|---|
| quadrant | 584 | 81, 29.3 | 6, 13.0 | {'spread': 8000, 'dp0': 6, 'rsr0': 3, 'rsn0': 256, 'rsk0': 8, 'owc0': 3, 'dp1': 28, 'rsmin1': 4, 'rsk1': 4, 'ows1': 8, 'pdp_max': True} |

### Tile rule: depth by leaving shapes out

| depth, min leaf | shapes | median | 90th | worst | geomean | > 1.15 | > sigma |
|---|---|---|---|---|---|---|---|
| 0, 5 | 198 | 1.006 | 1.207 | 1.871 | 1.053 | 27 | 58 |
| 1, 5 | 198 | 1.006 | 1.148 | 1.612 | 1.036 | 19 | 57 |
| 1, 10 | 198 | 1.006 | 1.148 | 1.612 | 1.036 | 19 | 57 |
| 1, 20 | 198 | 1.006 | 1.148 | 1.612 | 1.036 | 19 | 57 |
| 2, 5 | 198 | 1.006 | 1.090 | 1.277 | 1.027 | 10 | 59 |
| 2, 10 | 198 | 1.006 | 1.077 | 1.277 | 1.026 | 10 | 57 |
| 2, 20 | 198 | 1.006 | 1.094 | 1.465 | 1.027 | 11 | 57 |
| 3, 5 | 198 | 1.006 | 1.062 | 1.277 | 1.024 | 9 | 59 |
| 3, 10 | 198 | 1.007 | 1.072 | 1.277 | 1.026 | 9 | 60 |
| 3, 20 | 198 | 1.007 | 1.090 | 1.465 | 1.027 | 10 | 60 |
| 4, 5 | 198 | 1.006 | 1.063 | 1.411 | 1.028 | 13 | 61 |
| 4, 10 | 198 | 1.006 | 1.082 | 1.277 | 1.027 | 11 | 60 |
| 4, 20 | 198 | 1.006 | 1.094 | 1.465 | 1.027 | 10 | 59 |

Chosen: depth 3, min leaf 5. On all training shapes:

```
if log2 K <= 12.754:
    if log2 tiles@256 <= 5.146:
        if log2 K <= 11.643:
            -> area rule, theta 0.75
        else:
            -> 64x128x128
    else:
        if log2 M <= 12.129:
            -> area rule, theta 0.5
        else:
            -> area rule, theta 1.5
else:
    if log2 tiles@256 <= 3.989:
        -> 128x128x64
    else:
        if log2 tiles@256 <= 4.882:
            -> 128x256x64
        else:
            -> area rule, theta 0.25
```

### Regret

| rule | shapes | median | 90th | worst | geomean | > 1.15 | > sigma |
|---|---|---|---|---|---|---|---|
| Tier 1, training, leaving shapes out (tile; splits fitted on all) | 198 | 1.006 | 1.062 | 1.277 | 1.024 | 9 | 59 |
| held-out, post hoc: v26 (cap at 256x256x64) | 617 | 1.004 | 1.315 | 2.599 | 1.093 | 139 | 235 |
| held-out, post hoc: auto (auto_tile, cap) | 617 | 1.003 | 1.158 | 1.573 | 1.042 | 63 | 188 |
| held-out, post hoc: Tier 1 (158 not timed) | 459 | 1.004 | 1.171 | 1.402 | 1.041 | 53 | 148 |
| held-out, post hoc: Tier 1's tile, cap | 617 | 1.003 | 1.195 | 1.492 | 1.048 | 84 | 208 |

### Tier 2: the short list (Tier 1's tile, then the 8 nearest training shapes' best-scored)

| set | k | has the best tile | a tile within sigma | within 1.15 | Tier 1 unsure |
|---|---|---|---|---|---|
| training, leaving shapes out | 1 | 125 / 198 | 170 / 198 | 196 / 198 | 50 |
| training, leaving shapes out | 2 | 147 / 198 | 192 / 198 | 198 / 198 |  |
| training, leaving shapes out | 3 | 149 / 198 | 194 / 198 | 198 / 198 |  |
| training, leaving shapes out | 4 | 153 / 198 | 197 / 198 | 198 / 198 |  |
| training, leaving shapes out | 5 | 153 / 198 | 198 / 198 | 198 / 198 |  |

## memory-bound: 152 training shapes, 0 test

### Split rule (capped_policy's constants per tile class)

| class | pairs (tile within 1.15 of the shape's best) | default: > 1.15, sum log | fitted | changed |
|---|---|---|---|---|
| quadrant | 324 | 34, 16.4 | 7, 5.8 | {'spread': 8000, 'rsr0': 3, 'rsn0': 256, 'dp1': 40, 'rsr1': 4, 'rsh1': 1, 'ows1': 8, 'pdp_max': True} |
| small P=1024 | 463 | 138, 58.7 | 51, 23.9 | {'spread': 1099511627776, 'rsr0': 8, 'rsn0': 384, 'owc0': 3, 'dp1': 10, 'rsh1': 1, 'ows1': 12} |
| small P=256 | 64 | 16, 5.5 | 2, 1.8 | {'dp0': 6, 'rsn0': 256, 'owc0': 3} |
| small P=512 | 125 | 40, 13.6 | 15, 6.7 | {'dp0': 20, 'rsr0': 8, 'rsn0': 256, 'rsk0': 2} |

### Tile rule: depth by leaving shapes out

| depth, min leaf | shapes | median | 90th | worst | geomean | > 1.15 | > sigma |
|---|---|---|---|---|---|---|---|
| 0, 5 | 152 | 1.082 | 2.264 | 4.659 | 1.269 | 61 | 109 |
| 1, 5 | 152 | 1.103 | 1.451 | 1.767 | 1.152 | 58 | 117 |
| 1, 10 | 152 | 1.103 | 1.451 | 1.767 | 1.152 | 58 | 117 |
| 1, 20 | 152 | 1.103 | 1.451 | 1.767 | 1.152 | 58 | 117 |
| 2, 5 | 152 | 1.080 | 1.330 | 1.868 | 1.123 | 50 | 112 |
| 2, 10 | 152 | 1.080 | 1.330 | 1.868 | 1.123 | 50 | 112 |
| 2, 20 | 152 | 1.080 | 1.338 | 1.868 | 1.126 | 51 | 111 |
| 3, 5 | 152 | 1.098 | 1.290 | 1.653 | 1.122 | 58 | 115 |
| 3, 10 | 152 | 1.083 | 1.301 | 1.653 | 1.115 | 51 | 114 |
| 3, 20 | 152 | 1.104 | 1.338 | 1.868 | 1.136 | 58 | 112 |
| 4, 5 | 152 | 1.099 | 1.301 | 1.804 | 1.124 | 60 | 116 |
| 4, 10 | 152 | 1.077 | 1.290 | 1.653 | 1.107 | 48 | 110 |
| 4, 20 | 152 | 1.104 | 1.338 | 1.868 | 1.136 | 58 | 112 |

Chosen: depth 4, min leaf 10. On all training shapes:

```
if log2 M <= 5.500:
    if log2 N <= 14.050:
        if log2 K <= 13.443:
            if log2 M <= 1.000:
                -> 1x16x256
            else:
                -> 16x16x256
        else:
            -> 16x32x128
    else:
        -> 32x128x128
else:
    if log2 N <= 12.209:
        if log2 K <= 12.044:
            if log2 N <= 10.453:
                -> 16x16x256
            else:
                -> 32x32x128
        else:
            -> 128x64x128
    else:
        if log2 K <= 12.315:
            -> 128x64x128
        else:
            -> 128x128x128
```

### Regret

| rule | shapes | median | 90th | worst | geomean | > 1.15 | > sigma |
|---|---|---|---|---|---|---|---|
| Tier 1, training, leaving shapes out (tile; splits fitted on all) | 152 | 1.077 | 1.290 | 1.653 | 1.107 | 48 | 110 |

### Tier 2: the short list (Tier 1's tile, then the 8 nearest training shapes' best-scored)

| set | k | has the best tile | a tile within sigma | within 1.15 | Tier 1 unsure |
|---|---|---|---|---|---|
| training, leaving shapes out | 1 | 40 / 152 | 76 / 152 | 133 / 152 | 107 |
| training, leaving shapes out | 2 | 67 / 152 | 118 / 152 | 149 / 152 |  |
| training, leaving shapes out | 3 | 78 / 152 | 130 / 152 | 150 / 152 |  |
| training, leaving shapes out | 4 | 90 / 152 | 137 / 152 | 150 / 152 |  |
| training, leaving shapes out | 5 | 98 / 152 | 143 / 152 | 151 / 152 |  |
