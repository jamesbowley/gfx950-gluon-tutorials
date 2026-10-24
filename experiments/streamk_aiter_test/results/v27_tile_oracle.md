# v27 at a per-shape tile against v26 as deployed (AITER shapes)

Oracle tile choice over {256x256 (v26 deployed), v27 capped at 256x128, 128x256, 128x128x128}; time / v26 deployed.

| shapes | count | oracle / v26: median / 10th pct / best | shapes > 2% faster | tile picked (256x256 / 256x128 / 128x256 / 128x128x128) |
|---|---|---|---|---|
| less than one wave, S <= 128 | 254 | 0.856 / 0.637 / 0.496 | 200 | 48 / 37 / 42 / 127 |
| less than one wave, S > 128 | 60 | 1.000 / 1.000 / 0.970 | 2 | 58 / 0 / 2 / 0 |
| one full wave or more, S <= 128 | 95 | 1.000 / 1.000 / 0.938 | 2 | 91 / 1 / 3 / 0 |
| one full wave or more, S > 128 | 31 | 1.000 / 1.000 / 1.000 | 0 | 31 / 0 / 0 / 0 |

- 1-32 tiles of 256x256 (71 shapes): oracle / v26 median 0.789, 256x256 kept on 1

- 33-64 tiles of 256x256 (84 shapes): oracle / v26 median 0.856, 256x256 kept on 10

- 65-128 tiles of 256x256 (99 shapes): oracle / v26 median 0.947, 256x256 kept on 37

- 129-255 tiles of 256x256 (60 shapes): oracle / v26 median 1.000, 256x256 kept on 58
