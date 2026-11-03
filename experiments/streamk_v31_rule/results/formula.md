# Stream-K formula given the tile

Train: 7368 (shape, spec) pairs; test: 7721 pairs.

Measured full-wave TFLOPS per spec (dp, S = 0 or W >= 2): 256x256x64 1394, 256x128x64 1156, 128x256x64 1148, 128x128x128 1005, 128x128x64 905, 256x64x64 770, 64x256x64 763, 128x64x128 721, 64x128x128 701, 128x64x64 634, 64x128x64 614, 64x64x128 522, 64x64x64 405

## Form selection (leave one spec out: time over the pair's best, geomean / median / 90th / worst)

| form | in-sample | leave one spec out |
|---|---|---|
| shared | 1.0104 / 1.000 / 1.036 / 1.355 | 1.0104 / 1.000 / 1.036 / 1.355 |
| regimes | 1.0112 / 1.000 / 1.036 / 1.350 | 1.0120 / 1.000 / 1.041 / 1.350 |
| shared + 64/BK | 1.0146 / 1.000 / 1.050 / 1.477 | 1.0157 / 1.000 / 1.055 / 1.477 |
| regimes + 64/BK | 1.0123 / 1.000 / 1.042 / 1.331 | 1.0133 / 1.000 / 1.047 / 1.331 |
| shared + tf | 1.0194 / 1.000 / 1.067 / 1.831 | 1.0195 / 1.000 / 1.067 / 1.831 |
| regimes + tf | 1.0126 / 1.000 / 1.050 / 1.307 | 1.0126 / 1.000 / 1.050 / 1.307 |

Chosen form: **shared**.

## Constants (fitted, then rounded)

```json
{
 "form": {
  "name": "shared",
  "regimes": false,
  "tile": "none"
 },
 "fitted": {
  "all": {
   "m_ow": 3,
   "c_ow": 1.5,
   "pairs_ow": 10,
   "b": 6,
   "gain": 6
  }
 },
 "rounded": {
  "all": {
   "m_ow": 3,
   "c_ow": 1,
   "pairs_ow": 10,
   "b": 6,
   "gain": 6
  }
 }
}
```

- train, fitted: 1.0104 / 1.000 / 1.036 / 1.355; vs dp: >5% slower 92, >10% 27 of 7368, worst 1.196
- train, rounded: 1.0104 / 1.000 / 1.036 / 1.355; vs dp: >5% slower 92, >10% 27 of 7368, worst 1.196
- leave one k bin out: 1.0109 / 1.000 / 1.040 / 1.355
- leave one m bin out: 1.0173 / 1.000 / 1.060 / 1.458

## Bootstrap over shapes (100 resamples): 10th / median / 90th percentile

- all b: 6 / 6.0 / 6
- all c_ow: 1.5 / 1.5 / 1.5
- all gain: 5 / 6.0 / 6
- all m_ow: 3 / 3.0 / 3
- all pairs_ow: 10 / 10.0 / 16

## Test: held-out AITER shapes, every spec (time over the pair's best)

- all specs: 1.0024 / 1.000 / 1.000 / 1.292; vs dp: >5% slower 30, >10% 8 of 7721, worst 1.195
- dp for comparison: 1.0776 / 1.000 / 1.320 / 4.969
- 256x256x64: 1.0036 / 1.000 / 1.004 / 1.145; vs dp: >5% slower 0, >10% 0 of 576, worst 1.036
- 256x128x64: 1.0027 / 1.000 / 1.000 / 1.164; vs dp: >5% slower 6, >10% 2 of 585, worst 1.137
- 128x256x64: 1.0032 / 1.000 / 1.000 / 1.257; vs dp: >5% slower 4, >10% 1 of 597, worst 1.147
- 128x128x64: 1.0018 / 1.000 / 1.000 / 1.227; vs dp: >5% slower 2, >10% 1 of 597, worst 1.192
- 128x128x128: 1.0033 / 1.000 / 1.000 / 1.228; vs dp: >5% slower 0, >10% 0 of 590, worst 1.050
- 256x64x64: 1.0014 / 1.000 / 1.000 / 1.130; vs dp: >5% slower 3, >10% 1 of 597, worst 1.117
- 64x256x64: 1.0021 / 1.000 / 1.000 / 1.292; vs dp: >5% slower 0, >10% 0 of 597, worst 1.041
- 128x64x64: 1.0009 / 1.000 / 1.000 / 1.140; vs dp: >5% slower 0, >10% 0 of 597, worst 1.046
- 128x64x128: 1.0020 / 1.000 / 1.000 / 1.133; vs dp: >5% slower 1, >10% 0 of 597, worst 1.055
- 64x128x64: 1.0019 / 1.000 / 1.000 / 1.228; vs dp: >5% slower 1, >10% 0 of 597, worst 1.094
- 64x128x128: 1.0039 / 1.000 / 1.000 / 1.245; vs dp: >5% slower 1, >10% 0 of 597, worst 1.081
- 64x64x64: 1.0016 / 1.000 / 1.000 / 1.195; vs dp: >5% slower 5, >10% 2 of 597, worst 1.195
- 64x64x128: 1.0025 / 1.000 / 1.000 / 1.192; vs dp: >5% slower 7, >10% 1 of 597, worst 1.164
