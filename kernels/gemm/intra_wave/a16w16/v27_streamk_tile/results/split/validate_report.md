# The merged tables' picks against cap on tilesel's shapes (tmp/validate_new.py)

New: both geometries, bracket lookup; old: the tables tilesel.py was timed with. Pick over cap at the same tile, within one block (tilesel's or stage V's).

| tile class | waves | pairs (of which stage V) | new / cap: geomean | better by > 2% | worse by > 2% | worst | old / cap |
|---|---|---|---|---|---|---|---|
| quadrant | 0 | 1086 (107) | 0.984 | 334 | 134 | 1.62 | 1.005 |
| quadrant | 1 | 748 (126) | 0.980 | 211 | 48 | 1.18 | 0.990 |
| quadrant | 4 | 1088 (116) | 0.962 | 360 | 27 | 1.46 | 0.967 |
| small | 0 | 595 (119) | 0.930 | 287 | 75 | 1.63 | 0.944 |
| small | 1 | 293 (58) | 0.992 | 60 | 27 | 1.29 | 1.012 |
| small | 4 | 279 (47) | 0.998 | 20 | 14 | 1.10 | 1.005 |
| all | | 4089 | 0.971 | 1272 | 325 | 1.63 | |
