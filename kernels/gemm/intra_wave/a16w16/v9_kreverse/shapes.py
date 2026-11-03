"""The v9_kreverse study shapes (M, N, K) and what each isolates (see WORKLOG.md)."""

SHAPES = [
    (4096, 4096, 8192),  # one dispatch wave: no next wave to reuse anything, isolates the cost
    (8192, 8192, 8192),  # 4 waves, a wave's 160 MB fits the MALL: L2 effect only
    (16384, 16384, 8192),  # 16 waves: the largest modelled L2 gain
    (8192, 8192, 16384),  # 4 waves, a wave's 320 MB exceeds the MALL: serpentine MALL reuse
    (16384, 16384, 16384),  # 16 waves, 320 MB per wave: serpentine MALL reuse
    (6144, 6144, 8192),  # 2.25 waves: partial last wave
]
