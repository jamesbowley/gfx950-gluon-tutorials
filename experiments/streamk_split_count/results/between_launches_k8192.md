# Clean-cache cross-check of the best split at K = 8192 (between_launches.py, HIP_VISIBLE_DEVICES=2)

num_programs is STREAMK_NUM_PROGRAMS = S * n: 32 = n 8 at S = 4, 64 = n 16 (rs) or n 4 (owner, S = 16),
128 = n 8 at S = 16, 152 = n 4 at S = 38, 192 = n 3 at S = 64. v19 is two-tile on all four shapes.

| shape | version | nothing | zero 256 MB (do_bench) | read 256 MB | zero 64 MB | sleep 100 us |
|---|---|---|---|---|---|---|
| 3328x5120x8192 | v13 | 274.6 | 278.2 | 273.6 | 275.7 | 272.6 |
| 3328x5120x8192 | v20 policy=one_tile fixup=owner | 334.2 | 347.0 | 342.9 | 339.2 | 334.8 |
| 3328x5120x8192 | v20 policy=one_tile fixup=owner num_programs=32 | 203.7 | 206.4 | 202.3 | 205.1 | 201.7 |
| 3328x5120x8192 | v20 policy=one_tile fixup=rs | 203.2 | 210.5 | 202.7 | 203.8 | 195.4 |
| 3328x5120x8192 | v20 policy=one_tile fixup=rs num_programs=64 | 190.0 | 192.0 | 183.6 | 190.5 | 182.1 |
| 3328x5120x8192 | v19 | 204.6 | 211.6 | 185.0 | 199.0 | 180.5 |
| 4352x4096x8192 | v13 | 273.7 | 280.4 | 273.4 | 278.7 | 273.8 |
| 4352x4096x8192 | v20 policy=one_tile fixup=owner | 233.3 | 241.9 | 234.9 | 237.2 | 227.6 |
| 4352x4096x8192 | v20 policy=one_tile fixup=owner num_programs=64 | 207.9 | 215.4 | 206.8 | 212.1 | 206.1 |
| 4352x4096x8192 | v20 policy=one_tile fixup=rs | 212.1 | 218.0 | 207.6 | 215.2 | 203.0 |
| 4352x4096x8192 | v20 policy=one_tile fixup=rs num_programs=128 | 209.0 | 210.4 | 201.6 | 207.2 | 196.6 |
| 4352x4096x8192 | v19 | 211.7 | 228.0 | 200.3 | 211.1 | 188.7 |
| 3584x5376x8192 | v13 | 278.9 | 285.0 | 274.7 | 275.8 | 274.1 |
| 3584x5376x8192 | v20 policy=one_tile fixup=owner | 241.4 | 236.2 | 225.6 | 239.8 | 223.3 |
| 3584x5376x8192 | v20 policy=one_tile fixup=owner num_programs=152 | 228.1 | 226.1 | 213.4 | 229.7 | 212.5 |
| 3584x5376x8192 | v20 policy=one_tile fixup=rs | 233.7 | 234.5 | 229.4 | 237.5 | 221.1 |
| 3584x5376x8192 | v20 policy=one_tile fixup=rs num_programs=152 | 226.4 | 228.8 | 215.5 | 228.7 | 214.5 |
| 3584x5376x8192 | v19 | 231.4 | 249.1 | 218.6 | 235.5 | 209.3 |
| 4096x5120x8192 | v13 | 283.6 | 298.4 | 283.3 | 286.3 | 274.8 |
| 4096x5120x8192 | v20 policy=one_tile fixup=owner | 243.1 | 245.1 | 228.6 | 237.1 | 222.9 |
| 4096x5120x8192 | v20 policy=one_tile fixup=owner num_programs=192 | 236.0 | 241.8 | 229.4 | 235.6 | 221.8 |
| 4096x5120x8192 | v20 policy=one_tile fixup=rs | 237.7 | 250.6 | 237.0 | 243.3 | 229.1 |
| 4096x5120x8192 | v19 | 248.4 | 276.3 | 252.4 | 239.7 | 219.5 |
