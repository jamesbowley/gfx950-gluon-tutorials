HIP_VISIBLE_DEVICES=5 launches/point=20

## pmc_detail

(T, n) = (1, 8): 64 programs, 131K 128-byte lines of partials, 131K of src. Thousands per launch; 'no slice' skips step 4. Reads (TCC_READ, EA reads) count 128-byte requests, writes 64-byte ones.

| variant | slice | TCC_READ | TCC_WRITE | TCC_EA0_RDREQ | TCC_EA0_WRREQ | TCC_HIT | TCC_MISS | TCC_RW_REQ | TCC_NC_REQ | TCC_PROBE | TCC_BYPASS_REQ | TCC_STREAMING_REQ | TCC_ALL_TC_OP_WB_WRITEBACK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | yes | 272 | 295 | 263 | 295 | 26 | 541 | 568 | 0 | 262 | 262 | 0 | 16 |
| mem | no | 139 | 262 | 132 | 262 | 8 | 394 | 402 | 0 | 262 | 262 | 0 | 0 |
| wt_relaxed | yes | 272 | 295 | 263 | 295 | 26 | 541 | 567 | 0 | 262 | 262 | 0 | 16 |
| wt_relaxed | no | 139 | 262 | 132 | 262 | 8 | 394 | 401 | 0 | 262 | 262 | 0 | 0 |
| l2_cg | yes | 272 | 295 | 263 | 295 | 288 | 279 | 567 | 0 | 0 | 0 | 131 | 131 |
| l2_cg | no | 139 | 262 | 132 | 262 | 139 | 263 | 401 | 0 | 0 | 0 | 0 | 131 |
| l2_pre | yes | 405 | 295 | 264 | 295 | 420 | 279 | 700 | 0 | 0 | 0 | 131 | 131 |
| l2_pre | no | 272 | 262 | 263 | 262 | 271 | 263 | 534 | 0 | 0 | 0 | 0 | 131 |
