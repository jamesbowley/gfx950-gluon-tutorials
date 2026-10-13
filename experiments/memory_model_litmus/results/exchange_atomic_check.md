HIP_VISIBLE_DEVICES=0, AMD Instinct MI355X, triton 3.8.0, 2026-10-07 18:41


## atomic_check

Atomic instructions in each ending's compiled code (distinct forms; `sc0` would be the returning form), then the tiles after 3 launches with every partial added once and, as a control, twice (REPEAT 2). W = 256, 256 KiB partials, a 1 GiB read before each launch.

| pattern | atomic instructions | returning form |
|---|---|---|
| atomic bf16 | `buffer_atomic_pk_add_bf16 0 offen` | no |
| atomic fp32 | `buffer_atomic_add_f32 0 offen` | no |
| atomic fp32 + finalize | `buffer_atomic_add_f32 0 offen` | no |

| pattern | placement | n | adds | elements checked | wrong | expected wrong |
|---|---|---|---|---|---|---|
| atomic bf16 | one XCD | 4 | once | 12582912 | 0 | 0 |
| atomic bf16 | one XCD | 4 | twice | 12582912 | 12582912 | 12582912 |
| atomic bf16 | one XCD | 16 | once | 3145728 | 0 | 0 |
| atomic bf16 | one XCD | 16 | twice | 3145728 | 3145728 | 3145728 |
| atomic bf16 | spread | 4 | once | 12582912 | 0 | 0 |
| atomic bf16 | spread | 4 | twice | 12582912 | 12582912 | 12582912 |
| atomic bf16 | spread | 16 | once | 3145728 | 0 | 0 |
| atomic bf16 | spread | 16 | twice | 3145728 | 3145728 | 3145728 |
| atomic fp32 | one XCD | 4 | once | 12582912 | 0 | 0 |
| atomic fp32 | one XCD | 4 | twice | 12582912 | 12582912 | 12582912 |
| atomic fp32 | one XCD | 16 | once | 3145728 | 0 | 0 |
| atomic fp32 | one XCD | 16 | twice | 3145728 | 3145728 | 3145728 |
| atomic fp32 | spread | 4 | once | 12582912 | 0 | 0 |
| atomic fp32 | spread | 4 | twice | 12582912 | 12582912 | 12582912 |
| atomic fp32 | spread | 16 | once | 3145728 | 0 | 0 |
| atomic fp32 | spread | 16 | twice | 3145728 | 3145728 | 3145728 |
| atomic fp32 + finalize | one XCD | 4 | once | 37748736 | 0 | 0 |
| atomic fp32 + finalize | one XCD | 4 | twice | 37748736 | 25165824 | 25165824 |
| atomic fp32 + finalize | one XCD | 16 | once | 9437184 | 0 | 0 |
| atomic fp32 + finalize | one XCD | 16 | twice | 9437184 | 6291456 | 6291456 |
| atomic fp32 + finalize | spread | 4 | once | 37748736 | 0 | 0 |
| atomic fp32 + finalize | spread | 4 | twice | 37748736 | 25165824 | 25165824 |
| atomic fp32 + finalize | spread | 16 | once | 9437184 | 0 | 0 |
| atomic fp32 + finalize | spread | 16 | twice | 9437184 | 6291456 | 6291456 |
