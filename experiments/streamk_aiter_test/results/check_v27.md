# AITER test set: correctness (contiguous B and C)

1281 of 1320 cases ran, 2 launches each against torch.matmul (atol 0.1, rtol 0.01).


FAIL

The 39 cases at K = 512, 640 and 768 were timed before their check worker started, so `untimed()` skipped
them (the FAIL above is "1281 of 1320 ran", no wrong case). Checked afterwards with the same worker
(check_worker on exactly those v27 cases): 39 of 39 correct, 0 errors. All 1320 v27 cases are correct.
