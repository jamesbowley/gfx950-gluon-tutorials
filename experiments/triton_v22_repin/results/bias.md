| Dtype | Config | Kernel | TFLOPS no bias | TFLOPS bias | bias cost | MFMA eff no bias | MFMA eff bias | VGPRs / spills no bias | VGPRs / spills bias |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| bf16 | llir | v9_beyond_hotloop | 1432 | 1426 | -0.5% | 95.94% | 94.55% | 448/0 | 448/0 |
| bf16 | llir | v10_persistant | 1429 | 1421 | -0.6% | 96.40% | 95.00% | 480/0 | 488/0 |
| bf16 | llir | v11_persistant_overlap_global | 1428 | 1419 | -0.6% | 96.09% | 95.02% | 452/0 | 464/0 |
| bf16 | llir | v12_persistant_overlap_lds | 1427 | 1418 | -0.6% | 96.09% | 95.03% | 440/0 | 444/0 |
| bf16 | llir | v13_persistant_peel_acc | 1426 | 1422 | -0.2% | 96.41% | 96.03% | 480/0 | 488/0 |
| bf16 | llir+amdgcnas | v9_beyond_hotloop | 1438 | 1432 | -0.4% | 98.12% | 97.78% | 448/0 | 448/0 |
| bf16 | llir+amdgcnas | v10_persistant | 1432 | 1426 | -0.5% | 98.75% | 97.40% | 480/0 | 488/0 |
| bf16 | llir+amdgcnas | v11_persistant_overlap_global | 1433 | 1424 | -0.6% | 98.06% | 97.33% | 452/0 | 464/0 |
| bf16 | llir+amdgcnas | v12_persistant_overlap_lds | 1431 | 1426 | -0.3% | 97.97% | 97.63% | 440/0 | 444/0 |
| bf16 | llir+amdgcnas | v13_persistant_peel_acc | 1429 | 1426 | -0.2% | 97.76% | 97.80% | 480/0 | 488/0 |
| fp16 | llir | v9_beyond_hotloop | 1335 | 1322 | -0.9% | 95.84% | 94.52% | 448/0 | 448/0 |
| fp16 | llir | v10_persistant | 1330 | 1322 | -0.5% | 95.60% | 94.83% | 480/0 | 488/0 |
| fp16 | llir | v11_persistant_overlap_global | 1328 | 1320 | -0.5% | 96.05% | 95.00% | 452/0 | 464/0 |
| fp16 | llir | v12_persistant_overlap_lds | 1328 | 1320 | -0.6% | 95.72% | 95.05% | 440/0 | 444/0 |
| fp16 | llir | v13_persistant_peel_acc | 1326 | 1326 | -0.0% | 95.81% | 95.87% | 480/0 | 488/0 |
| fp16 | llir+amdgcnas | v9_beyond_hotloop | 1338 | 1328 | -0.7% | 98.08% | 97.96% | 448/0 | 448/0 |
| fp16 | llir+amdgcnas | v10_persistant | 1333 | 1326 | -0.5% | 97.72% | 97.47% | 480/0 | 488/0 |
| fp16 | llir+amdgcnas | v11_persistant_overlap_global | 1334 | 1330 | -0.3% | 98.06% | 97.50% | 452/0 | 464/0 |
| fp16 | llir+amdgcnas | v12_persistant_overlap_lds | 1330 | 1328 | -0.2% | 97.63% | 97.35% | 440/0 | 444/0 |
| fp16 | llir+amdgcnas | v13_persistant_peel_acc | 1328 | 1330 | +0.2% | 97.78% | 98.12% | 480/0 | 488/0 |
