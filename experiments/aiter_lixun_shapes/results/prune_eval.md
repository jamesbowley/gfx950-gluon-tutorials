# Search-space reductions replayed on the exhaustive v27 sweeps

**full sweep (reference)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 100% (254/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| heldout | 617 | 100% (59/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| worst10 | 60 | 100% (140/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| **all** | 1027 | 100% (130/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |

**exact duplicates removed (same launch)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 94% (238/shape) | 0.00% | 0.27% | 5.14% | 9 | 5 |
| heldout | 617 | 83% (49/shape) | 0.00% | 0.44% | 2.23% | 15 | 2 |
| worst10 | 60 | 86% (120/shape) | 0.00% | 0.11% | 0.97% | 0 | 0 |
| **all** | 1027 | 90% (117/shape) | 0.00% | 0.38% | 5.14% | 24 | 7 |

**n in {2, 3, 4, 6, 8, 12, 16, all}**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 84% (213/shape) | 0.00% | 0.00% | 4.74% | 14 | 11 |
| heldout | 617 | 100% (59/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| worst10 | 60 | 92% (129/shape) | 0.00% | 1.92% | 5.04% | 9 | 6 |
| **all** | 1027 | 89% (116/shape) | 0.00% | 0.00% | 5.04% | 23 | 17 |

**reduce-scatter only (no owner ending)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 66% (169/shape) | 0.00% | 0.00% | 5.71% | 7 | 3 |
| heldout | 617 | 96% (57/shape) | 0.00% | 0.00% | 1.18% | 1 | 0 |
| worst10 | 60 | 76% (106/shape) | 0.00% | 0.01% | 2.07% | 4 | 1 |
| **all** | 1027 | 75% (98/shape) | 0.00% | 0.00% | 5.71% | 12 | 4 |

**tile screen (dp, cap, rsall, owall at every tile; all kinds within 2%)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 31% (78/shape) | 0.00% | 0.00% | 14.97% | 26 | 17 |
| heldout | 617 | 57% (34/shape) | 0.00% | 0.00% | 6.38% | 12 | 9 |
| worst10 | 60 | 40% (56/shape) | 0.00% | 0.00% | 8.27% | 4 | 3 |
| **all** | 1027 | 39% (50/shape) | 0.00% | 0.00% | 14.97% | 42 | 29 |

**tile screen (dp, cap, rsall, owall at every tile; all kinds within 5%)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 33% (83/shape) | 0.00% | 0.00% | 14.97% | 16 | 6 |
| heldout | 617 | 58% (34/shape) | 0.00% | 0.00% | 2.57% | 3 | 1 |
| worst10 | 60 | 46% (65/shape) | 0.00% | 0.00% | 1.56% | 1 | 0 |
| **all** | 1027 | 41% (53/shape) | 0.00% | 0.00% | 14.97% | 20 | 7 |

**tile screen (dp, cap, rsall, owall at every tile; all kinds within 10%)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 37% (93/shape) | 0.00% | 0.00% | 11.77% | 4 | 2 |
| heldout | 617 | 60% (36/shape) | 0.00% | 0.00% | 1.73% | 2 | 0 |
| worst10 | 60 | 54% (76/shape) | 0.00% | 0.00% | 0.21% | 0 | 0 |
| **all** | 1027 | 44% (58/shape) | 0.00% | 0.00% | 11.77% | 6 | 2 |

**tile screen (dp, cap, rsall, owall at every tile; all kinds within 20%)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 47% (119/shape) | 0.00% | 0.00% | 2.11% | 1 | 1 |
| heldout | 617 | 66% (39/shape) | 0.00% | 0.00% | 1.48% | 1 | 0 |
| worst10 | 60 | 68% (95/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| **all** | 1027 | 53% (69/shape) | 0.00% | 0.00% | 2.11% | 2 | 1 |

**second round only within 3% of round 1's best**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 51% (130/shape) | 0.00% | 0.00% | 5.34% | 5 | 3 |
| heldout | 617 | 53% (31/shape) | 0.00% | 0.00% | 1.89% | 1 | 0 |
| worst10 | 60 | 51% (72/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| **all** | 1027 | 51% (67/shape) | 0.00% | 0.00% | 5.34% | 6 | 3 |

**second round only within 5% of round 1's best**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 51% (131/shape) | 0.00% | 0.00% | 1.37% | 3 | 0 |
| heldout | 617 | 53% (31/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| worst10 | 60 | 52% (73/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| **all** | 1027 | 52% (68/shape) | 0.00% | 0.00% | 1.37% | 3 | 0 |

**origami top 1 tiles, every kind**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 5% (14/shape) | 0.00% | 11.51% | 113.10% | 121 | 104 |
| heldout | 617 | 7% (4/shape) | 0.00% | 5.69% | 22.12% | 129 | 103 |
| worst10 | 60 | 9% (12/shape) | 0.96% | 8.61% | 14.11% | 29 | 26 |
| **all** | 1027 | 6% (8/shape) | 0.00% | 7.95% | 113.10% | 279 | 233 |

**origami top 3 tiles, every kind**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 16% (41/shape) | 0.00% | 5.11% | 22.59% | 75 | 58 |
| heldout | 617 | 21% (12/shape) | 0.00% | 0.00% | 13.31% | 41 | 36 |
| worst10 | 60 | 26% (36/shape) | 0.00% | 4.00% | 8.39% | 16 | 12 |
| **all** | 1027 | 18% (24/shape) | 0.00% | 2.10% | 22.59% | 132 | 106 |

**origami top 5 tiles, every kind**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 27% (67/shape) | 0.00% | 0.43% | 19.54% | 32 | 25 |
| heldout | 617 | 36% (21/shape) | 0.00% | 0.00% | 8.03% | 18 | 14 |
| worst10 | 60 | 42% (59/shape) | 0.00% | 0.45% | 4.18% | 6 | 6 |
| **all** | 1027 | 30% (39/shape) | 0.00% | 0.00% | 19.54% | 56 | 45 |

**origami top 8 tiles, every kind**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 41% (105/shape) | 0.00% | 0.00% | 9.53% | 12 | 8 |
| heldout | 617 | 57% (34/shape) | 0.00% | 0.00% | 7.50% | 13 | 11 |
| worst10 | 60 | 64% (89/shape) | 0.00% | 0.00% | 3.02% | 4 | 4 |
| **all** | 1027 | 47% (61/shape) | 0.00% | 0.00% | 9.53% | 29 | 23 |

**origami top 5 tiles, coarse n**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 22% (57/shape) | 0.00% | 1.84% | 19.54% | 44 | 34 |
| heldout | 617 | 36% (21/shape) | 0.00% | 0.00% | 8.03% | 18 | 14 |
| worst10 | 60 | 38% (53/shape) | 0.00% | 2.98% | 5.20% | 15 | 12 |
| **all** | 1027 | 27% (35/shape) | 0.00% | 0.05% | 19.54% | 77 | 60 |

**screen within 10%, then coarse n**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 37% (93/shape) | 0.00% | 0.00% | 11.77% | 18 | 13 |
| heldout | 617 | 60% (36/shape) | 0.00% | 0.00% | 1.73% | 2 | 0 |
| worst10 | 60 | 54% (76/shape) | 0.00% | 1.92% | 5.04% | 9 | 6 |
| **all** | 1027 | 44% (58/shape) | 0.00% | 0.00% | 11.77% | 29 | 19 |

**no pdp512 / pdp768 / pdp1024**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 97% (247/shape) | 0.00% | 0.00% | 1.76% | 1 | 0 |
| heldout | 617 | 88% (52/shape) | 0.00% | 0.00% | 4.65% | 2 | 1 |
| worst10 | 60 | 95% (133/shape) | 0.00% | 0.00% | 0.00% | 0 | 0 |
| **all** | 1027 | 95% (123/shape) | 0.00% | 0.00% | 4.65% | 3 | 1 |

**no two-tile or spread**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 92% (234/shape) | 0.00% | 0.00% | 5.00% | 7 | 4 |
| heldout | 617 | 90% (53/shape) | 0.00% | 0.00% | 11.92% | 24 | 18 |
| worst10 | 60 | 89% (125/shape) | 0.00% | 0.00% | 8.33% | 4 | 1 |
| **all** | 1027 | 91% (119/shape) | 0.00% | 0.00% | 11.92% | 35 | 23 |

**no cap or all (rules that repeat other points' launches)**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 93% (236/shape) | 0.00% | 0.37% | 5.14% | 10 | 5 |
| heldout | 617 | 78% (46/shape) | 0.00% | 1.91% | 19.21% | 95 | 57 |
| worst10 | 60 | 83% (116/shape) | 0.00% | 0.11% | 2.41% | 1 | 1 |
| **all** | 1027 | 88% (115/shape) | 0.00% | 1.03% | 19.21% | 106 | 63 |

**pipeline: duplicates removed, tile screen 20%, second round within 5%**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 23% (58/shape) | 0.00% | 0.30% | 5.14% | 10 | 6 |
| heldout | 617 | 33% (19/shape) | 0.00% | 0.46% | 2.23% | 16 | 2 |
| worst10 | 60 | 30% (43/shape) | 0.00% | 0.11% | 0.97% | 0 | 0 |
| **all** | 1027 | 26% (34/shape) | 0.00% | 0.40% | 5.14% | 26 | 8 |

**pipeline: duplicates removed, tile screen 10%, second round within 5%**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 18% (45/shape) | 0.00% | 0.43% | 11.77% | 13 | 7 |
| heldout | 617 | 31% (18/shape) | 0.00% | 0.48% | 2.23% | 17 | 2 |
| worst10 | 60 | 24% (34/shape) | 0.00% | 0.11% | 0.97% | 0 | 0 |
| **all** | 1027 | 22% (28/shape) | 0.00% | 0.43% | 11.77% | 30 | 9 |

**pipeline: duplicates removed, tile screen 20%, second round within 3%**

| dataset | shapes | points kept | loss median | 90th | max | > 1% | > 2% |
|---|---:|---:|---:|---:|---:|---:|---:|
| refit | 350 | 22% (57/shape) | 0.00% | 0.35% | 5.14% | 11 | 6 |
| heldout | 617 | 32% (19/shape) | 0.00% | 0.48% | 2.23% | 17 | 2 |
| worst10 | 60 | 30% (42/shape) | 0.00% | 0.11% | 0.97% | 0 | 0 |
| **all** | 1027 | 26% (33/shape) | 0.00% | 0.43% | 5.14% | 28 | 8 |

**fastest kind (fp32 partials), all datasets:** cap 377, dp 185, rsN 132, pdp 118, rsall 88, sp 47, owN 33, 2t 27, owall 9, pdp512 7, all 4