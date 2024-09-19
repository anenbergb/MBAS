# Paper tables
## First Stage Binary Segmentation Model Experiments

|    | model                                                                        |   Rank |   Avg_Rank |   DSC_atrium |   OVERLAP_atrium |   HD95_atrium |
|----|------------------------------------------------------------------------------|--------|------------|--------------|------------------|---------------|
|  8 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256_nblocks3 |      1 |          1 |     0.933403 |         0.940038 |       3.48028 |
|  7 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.25_M_16_256          |      2 |          2 |     0.932349 |         0.939705 |       3.58482 |
|  6 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_40_256_nblocks3  |      3 |          3 |     0.933596 |         0.935365 |       3.32731 |
|  4 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256           |      4 |          4 |     0.933949 |         0.934746 |       3.41017 |
|  1 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres                    |      5 |          5 |     0.934025 |         0.934561 |       3.39874 |
|  5 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.0_M_16_256_nblocks3  |      6 |          6 |     0.933803 |         0.934545 |       3.4339  |
|  0 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_fullres                   |      7 |          7 |     0.932549 |         0.930944 |       3.51358 |
| 10 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256_nblocks3  |      8 |          8 |     0.931221 |         0.929839 |       3.54931 |
|  9 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__lowres1.5_M_16_256           |      9 |          9 |     0.930668 |         0.928931 |       3.63163 |
|  2 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256             |     10 |         10 |     0.472544 |         0.475942 |     137.721   |
|  3 | mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__fullres_M_32_256_nblocks3    |     11 |         11 |     0.473494 |         0.475621 |     119.312   |





5-fold cross validation
|    | model                                                                                       |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  9 | nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres                                           |      1 |    2.16667 |   0.713867 |     3.03584 |    0.920676 |      3.55649 |   0.929487 |     4.0353  |
|  0 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27  |      2 |    2.83333 |   0.711028 |     3.02579 |    0.92025  |      3.59335 |   0.929837 |     3.9958  |
|  1 | nnUNetTrainer_MedNeXt__MedNeXtPlans__3d_fullres                                             |      3 |    2.83333 |   0.717269 |     3.45771 |    0.920714 |      3.58138 |   0.927072 |     4.5996  |
|  8 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_fullres                                 |      4 |    3.33333 |   0.71543  |     3.40591 |    0.918461 |      3.64616 |   0.927906 |     4.43022 |
|  7 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres_from_only_atrium_orig01 |      5 |    5.16667 |   0.713451 |     3.40225 |    0.916274 |      3.78965 |   0.92633  |     4.6792  |
|  2 | nnUNetTrainer_250epochs__nnUNetPlans__3d_cascade_fullres_from_only_atrium                   |      6 |    6.33333 |   0.714103 |     3.57315 |    0.916853 |      3.87904 |   0.92456  |     4.73016 |
|  6 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres_from_only_atrium        |      7 |    6.83333 |   0.711161 |     3.5368  |    0.914908 |      3.94891 |   0.92515  |     4.69839 |
|  3 | nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres                                            |      8 |    7.16667 |   0.711427 |     3.64184 |    0.916289 |      3.68955 |   0.924737 |     4.86724 |
|  4 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__2d                                         |      9 |    9       |   0.68787  |     3.58382 |    0.909613 |      3.9713  |   0.921406 |     4.71905 |
|  5 | nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans__2d_cascade_fullres_from_only_atrium        |     10 |    9.33333 |   0.688681 |     3.60335 |    0.910178 |      3.95708 |   0.919544 |     4.91413 |

