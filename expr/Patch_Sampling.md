# Patch Sampling during training
The neural network is trained on 3D patches sampled from the larger 3D MRI volume.
For example, the MRI volume may be of shape `(1, 44, 638, 638)`, but the neural network is trained on patches of size
`(1,20,256,256)`.

One of the training strategies that has proven effective has been to overample the patches centered on the ground truth
segmentation region, rather than randomly sampling patches from anywhere in the input volume.
This patch sampling strategy ensures there are non-zero segmentation values within the volume.
Sampled patches can be centered on points from any of the 3 segmentation regions -- atrium wall, left atrum cavity, or right atrium cavity.

A `(1,20,256,256)` sampled patch is quite large. Realistically, most sampled patches will contain most of the heart volume. 
![image](https://github.com/user-attachments/assets/17ccf07c-dff1-4187-af45-1caf22dd4127)


# Patch sampling that ensures full coverage of the depth (z) dimension
By default, the oversampling patch sampler will randomly sample patches centered at one of the predefined sampling points (the dots in the above figure). 
There area of the heart region is smaller for slices towards the edge of the 3d volume. For example, for a volume of shape (1,44,638,638), the area of the visible shape in slice 5 is significantly smaller than the area in slice 25.
Therefore, the random patch sampler is less likely to sample a patch centered on a slice with an index towards the edge of the 3d volume. 

I introduced a patch sampling policy to ensure that the depth (z) dimension is equally sampled by enforcing all slices are sampled.

|    | model                                                                                                         |   Rank |   Avg_Rank |   DSC_wall |   HD95_wall |   DSC_right |   HD95_right |   DSC_left |   HD95_left |
|----|---------------------------------------------------------------------------------------------------------------|--------|------------|------------|-------------|-------------|--------------|------------|-------------|
|  8 | nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres2                                                            |      9 |   14       |   0.725331 |     2.76333 |   0.92567   |      3.20032 |   0.930359 |     3.68754 |
|  9 | mbasTrainer__plans_2024_08_30__ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27                    |     10 |   15.3333  |   0.723445 |     2.72685 |   0.925043  |      3.21474 |   0.931448 |     3.71448 |
| 83 | mbasTrainer__plans_2024_09_04__ResEncUNet_p20_256_dil2_bd_zcov_cascade_ResEncUNet_08_27                       |     12 |   16.6667  |   0.722161 |     2.74292 |   0.925147  |      3.24111 |   0.931686 |     3.69314 |
| 85 | mbasTrainer__plans_2024_09_04__ResEncUNet_p16_192_dil2_bd_zcov_cascade_ResEncUNet_08_27                       |     20 |   23       |   0.721729 |     2.77263 |   0.924553  |      3.28032 |   0.931161 |     3.8619  |
| 27 | mbasTrainer__plans_2024_09_02__MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27             |     29 |   32.5     |   0.718305 |     2.80845 |   0.921939  |      3.31858 |   0.929709 |     4.06074 |
| 33 | mbasTrainer__plans_2024_09_04__MedNeXtV2_p16_256_dil2_bd_zcov_nblocks346_slim128_cascade_ResEncUNet_08_27     |     35 |   38.6667  |   0.71979  |     2.97324 |   0.920986  |      3.4197  |   0.929486 |     4.07476 |

Comparing `ResEncUNet_p20_256_dil2_batch_dice_cascade_ResEncUNet_08_27  vs. `ResEncUNet_p20_256_dil2_bd_zcov_cascade_ResEncUNet_08_27`.
- training standard sampling without forcing the full depth (z) dimension coverage appears to be better.
Comparing `ResEncUNet_p20_256_dil2_bd_zcov_cascade_ResEncUNet_08_27` vs. `ResEncUNet_p16_192_dil2_bd_zcov_cascade_ResEncUNet_08_27`
- I hypothesized that the smaller patch size would make it less likely for the model to overfit since there's increased diversity between small patches.
- Models trained on the samller patches performed worse.
Comparing `MedNeXtV2_p16_256_dil2_nblocks346_slim128_cascade_ResEncUNet_08_27` vs. `MedNeXtV2_p16_256_dil2_bd_zcov_nblocks346_slim128_cascade_ResEncUNet_08_27`
- Again, the standard sampling strategy performed better than the new z-dimension coverage sampling.

Unfortunately the new sampling policy doesn't seem to have improved performance.
