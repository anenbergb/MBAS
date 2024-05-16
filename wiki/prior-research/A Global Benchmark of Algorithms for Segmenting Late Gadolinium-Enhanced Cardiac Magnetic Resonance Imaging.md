# A Global Benchmark of Algorithms for Segmenting Late Gadolinium-Enhanced Cardiac Magnetic Resonance Imaging

# Reference
* https://pubmed.ncbi.nlm.nih.gov/33166776/
## Overview
This paper published by the challenge organizers compiles the results from the 2018 "Left Atrium Segmentation Challenge" using 154 LGE-MRI.\
LGE = Late Gadolinium-Enhanced Magnetic Resonance Imaging\
The best methods in the challenge were double CNNs
* 1st CNN for automatic region-of-interest localization (ROI)
* 2nd CNN for refined regional segmentation\
The top method achieved dice score 93.2% and mean surface to a surface distance of 0.7 mm.

# Background
* Segmentation is an important task for the quantitative analysis of medical images.
* Useful for delineation of a patient’s internal organ and tissue structure from 3D images, such as those obtained from computed tomography (CT) and magnetic resonance imaging (MRI), is often a necessity for medical diagnosis, patient stratification, and clinical treatment (Medrano-Gracia et al. 2015, Oakes et al. 2009, Csepe et al. 2017)
* gadolinium-based contrast-enhancing agents are used in a third of all MRI scans (LGE-MRIs) worldwide and are proved to be very effective in providing clinical diagnosis of cardiac diseases (Oakes et al. 2009, Higuchi et al. 2017, Hennig et al. 2017, Figueras i Ventura et al. 2018)
* direct segmentation and analysis of cardiac LGE-MRIs remain challenging due to the attenuated contrast in nondiseased tissue and imaging artifacts, as well as the varying quality of imaging. It is more challenging than segmenting non-contrast images due to the attenuated color contrast causing a lack of clarity between the atrial tissue and background pixels.
* Contrary to non-enhanced images, contrast-enhanced MRIs/CTs have received significantly less attention for research despite their important role in clinics.

# Dataset and Labels
LA = Left Atrial\
PV = Pulmonary Sleeves
* 154 independently acquired 3D LGE-MRIs from 60 de-identified patients with atrial
fibrillation were used in this challenge.
* All patients underwent LGE-MRI scanning to define the atrial structure and fibrosis distribution prior to and post-ablation treatment at Utah (McGann et al. 2014, McGann et al. 2011).
* High-resolution LGE-MRIs of bi-atrial chambers were acquired approximately 20-25 minutes after the injection of 0.1 mmol/kg gadolinium contrast (Multihance, Bracco Diagnostics Inc., Princeton, NJ) using a 3D respiratory navigated, inversion recovery prepared gradient echo pulse sequence. An inversion pulse was applied every heartbeat, and fat saturation was applied immediately before data acquisition. To preserve magnetization in the image volume, the navigator was acquired immediately after the data acquisition block. Typical scan times for the LGE-MRI study were between 8-15 minutes at 1.5 T and 6-11 minutes using the 3T scanner (for Siemens sequences) depending on patient respiration (McGann et al. 2014, McGann et al. 2011).
* The spatial resolution of one 3D LGE-MRI scan was 0.625×0.625×0.625 mm³ with spatial dimensions of either 576 × 576 × 88 or 640 × 640 × 88 pixels.
* The LA cavity volumes were manually segmented in consensus with three trained observers for each LGE-MRI scan using the Corview image processing software (Merrk Inc, Salt Lake City, UT) (McGann et al. 2014).
* The LA cavity was defined as the pixels contained within the LA endocardial surface including the mitral valve and LA appendage as well as an extent of the pulmonary vein (PV) sleeves. The endocardial surface border of the LA was segmented by manually tracing the LA and PV blood pool which were regions with enhanced pixel intensities in each slice of the LGE-MRI volume. The extent of the PV sleeves in the endocardial segmentations was limited to the PV antrum region, and was defined as the point where the PVs stopped narrowing and remained constant in diameter. On average, the PV antra were limited to less than 10 mm extending out from the endocardial surface, or approximately three times the thickness of the LA wall. The LGE-MRI image volumes and associated LA segmentations were stored in the nearly-raw raster data (nrrd) format. The LGE-MRIs were grayscale and the segmentation labels were binary.
* Less than 15% of the data was of high quality (SNR < 1), 70% of the data was of medium quality (SNR = 1 to 3), and over 15% of the data was of low quality (SNR > 3).
![overview-lge-mri](https://github.com/anenbergb/MBAS/assets/5284312/c9d25faa-6d79-4eb2-b38a-9287af04425b)
# 2018 LA Segmentation Challenge
* 3D LGE-MRI dataset was randomly split into training (N = 100) and testing (N = 54) sets, with the entire training set published at the start of the challenge for participants to develop their algorithms.
* Dice score was used as the only evaluation metric in the challenge for simplicity
* Subsequent analyses with the surface to surface distance and the Euclidean distance error of the LA diameter and volume measurements were conducted after the challenge.

# Results

Double CNN significantly outperform single 2D and 3D CNN methods in terms of Dice score
![Double-3D-CNN](https://github.com/anenbergb/MBAS/assets/5284312/0e5e6ee0-4fd7-4115-bf99-b4366d92db3c)

![Double-3D-CNN](https://github.com/anenbergb/MBAS/assets/5284312/b40a2d11-421e-4edb-91e5-ef144c74b30f)

3D U-Net architecture
* batch-normalization in each layer and residual connections along the length of the network
* first CNN detected the centroid of the ROI from a down-sampled version of the initial late gadolinium-enhanced magnetic resonance imaging (LGE-MRI).
* A 240×160×96 region centered in the LA cavity was the output and was then processed by the second CNN to segment LA in 3D.
* The output was padded to obtain the original resolution of the input LGE-MRI


Improvements to the baseline 3D U-Net
* residual connections - residual connections were added to each block of two to three sequential convolutional layers along the entire length of the networks to improve
gradient flow during backpropagation when training the CNNs. The type of residual connections varied from a simple connection without any additional operations to more advanced connections containing convolutional and pooling layers
  * extra residual connections increased Dice score by 0.7%
* 5x5x5 conv kernels (rather than 3x3x3) to increase receptive field size. Increased Dice score by 4%
* dense connections + dilated convolutions to increase receptive field of CNNs
* dense supervision training scheme
* customized losss function - ensemble of the Dice score, pixel thresholding to improve sensitivity, and an overlap metric for improving segmentations at boundary locations.
  * Dice loss improved accuracy by 2.1% over traditional cross-entropy loss, which doesn't account for the major class imbalance present in the dataset.
* dropout and parametric rectified linear unit (PReLU) increased Dice score by 0.5%
* Color-intensity normalization or CLAHE (contrast limited adaptive histogram equilization) improved Dice score by 0.7%
* standard data augmentation techniques - random rotation, elastic deformations, perspective scaling, and random flipping improved the performance by over 2%
* blurring, affine transformations, sheering did not improve performance.
* online data augmentation is better than offline.

## Key Factors Influencing the Performance of Double CNN Approaches
* It's very important that the 1st CNN crops the LA such that it's centered for the 2nd CNN.
  * Offsetting the patch reduces the Dice score.
* The smaller the image patch of ROI as the output of the 1st CNN, the higher the final segmentation accuracy. True for all ROI sizes that were greater than 240x160. Best ROI size was 240x160.
  * The decreased input size of ROIs generated from the first CNN of the double CNN methods reduced the class imbalance as there were significantly fewer background pixels present in the original input images, resulting in better performances as seen in our experiments using input sizes with X/Y dimensions of 240×160 to 400×400.
  * Dice accuracy of the CNN decreased when the size of the ROI was less than 240×160 even though the LA was fully contained within the ROI. The decreased performance is likely due to the boundary of the LA being too close to the edge of the ROI inputted into the CNN.
  * U-Net is known to perform poorly when segmenting boundary regions.
![factors](https://github.com/anenbergb/MBAS/assets/5284312/10c24947-bcca-4a2b-89a9-965fb888c143)
# Error Analysis
* The methods performed poorly when segmenting the regions containing the PVs located at the superior slices of the 3D LGE-MRIs and the mitral valve at the bottom connecting the LA with the left ventricle. The errors at the mitral valve were attributed to the fact that there are no clear landmarks to separate the two chambers. This leads the experts to label this region with a flat plane which potentially contains large inter-observer variability, making it difficult to be reproduced by the CNNs. On the other hand, the errors at the PVs could be explained by the fact that these structures are often very small in size and vary greatly in shape between patients, making them difficult to detect.
* low contrast images are challenging. accuracy for each LGE-MRI was directly correlated to the quality of the particular LGE-MRI measured in the signal-to-noise ratio for all approaches.
![error-analysis](https://github.com/anenbergb/MBAS/assets/5284312/8e644589-bd7d-4537-9f11-d00e5e3ff264)
# Next Steps
* More data -- Address some of the issues with an increased number of LGE-MRIs and the use of multi-center LGE-MRIs to improve the robustness of the CNNs on a greater diversity of datasets
* Extend current methodologies in this study to a concurrent multi-label problem,
such as the segmentation of both atrial chambers and cavities simultaneously.
