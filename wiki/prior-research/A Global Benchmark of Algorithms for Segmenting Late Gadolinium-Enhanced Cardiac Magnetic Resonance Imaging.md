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
* direct segmentation and analysis of cardiac LGE-MRIs remain challenging due to the attenuated contrast in nondiseased tissue and imaging artifacts, as well as the varying quality of imaging
* Contrary to non-enhanced images, contrast-enhanced MRIs/CTs have received significantly less
attention for research despite their important role in clinics.

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

# 2018 LA Segmentation Challenge
* 3D LGE-MRI dataset was randomly split into training (N = 100) and testing (N = 54) sets, with the entire training set published at the start of the challenge for participants to develop their algorithms.
* Dice score was used as the only evaluation metric in the challenge for simplicity
* Subsequent analyses with the surface to surface distance and the Euclidean distance error of the LA diameter and volume measurements were conducted after the challenge.

# Results

Double CNN significantly outperform single 2D and 3D CNN methods in terms of Dice score