# 3D Cardiac Structure Segmentation with Two Stage Cascaded Neural Networks

This project presents a two-stage cascaded neural network approach for automatic 3D semantic segmentation of cardiac structures in Late Gadolinium Enhancement MRI (LGE-MRI). The method was developed as a submission to the Multi-class Bi-Atrial Segmentation (MBAS) Challenge at MICCAI 2024 where it achieved **3rd place**.

This project was carried out with guidance from [Peter Chang, MD](https://www.faculty.uci.edu/profile/?facultyId=6569) and the [Center for Artificial Intelligence in Diagnostic Medicine](https://www.caidm.som.uci.edu/).

## Task Description & Motivation

Late Gadolinium Enhancement (LGE) MRI is a specialized cardiac imaging modality frequently acquired in the management of patients with atrial fibrillation (AF). It enhances fibrotic myocardial tissue after the injection of a gadolinium-based contrast agent, making it particularly useful for visualizing structural heart abnormalities.

This project addresses the critical task of **automated 3D semantic segmentation** of the heart atria in LGE-MRI images, specifically the:
- **Right Atrium Cavity**
- **Left Atrium Cavity**
- **Left & Right Atrium Walls**

Accurate segmentation of these structures is essential for building patient-specific 3D anatomical models, which clinicians use for diagnostic review and planning of catheter ablation procedures.

The motivation for this task stems from the challenges posed by LGE-MRI:
- Attenuated contrast in non-diseased tissue
- Imaging artifacts and poor resolution
- Significant variability in atrial anatomy

These factors make manual segmentation labor-intensive and error-prone, highlighting the need for robust automated solutions—such as those developed for the MBAS Challenge at MICCAI 2024**.

## Clinical Relevance

Atrial fibrillation (AF) is a common cardiac arrhythmia that can be treated using catheter ablation. This minimally invasive procedure uses radiofrequency energy or cryotherapy to scar abnormal cardiac tissue, disrupting the errant electrical signals causing the arrhythmia.

Key roles of LGE-MRI in AF management include:

- Assessment of atrial fibrosis before treatment
- Evaluation of ablation lesion quality post-procedure
- Prediction of long-term success of ablation
- Guidance of patient-specific ablation strategies

The ablated (injured) tissue is replaced by fibrotic scar tissue, altering the electrical conduction pathways of the heart in a controlled manner.

However, to leverage the full potential of LGE-MRI, clinicians must first isolate atrial structures from complex 3D scans. High-quality segmentation enables:
- Volumetric and wall-thickness measurements
- Electrophysiological modeling
- Visualization for interventional planning

Thus, automated and accurate segmentation of the atria is critical for improving outcomes in AF treatment workflows.

## Methodology

### Final Model: Two-Stage Cascaded nnU-Net ResEnc

1. **Stage 1 (Localization)**  
   A lightweight nnU-Net Residual Encoder (ResEnc) model segments the atrial region as a binary mask optimized for high recall.
   - Post-processing: largest connected component selection + morphological dilation
   - Overlap: **0.988**, Dice: **0.825**, HD95: **6.57**

2. **Stage 2 (Refined Multi-Class Segmentation)**  
   A second ResEnc model performs detailed segmentation within the foreground mask from Stage 1.
   - Masked loss computation (Soft Dice + Cross Entropy)
   - Foreground-focused training with class-balanced sampling

3. **Architecture Highlights**  
   - 3D U-Net with residual connections
   - Deep supervision
   - Adaptive patch-based training with nnU-Net heuristics

## Experimental Design

### Dataset

- **MBAS Challenge Dataset**  
  - 70 training, 30 validation, 100 test images (held out)
  - Resolution: 44×640×640 or 44×576×576  
  - Voxel spacing: 2.5×0.625×0.625 mm

### Architectures Evaluated

- **nnU-Net ResEnc**: Residual U-Net within nnU-Net framework
- **MedNeXt**: ConvNeXt-based architecture adapted for medical segmentation

### Ablation Study Results

| Model         | Wall Dice | RA Dice | LA Dice | Wall HD95 | RA HD95 | LA HD95 |
|---------------|-----------|---------|---------|------------|----------|----------|
| ResEnc (M)    | 0.7235    | 0.9231  | 0.9305  | 2.88       | 3.40     | 3.76     |
| ResEnc (L)    | 0.7253    | 0.9257  | 0.9304  | 2.76       | 3.20     | 3.69     |
| MedNeXt       | 0.724     | 0.926   | 0.932   | 2.84       | 3.03     | 3.94     |
| Two-stage     | 0.711     | 0.920   | 0.930   | 3.03       | 3.61     | 3.99     |

### Key Observations

- 3D convolutions outperformed 2D slice-based models
- 100% foreground patch sampling improves performance
- Largest component filtering and binary dilation reduce HD95 significantly
- MedNeXt performed well but slightly under ResEnc

### Training Configuration

- **Epochs**: 1000  
- **Optimizer**: SGD (lr=1e-2, weight decay=3e-5)  
- **Loss**: Soft Dice + Cross Entropy  
- **Augmentations**: Rotation, scaling, noise, contrast, gamma  
- **Evaluation**: 5-fold cross-validation

## Inference Optimization

To accelerate inference:
- Increased patch stride from 0.5 to 1.0
- Reduced per-image runtime from **20.8s → 11.8s** (NVIDIA RTX 4090)
- Maintained similar segmentation performance

| Stride Size | Wall Dice | RA Dice | LA Dice | Wall HD95 | RA HD95 | LA HD95 |
|-------------|-----------|---------|---------|------------|----------|----------|
| 0.5         | 0.711     | 0.920   | 0.930   | 3.03       | 3.61     | 3.99     |
| 1.0         | 0.709     | 0.919   | 0.929   | 3.22       | 3.59     | 4.09     |

## Conclusion

This project developed a robust and modular two-stage cascaded CNN system for segmenting atrial structures from LGE-MRI. The proposed architecture:
- Achieved strong results on a challenging dataset
- Showed benefits of foreground-guided loss masking
- Demonstrated inference-time efficiency improvements

While the final two-stage model's metrics were close to the ResEnc baseline, the framework enables structured experimentation and future improvements—especially with enhanced binary localization in Stage 1.

## Installation 
```
# create virtual environment
pip install -e .[dev,notebook]
```
### Jupyter notebook development
```
jupyter nbextension install itkwidgets --user --py
jupyter nbextension enable itkwidgets --user --py

jupyter notebook
```
### Tensorboard
```
tensorboard --logdir=/path/to/logs --port 6006
```
