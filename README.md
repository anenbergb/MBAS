# 3D Cardiac Structure Segmentation with Two Stage Cascaded Neural Networks

This project presents a two-stage cascaded neural network approach for automatic 3D semantic segmentation of cardiac structures in Late Gadolinium Enhancement MRI (LGE-MRI). The method was developed as a submission to the [Multi-class Bi-Atrial Segmentation (MBAS) Challenge at MICCAI 2024](https://codalab.lisn.upsaclay.fr/competitions/18516) where it achieved **3rd place**.

This project was carried out with guidance from [Peter Chang, MD](https://www.faculty.uci.edu/profile/?facultyId=6569) and the [Center for Artificial Intelligence in Diagnostic Medicine](https://www.caidm.som.uci.edu/).

![mbas-voxel51](https://github.com/user-attachments/assets/4e824dac-7fba-40a2-b0e4-13f5f58baf27)

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
   - trained on 3D patches sampled from the LGE-MRI volume because the full resolution 3D volume is too large to fit into GPU memory, especially when using deep 3D CNNs. 

### Final Model: Results

The table below summarizes the results of the proposed two stage cascaded model based on 5-fold cross-validation using Dice Similarity Coefficient (↑ better) and 95th percentile Hausdorff Distance (HD95 ↓ better). The final two-stage model's metrics were close to the ResEnc baseline.

| Model             | Wall Dice | RA Dice | LA Dice | Wall HD95 | RA HD95 | LA HD95 | Description |
|------------------|-----------|---------|---------|-----------|---------|---------|-------------|
| Two-stage Final   | 0.711     | 0.920   | 0.930   | 2.99     | 3.59    | 3.98    | Final cascaded model submitted to MBAS |
| ResEnc (M) baseline | 0.714   | 0.921  | 0.929  | 3.04  |  3.56  | 4.04 |  single stage nnU-Net ResEnc (M) baseline model |

## Experimental Design

### Dataset

- **MBAS Challenge Dataset**  
  - 70 training, 30 validation, 100 test images (held out)
  - Resolution: 44×640×640 or 44×576×576  
  - Voxel spacing: 2.5×0.625×0.625 mm

### Architectures Evaluated

- **nnU-Net ResEnc**: Residual U-Net within nnU-Net framework
- **MedNeXt**: ConvNeXt-based architecture adapted for medical segmentation


## Ablation Study Results

Extensive experiments were conducted to explore the impact of model architecture, training configuration, and post-processing strategies. 

2D vs. 3D Convolution
- 3D convolution improved spatial continuity and outperformed 2D slice-based models (Dice +1–2%, HD95 −0.5 mm).

Architecture and Size
- The ResEnc (M) model provided the best balance between segmentation quality and computational efficiency.
- Larger patch sizes, as used in ResEnc (L) and (XL), resulted in slightly better metrics but significantly higher GPU memory usage and training time.
- MedNeXt architectures were competitive but did not consistently outperform ResEnc models on this specific segmentation task.

Cascaded Architecture
- The two-stage cascaded approach separates the problem into two subtasks:
  - Stage 1 focuses on coarse localization of the atrial region (high recall).
  - Stage 2 performs fine-grained multi-class segmentation within the localized mask (high precision).
- While the cascaded model achieved similar metrics to the best single-stage baseline, theoretical analysis showed that improved binary masks from Stage 1 could significantly enhance final segmentation quality.

Post-Processing and Loss Masking
- Applying connected-component filtering in Stage 1 reduced large false-positive regions, leading to lower HD95 scores.
- Morphological dilation (radius = 2) applied to the binary mask improved coverage of atrial boundaries at the cost of slight Dice reduction.
- In Stage 2, masking the loss function using the Stage 1 binary mask enabled the network to focus only on the relevant foreground regions, improving learning efficiency and robustness.
-  Using ground truth masks in Stage 2 (simulating a perfect first stage) boosted Dice to 0.95 and HD95 to approximately 2.4 mm.

Training Strategies
- Different sample input patch sizes (e.g. 28x256x224) and voxel spacing (e.g. 25x0.97x0.97) were evaluated. 
- Foreground sampling probabilities of 25% or higher were critical for dealing with the extreme class imbalance in the data.
- Batch Dice loss and deep supervision contributed to improved convergence and stability.
- Data augmentation settings provided by nnU-Net (rotation, scaling, noise, contrast) worked well; custom augmentations provided no significant additional benefit.

Inference Optimization
- Changing the inference patch stride from 0.5 to 1.0 reduced runtime by approximately 43% per image on a NVIDIA RTX 4090 GPU.
- This change had negligible effect on Dice and HD95 scores, making it a practical speedup for deployment.

These experiments provide empirical justification for model choices and highlight areas for future improvement, particularly in enhancing Stage 1 localization performance.


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

This project developed a robust, empirically optimized two-stage cascaded neural network for automatic 3D segmentation of key atrial structures in LGE-MRI, including the right atrium cavity, left atrium cavity, and atrial walls. The approach follows a coarse-to-fine strategy: the first stage localizes the atrial region with high recall, and the second stage performs focused multi-class segmentation within that localized region to improve precision.

The final model was designed through systematic ablation studies, exploring architectural variations, training strategies, and post-processing techniques. While the two-stage approach achieved similar metrics to the strongest single-stage baseline (nnU-Net ResEnc), it offers a modular and theoretically extensible framework. Simulated experiments using perfect binary masks in Stage 1 suggest that improved localization could further enhance performance.

Despite strong results, the project faced several technical challenges:
- Thin-walled atrial tissue is particularly difficult to segment due to low contrast and partial volume effects in LGE-MRI.
- Anatomical ambiguity in the ground truth labels, especially at structure boundaries, posed limits to achievable precision and introduced inter-observer variability.

Nevertheless, the proposed solution achieved **3rd place** in the [Multi-class Bi-Atrial Segmentation (MBAS) Challenge](https://codalab.lisn.upsaclay.fr/competitions/18516) at MICCAI 2024 (STACOM workshop) and was included in the official benchmarking study paper, validating both its technical rigor and clinical relevance.

Future directions include enhancing Stage 1 accuracy, improving robustness to labeling noise, and integrating the segmentation system into full clinical pipelines for AF ablation planning.



## Installation 

```
pip install -e .[dev,notebook]
```
Also install the [MBAS branch of nnUNet](https://github.com/anenbergb/nnUNet/tree/mbas). The MBAS project relies on the `nnUNetv2` routines such as `nnUNetv2_train` and `nnUNetv2_find_best_configuration` to train the custom 3D CNN models defined in this repo.
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

## ML Training
The [./expr](expr) folder records the expansive experiments and ablation studies that were performed for this project.

The [mbas/tasks](mbas/tasks) directory defines scripts to convert labels between formats, run inference, generate tables to report the metrics, and render videos.

### Voxel51
The opensource [Voxel51](https://voxel51.com/) application can be used to visualize the ground truth and predicted segmentation masks overlayed on the 3D MRI volume.
```
python mbas/tasks/launch_fiftyone.py \
--data-dir /path/to/MBAS/dataset/with/videos \
--dataset-name mbas_videos \
-p /path/to/postprocessed/predictions \
--port 5151
```

`/path/to/MBAS/dataset/with/videos` refers to the dataset directory containing the ground truth 3D segmentation masks along with a video rendered from the axial perspective for each MRI volume using the [mbas/tasks/render_subject_videos.py](mbas/tasks/render_subject_videos.py) script.

