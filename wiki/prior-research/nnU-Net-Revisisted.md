# nnU-Net Revisited:A Call for Rigorous Validation in 3D Medical Image Segmentation
Reference: https://arxiv.org/pdf/2404.09556

  
- Demonstrates that many recent architectures that claim superior performance over the U-Net baseline actually were not adequately tested. Inadequate baselines, insufficient datasets, neglected computational resources.
- Comprehensive benchmark of CNN-based, Transformer-based, and Mamba-based approaches.\
### Conclusion
The recipe for state-of-the-art performance is
1. CNN-based U-Net models like ResNet and ConvNeXt variants
2. using nnU-Net framework
3. scaling models to modern hardware resources
## Introduction
Identify a prevailing attention bias in medical image segmentation towards novel architectures. \
They call for a systemic change emphasizing rigorous validation practices
## Validation Pitfalls
### Baseline-related Pitfalls
1.  Coupling the claimed innovation with confounding performance boosters
- Add residual connections (extra architectural improvements) + claimed innovation, whereas baseline doesn't use residual connections
- Add additional training data + claimed innovation, whereas baseline doesn't
- Add self-supervised pretraining
- Use better hardware to train the model vs. the baseline\
#### Recommendation: Meaningful validation entirely isolates the effect of the claimed innovation by ensuring a fair comparison to baselines where the proposed method is not coupled with confounding performance booster

### Lack of well-configured and standardized baselines
