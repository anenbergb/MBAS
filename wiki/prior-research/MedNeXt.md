# MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation

References:
* https://arxiv.org/pdf/2303.09975
* https://github.com/MIC-DKFZ/MedNeXt
* "A ConvNet for the 2020s" https://arxiv.org/pdf/2201.03545


## Abstract
* ConvNeXt architecture attempted to modernize the standard ConvNet by mirroring Transformer blocks.

MedNeXt
1. A fully ConvNeXt 3D Encoder-Decoder Network for medical image segmentation
2.  Residual ConvNeXt up and downsampling blocks to preserve semantic richness across scales
3. novel technique to iteratively increase kernel sizes by upsampling small kernel networks, to prevent performance saturation on limited medical data
4. Compound scaling at multiple levels (depth, width, kernel size) of MedNeXt

## Introduction
* Transformers are plagued by the necessity of large annotated datasets to maximize performance benefits owing to their limited inductive bias.

ConvNeXt architecture
*  inverted bottleneck mirroring that of Transformers. allows us to scale width (increase channels) while not being affected by kernel sizes.
* composed of a depthwise layer, an expansion layer and a contraction layer
* large depthwise kernels to replicate their scalability and long-range representation learning

Benefits from inverted bottleneck
1. learning long-range spatial dependencies via large kernels,
2. simultaneously scaling multiple network levels

Contributions of this paper
* Architecture composed purely of ConvNeXt block
* Residual Inverted Bottlenecks in place of regular up and downsampling blocks, to preserve contextual richness while resampling to 
benefit dense segmentation tasks. The modified residual connection in particular improves gradient flow during training.
* UpKern (technique to iteratively increase kernel size) to prevent performance saturation on large kernel MedNeXts
by initializing with trained upsampled small kernel networks.
* Compound Scaling multiple network parameters such as width (channels), receptive field (kernel size), and depth (number of layers)

## ConvNeXt 3D Segmentation Architecture
1. Depthwise Convolution Layer: This layer contains a Depthwise Convolution with kernel size k × k × k, followed by normalization, with C output channels. We use channel-wise GroupNorm for stability with small
batches instead of the original LayerNorm. The depthwise nature of
convolutions allow large kernels in this layer to replicate a large attention
window of Swin-Transformers, while simultaneously limiting compute and
thus delegating the “heavy lifting" to the Expansion Layer
2. Expansion Layer: Corresponding to a similar design in Transformers, this
layer contains an overcomplete Convolution Layer with CR output channels,
where R is the expansion ratio, followed by a GELU [12] activation. Large
values of R allow the network to scale width-wise while 1×1×1 kernel limits
compute. It is important to note that this layer effectively decouples width
scaling from receptive field (kernel size) scaling in the previous layer.
3. Compression Layer: Convolution layer with 1×1×1 kernel and C output
channels performing channel-wise compression of the feature maps.

![image](https://github.com/anenbergb/MBAS/assets/5284312/e0d1af4f-bb1a-4d55-8759-a80d10cb5517)
![image](https://github.com/anenbergb/MBAS/assets/5284312/3351befc-7ef0-44db-98db-34557ab469b2)

Resampling with Residual Inverted Bottlenecks\
UpKern: Large Kernel Convolutions without Saturation
* Want to avoid saturating kernel of size 7x7x7
* Borrow idea from Swin Transformer V2 - large attention-window network is initialized with another network trained with a
smaller attention window
* UpKern allows us to iteratively increase kernel size by initializing a large kernel network with a compatible pretrained small
kernel network by trilinearly upsampling convolutional kernels (represented as
tensors) of incompatible size.
All other layers with identical tensor sizes (including normalization layers) are initialized by copying the unchanged pretrained
weights. This leads to a simple but effective initialization technique for MedNeXt which helps large kernel networks overcome performance saturation in the
comparatively limited data scenarios common to medical image segmentation.

Compound Scaling of Depth, Width and Receptive Field
* simultaneous scaling on multiple levels (depth, width, receptive field, resolution etc)

## Experimental Design
1. Mixed precision training with PyTorch AMP
2. Gradient Checkpointing
3. nnUNet backbone
Training schedule - (epochs=1000, batches per epoch=250), inference (50% patch overlap) and data augmentation from nnUNet\
Optimizer  = AtomW\
Data is resampled to 1.0 mm isotropic spacing during training and inference (with results on original spacing) \
Patch size = 128 × 128 × 128 and 512 × 512, and batch size 2 and 14, for 3D and 2D networks respectively\
Learning rate = 0.001, except kernel:5 in KiTS19, which uses 0.0001 for stability\
Metrics =  Dice Similarity Coefficient (DSC) and Surface Dice Similarity (SDC) at 1.0mm tolerance for volumetric and surface accuracy\
5-fold cross-validation (CV)

Baseline models
* nnUNet
* 4 convolution-transformer hybrid networks with
transformers in the encoder (UNETR [9], SwinUNETR [8]) and in intermediate
layers (TransBTS [31], TransUNet [3] 2D network)
* fully transformer network (nnFormer)
* partially ConvNeXt network (3D-UX-Net [17])

## Datasets
1. Beyond-the-Cranial-Vault (BTCV) Abdominal CT Organ Segmentation - 30 CT volumes, 13 classes
2. AMOS22 Abdominal CT Organ Segmentation - 200 CT volumes, 15 classes
3. Kidney Tumor Segmentation Challenge 2019 Dataset (KiTS19) - 210 CT volumes, 2 classes
4. Brain Tumor Segmentation Challenge 2021 (BraTS21) - 1251 MRI volumes, 3 classes

# Ablation study
1. Residual Inverted Bottlenecks (MedNeXt-B Resampling vs Standard Resampling), specifically in Up and Downsampling layers is very important. Preserves semantic richness
2. UpKern improves performance in kernel 5 × 5 × 5 on both BTCV and AMOS22, whereas large kernel performance is indistinguishable from small kernels without it.
3. Performance boost in large kernels is seen to be due to the combination of UpKern with a larger kernel and not merely a longer effective training
schedule (Upkern vs Trained 2x), as a trained MedNeXt-B with kernel 3 ×
3 × 3 retrained again is unable to match its large kernel counterpart.

![image](https://github.com/anenbergb/MBAS/assets/5284312/27547893-bfcb-41f9-a537-47ec1afdd247)
