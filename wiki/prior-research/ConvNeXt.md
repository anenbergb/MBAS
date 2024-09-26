# A ConvNet for the 2020s
References:
* https://arxiv.org/abs/2201.03545

<img width="575" alt="first" src="https://github.com/user-attachments/assets/94d3f7df-fcee-4d92-bae1-c7b029b624a0">

## Abstract
* Swin Transformers (that reintroduced several ConvNet priors) and made Transformers practically viable as a generic vision backbone.
* The superior performance of Swin Transformers can probably be attributed to the inductive bias of convolutions, rather than the inherit superiority of Transformers. 
* They gradually "modernize" a ResNet towards the design of a vision Transformer 
* ConvNeXts compete favorably with Transformers in term sof accuracy and scalability, achieving 87.8% ImageNet Top-1 accuracy, outperforming Swin Transformers on COCO detection and ADE20k segmentation.
## Introduction
ConvNet's inductive biases
  * translation equivariance
  * inherently efficient due to sliding-window application. shared computation.
Vision Transformer (ViT)
* patchify initial layer
* generic Transformer backbone
* primary focus of ViT is on scaling behavior -- large models and dataset sizes.
* Transformers have superior scaling behavior -- multi-head self-attention being key component.
* The quadratic complexity with respect to input size = very slow, won't work for higher resolution inputs.
Transformer improvements
* Hierarchical Transformers. Sliding window
* Swin Transformers

ConvNext
* How do design decisions in Transformers impact ConvNets' performance?

<img width="551" alt="im2" src="https://github.com/user-attachments/assets/4beeb201-a0aa-4e16-83fa-70901b18d63d">

## Modernizing a ConvNet: a Roadmap
1. ResNet-50 model
2. Train it using the latest vision Transformer training techniques.
Apply various design decisions
1. Macro design
2. ResNeXt
3. Inverted bottleneck
4. Large kernel size
5. Various layer-wise micro design

## Training Techniques
2.7% percent of improvement comes from training techniques
* AdamW optimizer
* training recipe close to DeiT's and Swin Transformer's
* training extended to 300 epochs from the original  90 for ResNets
* data augmentation techniques such as Mixup, Cutmix, RandAugment, Random Erasing
* regularization such as Stochastic Depth
* label smoothing

## Macro Design
### Changing stage compute ratio
* Swin Transformer stage compute ratio 1:1:3:1, or 1:1:9:1.
* Adjust the number of ResNet blocks from (3,4,6,3) to (3,3,9,3).
78.8% -> 79.4%
### Changing stem to Patchify
* standard stem in ResNet is a 7x7 conv with stride 2, max pool -> total of 4x downsampling
* vision Transformer uses patchify stem which is like a large kernel 14x14 or 16x16 and non-overlapping convolution
* Swin Transformer uses 4x4 kernel
79.4% -> 79.5%
### ResNeXt-ify
* significantly reduce FLOPs by using grouped convolutions
* Depthwise convolutions. Similar to the weighted sum operation in self-attention which operates on per-channel basis, only mixing information in the spatial dimensinos
* Reduces FLOPs and the accuracy, so to compensate, increase the number of channels from 64 to 96.
79.5% -> 80.5%
### Inverted Bottleneck
* inverted bottleneck = hidden dimension of the MLP block is 4x wider thant he input dimension
* actually reduces # of FLOPs
80.5% -> 80.6%
### Large Kernel Size
* vision Transformers have non-local self-attention, which allows each layer to have a global receptive field.
* Swin Transformer local window size is still 7x7, which is bigger than the standard 3x3 kernel of ResNet
#### Moving up depthwise conv layer
* moving the larger kernels to the beginning of network is more efficient since there are fewer channels. This hurts performance slightly
#### Increasing the kernel size
* 79.9% -> 80.6% when going from 3x3 to 7x7 kernel
## Micro Design
### Replacing ReLU with GELU
GELU = Gaussian Error Linear Unit
Doesn't affect accuracy.
### Fewer activation functions
* Transformers have fewer activation functions
* better performance if only use a single GELU activation in each block. GELU between the two 1x1 layers
0.7% improvement from 80.6% to 81.3%
### Fewer normalization layers
* Leave one BN layer before conv 1x1 layer
81.3% to 81.4%
### Substituting BN with LN
* Layer Normalization is used in Transformers
* Use LN now
81.4% to 81.5%
### Separate downsampling layers
* Swin Transformers have separate downsampling layer between stages
* They try to add a 2x2 conv layer with stride 2 for spatial downsampling between stages (rather than using 3x3 conv with stride 2), and fiund that it works if you also add LN layer (normalization) whenever spatial resolution is changing.
81.5% to 82.0%


### Closing Remarks
* Demonstrate that a pure ConvNet can outperform the Swin Transformer for ImageNet-1k classification by modernizing the ConvNet by applying many of the vision Transformer design choices.
* The new ConvNet has approximately the same number of FLOPs, #params, throughput, and memory usage as Swin Transformer but doesn't require the specialized modules such as shifited window attention or relative position biases.

## Emperical Evaluation on Downstream Tasks
### Object detection and segmentation on COCO
Use Mask R-CNN and Cascade Mask R-CNN on COCO dataset with ConvNeXt backbones.
ConvNeXt is on-par or better than Swin Transformers
### Semantic segmentation on ADE20k
Use UperNet
ConvNeXt achieves competitive performance across different model capacities
### Remarks on model efficiency
* Under similar FLOPs, depthwise convs are known to be slower and consume more memory than ConvNets with dense convs
* However, ConvNeXt models are comparable speed to Swin Transformers
* ConvNeXt models require less memory than Swin Transformers
* Both ConvNeXt and Swin Transformers have more favorable accuracy-FLOPs trade-off than vanilla ViT due to the local computations. This improved efficiency is the "ConvNet inductive bias".

## Conclusion
* the prior widely held belief was that vision Transforemrs are more accurate, efficient, and scalable than ConvNets
* ConvNeXt model can compete favorably with state-of-the-art hierarchical vision Transformers (Swin) across multiple CV benchmarks
