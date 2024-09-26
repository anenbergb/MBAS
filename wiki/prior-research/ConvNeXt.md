# A ConvNet for the 2020s
References:
* https://arxiv.org/abs/2201.03545

<img width="575" alt="Screenshot 2024-09-25 at 5 14 13 PM" src="https://github.com/user-attachments/assets/94d3f7df-fcee-4d92-bae1-c7b029b624a0">

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
