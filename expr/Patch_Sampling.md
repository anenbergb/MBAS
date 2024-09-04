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
