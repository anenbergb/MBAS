# nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
distilling domain knowledge into 3 parameter groups
- fixed
- rule-based
- empirical parameters\
**dataset fingerprint** - standardized dataset representation comprising key properties such as image size, voxel spacing information or class ratios\
**pipeline fingerprint** - entirity of choices being made during method design\

- Larger batch sizes allow for more accurate gradient estimates and are thus preferable (up to a sweet spot typically not reached in our domain), but in practice any batch size larger than one already results in robust training.
- Larger patch size during training increases the contextual information absorbed by the network and is thus crucial for performance
- The topology of the network should be deep enough to guarentee an effective receptive field sie at least as large as the patch size, so that contextual information is not discarded.
- Initialize the patch size to median image shape and iteratively reduce it while adapting the network topology accordingly (including network depth, number and position of pooling operations along each axis, feature map sizes and convolutional kernel sizes) until the network can be trained with a batch size of at least two given GPU memory constraints.
- 3 different U-Net configurations
  - two-dimensional (2D) U-Net
  - 3D U-Net that operates at full image resolution
  - 3D U-Net cascade in which the first U-Net operates on downsampled images, and the second is trained to refine the segmentation maps created by the former at full resolution.
- nnU-Net empirically opts for ‘non-largest component suppression’ as a post-processing step if performance gains are measured
![image](https://github.com/anenbergb/MBAS/assets/5284312/b03c7b1f-d902-4fa0-8437-9995c3ff42e6)

# Discussion of methods applied for challenges such as KiTS
-  of the commonly used architectural modifications (for example, residual connections27,28, dense connections29,30, atten- tion mechanisms31 or dilated convolutions32,33) represented a neces- sary condition for good performance on the KiTS task.
