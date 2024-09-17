

# nnU-Net
- summarize the findings of nnU-Net Revisisted

# Two Stage Architecture
- decouple the task of localization and that of fine-grain classification
- train the 1st stage network to maximize the overlap
- train the 2nd stage architecture to maximize DICE and minimize HD95
- difference between two stage architecture loss, and nnU-Net style cascade where the 1st stage segmentation is provided as additional channel input to the network.
- negative results using cascaded model
- experiments training 2nd stage model using GT binary mask as 1st stage segmentation

# Model Architecture
- experimented with 2 primary architectures, nnU-Net ResEnc and MedNeXt.
- emperically found that nnU-Net ResEnc performed better. Used nnU-Net ResEnc for both 1st and 2nd stage architectures.
