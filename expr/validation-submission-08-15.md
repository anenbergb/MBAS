# Validation set submission for 08/15/2024

I went with a cascaded 2-stage model for the validation set submission.
### 1st stage model
For the first stage, I used `mbasTrainer__nnUNetResEncUNetMPlans_2024_08_10__3d_lowres` model. 
This was trained for all 5 cross-validation folds such that the inference could be run on all of the images in the training dataset.
The model used in the final submission was trained on the "full" training dataset.

The Dice and HD95 results across all 70 training samples was
```
|             |   Average |       STD |
|-------------|-----------|-----------|
| DSC_atrium  |  0.930169 | 0.0184425 |
| HD95_atrium |  3.9864   | 1.92598   |
```
