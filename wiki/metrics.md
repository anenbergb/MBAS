# Metrics

## Dice Score
A: 3D prediction\
B: 3D ground truth\
$$DICE(A,B) = \frac{2|A\cap B|}{|A| + |B|}$$

## IoU (Jaccard Index)
$$IoU(A,B) = \frac{|A\cap B|}{|A\cup B|}$$

## Sensitivity and Specificity
Sensitivity = Recall
$$Sensitivity = \frac{TP}{TP + FN}$$

$$Specificity = \frac{TN}{TN + FP}$$

$$Precision = \frac{TP}{TP + FP}$$

## Hausdorff Distance (HD)
* HD measures the local maximum distance between the surfaces of the predicted LA volume and the ground truth
* This metric evaluates geometrical characteristics, unlike the Dice or IoU which purely evaluates pixel-by-pixel comparisons. The 3D version of the HD was used in this study to measure the largest error distance of the 3D segmentation defined for a prediction of LA volume, A, and ground truth, B.\
a = all pixels within A\
b = all pixels within B

$$HD(A,B) = \max_{b\in B}\{\min_{a \in A}\{\sqrt{a^2 + b^2} \}\}\$$

$HD95$ = 95th percentile Hausdorff distance
* measures the maximum of the minimum distances between the predicted segmentation and the ground truth at the 95th percentile. The HD95 is a non-negative real number measured in millimeters, with a value of 0mm indicating a perfect prediction.
## STSD
STSD measures the average distance error between the
surfaces of the predicted LA volume and the ground truth\
$n_A$ = number of pixels in A\
$n_B$ = number of pixels in B\
$p$ = all points in A\
$p'$ = all points in B
$$STSD(A,B) = \frac{1}{n_A + n_B}\Bigl( \sum^{n_A}_{p=1}\sqrt{p^2 - B^2}+ \sum^{n_B}_{p'=1}\sqrt{p'^2 - A^2}\Bigr)$$

## Biological Measures for Evaluating Performance

LA diameter and volume are the two widely used clinical measures during the clinical diagnosis and treatment of patients with AF.

### Error of the LA anterior-posterior diameter
* The LA diameter, measured in millimeters, is calculated by finding the maximum Euclidean distance along the x-axis of each MRI scan to estimate the distance from the anterior LA to the posterior.

### Error of the 3D LA volume between predictions and ground truth
* The LA volume, measured in $cm^3$, is calculated by summing the total number of positive (LA cavity) pixels.