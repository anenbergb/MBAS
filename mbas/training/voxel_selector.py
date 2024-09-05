import random
from collections import defaultdict
import numpy as np


class VoxelSelector:
    def __init__(self, z_coverage=False):
        self.sampled_z_indices = defaultdict(set)
        self.z_coverage = z_coverage

    def select_voxel(self, voxels_of_that_class, class_index=None):
        if self.z_coverage:
            all_z_indices = set(np.unique(voxels_of_that_class[:, 1]))
            available_indices = all_z_indices - self.sampled_z_indices[class_index]
            if len(available_indices) == 0:
                available_indices = all_z_indices
                self.sampled_z_indices[class_index] = set()
            available_indices = list(available_indices)

            selected_z = available_indices[np.random.choice(len(available_indices))]
            filtered_voxels = voxels_of_that_class[
                voxels_of_that_class[:, 1] == selected_z
            ]
            selected_voxel = filtered_voxels[np.random.choice(len(filtered_voxels))]
            self.sampled_z_indices[class_index].add(selected_z)
        else:
            selected_voxel = voxels_of_that_class[
                np.random.choice(len(voxels_of_that_class))
            ]
        return selected_voxel
