# Hippocampus multi-label segmentation of 3D MRI images with Pytorch and Nitrain

# This example shows how to fit a segmentation model on 3D images with Pytorch
# and Nitrain. It uses 394 cropped images of the area around the hippocampus ang
# ground-truth segmentations of the left and right hippocampus. This example 
# demonstrates a wide range of nitrain's capabilities such as transforms,
# predictors, and samplers.

import nitrain as nt
from nitrain import readers, samplers, transforms as tx
import ants

# First, download dataset from http://medicaldecathlon.com/dataaws/
# It is the "Task04_Hippocampus" dataset which is only 27 MB in size.
# The dataset looks like this:
# 
# imagesTr/ (training images)
#     hippocampus_001.nii.gz
#     ...
# imagesTs/ (test images; no ground-truth labels given)
#     hippocampus_256.nii.gz
#     ...
# labelsTr/ (ground-truth labelsfor training images; vals: 0, 1, 2)
#     hippocampus_001.nii.gz
#     ...

# create dataset
# transforms include:
# - converting output segmentation to integer type
# - reorienting and resampling all images to a common space
# - standard normalizing input image
base_dir = '~/Downloads/Task04_Hippocampus'
dataset = nt.Dataset(inputs=readers.ImageReader('imagesTr/*.nii.gz'),
                     outputs=readers.ImageReader('labelsTr/*.nii.gz'),
                     base_dir=base_dir,
                     transforms={
                         'outputs': tx.Astype('uint8'),
                         ('inputs','outputs'): [tx.Reorient('RAS'),
                                                tx.Resample((40,60,40), use_voxels=True)],
                         'inputs': tx.StandardNormalize()
                     })

# optional: get example record and visualize it
x, y = dataset[0]
#ants.plot(x, y, overlay_alpha=0.7)

loader = nt.Loader(dataset,
                   images_per_batch=5,
                   channels_first=None)

xb, yb = next(iter(loader))