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

# create dataset with preprocessing transforms
base_dir = '~/Downloads/Task04_Hippocampus'
dataset = nt.Dataset(inputs=readers.ImageReader('imagesTr/*.nii.gz'),
                     outputs=readers.ImageReader('labelsTr/*.nii.gz'),
                     base_dir=base_dir,
                     transforms={
                         'outputs': tx.Astype('uint8'),
                         ('inputs','outputs'): [tx.Resample((40,60,40), use_voxels=True),
                                                tx.Reorient('RAS')],
                         'inputs': tx.StandardNormalize()
                     })

train_dataset, test_dataset = dataset.split((0.8, 0.2), random=True)

# optional: get example record and visualize it
x, y = dataset[0]
# ants.plot(x, y, overlay_alpha=0.7)

# create loader with a block sampler and random augmentation transforms
sampler = samplers.BlockSampler(block_size=(30,30,30), stride=(8,8,8), batch_size=20, shuffle=True)
train_loader = nt.Loader(train_dataset,
                         images_per_batch=4,
                         channels_first=True,
                         sampler=sampler,
                         transforms={
                             
                         })

test_loader = train_loader.copy(test_dataset, drop_transforms=True)

# optional: get example batch and visualize one record
xb, yb = next(iter(train_loader))

ants.plot(ants.from_numpy(xb[0,0,:,:,:]),
          ants.from_numpy(yb[0,0,:,:,:]),
          overlay_alpha=0.2, axis=1)


# create model and other settings
import torch
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete

model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)

# alternative: reduction="mean_batch"
dice_metric = DiceMetric(include_background=True, reduction="mean") 
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# create trainer
from nitrain.trainers import TorchTrainer
trainer = TorchTrainer(model,
                       optimizer=optimizer,
                       loss=loss_function,
                       metrics=[dice_metric])

results = trainer.fit(train_loader, epochs=20, validation=test_loader)