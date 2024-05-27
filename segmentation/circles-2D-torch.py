import os
import ants
import nitrain as nt
from nitrain import readers, samplers, transforms as tx

# create example data - 40 image + segmentation pairs
base_dir = os.path.expanduser('~/Desktop/segmentation-example/')
if not os.path.exists(os.path.join(base_dir, 'img-1.nii.gz')):
    from monai.data import create_test_image_2d
    for i in range(40):
        img_arr, seg_arr = create_test_image_2d(128, 128, num_seg_classes=1)
        ants.from_numpy(img_arr).image_write(os.path.join(base_dir, f'img-{i}.nii.gz'))
        ants.from_numpy(seg_arr).image_write(os.path.join(base_dir, f'seg-{i}.nii.gz'))


# create dataset
dataset = nt.Dataset(readers.ImageReader('img-*.nii.gz'),
                     readers.ImageReader('seg-*.nii.gz'),
                     base_dir=base_dir,
                     transforms={
                         'inputs': tx.RangeNormalize()
                     })

train_dataset, test_dataset = dataset.split((0.8, 0.2), random=True)

## optional: read an example record
x, y = dataset[0]

# create loader with random augmentation transforms
# notice that the same random rotation is applied to the original and segmented image
train_loader = nt.Loader(train_dataset,
                         images_per_batch=4,
                         channels_first=True,
                         transforms={
                             ('inputs', 'outputs'): [tx.RandomRotate(-90, 90, p=1),
                                                     tx.RandomCrop((96,96))]
                         })

test_loader = train_loader.copy(test_dataset)

## optional: read an example batch
xb, yb = next(iter(train_loader))
    
## optional: plot sampled patches
#ants.plot_grid([ants.from_numpy(xb[i,:,:,0]) for i in range(4)])

# create model
import monai
import torch

model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

# create loss, optimizer, metrics
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
def dice_metric(ytrue, ypred):
    from monai.data import decollate_batch
    from monai.transforms import Activations, AsDiscrete, Compose
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    dice_metric = monai.metrics.DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    ypred_act = [post_trans(i) for i in decollate_batch(ypred)]
    ytrue_act = decollate_batch(ytrue)
    dice_metric(y_pred=ypred_act, y=ytrue_act)
    result = dice_metric.aggregate().item()
    dice_metric.reset()
    return result

# create trainer
from nitrain.trainers import TorchTrainer
trainer = TorchTrainer(model,
                       optimizer=optimizer,
                       loss=loss_function,
                       metrics=[dice_metric])

# fit model
results = trainer.fit(train_loader, epochs=50, validation=test_loader)

# evaluate model
test_results = trainer.evaluate(test_loader)