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

## optional: read an example record
x, y = dataset[0]

# create loader with random augmentation transforms
# notice that the same random rotation must be applied to the original and segmented image
loader = nt.Loader(dataset,
                   images_per_batch=4,
                   sampler=samplers.RandomPatchSampler(patch_size=(96,96),
                                                       patches_per_image=4,
                                                       batch_size=8),
                   transforms={
                       ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=1)
                   })

## optional: read an example batch
xb, yb = next(iter(loader))

## optional: plot sampled patches
ants.plot_grid([ants.from_numpy(xb[i,:,:,0]) for i in range(8)])