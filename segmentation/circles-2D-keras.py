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
                         transforms={
                             ('inputs', 'outputs'): [tx.RandomRotate(-90, 90, p=1),
                                                     tx.RandomCrop((96,96))]
                         })

test_loader = train_loader.copy(test_dataset)

## optional: read an example batch
xb, yb = next(iter(train_loader))
    
## optional: plot sampled patches
#ants.plot_grid([ants.from_numpy(xb[i,:,:,0]) for i in range(4)])

# create unet model in keras via antspynet
arch_fn = nt.fetch_architecture('unet', dim=2)
model = arch_fn((96,96,1), 
                number_of_outputs=1,
                number_of_layers=4,
                mode='sigmoid')

# create trainer
trainer = nt.Trainer(model,
                     task='segmentation')

# fit model
results = trainer.fit(train_loader, epochs=15, validation=test_loader)

# evaluate model
test_results = trainer.evaluate(test_loader)

# predict model
xb, yb = next(iter(train_loader))
yb_pred = trainer.predict(xb)

## optional: plot predictions
#ants.plot(ants.from_numpy(xb[0,:,:,0]), ants.from_numpy(yb_pred[0,:,:,0]), overlay_alpha=0.7)