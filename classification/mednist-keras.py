## MedNIST clasification with a Keras model from ANTsPyNet

# About:
# MedNIST is a collection of 2d images from 6 different modalities / image types
# where the goal is to predict the class from the image. 

# The folder structure looks like this:
# AbdomenCT/
#    000000.jpeg
#    000001.jpeg
#    ...
# BreastMRI/
#    000000.jpeg
#    000001.jpeg
#    ...
# ...

import nitrain as nt
from nitrain import readers, transforms as tx

## Download data and unzip from this link:
## https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz

# set path to download
mednist_path = '~/Downloads/MedNIST/'

# create dataset with image as input and folder name (class) as output
dataset = nt.Dataset(inputs=readers.ImageReader('*/*.jpeg'),
                     outputs=readers.FolderNameReader('*/*.jpeg', format='onehot'),
                     base_dir=mednist_path,
                     transforms={
                         'inputs': tx.RangeNormalize(0, 1)
                     })

## optional: select 1000 random records to have a smaller dataset
dataset = dataset.select(1000, random=True)

## optional: get example record by indexing the dataset
x, y = dataset[0]

# random split: 80% training, 10% validation, 10% testing
train_ds, test_ds, val_ds  = dataset.split((0.8, 0.1, 0.1), random=True)

# create batch generator with random augmentation transforms
train_loader = nt.Loader(train_ds,
                         images_per_batch=100,
                         shuffle=True,
                         transforms={
                             'inputs': [tx.RandomRotate(-15, 15, p=0.5),
                                        tx.RandomFlip(axis=0),
                                        tx.RandomZoom(0.9, 1.1, p=0.5)]
                         })

# create loaders for test and validation data without the random transforms
test_loader = train_loader.copy(test_ds, drop_transforms=True)
val_loader = train_loader.copy(val_ds, drop_transforms=True)

## optional: get example batch by calling next on iterator
xb, yb = next(iter(train_loader))

# fetch architecture from antspynet
architecture_fn = nt.fetch_architecture('alexnet', dim=2)
model = architecture_fn(input_image_size=(64,64,1),
                        number_of_outputs=6,
                        number_of_dense_units=512)

# create trainer
trainer = nt.Trainer(model, task='classification')

# fit model
val_results = trainer.fit(train_loader, epochs=5, validation=val_loader)

# evaluate test dataset
test_results = trainer.evaluate(test_loader)