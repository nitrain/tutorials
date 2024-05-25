## MedNIST clasification with a Pytorch model from MONAI

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

## download data and unzip from this link:
## https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz

# set path to download
mednist_path = '~/Downloads/MedNIST/'

# create dataset with image as input and label as output
dataset = nt.Dataset(inputs=readers.ImageReader('*/*.jpeg'),
                     outputs=readers.FolderNameReader('*/*.jpeg', format='integer'),
                     base_dir=mednist_path,
                     transforms={
                         'inputs': tx.RangeNormalize()
                     })

# select 1500 random records to have a smaller dataset
dataset = dataset.select(1000, random=True)

## get example record
x, y = dataset[0]

# random split: 80% training, 10% validation, 10% testing
train_ds, test_ds, val_ds  = dataset.split((0.8, 0.1, 0.1), random=True)

# create loader with random augmentation transforms
train_loader = nt.Loader(train_ds,
                         images_per_batch=100,
                         channels_first=True,
                         shuffle=True,
                         transforms={
                             'inputs': [tx.RandomRotate(-15, 15, p=0.5),
                                        tx.RandomFlip(axis=0),
                                        tx.RandomZoom(0.9, 1.1, p=0.5)]
                         })

# create loaders for test and validation data without the random transforms
test_loader = train_loader.copy(test_ds, drop_transforms=True)
val_loader = train_loader.copy(val_ds, drop_transforms=True)

## get example batch
xb, yb = next(iter(train_loader))

# create model
import torch
from monai.networks.nets import DenseNet121
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6)

# create metrics
def auc_metric(y, ypred):
    from monai.data import decollate_batch
    from monai.transforms import (
        Activations,
        Compose,
        AsDiscrete
    )
    from monai.metrics import ROCAUCMetric
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=6)])
    metric_fn = ROCAUCMetric()
    y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
    y_pred_act = [y_pred_trans(i) for i in decollate_batch(ypred)]
    metric_fn(y_pred_act, y_onehot)
    result = metric_fn.aggregate()
    metric_fn.reset()
    return result

def acc_metric(y, ypred):
    acc_value = torch.eq(ypred.argmax(dim=1), y)
    acc_metric = acc_value.sum().item() / len(acc_value)
    return acc_metric

# create trainer
from nitrain.trainers import TorchTrainer
trainer = TorchTrainer(model=model, 
                       loss=torch.nn.CrossEntropyLoss(),
                       optimizer=torch.optim.Adam(model.parameters(), 1e-5),
                       metrics=[auc_metric, acc_metric],
                       device='cpu')

# fit model
val_results = trainer.fit(train_loader, epochs=3, validation=val_loader)

# evaluate test dataset
test_results = trainer.evaluate(test_loader)