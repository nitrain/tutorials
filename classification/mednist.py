## Mednist clasification

import nitrain as nt
from nitrain import samplers, readers, transforms as tx
import ants
import numpy as np

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
dataset = dataset.select(1500, random=True)

## get example record
x, y = dataset[0]

# random split: 80% training, 10% validation, 10% testing
train_ds, test_ds, val_ds  = dataset.split((0.8, 0.1, 0.1), random=True)

# create loader with random augmentation transforms
train_loader = nt.Loader(train_ds,
                         images_per_batch=300,
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

## plot original images and augmented images
#ants.plot_grid(np.array([[ants.from_numpy(xb[i,0,:,:]), dataset[i][0]] for i in range(7)]).T)

###############
#### MODEL ####
###############
import torch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()


#### TRAINING ####
root_dir = '/Users/ni5875cu/Desktop/mednist_test/'

import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from monai.data import decollate_batch

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

from monai.transforms import (
    Activations,
    Compose,
    AsDiscrete
)
y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=6)])

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = torch.Tensor(batch_data[0]).to(device), torch.Tensor(batch_data[1]).long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.images_per_batch}, " f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.images_per_batch
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    torch.Tensor(val_data[0]).to(device),
                    torch.Tensor(val_data[1]).long().to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            writer.add_scalar("val_accuracy", acc_metric, epoch + 1)

print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
writer.close()



model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            torch.Tensor(test_data[0]).to(device),
            torch.Tensor(test_data[1]).long().to(device),
        )
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
            
print(classification_report(y_true, y_pred, digits=4))
