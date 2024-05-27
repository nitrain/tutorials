# Affine registration model fit on 2D images from MedNIST with GlobalNet in Pytorch

# This example shows how nitrain can be used to train a model to learn an affine
# transformation. Random augmentation transforms from nitrain are used to create
# an infinite number of affine transformed images that can be used for training.

import nitrain as nt
from nitrain import readers, transforms as tx
import ants

## download data and unzip from this link:
## https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz

# set path to download
mednist_path = '~/Downloads/MedNIST/'

# create dataset with image as input and randomly zoomed / rotated image as output
# notice that we give the inputs / outputs specific labels that we use in the transforms
dataset = nt.Dataset({'moving': readers.ImageReader('CXR/*.jpeg')},
                     {'fixed': readers.ImageReader('CXR/*.jpeg')},
                     base_dir=mednist_path,
                     transforms={
                         ('moving','fixed'): tx.RangeNormalize(),
                         'moving': [tx.RandomZoom(0.7, 1.3),
                                    tx.RandomRotate(-45, 45)]
                     }).select(300, random=True)

train_dataset, val_dataset = dataset.split((0.8, 0.2), random=True)

## optional: get example record
x, y = dataset[0]

## optional: plot example of original and transformed image
ants.plot_grid([x, y])

# create loader
train_loader = nt.Loader(train_dataset,
                         images_per_batch=50,
                         channels_first=True,
                         shuffle=True)

val_loader = train_loader.copy(val_dataset)

# create model
import torch
from torch.nn import MSELoss
from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp

model = GlobalNet(image_size=(64, 64), spatial_dims=2, 
                  in_channels=2, num_channel_initial=4, 
                  depth=2)
warp_layer = Warp("bilinear", "border")

optimizer = torch.optim.Adam(model.parameters(), 1e-5)
image_loss = MSELoss()

# train model
max_epochs = 20
epoch_loss_values = []
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()
        
        moving = torch.tensor(batch_data[0])
        fixed = torch.tensor(batch_data[1])
        ddf = model(torch.cat((moving, fixed), dim=1))
        pred_image = warp_layer(moving, ddf)

        loss = image_loss(pred_image, fixed)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


# predict on validation data
xb, yb = next(iter(val_loader))
moving = torch.tensor(batch_data[0])
fixed = torch.tensor(batch_data[1])
ddf = model(torch.cat((moving, fixed), dim=1))
pred_image = warp_layer(moving, ddf)


# visualize predictions
fixed_images = [ants.from_numpy(fixed.detach().numpy()[i,0,:,:]) for i in range(5)]
moving_images = [ants.from_numpy(moving.detach().numpy()[i,0,:,:]) for i in range(5)]
pred_images = [ants.from_numpy(pred_image.detach().numpy()[i,0,:,:]) for i in range(5)]

all_images = []
for i in range(5):
    all_images.append([fixed_images[i], moving_images[i], pred_images[i]])

ants.plot_grid(all_images)