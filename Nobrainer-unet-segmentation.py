## Example: Train aparc segmentation model using nitrain with UNET architecture from nobrainer

import nitrain as nt
from nitrain.readers import ColumnReader, ComposeReader
from nitrain import transforms as tx

import nobrainer
from nobrainer.models import unet

#### Start data download ####

# download dataset using nobrainer
csv_path = nobrainer.utils.get_data()

# note: have to convert mgz to nii.gz manually because mgz is not supported by ntimage yet
# only have to do this once
if True:
    import pandas as pd
    import nibabel
    df = pd.read_csv(csv_path)

    for fname in df['features']:
        nibabel.save(nibabel.load(fname), fname.replace('.mgz','.nii.gz'))

    for fname in df['labels']:
        nibabel.save(nibabel.load(fname), fname.replace('.mgz','.nii.gz'))

    df['features'] = [f.replace('.mgz','.nii.gz') for f in df['features']]
    df['labels'] = [f.replace('.mgz','.nii.gz') for f in df['labels']]
    df.to_csv(csv_path)

#### Start nitrain code ####

dataset = nt.Dataset(
    inputs=ColumnReader(csv_path, 'features', is_image=True),
    outputs=ColumnReader(csv_path, 'labels', is_image=True),
    transforms={
        ('inputs','outputs'): [tx.Resample((128,128,128))],
        'outputs': [tx.Cast('uint8'),
                    tx.CustomFunction(lambda img: (img == 4) | (img == 43))]
    }
)

# optional: get an example input + output from the dataset
x, y = dataset[0]

# optional: print an image - it is an ntimage class
print(x)

# optional: plot an image with segmentation overlay
x.plot(y)

loader = nt.Loader(dataset,
                   images_per_batch=4,
                   sampler=nt.SliceSampler(axis=0, batch_size=24))

#  optional: get an example batch from the loader
xbatch, ybatch = next(iter(loader))

model = unet(n_classes=1,
             input_shape=(128,128,1))

trainer = nt.Trainer(model,
                     task='segmentation')

# optional: see information about how the trainer is compiled
print(trainer)

trainer.fit(loader, epochs=5)