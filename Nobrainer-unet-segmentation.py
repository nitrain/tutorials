## Example: Train aparc segmentation model using nitrain with UNET architecture from nobrainer

import os
import nitrain
from nitrain.readers import ColumnReader, ComposeReader
from nitrain import transforms as tx

# note: have to convert mgz to nii.gz manually because mgz is not supported by ntimage yet
csv_path = os.path.expanduser('~/Desktop/nobrainer/filepaths.csv')

dataset = nitrain.Dataset(
    inputs=ColumnReader(csv_path, 'features', is_image=True),
    outputs=ColumnReader(csv_path, 'labels', is_image=True),
    transforms={
        ('inputs','outputs'): [tx.Resample((128,128,128))],
        'outputs': [tx.Cast('uint8'),
                    tx.CustomFunction(lambda img: (img == 4) | (img == 44))]
    }
)

# optional: get an example input + output from the dataset
x, y = dataset[0]

loader = nitrain.Loader(dataset,
                        images_per_batch=4)

#  optional: get an example batch from the loader
xbatch, ybatch = next(iter(loader))

from nobrainer.models import unet
model = unet(n_classes=1,
             input_shape=(128,128,128,1))

trainer = nitrain.Trainer(model,
                          task='segmentation')

# optional: see information about how the trainer is compiled
print(trainer)

trainer.fit(loader, epochs=5)