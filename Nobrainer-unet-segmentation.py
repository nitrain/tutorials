## Example: Train aparc segmentation model using nitrain with UNET architecture from nobrainer

import nitrain
from nitrain.readers import ColumnReader, ComposeReader
from nitrain import transforms as tx

# note: have to convert mgz to nii.gz manually because mgz is not supported by ntimage yet
csv_path = '/Users/ni5875cu/Desktop/nobrainer/filepaths.csv'

dataset = nitrain.Dataset(
    inputs=ColumnReader(csv_path, 'features', is_image=True),
    outputs=ColumnReader(csv_path, 'labels', is_image=True),
    transforms={
        ('inputs','outputs'): [tx.Resample((128,128,128))],
        'outputs': [tx.Cast('uint8'),
                    tx.CustomFunction(lambda img: (img == 4) | (img == 44))]
    }
)

x, y = dataset[0]

loader = nitrain.Loader(dataset,
                        images_per_batch=2)


from nobrainer.models import unet
model = unet(n_classes=1,
             input_shape=(256,256,256,1))


trainer = nitrain.Trainer(model,
                          task='segmentation')
trainer.fit(loader)