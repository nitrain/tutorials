## Example: Train aparc segmentation model using nitrain with UNET architecture from nobrainer

import nitrain as nt
from nitrain.samplers import SliceSampler
from nitrain.readers import ColumnReader
from nitrain import transforms as tx

# note: have to convert mgz to nii.gz manually because mgz is not supported by ntimage yet
csv_path = '/Users/ni5875cu/Desktop/nobrainer/filepaths.csv'

dataset = nt.Dataset(
    inputs=ColumnReader(csv_path, 'features', is_image=True),
    outputs=ColumnReader(csv_path, 'labels', is_image=True),
    transforms={
        ('inputs','outputs'): [tx.Resample((128,128,128))],
        'outputs': [tx.Astype('uint8'),
                    tx.CustomFunction(lambda img: (img == 4) | (img == 43))]
    }
)

# get an example input + output from the dataset
x, y = dataset[0]

# plot example with segmentation overlay
x.plot(y)

loader = nt.Loader(dataset,
                   images_per_batch=4,
                   sampler=SliceSampler(batch_size=24))

# get an example batch from the loader
xbatch, ybatch = next(iter(loader))


arch_fn = nt.fetch_architecture('unet', dim=2)
model = arch_fn((128,128,1), 
                number_of_outputs=1,
                number_of_layers=2, 
                number_of_filters_at_base_layer=12,
                mode='sigmoid')

trainer = nt.Trainer(model,
                     task='segmentation')

# see information about how the trainer is compiled
print(trainer)

trainer.fit(loader, epochs=5)


pred = model.predict(loader.to_keras())