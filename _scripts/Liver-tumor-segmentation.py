# DATASET #
from nitrain.datasets import Dataset
from nitrain.readers import PatternReader
from nitrain.samplers import SliceSampler
from nitrain.loaders import DatasetLoader
from nitrain.models import fetch_architecture
from nitrain.trainers import LocalTrainer
from nitrain import transforms as tx

base_dir = '~/Desktop/kaggle-liver-ct'
dataset = Dataset(inputs=PatternReader(base_dir, 'volumes/volume-{id}.nii'),
                  outputs=PatternReader(base_dir, 'segmentations/segmentation-{id}.nii'),
                  transforms=[
                      {['inputs','outputs']: [tx.Resample((128, 128, 64)),
                                              tx.Reorient('IPR')]},
                      {'inputs': [tx.RangeNormalize(0, 1)]},
                      {'outputs': [tx.CustomFunction(lambda x: x > 0),
                                   tx.SplitLabels()]}
                      ])
x, y = dataset[0]

# SAMPLER #
sampler = SliceSampler(batch_size=24, axis=0)

# TRANSFORMS #
x_transforms = []
y_transforms = []
co_transforms = []

# LOADER #
loader = DatasetLoader(dataset, 
                       images_per_batch=2,
                       x_transforms=x_transforms,
                       y_transforms=y_transforms,
                       co_transforms=co_transforms,
                       sampler=sampler)

x_batch, y_batch = next(iter(loader))

# MODEL #
arch_fn = fetch_architecture('unet', dim=2)
model = arch_fn((128, 128, 1), number_of_outputs=1)

# TRAINER #
trainer = LocalTrainer(model=model, task='classification')

# FIT #
trainer.fit(loader, epochs=20)