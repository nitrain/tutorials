## Overview of nitrain

The 10-minute overview presented below will take you through the key components of nitrain:

- [Datasets](#datasets)
- [Loaders](#loaders)
- [Architectures and pretrained models](#architectures-and-pretrained-models)
- [Model Trainers](#model-trainers)
- [Explainers](#explainers)

<br />

### Datasets

Datasets help you read in your images from wherever they are stored -- in a local folder, in memory, on a cloud service. You can flexibly specify the inputs and outputs using glob patterns, etc. Transforms can also be passed to your datasets as a sort of preprocessing pipeline that will be applied whenever the dataset is accessed.

```python
import nitrain as nt
from nitrain import transforms as tx

dataset = datasets.FolderDataset(x={'pattern': 'sub-*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                 y={'file': 'participants.tsv', 'column': 'age'},
                                 x_transforms=[tx.Resample((64,64,64))])
```

Although you will rarely need to do this, data can be read into memory by indexing the dataset:

```python
x_raw, y_raw = dataset[:3]
```

#### Readers

Notice that we used a `FolderReader` to specify that we wanted to read images from a local folder.

#### Fixed Transforms

You also saw that we passed in transforms to our dataset using a dictionary. We call these "fixed transforms" because they will only be applied once to your images (when they are first loaded from file) and their result never changes.

<br />

### Loaders

To prepare your images for batch generation during training, you pass the dataset into one the loaders. Here is where you can also pass in random transforms that will act as data augmentation. If you want to train on slices, patches, or blocks of images then you will additionally provide a sampler. The different samplers are explained later.

```python
from nitrain import loaders, samplers

loader = loaders.DatasetLoader(dataset,
                               images_per_batch=32,
                               x_transforms=[tx.RandomSmoothing(0, 1)])

# loop through all images in batches for one epoch
for x_batch, y_batch in loader:
        print(y_batch)
```

The loader can be be used directly as a batch generator to fit models in tensorflow, keras, pytorch, or any other framework.

#### Samplers

Samplers allow you to keep the same dataset + loader workflow that batches entire images and applies transforms to them, but then expand on those transformed image batches to create special "sub-batches".

For instance, samplers let you serve batches of 2D slices from 3D images, or 3D blocks from 3D images, and so forth. Samplers are essntial for common deep learning workflows in medical imaging where you often want to train a model on only parts of the image at once.

```python
from nitrain import loaders, samplers, transforms as tx
loader = loaders.DatasetLoader(dataset,
                               images_per_batch=4,
                               x_transforms=[tx.RandomSmoothing(0, 1)],
                               sampler=samplers.SliceSampler(batch_size=24, axis=2))
```

What happens is that we start with the ~190 images from the dataset, but 4 images will be read in from file at a time. Then, all possible 2D slices will be created from those 4 images and served in shuffled batches of 24 from the loader. Once all "sub-batches" (sets of 24 slices from the 4 images) have been served, the loader will move on to the next 4 images and serve slices from those images. One epoch is completed when all slices from all images have been served.

#### Random Transforms

The philosophy of nitrain is to be medical imaging-native. This means that all transforms are applied directly on images - specifically, `antsImage` types from the [ANTsPy](https://github.com/antsx/antspy) package - and only at the very end of batch generator are the images converted to arrays / tensors for model consumption.

The nitrain package supports an extensive amount of medical imaging transforms:

- Affine (Rotate, Translate, Shear, Zoom)
- Flip, Pad, Crop, Slice
- Noise
- Motion
- Intensity normalization

You can create your own transform with the `CustomTransform` class:

```python
from nitrain import transforms as tx

my_tx = tx.CustomTransform(lambda x: x * 2)
```

If you want to explore what a transform does, you can take a sample of it over any number of trials on the same image and then plot the results:

```python
import ants
import numpy as np
from nitrain import transforms as tx

img = ants.image_read(ants.get_data('r16'))

my_tx = tx.RandomSmoothing(0, 2)
imgs = my_tx.sample(img, n=12)

nt.plot_grid(imgs, shape=(4,3))
```

<br />

### Architectures and pretrained models

The nitrain package provides an interface to an extensive amount of deep learning model architectures for all kinds of tasks - regression, classification, image-to-image generation, segmentation, autoencoders, etc.

The available architectures can be listed and explored:

```python
from nitrain import models
print(models.list_architectures())
```

You first fetch an architecture function which provides a blueprint on creating a model of the given architecture type. Then, you call the fetched architecture function in order to actually create a specific model with you given parameters.

```python
from nitrain import models

vgg_fn = models.fetch_architecture('vgg', dim=3)
vgg_model = vgg_fn((128, 128, 128, 1))

autoencoder_fn = models.fetch_architecture('autoencoder')
autoencoder_model = autoencoder_fn((784, 500, 500, 2000, 10))
```

<br />

### Trainers

After you have created a model from a nitrain architecture, fetched a pretrained model, or created a model yourself in your framework of choice, then it's time to actually train the model on the dataset / loader that you've created.

Although you are free to train models on loaders using standard pytorch, keras, or tensorflow workflows, we also provide the `LocalTrainer` class to make training even easier. This class provides sensible defaults for key training parameters based on your task.

```python
trainer = trainers.LocalTrainer(model=vgg_model, task='regression')
trainer.fit(loader, epochs=10)

# access fitted model
print(trainer.model)
```

If you have signed up for an account at [nitrain.dev](https://www.nitrain.dev) then you can also train your model in the cloud using the `PlatformTrainer` class. All training takes place on HIPAA-compliant GPU servers with competitive pricing.

```python
trainer = trainers.PlatformTrainer(model=model, task='regression',
                                name='brain-age', resource='gpu-small')
trainer.fit(loader, epochs=10)

# check job status
print(trainer.status)
```

<br />

### Explainers

The idea that deep learning models are "black boxes" is out-dated, particularly when it comes to images. There are numerous techiques to help you understand which parts of the brain a trained model is weighing most when making predictions.

Nitrain provides tools to perform this techique - along with many others - and can help you visualize the results of such explainability experiments directly in brain space. Here is what that might look like:
