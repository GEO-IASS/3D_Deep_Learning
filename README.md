# 3D Deep learning

This repository contains a varied set of resources to make deep learning on three-dimensional data easier.


## Generators

You can use `VoxelGridDataGenerator` from threeDlearn.keras_generator.py as you would use kera's ImageDataGenerator.

```
from threeDlearn import VoxelGridDataGenerator

gen = VoxelGridDataGenerator(z_rotation_range=10)
train_batches = gen.flow_from_directory("your_train_directory")

your_model.fit_generator(train_batches, train_batches.samples / train_batches.batch_size)
```

