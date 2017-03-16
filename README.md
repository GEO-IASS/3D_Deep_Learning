# 3D Deep learning

Python module for making 3D Deep Learning easier.

threeDlearn content:

- load3D
- keras_generator
- transforms
- visualization
- models
- weights
- datasets

## load3D

Functions for reading several 3D file formats and generic array formats:

- [.mat](https://es.mathworks.com/help/matlab/import_export/mat-file-versions.html)

- [.npy / .npz](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html)
- [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
- [.off](https://en.wikipedia.org/wiki/OFF_(file_format))
- [.ply](https://en.wikipedia.org/wiki/PLY_(file_format))

## keras_generator

For generating augmented voxelizations of 3D data on-the-fly.

You can use `VoxelGridDataGenerator` as you would use kera's ImageDataGenerator:

```python
from threeDlearn.models import Voxnet
from threeDlearn.keras_generator import VoxelGridDataGenerator

model = Voxnet(10, weights="voxnet10.h5")

# Finetune last layer
for layer in model.layers[:-1]:
    layer.trainable = False
    
gen = VoxelGridDataGenerator(z_rotation_range=10)
train_batches = gen.flow_from_directory("3DMNIST/train")

model.fit_generator(train_batches, train_batches.samples // train_batches.batch_size)
```

Current supported augmentations:

```
x_rotation_range : float, optional (Default None)
    Rotation range in Degrees (0-180) along the x axis.
    Equivalent to 'Roll' in aircraft principal axis.
    
y_rotation_range : float, optional (Default None)
    Rotation range in Degrees (0-180) along the y axis.
    Equivalent to 'Pitch' in aircraft principal axis.
    
z_rotation_range : float, optional (Default None)
    Rotation range in Degrees (0-180) along the z axis.
    Equivalent to 'Yaw' in aircraft principal axis.

x_shift_voxel_range : uint, optional (Default None)
    Number of voxels to be shifted along x axis.
    
y_shift_voxel_range : uint, optional (Default None)
    Number of voxels to be shifted along y axis.
    
z_shift_voxel_range : uint, optional (Default None)
    Number of voxels to be shifted along z axis.

x_flip : bool, optional (Default False)
    Flip around x axis with random probability
    
y_flip : bool, optional (Default False)
    Flip around y axis with random probability

z_flip : bool, optional (Default False)
    Flip around z axis with random probability
```

## transforms

Functions used to generate augmented voxelgrids (see keras_generator above).

## visualization

Use `plot_feature_vector` to visualize a voxelgrid sliced along the "z" axis.

## models

Pre-defined models.

Currently avaliable:

- VoxNet ('VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.')

## datasets

- 3DMNIST

