# 3D Deep learning

This repository contains a varied set of resources to make deep learning on three-dimensional data easier.

## Tools

Raw 3D data (point clouds, meshes) don't present the required structure to be processed by traditional convolutional neural networks. 

Under this directory you can find different utilities to transform raw 3D data into 3D feature vectors ready to serve as input to 3D Convolutional neural networks.

----

In `io_3D.py` you can find read and write functions for specific 3D formats and generic array formats:

- [.mat](https://es.mathworks.com/help/matlab/import_export/mat-file-versions.html)
- [.npy / .npz](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html)
- [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
- [.off](https://en.wikipedia.org/wiki/OFF_(file_format))
- [.ply](https://en.wikipedia.org/wiki/PLY_(file_format))

---

The usual procedures to transform raw **3D data into 3D feature vectors** are:

- Point cloud -> compute voxelgrid -> compute feature vector 
- Mesh -> convert to point cloud by sampling -> compute voxelgrid -> compute feature vector

Functions for this pipeline are inside `mesh_to_pointcloud.py` and `pointcloud_to_feature_vector.py`

----

You can use `dataset_to_feature_vector.py` to compute a new version of the dataset by transforming the original data into 3D feature vectors; ready to be used by `generator3D.py`.

```
python dataset_to_feature_vector.py -h

usage: dataset_to_feature_vectors.py [-h] [--n_sampling N_SAMPLING]
                                     [--n_voxelgrid N_VOXELGRID]
                                     [--size_voxelgrid SIZE_VOXELGRID]
                                     [--mode MODE]
                                     input_path output_path

Convert input dataset to feature vectors. Check function docstring for more
info.

positional arguments:
  input_path            Path to the input dataset.
  output_path           Path where the new dataset of feature vectors will be
                        written.

optional arguments:
  -h, --help            show this help message and exit
  --n_sampling N_SAMPLING
                        The number of points that will be sampled from the
                        mesh, in case mesh exists.
  --n_voxelgrid N_VOXELGRID
                        The number of voxels per axis.
  --size_voxelgrid SIZE_VOXELGRID
                        The desired ouput voxel size. The number of voxels
                        will be infered.
  --mode MODE           Type of feature vector to be computed from voxelgrid.
```

The expected format for the input dataset is:

```
input_path/
    train/
        class_1/
            class_1_001.ply  # .ply or any of the formats inside io_3D
            class_1_002.ply
            ...
        class_2/
            class_2_001.ply
            class_2_002.ply
            ...

    test/
        class_1/
            class_1_001.ply
            class_1_002.ply
            ...
        class_2/
            class_2_001.ply
            class_2_002.ply
            ...
```

The output dataset will be:

```
output_path/
    train/
        class_1/
            class_1_001.npy  # this is a feature vector constructed with given parameters
            class_1_002.npy
            ...
        class_2/
            class_2_001.npy
            class_2_002.npy
            ...

    test/
        class_1/
            class_1_001.npy
            class_1_002.npy
            ...
        class_2/
            class_2_001.npy
            class_2_002.npy
            ...
```

## Datasets

Under this directory you can find some "toy" 3D datasets created by myself.

- 3DMNIST
    
A 3D version of the MNIST database of handwritten digits. This dataset can also be found [at kaggle](https://www.kaggle.com/daavoo/3d-mnist).

