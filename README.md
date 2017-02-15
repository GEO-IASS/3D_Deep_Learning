# 3D Deep learning

This repository contains a varied set of resources to perform deep learning on three-dimensional data.

## I-O 

Under this directory you can find read and write functions for specific 3D formats and generic array formats:

- [.mat](https://es.mathworks.com/help/matlab/import_export/mat-file-versions.html)
- [.npy /.npz](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html)
- [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
- [.off](https://en.wikipedia.org/wiki/OFF_(file_format))
- [.ply](https://en.wikipedia.org/wiki/PLY_(file_format))

## Datasets

Under this directory you can find some "toy" 3D datasets created by myself.

- 3DMNIST
    
A 3D version of the MNIST database of handwritten digits. This dataset can also be found [at kaggle](https://www.kaggle.com/daavoo/3d-mnist).

## Feature vectors

Raw 3D data (point clouds, meshes) don't present the required structure to be processed by traditional convolutional neural networks. 

Under this directory you can find different utilities to transform raw 3D data into 3D feature vectors ready to serve as input to 3D Convolutional neural networks.

The usual procedures to transform raw 3D data into 3D feature vectors are:

- Point cloud -> compute voxelgrid -> compute feature vector 
- Mesh -> convert to point cloud by sampling -> compute voxelgrid -> compute feature vector
- Mesh -> compute feature vector (only distance function based are possible)

