# 3D Deep learning

This repository contains a varied set of resources to make deep learning on three-dimensional data easier.

## tools

Raw 3D data (point clouds, meshes) don't present the required structure to be processed by traditional convolutional neural networks. 

Under this directory you can find different utilities to transform raw 3D data into 3D feature vectors ready to serve as input to 3D Convolutional neural networks.

-------

The usual procedures to transform raw **3D data into 3D feature vectors** are:

- Point cloud -> compute voxelgrid -> compute feature vector 
- Mesh -> convert to point cloud by sampling -> compute voxelgrid -> compute feature vector

In `data_3D_to_feature_vector.py` you can find all the functions nedded for the previous transformation pipelines.

------

You can use `convert_dataset_py` to compute a new version of the dataset by transforming the original data into 3D feature vectors; ready to be used by `generator3D.py`.

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

In `notebooks\rearrange_modelnet40` you can find an example of rearranging the original Modelnet40 dataset into this expected format.

The output dataset will be created at `output_path` with the same structure. All files will contain the feature vector information and will be saved in `.npy` format.

------

Under `io_3D` you can find read and write functions for specific 3D formats and generic array formats:

- [.mat](https://es.mathworks.com/help/matlab/import_export/mat-file-versions.html)
- [.npz](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html)
- [.obj](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
- [.off](https://en.wikipedia.org/wiki/OFF_(file_format))
- [.ply](https://en.wikipedia.org/wiki/PLY_(file_format))

## Datasets

Under this directory you can find some "toy" 3D datasets created by myself.

- 3DMNIST
    
A 3D version of the MNIST database of handwritten digits. This dataset can also be found [at kaggle](https://www.kaggle.com/daavoo/3d-mnist).

