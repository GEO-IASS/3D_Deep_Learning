# 3DMNIST

A 3D version of the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

You can also find and download the [dataset at Kaggle](https://www.kaggle.com/daavoo/3d-mnist)

This dataset contains 3D point clouds generated from the original images of the MNIST dataset to bring a familiar introduction of 3D for people used to work with 2D datasets (images).

# Content

You can use `img_to_pointcloud.ipynb` to generate the dataset from the original images. Following this notebook will let you choose the number of point cloud you wish to generate. Default is as many point clouds as images in the original dataset.

You can also download the dataset from [this link](https://mega.nz/#!LAZmXZKB).

The dataset's format is ready to be used by `dataset_to_feature_vector.py`.

```
data/
    train/
        0/
            0.npy
            1.npy
            ...
        ...

        9/
            0.npy
            ...
    test/
        0/
            0.npy
            1.npy
            ...
        ...

        9/
            0.npy
            ...
    valid/
        0/
            0.npy
            1.npy
            ...
        ...

        9/
            0.npy
            ...
```

# Acknowledgements

- Website of the [original MNIST dataset](http://yann.lecun.com/exdb/mnist/)

