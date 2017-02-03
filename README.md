# 3DMNIST

A 3D version of the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

You can also find and download an already [computed dataset at Kaggle](https://www.kaggle.com/daavoo/3d-mnist)

![3D MNIST](data/3Dmnist.png)

The aim of this dataset is to provide a simple way to get started with 3D computer vision problems such as 3D shape recognition.

Accurate [3D point clouds](https://en.wikipedia.org/wiki/Point_cloud) can (easily and cheaply) be adquired nowdays from different sources:

- RGB-D devices: [Google Tango](http://get.google.com/tango/), [Microsoft Kinect](https://developer.microsoft.com/en-us/windows/kinect), etc.
- [Lidar](https://en.wikipedia.org/wiki/Lidar).
- [3D reconstruction from multiple images](https://en.wikipedia.org/wiki/3D_reconstruction_from_multiple_images).

However there is a lack of large 3D datasets (you can find a [good one here](http://shapenet.cs.stanford.edu/) based on triangular meshes); it's especially hard  to find datasets based on point clouds (wich is the raw output from every 3D sensing device).

This dataset contains 3D point clouds generated from the original images of the MNIST dataset to bring a familiar introduction to 3D to people used to work with 2D datasets (images).

# Content
## full_dataset_vectors.h5

The entire dataset stored as 4096-D vectors obtained from the voxelization (x:16, y:16, z:16) of all the 3D point clouds.

In adition to the original point clouds, it contains randomly rotated copies with noise.

The full dataset is splitted into arrays:

- X_train (10000, 4096)
- y_train (10000)
- X_test(2000, 4096)
- y_test (2000)


In the [3D_from_2D notebook](http://nbviewer.jupyter.org/github/daavoo/3DMNIST/blob/master/3D_from_2D.ipynb) you can find the code used to generate the dataset.

You can use the code in the notebook to generate a bigger 3D dataset from the original.


## train_point_clouds.h5 & test_point_clouds.h5

5000 (train),  and 1000 (test) [3D point clouds](https://en.wikipedia.org/wiki/Point_cloud) stored in [HDF5 file format](https://support.hdfgroup.org/HDF5/whatishdf5.html). The point clouds have zero mean and a maximum dimension range of 1.

Each file is divided into [HDF5 groups](https://support.hdfgroup.org/HDF5/Tutor/fileorg.html)

Each group is named as its corresponding array index in the original mnist dataset and it contains:

- "points" dataset: `x, y, z` coordinates of each 3D point in the point cloud.
- "normals" dataset: `nx, ny, nz` components of the unit normal associate to each point.
- "img" dataset: the original mnist image.
- "label" attribute: the original mnist label.

Example python code reading 2 digits and storing some of the group content in tuples:

    with h5py.File("../input/train_point_clouds.h5", "r") as hf:    
        a = hf["0"]
        b = hf["1"]    
        digit_a = (a["img"][:], a["points"][:], a.attrs["label"]) 
        digit_b = (b["img"][:], b["points"][:], b.attrs["label"]) 
 
## voxelgrid.py
Simple Python class that generates a grid of voxels from the 3D point cloud. Check kernel for use.

## plot3D.py
Module with functions to plot point clouds and voxelgrid inside jupyter notebook.
You have to run this locally due to Kaggle's notebook lack of support to rendering Iframes. [See github issue here](https://github.com/Kaggle/docker-python/issues/36)

Functions included:

- `array_to_color`
Converts 1D array to rgb values use as kwarg `color` in plot_points()

- `plot_points(xyz, colors=None, size=0.1, axis=False)`

![plot_points][2]

- `plot_voxelgrid(v_grid, cmap="Oranges", axis=False)`

![plot_vg][3]

# Acknowledgements

- Website of the [original MNIST dataset](http://yann.lecun.com/exdb/mnist/)


# Have fun!
  [2]: https://raw.githubusercontent.com/daavoo/3DMNIST/master/data/plot_points.gif
  [3]: https://raw.githubusercontent.com/daavoo/3DMNIST/master/data/plot_vg.gif
