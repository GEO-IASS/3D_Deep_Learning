
import os

import numpy as np

from .io_3D import FROM 
from .mesh_to_pointcloud import (
    get_vertices,
     mesh_sampling
)
from .pointcloud_to_feature_vector import (
    voxelgrid,
    binary_vector,
    density_vector,
    truncated_distance_function
)

def dataset_to_feature_vectors(input_path, output_path, 
                                n_sampling=None,
                                n_voxelgrid, size_voxelgrid=None,
                                mode="binary"):
     """ Create new version of dataset with point clouds and/or meshes transformed into feature vectors.

    Parameters
    ----------    
    input_path: str
        Expected dataset format:
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
    output_path: str
        New dataset will be:
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
    n_sampling: int, optional
        Default: None
        The number of points that will be sampled from the mesh, in case mesh exists.
        This is used to convert the mesh into pointcloud in order to compute voxelgrid
        and feature vector.
        If n_sampling is None and mesh is found, n_sampling will be assigned to number
        of vertices * 10.
        
    n_voxelgrid:  int
        The number of voxels per axis. i.e:
        n_voxelgrid = 2 results in a 2x2x2 voxelgrid
        The bounding box will be adjusted in order to have all sizes of equal length.
    
    size_voxelgrid: float, optional
        Default: None
        The desired voxel size. The number of voxels will be infered. i.e:
        size = 0.2 results in a IxJxK voxelgrid ensuring that each voxel is 0.2 x 0.2 x 0.2
        If size_voxelgrid is not None, n_voxelgrid will be ignored.
        The bounding box will be adjusted in order to make each axis divisible by size.
    
    mode: {"binary", "density", "truncated"}
        Type of feature vector to be computed from voxelgrid.
        See correspoding function for details.

    """
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    # train / val / test split
    for split_folder in os.listdir(input_path):

        try:
            new_dir = "{}\\{}".format(output_path, split_folder)
            os.mkdir(new_dir)
        except FileExistsError:
            pass

        for class_folder in os.listdir("{}\\{}".format(input_path, split_folder)):
            print("CONVERTING FOLDER: {}\\{}".format(split_folder, class_folder))

            try:
                new_dir = "{}\\{}\\{}".format(output_path, split_folder, class_folder)
                os.mkdir(new_dir)
            except FileExistsError:
                pass

            for file_3D in os.listdir("{}\\{}\\{}".format(input_path, split_folder, class_folder)):

                fname = "{}\\{}\\{}\\{}".format(input_path, split_folder, class_folder, file_3D)
                    
                ext = file_3D.split(".")[-1].upper()

                new_fname= "{}\\{}\\{}\\{}".format(output_path, split_folder, class_folder,
                                                     file_3D.replace(".{}".format(ext.lower()), ".npy"))
                if not os.path.exists(new_fname):
                    data_3D = FROM[ext](fname)

                    if "mesh" in data_3D:
                        if n_sampling is None:
                            n_sampling = len(data_3D["mesh"]) * 10

                        v1, v2, v3 = get_vertices(data_3D["points"], data_3D["mesh"])
                        point_cloud = mesh_sampling(v1, v2, v3, n_sampling)

                    else:
                        point_cloud = data_3D["points"]

                    v_grid, centers, dimensions, x_y_z = voxelgrid(point_cloud, n_voxelgrid, size_voxelgrid)

                    if mode == "binary":
                        feature_vector = binary_vector(v_grid, x_y_z)
                    elif mode == "density":
                        feature_vector = density_vector(v_grid, x_y_z)
                    elif mode == "truncated":
                        feature_vector = truncated_distance_function(point_cloud, centers, x_y_z, dimensions)
                    else:
                        raise ValueError("Unvalid mode; avaliable modes are: {}".format({"binary", "density", "truncated"}))

                    np.save(new_fname, feature_vector)

if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()