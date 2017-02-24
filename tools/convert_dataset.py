
import os

import numpy as np

from .feature_vectors import (
    binary_vector,
    density_vector,
    truncated_distance_function
)

from .io_3D import FROM 
from .mesh_to_point_cloud import get_vertices, mesh_sampling
from .voxelgrid import voxelgrid

def dataset_to_feature_vectors(dataset_path, output_path, n_voxelgrid,
                                mode="binary",
                                n_sampling=None,
                                out_type=np.float32,
                                size_voxelgrid=None
                                ):

    try:
        os.mkdir(output_path)
    except FileExistsError:
        print("{} already exist".format(output_path))

    # train / val / test split
    for split_folder in os.listdir(dataset_path):

        try:
            new_dir = "{}\\{}".format(output_path, split_folder)
            os.mkdir(new_dir)
        except FileExistsError:
            print("{} already exist".format(new_dir))

        for class_folder in os.listdir("{}\\{}".format(dataset_path, split_folder)):
            print("CONVERTING FOLDER: {}\\{}".format(split_folder, class_folder))

            try:
                new_dir = "{}\\{}\\{}".format(output_path, split_folder, class_folder)
                os.mkdir(new_dir)
            except FileExistsError:
                print("{} already exist".format(new_dir))

            for file_3D in os.listdir("{}\\{}\\{}".format(dataset_path, split_folder, class_folder)):
                
                fname = "{}\\{}\\{}\\{}".format(dataset_path, split_folder, class_folder, file_3D)

                ext = file_3D.split(".")[-1].upper()
                data_3D = FROM[ext](fname)

                feature_vector = data_3D_to_feature_vector(data_3D, n_voxelgrid, size_voxelgrid, n_sampling, mode)

                new_file = "{}\\{}\\{}\\{}".format(output_path, split_folder, class_folder, file_3D.replace(ext, ".npy"))

                np.save(new_file, feature_vector.astype(out_type))


def data_3D_to_feature_vector(data_3D, n_voxelgrid, size_voxelgrid=None, n_sampling=None, mode="binary"):

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
    
    return feature_vector


