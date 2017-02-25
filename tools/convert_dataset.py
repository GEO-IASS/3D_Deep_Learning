
import os

import numpy as np

from .io_3D import FROM 
from .data_3D_to_feature_vector import data_3D_to_feature_vector

def dataset_to_feature_vectors(input_path, output_path, n_voxelgrid,
                                mode="binary",
                                n_sampling=None,
                                out_type=np.float32,
                                size_voxelgrid=None
                                ):

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
                    feature_vector = data_3D_to_feature_vector(data_3D, n_voxelgrid, size_voxelgrid, n_sampling, mode)
                    np.save(new_fname, feature_vector.astype(out_type))
