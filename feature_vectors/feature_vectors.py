import numpy as np

def binary_vector(voxelgrid, x_y_z):
    vector = np.zeros(len(voxelgrid))
    vector[np.unique(voxelgrid)] = 1
    return vector.reshape(x_y_z[2], x_y_z[1], x_y_z[0])

def density_vector(voxelgrid, x_y_z):
    vector = np.zeros(len(voxelgrid))
    count = np.bincount(voxelgrid)
    vector[:len(count)] = count
    vector /= len(voxelgrid)
    return vector.reshape(x_y_z[2], x_y_z[1], x_y_z[0])