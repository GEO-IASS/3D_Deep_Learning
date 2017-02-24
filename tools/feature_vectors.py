import numpy as np
from scipy.spatial.distance import cdist

def binary_vector(voxelgrid, x_y_z):
    n_x, n_y, n_z = x_y_z
    vector = np.zeros(n_x * n_y * n_z)
    vector[np.unique(voxelgrid)] = 1
    return vector.reshape(x_y_z)

def density_vector(voxelgrid, x_y_z):
    n_x, n_y, n_z = x_y_z
    vector = np.zeros(n_x * n_y * n_z)
    count = np.bincount(voxelgrid)
    vector[:len(count)] = count
    vector /= len(voxelgrid)
    return vector.reshape(x_y_z)

def truncated_distance_function(points, voxelgrid_centers, x_y_z, voxelgrid_sizes):
    truncation = np.linalg.norm(voxelgrid_sizes)
    vector = cdist(voxelgrid_centers, points).min(1)
    return vector.reshape(x_y_z)

def plot_feature_vector(feature_vector, cmap="Oranges"):
    fig, axes= plt.subplots(int(np.ceil(feature_vector.shape[0] / 4)), 4, figsize=(8,20))
    plt.tight_layout()
    for i, ax in enumerate(axes.flat):
        if i >= len(feature_vector):
            break
        im = ax.imshow(feature_vector[:, :, i], cmap=cmap, interpolation="none")
        ax.set_title("Level " + str(i))