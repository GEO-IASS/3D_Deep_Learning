import numpy as np
import pandas as pd 

def voxelgrid(points, x_y_z=[1,1,1]):
    xyzmin = np.min(points, axis=0) - 1e-5
    xyzmax = np.max(points, axis=0) + 1e-5

    # adjust to obtain a  minimum bounding box with all sides of equal lenght 
    diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
    xyzmin = xyzmin - diff / 2
    xyzmax = xyzmax + diff / 2 

    # segment each axis according to number of voxels
    segments = [np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1)) for i in range(3)]

    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based
    x = np.searchsorted(segments[0], points[:,0]) - 1
    y = np.searchsorted(segments[1], points[:,1]) - 1
    y = np.searchsorted(segments[2], points[:,2]) - 1

    n_x, n_y, n_z = x_y_z
    # i = ((y * n_x) + x) + (z * (n_x * n_y))
    voxelgrid = ((y * n_x) + x) + (z * (n_x * n_y))

    return voxelgrid

def get_centroids(points, voxelgrid)
    st = pd.DataFrame(voxelgrid, columns=["voxel_n"])
    for n, i in enumerate(["x", "y", "z"]):
        st[i] = points[:, n]
    return st.groupby("voxel_n").mean().values
