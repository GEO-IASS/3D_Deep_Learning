import numpy as np

def voxelgrid(points, x_y_z=[1,1,1], bb_cuboiud=True):
    """ Build a voxelgrid and compute the corresponding index for each point.

    Parameters
    ----------
    points: (N,3) ndarray
                The point cloud from wich we want to construct the VoxelGrid.
                Where N is the number of points in the point cloud and the second
                dimension represents the x, y and z coordinates of each point.
        
    x_y_z:  list
            The segments in wich each axis will be divided.
            x_y_z[0]: x axis 
            x_y_z[1]: y axis 
            x_y_z[2]: z axis

    bb_cuboid(Optional): bool
            If True(Default):   
                The bounding box of the point cloud will be adjusted
                in order to have all the dimensions of equal lenght.                
            If False:
                The bounding box is allowed to have dimensions of different sizes.

    Returns
    -------
    voxelgrid_indices : ndarray
        (N,) array where the nth entry correspond to the voxelgrid index where
        the nth point (from the given points) lies.
    centers: ndarray
        (x_y_z[0] * x_y_z[1] *x_y_z[2],) array representing the centroid of each
        of the voxels in the voxelgrid.
    sizes: list 
        Voxel size along x, y and z dimensions.

    Examples
    --------
    >>> points = np.array([[0,0,0],[1,1,1]])
    >>> voxelgrid_indices, centers, sizes = voxelgrid(points, [2,2,2])
    >>> voxelgrid_indices
    array([0, 7], dtype=int64)
    >>> centers
    array([
        [ 0.25,  0.25,  0.25],
        [ 0.25,  0.25,  0.75],
        [ 0.25,  0.75,  0.25],
        [ 0.25,  0.75,  0.75],
        [ 0.75,  0.25,  0.25],
        [ 0.75,  0.25,  0.75],
        [ 0.75,  0.75,  0.25],
        [ 0.75,  0.75,  0.75]
        ])
    >>> sizes
    [0.5, 0.5, 0.5]
    
    """
    xyzmin = np.min(points, axis=0) 
    xyzmax = np.max(points, axis=0) 

    if bb_cuboid:
        #: adjust to obtain a  minimum bounding box with all sides of equal lenght 
        diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2 

    # segment each axis according to number of voxels
    sizes =[]
    segments = []
    for i in range(3):
        segment, size = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
        segments.append(segment)
        sizes.append(size)
    
    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based; clip for edge cases
    x = np.clip(np.searchsorted(segments[0], points[:,0]) - 1, 0, x_y_z[0])
    y = np.clip(np.searchsorted(segments[1], points[:,1]) - 1, 0, x_y_z[1])
    z = np.clip(np.searchsorted(segments[2], points[:,2]) - 1, 0, x_y_z[2])
    
    voxelgrid_indices = np.ravel_multi_index([x,y,z], x_y_z)

    # compute center of each voxel
    midsegments = [(segments[i][1:] + segments[i][:-1]) / 2 for i in range(3)]
    centers = cartesian(midsegments)

    return voxelgrid_indices, centers, sizes


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]
        
    return out