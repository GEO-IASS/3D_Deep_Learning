
def voxelgrid(points, n=2, size=None):
    """ Build a voxelgrid and compute the corresponding index for each point.

    Parameters
    ----------
    points: (N,3) ndarray
        The point cloud from wich we want to construct the VoxelGrid.
        Where N is the number of points in the point cloud and the second
        dimension represents the x, y and z coordinates of each point.
        
    n:  int
        The number of voxels per axis. i.e:
        n = 2 results in a 2x2x2 voxelgrid
        The bounding box will be adjusted in order to have all sizes of equal length.
    
    size: float, optional
        The desired voxel size. The number of voxels will be infered. i.e:
        size = 0.2 results in a IxJxK voxelgrid ensuring that each voxel is 0.2x0.2x0.2
        If size is not None, n will be ignored.
        The bounding box will be adjusted in order to make each axis divisible by size.

    Returns
    -------
    voxelgrid_indices : ndarray
        (N,) array where the nth entry correspond to the voxelgrid index where
        the nth point (from the given points) lies.

    centers: ndarray
        (x_y_z[0] * x_y_z[1] *x_y_z[2],) array representing the centroid of each
        of the voxels in the voxelgrid.

    dimensions: array-like 
        Voxel size along x, y and z dimensions.
    
    x_y_z: array-like
        Number of voxels along x, y and z axis

    Examples
    --------

    Using n:

    >>>  points = np.array([[0.,0.,0.], [1.,1.,1.]])
    >>> voxelgrid_indices, centers, dimensions = voxelgrid(points, n=2)
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
    >>> dimensions
        [0.5, 0.5, 0.5]
    
    Using size:

    >>>  points = np.array([[0.,0.,0.], [1.,1.,1.]])
    >>> voxelgrid_indices, centers, dimensions = voxelgrid(points, size=0.4)
    >>> voxelgrid_indices
        array([ 0, 26], dtype=int64)
    >>> centers
        array([
        [ 0.1,  0.1,  0.1],
        [ 0.1,  0.1,  0.5],
        [ 0.1,  0.1,  0.9],
        [ 0.1,  0.5,  0.1],
        [ 0.1,  0.5,  0.5],
        [ 0.1,  0.5,  0.9],
        [ 0.1,  0.9,  0.1],
        [ 0.1,  0.9,  0.5],
        [ 0.1,  0.9,  0.9],
        [ 0.5,  0.1,  0.1],
        [ 0.5,  0.1,  0.5],
        [ 0.5,  0.1,  0.9],
        [ 0.5,  0.5,  0.1],
        [ 0.5,  0.5,  0.5],
        [ 0.5,  0.5,  0.9],
        [ 0.5,  0.9,  0.1],
        [ 0.5,  0.9,  0.5],
        [ 0.5,  0.9,  0.9],
        [ 0.9,  0.1,  0.1],
        [ 0.9,  0.1,  0.5],
        [ 0.9,  0.1,  0.9],
        [ 0.9,  0.5,  0.1],
        [ 0.9,  0.5,  0.5],
        [ 0.9,  0.5,  0.9],
        [ 0.9,  0.9,  0.1],
        [ 0.9,  0.9,  0.5],
        [ 0.9,  0.9,  0.9]
        ])
    >>> dimensions
        [0.4, 0.4, 0.4]

    """

    xyzmin = points.min(0)
    xyzmax = points.max(0) 

    if size is None:
        # adjust to obtain all sides of equal lenght 
        margins = max(points.ptp(0)) - (points.ptp(0))
        xyzmin -= margins / 2
        xyzmax += margins / 2 
        x_y_z = [n, n, n]
        
    else:
        # adjust to obtain sides divisible by size
        margins = (((points.ptp(0) // size) + 1) * size) - points.ptp(0)
        xyzmin -= margins / 2
        xyzmax += margins / 2
        x_y_z = ((xyzmax - xyzmin) / size).astype(int) 
        
    dimensions = []
    segments = []
    for i in range(3):
        segment, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
        segments.append(segment)
        dimensions.append(step)
    
    # -1 so index are 0-based; clip for edge cases
    x = np.clip(np.searchsorted(segments[0], points[:,0]) - 1, 0, x_y_z[0])
    y = np.clip(np.searchsorted(segments[1], points[:,1]) - 1, 0, x_y_z[1])
    z = np.clip(np.searchsorted(segments[2], points[:,2]) - 1, 0, x_y_z[2])
    
    voxelgrid_indices = np.ravel_multi_index([x,y,z], x_y_z)

    midsegments = [(segments[i][1:] + segments[i][:-1]) / 2 for i in range(3)]
    centers = cartesian(midsegments)

    return voxelgrid_indices, centers, dimensions, x_y_z

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

def binary_vector(voxelgrid, x_y_z):
    """ 0 for empty space and 1 for occuped voxels.
    """
    n_x, n_y, n_z = x_y_z
    vector = np.zeros(n_x * n_y * n_z)
    vector[np.unique(voxelgrid)] = 1
    return vector.reshape(x_y_z)

def density_vector(voxelgrid, x_y_z):
    """ Number of points per voxel divided by total number of points
    """
    n_x, n_y, n_z = x_y_z
    vector = np.zeros(n_x * n_y * n_z)
    count = np.bincount(voxelgrid)
    vector[:len(count)] = count
    vector /= len(voxelgrid)
    return vector.reshape(x_y_z)

def truncated_distance_function(points, voxelgrid_centers, x_y_z, voxelgrid_sizes):
    """ Distance from voxel's center to closest surface point. Truncated and normalized.
    """
    truncation = np.linalg.norm(voxelgrid_sizes)
    kdt = cKDTree(points)
    dist, i =  kdt.query(voxelgrid_centers, n_jobs=-1)
    dist /= dist.max()
    dist[dist > truncation] = 1
    vector = 1 - dist
    return vector.reshape(x_y_z)