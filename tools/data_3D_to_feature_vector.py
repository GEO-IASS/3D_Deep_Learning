import numpy as np
from scipy.spatial import cKDTree

def data_3D_to_feature_vector(data_3D, n_voxelgrid, size_voxelgrid=None, n_sampling=None, mode="binary"):
    """ Converts point cloud or mesh into feature vector.

    Parameters
    ----------
    data_3D: dict
        The point cloud (and optinally mesh) extracted from file using io_3D readers
        
    n_voxelgrid:  int
        The number of voxels per axis. i.e:
        n_voxelgrid = 2 results in a 2x2x2 voxelgrid
        The bounding box will be adjusted in order to have all sizes of equal length.
    
    size_voxelgrid: float, optional
        Default: None
        The desired voxel size. The number of voxels will be infered. i.e:
        size = 0.2 results in a IxJxK voxelgrid ensuring that each voxel is 0.2x0.2x0.2
        If size_voxelgrid is not None, n_voxelgrid will be ignored.
        The bounding box will be adjusted in order to make each axis divisible by size.
    
    n_sampling: int, optional
        Default: None
        The number of points that will be sampled from the mesh, in case mesh exists.
        This is used to convert the mesh into pointcloud in order to compute voxelgrid
        and feature vector.
        If n_sampling is None and mesh is found, n_sampling will be assigned to number
        of vertices * 10.
    
    mode: {"binary", "density", "truncated"}
        Type of feature vector to be computed from voxelgrid.
        See correspoding function for details.

    Returns
    -------
    feature_vector: ndarray
        3D array representing the feature vector. Values vary according to selected mode.
    """
    
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

def get_vertices(points, mesh):
    """ Get vertices of mesh from points

    Parameters
    ----------
    points: (N, 3) ndarray
        Contains the x,y,z coordinates of each point.
    mesh: (N, 3) ndarray
        mesh[0] represents a triangle formed by 3 vertices.
        mesh[0, i] represents the index of the ith vertice in the
        associated points array.

    Returns
    -------
    v1, v2, v3: ndarray
        (N, 3) arrays of vertices so v1[i], v2[i], v3[i] represent the ith triangle

    """
    v1 = points.values[mesh["v1"]]
    v2 = points.values[mesh["v2"]]
    v3 = points.values[mesh["v3"]]
    return v1, v2, v3

def mesh_sampling(v1, v2, v3, n):
    """ Sample n points from the mesh defined by v1, v2, v3.

    Parameters
    ----------
    v1: (N, 3) ndarray
        Contains the x,y,z coordinates of points considered as the
        first vertex of each triangle.
    v2: (N, 3) ndarray
    v3: (N, 3) ndarray
    n: int
        Number of points to be sampled
    
    Returns
    -------
    sampled_points: (n, 3)
        Points sampled from the mesh triangles

    Notes
    -----
    v1[i], v2[i], v3[i] represent the ith triangle

    """
    # use area to make bigger triangles to be more likely choosen
    areas = triangle_area_multi(v1, v2, v3)
    probabilities = areas / np.sum(areas)
    random_idx = np.random.choice(np.arange(len(areas)) ,size=n, p=probabilities)
    
    v1 = v1[random_idx]
    v2 = v2[random_idx]
    v3 = v3[random_idx]
    
    # (n, 1) the 1 is for broadcasting
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1
    
    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    
    sampled_points = (v1 * u) + (v2 * v) + ((1 - (u + v)) * v3)
    
    return sampled_points

def triangle_area_multi(v1, v2, v3):
    """ Compute the area of given triangles.

    Notes
    -----
    v1[i], v2[i], v3[i] represent the ith triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1,
                                         v3 - v1), axis=1)

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

def plot_feature_vector(feature_vector, cmap="Oranges"):
    fig, axes= plt.subplots(int(np.ceil(feature_vector.shape[0] / 4)), 4, figsize=(8,20))
    plt.tight_layout()
    for i, ax in enumerate(axes.flat):
        if i >= len(feature_vector):
            break
        im = ax.imshow(feature_vector[:, :, i], cmap=cmap, interpolation="none")
        ax.set_title("Level " + str(i))
