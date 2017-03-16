import re
import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.io import loadmat
from scipy.spatial import cKDTree

#============================ 3D LOADERS =====================================#

def read_mat(filename, 
             points_name="points", 
             mesh_name="mesh", 
             points_columns="points_columns",
             mesh_columns="mesh_columns",
             points_dtypes="points_dtypes",
             mesh_dtypes="mesh_dtypes"):
    """ Extract 3D information from .mat file.
    Parameters
    ----------
    filename: str
        Path tho the filename

    Returns
    -------
    data: dict
        If possible, elements as pandas DataFrames else input format
    """

    data = {}

    mat = loadmat(filename)
    
    if points_columns in mat:
        columns = [mat[points_columns][i].strip() for i in range(len(mat[points_columns]))]
        data["points"] = pd.DataFrame(mat[points_name], columns=columns)
    else:
        columns = ["x", "y", "z"]
        for i in range(mat[points_name].shape[1] - 3):
            columns.append("sf{}".format(i))
    
    data["points"] = pd.DataFrame(mat[points_name], columns=columns)  
    
    if points_dtypes in mat:
        for i in range(len(mat[points_dtypes])):
            data["points"][columns[i]] = data["points"][columns[i]].astype(mat[points_dtypes][i].strip())          
    
    if mesh_name in mat:
        if mesh_columns in mat:
            columns= [mat[mesh_columns][i].strip() for i in range(len(mat[mesh_columns]))]
            data["mesh"] = pd.DataFrame(mat[mesh_name], columns=columns)
        else:
            columns = ["v1", "v2", "v3"]
            for i in range(mat[mesh_name].shape[1] - 3):
                columns.append("sf{}".format(i))
        data["mesh"] = pd.DataFrame(mat[mesh_name], columns=columns)

        if mesh_dtypes in mat:
            for i in range(len(mat[mesh_dtypes])):
                data["mesh"][columns[i]] = data["mesh"][columns[i]].astype(mat[mesh_dtypes][i].strip())     

            
    return data
    
def read_npy(filename, columns=["x", "y", "z"]):
    """ Extract 3D information from .npy file. 

    Parameters
    ----------
    filename: str
        Path tho the filename
    columns: list of str, optional
        Default: ["x", "y", "z"]
        Name of columns in the array. Used to create pandas DataFrame.

    Returns
    -------
    data: dict
        Points as pandas DataFrame.
    """

    data = {}
    points = np.load(filename)[:,:3]
    data["points"] = pd.DataFrame(points, columns=columns)
    return data
    
def read_npz(filename, points_name="points", mesh_name="mesh"):
    """ Extract 3D information from .npz file. 
    
    Parameters
    ----------
    filename: str
        Path tho the filename
        
    Returns
    -------
    data: dict
        Elements as pandas DataFrames.
    """

    data = {}
    with np.load(filename) as npz:
        data["points"] = pd.DataFrame(npz[points_name])
        if mesh_name in npz:
            data["mesh"] = pd.DataFrame(npz[mesh_name])
    return data
    
def read_obj(filename):
    """ Extract 3D information from .obj file.

    Parameters
    ----------
    filename: str
        Path to the obj file.

    Returns
    -------
     data: dict
        If possible, elements as pandas DataFrames else input format

    """
    v = []
    vn = []
    f = []
    
    with open(filename) as obj:
        for line in obj:
            if line.startswith('v '):
                v.append(line.strip()[1:].split())
                
            elif line.startswith('vn'):
                vn.append(line.strip()[2:].split())
                
            elif line.startswith('f'):
                f.append(line.strip()[2:])
                
                
    points = pd.DataFrame(v, dtype='f4', columns=['x', 'y', 'z'])
    vn = pd.DataFrame(vn, dtype='f4', columns=['nx', 'ny', 'nz'])
    
    if len(f) > 0 and "//" in f[0]:
        mesh_columns = ['v1', 'vn1', 'v2', 'vn2', 'v3', 'vn3']
    elif len(vn) > 0:
        mesh_columns = ['v1', 'vt1', 'vn1', 'v2', 'vt2', 'vn2', 'v3', 'vt3', 'vn3']
    else:
        mesh_columns = ['v1', 'vt1', 'v2', 'vt2', 'v3', 'vt3']
    
    f = [re.split(r'\D+', x) for x in f]
    
    mesh = pd.DataFrame(f, dtype='i4', columns=mesh_columns)
    
    data = {'points': points, 'mesh': mesh, "normals":vn}
    
    return data

def read_off(filename):
    """ Extract 3D information from .off file.

    Parameters
    ----------
    filename: str
        Path to the off file.

    Returns
    -------
     data: dict
        If possible, elements as pandas DataFrames else input format
        
    """
    with open(filename) as off:
        line = off.readline()
        if "OFF\n" not in line:
            numbers = line.split("OFF")[1].split()
            skip = 1
        else:
            numbers = off.readline().strip().split()
            skip = 2

    n_points = int(numbers[0])
    n_faces = int(numbers[1])

    data = {}

    data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                            skiprows=skip, skip_footer=n_faces,
                            names=["x", "y", "z"])

    data["mesh"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                        skiprows=(skip + n_points), usecols=[1,2,3],
                        names=["v1", "v2", "v3"])
    return data
    
def read_ply(filename):
    """ Extract 3D information from .ply file
    
    Parameters
    ----------
    filename: str
        Path tho the filename
        
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """
    ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
    ])
    
    valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}

    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start whith the word ply')
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]
                
        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None        
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size
                
            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:
                    mesh_names = ['n_points', 'v1', 'v2', 'v3']
                    
                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append((mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]
                    
                    for j in range(1, 4):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append((line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append((line[2].decode(), ext + ply_dtypes[line[1]]))
            count += 1
        
        # for bin
        end_header = ply.tell()

    data = {}

    if fmt == 'ascii':
        top = count
        bottom = 0 if mesh_size is None else mesh_size 

        names = [x[0] for x in dtypes["vertex"]]

        data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python", skiprows=top, skip_footer=bottom, usecols=names, names=names) 

        for n, col in enumerate(data["points"].columns):
            data["points"][col] = data["points"][col].astype(dtypes["vertex"][n][1])

        if mesh_size is not None:
            top = count + points_size

            names = [x[0] for x in dtypes["face"]][1:]
            usecols = [1,2,3]

            data["mesh"] = pd.read_csv(filename, sep=" ", header=None, engine="python", skiprows=top, usecols=usecols, names=names)

            for n, col in enumerate(data["mesh"].columns):
                data["mesh"][col] = data["mesh"][col].astype(dtypes["face"][n+1][1])    
            
    else:
        with open(filename, 'rb') as ply:
            ply.seek(end_header)
            data["points"] = pd.DataFrame(np.fromfile(ply, dtype=dtypes["vertex"], count=points_size))
            if mesh_size is not None:
                data["mesh"]  = pd.DataFrame(np.fromfile(ply, dtype=dtypes["face"], count=mesh_size))
                data["mesh"].drop('n_points', axis=1, inplace=True)
    
    return data
 
#
# MESH TO POINT CLOUD #########################################################
#

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
    
def triangle_area_multi(v1, v2, v3):
    """ Compute the area of given triangles.

    Notes
    -----
    v1[i], v2[i], v3[i] represent the ith triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1,
                                         v3 - v1), axis=1)
    
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
    
#
# POINT CLOUD TO FEATURE VECTOR ###############################################
#

def voxelgrid(points, x_y_z):
    """ Build a voxelgrid and compute the corresponding index for each point.

    Parameters
    ----------
    points: (N,3) ndarray
        The point cloud from wich we want to construct the VoxelGrid.
        Where N is the number of points in the point cloud and the second
        dimension represents the x, y and z coordinates of each point.
        
    x_y_z: array-like 
        Number of voxels along x, y and z axis. Example:
            x_y_z=[2,2,2] results in a 2x2x2 voxelgrid

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
    

    Examples
    --------

    Using n:

    >>>  points = np.array([[0.,0.,0.], [1.,1.,1.]])
    >>> voxelgrid_indices, centers, dimensions = voxelgrid(points, x_y_z=[2,2,2])
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
        
    """

    xyzmin = points.min(0)
    xyzmax = points.max(0) 

    
    # adjust to obtain all sides of equal lenght 
    margins = max(points.ptp(0)) - (points.ptp(0))
    xyzmin -= margins / 2
    xyzmax += margins / 2 
        
        
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

    return voxelgrid_indices, centers, dimensions

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
    return vector.reshape(x_y_z).astype(np.uint8)

def density_vector(voxelgrid, x_y_z):
    """ Number of points per voxel divided by total number of points
    """
    n_x, n_y, n_z = x_y_z
    vector = np.zeros(n_x * n_y * n_z)
    count = np.bincount(voxelgrid)
    vector[:len(count)] = count
    vector /= len(voxelgrid)
    return vector.reshape(x_y_z).astype(np.float16)

def truncated_distance_function(points, voxelgrid_centers, x_y_z, voxelgrid_sizes):
    """ Distance from voxel's center to closest surface point. Truncated and normalized.
    """
    truncation = np.linalg.norm(voxelgrid_sizes) * 2
    kdt = cKDTree(points)
    dist, i =  kdt.query(voxelgrid_centers, n_jobs=-1)
    dist /= dist.max()
    dist[dist > truncation] = 1
    vector = 1 - dist
    return vector.reshape(x_y_z).astype(np.float16)

#
# LOADER ######################################################################
#

FROM = {
"MAT": read_mat,
"NPY": read_npy,
"NPZ": read_npz,
"OBJ": read_obj,
"PLY": read_ply,
"OFF": read_off
}

def load_3D(path, 
            n_sampling=None,
            voxelize=True,
            voxel_mode="binary",
            target_size=(30,30,30)):
    """Loads 3D data into numpy array, voxelizing it.

    Parameters
    ----------
    path : srt
        Path to 3D file.
    n_sampling : int
        Number of points to be sampled in case the readed 3D data contains a mesh.
    voxelize : bool, optional (Default True)
        Indicates wheter the 3D data will be converted into voxelgrid or not.
    voxel_mode : {"binary", "density", "tdf"}, optional (Default "binary")
        The type of feature vector that will be generated from the voxelgrid.
        binary : uint8
            0 for unnocupied voxels 1 for occupied
        density : float16
            n_points_in_voxel / n_total_points
        truncated : float16
            Value between 0 and 1 indicating the distance between the voxel's center and
            the closest point. 1 on the surface, 0 on voxels further than 2 * voxel side.            
    target_size : [int, int, int], optional (Default [30, 30, 30])
        Dimensions of voxelgrid in case voxelize is True.

    Returns
    -------
    feature_vector : ndarray
        (target_size[0], target_size[1], target_size[2])
        
    Raises
    ------
    ValueError: if 3D format is not valid.
    
    """
    
    ext = path.split(".")[-1].upper()
    
    if ext not in FROM:
        raise ValueError('Invalid file format. Valid formats are: {}'.format([x for x in FROM]))
    
    data_3D = FROM[ext](path)
    
    if "mesh" in data_3D:
        if n_sampling is None:
            n_sampling = len(data_3D["mesh"]) * 10

        v1, v2, v3 = get_vertices(data_3D["points"], data_3D["mesh"])
        point_cloud = mesh_sampling(v1, v2, v3, n_sampling)

    else:
        point_cloud = data_3D["points"][["x", "y", "z"]].values
    
    if voxelize:
        v_grid, centers, dimensions = voxelgrid(point_cloud, target_size)
        
        if voxel_mode == "binary":
            feature_vector = binary_vector(v_grid, target_size)
        elif voxel_mode == "density":
            feature_vector = density_vector(v_grid, target_size)
        elif voxel_mode == "truncated":
            feature_vector = truncated_distance_function(point_cloud, centers, target_size, dimensions)
        else:
            raise ValueError("Unvalid mode; avaliable modes are: {}".format({"binary", "density", "truncated"}))
        
        return feature_vector
        
    else:
        return point_cloud

