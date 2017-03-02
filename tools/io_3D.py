import re
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.io import loadmat, savemat

FROM = {
"MAT": read_mat,
"NPZ": read_npz,
"OBJ": read_obj,
"PLY": read_ply,
"OFF": read_off
}

TO = {
"NPZ": write_npz,
"OBJ": write_obj,
"PLY": write_ply
}

def read_mat(filename, points_name="points", mesh_name="mesh", 
            points_columns="points_columns",
            mesh_columns="mesh_columns",
            points_dtypes="points_dtypes",
            mesh_dtypes="mesh_dtypes"):
    """ Read a .mat file and store all possible elements in pandas DataFrame 
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

def read_npz(filename, points_name="points", mesh_name="mesh"):
    """ Read a .npz file and store all possible elements in pandas DataFrame 
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
    with np.load(filename) as npz:
        data["points"] = pd.DataFrame(npz[points_name])
        if mesh_name in npz:
            data["mesh"] = pd.DataFrame(npz[mesh_name])
    return data


def write_npz(filename,  **kwargs):  
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this

    kwargs: Elements of the pyntcloud to be saved

    Returns
    -------
    boolean
        True if no problems
    """

    for k in kwargs:
        if isinstance(kwargs[k], pd.DataFrame):
            kwargs[k] = kwargs[k].to_records(index=False)
    np.savez_compressed(filename, **kwargs)
    return True

def read_obj(filename):
    """ Reads and obj file and return the elements as pandas Dataframes.

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
    """ Reads and off  file and return the elements as pandas Dataframes.

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

def read_ply(filename):
    """ Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path tho the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

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


def write_ply(filename, points=None, mesh=None, as_text=False):
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary

    Returns
    -------
    boolean
        True if no problems

    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                                                                encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                                                                encoding='ascii')

    else:
        # open in binary/append to use tofile
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)
                
    return True
    

def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]
    
    if name == 'face':
        element.append("property list uchar int points_indices")
        
    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element
