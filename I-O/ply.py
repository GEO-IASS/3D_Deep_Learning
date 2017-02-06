#       HAKUNA MATATA

import sys
import numpy as np
import pandas as pd
from collections import defaultdict

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
        
        if fmt not in valid_formats.keys():
            raise ValueError('A valid format could not be found')
                
        line = []
        comments = []
        obj_info = []
        names = []
        sizes = []
        dtypes = defaultdict(list)
        
        # we have already read 2 lines
        count = 2
        # use n to track the number of elements and build the dtypes
        n = -1
        
        while b'end_header' not in line and line != b'':
            line = ply.readline()
            if b'comment' in line:
                comments.append(line.decode())
            elif b'obj_info' in line:
                obj_info.append(line.decode())
            elif b'element' in line:
                n += 1
                line = line.split()
                name = line[1].decode()
                if name == "vertex":
                    name = "points"
                elif name == "face":
                    name = "mesh"
                size = int(line[2])
                names.append(name)
                sizes.append(size)
            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:
                    mesh_names = ['n_points', 'v1', 'v2', 'v3']
                    # the first number has different dtype than the list
                    dtypes[n].append((mesh_names[0], ext + ply_dtypes[line[2]]))
                    # rest of the numbers have the same dtype
                    dt = ext + ply_dtypes[line[3]]
                    for j in range(1, 4):
                        dtypes[n].append((mesh_names[j], dt))
                # regular elements
                else:
                    dtypes[n].append((line[2].decode(), ext + ply_dtypes[line[1]]))
            count += 1
        
        # for ascii
        sizes.insert(0, count)
        # for bin
        end_header = ply.tell()

    data = {'comments': comments, 'obj_info': obj_info}

    if fmt == 'ascii':
        top = 0
        bottom = 0
        for i in range(len(names)):
            top += sizes[i]
            bottom = 0
            # if there is more than 1 element adjust the bottom padding
            if len(sizes) > i + 2:
                bottom = sizes[i + 2]
                
            data[names[i]] = pd.DataFrame(np.genfromtxt(filename,
                            skip_header=top, skip_footer=bottom, dtype=dtypes[i]))

    else:
        with open(filename, 'rb') as ply:
            ply.seek(end_header)
            for i in range(len(names)):
                data[names[i]] = pd.DataFrame(np.fromfile(ply, dtype=dtypes[i],
                                                            count=sizes[i + 1]))


    return data


def write_ply(filename, points=None, mesh=None, comments=None, obj_info=None,
                                                        as_text=False, **kwargs):
    """

    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: DataFrame
    mesh: DataFrame
    comments: list[str] or str
    obj_info: list[str] or str
    as_text: boolean
        Set the write mode of the file. Default: binary
    kwargs: DataFrame
        Custom ply elements apart from the common ones

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

        if comments is not None:
            header.extend(comments)
        if obj_info is not None:
            header.extend(obj_info)
        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            header.extend(describe_element('face', mesh))
        for key in kwargs:
            header.extend(describe_element(key, kwargs[key]))

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
        for key in kwargs:
            kwargs[key].to_csv(filename, sep=" ", index=False, header=False, mode='a',
                                                                encoding='ascii')
    else:
        # open in binary/append to use tofile
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)
            for key in kwargs:
                kwargs[key].to_records(index=False).tofile(ply)
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
