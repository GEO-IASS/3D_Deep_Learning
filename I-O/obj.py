#       HAKUNA MATATA

import re
import numpy as np

def read_obj(filename):
    """ Reads and obj file and return the elements as pandas Dataframes.

    """
    v = []
    f = []
    
    with open(filename) as obj:
        for line in obj:                
            if line.startswith('v '):
                v.append(line.strip()[1:].split())
                
            elif line.startswith('f'):
                f.append(line.strip()[2:])
    
    f = [re.split(r'\D+', x) for x in f]
    
    return np.array(v), np.array(f)
            

def write_obj(filename, points, mesh=None):
    """
    """     

    if not filename.endswith('obj'):
        filename += '.obj'
    
    points = points.astype(str)
    new_points = np.empty((points.shape[0], points.shape[1]+1), dtype=points.dtype)
    new_points[:,0] = "v "
    new_points[:,1:] = points

    with open(filename, "ab") as obj:
        np.savetxt(obj, new_points, fmt="%s")

        if mesh is not None:
            mesh = mesh.astype(str)
            new_mesh = np.empty((mesh.shape[0], mesh.shape[1]+1), dtype=mesh.dtype)
            new_mesh[:,0] = "f "
            new_mesh[:,1:] = mesh
            np.savetxt(obj, new_mesh, fmt="%s")
                                      
    return True
            

            
                
