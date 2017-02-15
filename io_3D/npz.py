#       HAKUNA MATATA

import numpy as np
import pandas as pd


def read_npz(filename, points_name="points", mesh_name="mesh"):
    """ Reads a .npz file extracting points and mesh data 
    Parameters
    ----------
    filename: str
        Path tho the filename
        
    """
    
    with np.load(filename) as npz:
        points = npz[points_name]

        if mesh_name is not None:
            mesh = npz[mesh_name]
            return points, mesh

    return points


def write_npz(filename,  **kwargs):
    """
    Parameters
    ----------
    filename: str
        The created file will be named with this

    kwargs: Elements to be saved

    """
    for k in kwargs:
        if isinstance(kwargs[k], pd.DataFrame):
            kwargs[k] = kwargs[k].to_records(index=False)
    
    np.savez_compressed(filename, **kwargs)
    
    return True