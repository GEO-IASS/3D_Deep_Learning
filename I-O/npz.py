#       HAKUNA MATATA

import numpy as np
import pandas as pd


def read_npz(filename, points_name="points", mesh_name="mesh"):
    """ Reads a .npz file extracting points and mesh data 
    Parameters
    ----------
    filename: str
        Path tho the filename
    Returns
    -------
    data: dict

    """
    
    data = {}
    with np.load(filename) as npz:
        for i in {"points", "mesh"}:
            try:
                data[i] =  pd.DataFrame(npz[i])
            except KeyError:
                pass
    
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