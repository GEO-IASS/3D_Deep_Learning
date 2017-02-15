#       HAKUNA MATATA

from scipy.io import loadmat, savemat


def read_mat(filename, points_name="points", mesh_name="mesh"):
    """ Read a .mat file and return points and mesh(if exists)

    """
    mat = loadmat(filename)
    
    if mesh_name is not None:
        return mat[points_name], mat[mesh_name]
    else: 
        return mat[points_name]
