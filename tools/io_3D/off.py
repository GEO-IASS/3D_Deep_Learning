#       HAKUNA MATATA

import numpy as np 
import pandas as pd

def read_off(filename):
    
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