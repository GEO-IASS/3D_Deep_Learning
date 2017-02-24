#       HAKUNA MATATA

import numpy as np 
import pandas as pd

def read_off(filename):
    
    with open(filename) as off:
        line = off.readline()
        if "OFF\n" not in line:
            print("Error in format for file {}, trying to fix it".format(filename))
            numbers = line.split("OFF")[1].split()
        else:
            numbers = off.readline().strip().split()

    n_points = int(numbers[0])
    n_faces = int(numbers[1])

    data = {}

    data["points"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                            skiprows=2, skip_footer=n_faces,
                            names=["x", "y", "z"])

    data["mesh"] = pd.read_csv(filename, sep=" ", header=None, engine="python",
                        skiprows=(2 + n_points), usecols=[1,2,3],
                        names=["v1", "v2", "v3"])
    return data