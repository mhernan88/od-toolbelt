import numpy as np
from od_toolbelt.data_structures.bounding_box import BoundingBox

def smoke_test():
    coordinates = np.array([[1.01, 2.02], [3.03, 4.04]])
    bb = BoundingBox(coordinates, 0.5, 1, 2)
