import numpy as np

from nptyping import NDArray
from typing import Any

# from geometry.cube import intersection, union
from geometry import cube
from geometry.box import intersection, union
from metrics.base import Metric


class DefaultIntersectionOverTheUnion(Metric):
    """
    A default intersection over the union measure. This takes the intersection of two boxes (the "overlap") and divides
    it by the "union" of those two boxes (the total area covered by those two boxes combined).
    """
    def __init__(self):
        """Empty init method. No additional args, kwargs to pass here.
        """
        super().__init__()

    def compute(self, cube1: NDArray[(2, 2), np.float64], cube2: NDArray[(2, 2), np.float64]):
        """Method to compute intersection over the union.

        Args:
            cube1: An array of boxes to be evaluated.
            cube2: An array of boxes to be evaluated against cube1.

        Returns:
            An array of intersection of the union measures.
        """
        return np.divide(intersection(cube1, cube2), union(cube1, cube2))

    def compute_cube(self, cube1, cube2):
        return np.divide(cube.intersection(cube1, cube2), cube.union(cube1, cube2))
