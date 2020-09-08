import numpy as np
from nptyping import NDArray
from typing import Any


class Metric:
    """
    A Metric is how we will measure image overlap. It should have an __init__ method, where the developer can
    pass whatever arguments they deem fit. It also requires a compute method, which takes in two arrays and returns
    a single array of measures as an output.
    """
    def __init__(self, *args, **kwargs):
        pass

    def compute(self, cube1: NDArray[(Any, 2, 2), np.float64], cube2: NDArray[(Any, 2, 2), np.float64]) -> NDArray[(Any,), np.float64]:
        raise NotImplementedError
