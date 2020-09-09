import numpy as np
from nptyping import NDArray
from typing import Any


class Metric:
    """
    A Metric is how we will measure image overlap.
    """

    def __init__(self, *args, **kwargs):
        """Any configuration variables can be passed and stored here.
        """
        pass

    def compute(
        self,
        bounding_box1: NDArray[(2, 2), np.float64],
        bounding_box2: NDArray[(2, 2), np.float64],
    ) -> float:
        """Computes the metric value.

        A computation of each element of the argument bounding_box1 and the argument bounding_box2 is performed.
        The result is a float, which is your metric value.

        Args:
            bounding_box1: The first bounding box to compare.
            bounding_box2: The second bounding box to compare.

        Returns:
            The resulting metric value.
        """
        raise NotImplementedError

    def compute_many(
        self,
        bounding_boxes1: NDArray[(Any, 2, 2), np.float64],
        bounding_boxes2: NDArray[(Any, 2, 2), np.float64],
    ) -> NDArray[(Any,), np.float64]:
        """Computes the metric values.

        Same as compute(), but multiple bounding boxes can be passed. Elementwise computation will be performed.

        Args:
            bounding_boxes1: The first array of bounding boxes to compare.
            bounding_boxes2: The second array of bounding boxes to compare.

        Returns:
            An array of the resulting values.
        """
        raise NotImplementedError
