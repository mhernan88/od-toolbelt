import numpy as np

from nptyping import NDArray
from typing import Any

from geometry import cube, box
from od_toolbelt.nms.metrics.base import Metric


class DefaultIntersectionOverTheUnion(Metric):
    """
    A default intersection over the union measure. This takes the intersection of two boxes (the "overlap") and divides
    it by the "union" of those two boxes (the total area covered by those two boxes combined).
    """

    def __init__(self, threshold: float, direction: str):
        super().__init__(threshold, direction)

    def compute(
            self,
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64],
    ) -> float:
        """Method to compute intersection over the union.

        Args:
            bounding_box1: An bounding box to be evaluated.
            bounding_box2: An bounding box to be evaluated against bounding_box1.

        Returns:
            The intersection over the union measure.
        """
        return np.divide(
            box.intersection(bounding_box1, bounding_box2),
            box.union(bounding_box1, bounding_box2),
        )

    def compute_many(
            self,
            bounding_boxes1: NDArray[(Any, 2, 2), np.float64],
            bounding_boxes2: NDArray[(Any, 2, 2), np.float64],
    ) -> NDArray[(Any,), np.float64]:
        """

        Args:
            bounding_boxes1: The first array of bounding boxes to compare.
            bounding_boxes2: The second array of bounding boxes to compare.

        Returns:
            An array of intersection over the union measures.
        """
        return np.divide(
            cube.intersection(bounding_boxes1, bounding_boxes2),
            cube.union(bounding_boxes1, bounding_boxes2),
        )
