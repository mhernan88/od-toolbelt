import numpy as np

from nptyping import NDArray
from typing import Any

from od_toolbelt.nms.metrics.base import Metric


class DefaultIntersectionOverTheUnion(Metric):
    """
    A default intersection over the union measure. This takes the intersection of two boxes (the "overlap") and divides
    it by the "union" of those two boxes (the total area covered by those two boxes combined).
    """

    def __init__(self, threshold: float, direction: str):
        super().__init__(threshold, direction)

    @staticmethod
    def area(
            box: NDArray[(2, 2), np.float64]
    ) -> float:
        return (box[1, 0] - box[0, 0]) * (box[1, 1] - box[0, 1])

    def area_many(
            self,
            boxes: NDArray[(Any, 2, 2), np.float64]
    ) -> NDArray[
        (Any,), np.float64
    ]:
        # TODO: Optimize
        areas = np.zeros(boxes.shape[0])
        for i in np.arange(0, boxes.shape[0]):
            areas[i] = self.area(boxes[i, :, :])
        return areas

    @staticmethod
    def contains(
            box1: NDArray[(2, 2), np.float64],
            box2: NDArray[(2, 2), np.float64]
    ) -> bool:
        contains_flag = \
            box1[0, 0] >= box2[0, 0] and \
            box1[0, 1] >= box2[0, 1] and \
            box1[1, 0] <= box2[1, 0] and \
            box1[1, 1] <= box2[1, 1]
        return contains_flag

    def intersection(
            self,
            box1: NDArray[(2, 2), np.float64],
            box2: NDArray[(2, 2), np.float64],
    ) -> float:
        if (
            box2[0, 0] >= box1[1, 0] or
            box1[0, 0] >= box2[1, 0] or
            box2[0, 1] >= box1[1, 1] or
            box1[0, 1] >= box2[1, 1]
        ):
            # If boxes don't overlap, then return 0.0 intersection value.
            return 0.0

        if self.contains(box1, box2):
            # If box 1 contains box 2, then return 1.0 intersection value.
            return 1.0

        if self.contains(box2, box1):
            # If box 2 contains box 1, then return 1.0 intersection value.
            return 1.0

        common_pt1 = np.max((box1[0, :], box2[0, :]), axis=0)
        common_pt2 = np.max((box1[1, :], box2[1, :]), axis=0)
        wh = np.abs(np.subtract(common_pt2, common_pt1))
        return np.product(wh)

    def intersection_many(
            self,
            box1: NDArray[(Any, 2, 2), np.float64],
            box2: NDArray[(Any, 2, 2), np.float64]
    ) -> NDArray[
        (Any,), np.float64
    ]:
        # TODO: Optimize
        intersections = np.zeros(box1.shape[0])
        for i in np.arange(0, box1.shape[0]):
            intersections[i] = self.intersection(box1[i, :, :], box2[i, :, :])
        return intersections

    def union(
            self,
            box1: NDArray[(2, 2), np.float64],
            box2: NDArray[(2, 2), np.float64]
    ) -> float:
        box1_area = self.area(box1)
        box2_area = self.area(box2)
        overlap_area = self.intersection(box1, box2)
        return np.abs(box1_area + box2_area - overlap_area)

    def union_many(
            self,
            box1: NDArray[(Any, 2, 2), np.float64],
            box2: NDArray[(Any, 2, 2), np.float64]
    ) -> NDArray[
        (Any,), np.float64
    ]:
        # TODO: Optimize
        unions = np.zeros(box1.shape[0])
        for i in np.arange(0, box1.shape[0]):
            unions[i] = self.union(box1[i, :, :], box2[i, :, :])
        return unions

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
        assert len(bounding_box1.shape) == 2
        assert len(bounding_box2.shape) == 2
        return np.divide(
            self.intersection(bounding_box1, bounding_box2),
            self.union(bounding_box1, bounding_box2),
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
        assert len(bounding_boxes1.shape) == 3
        assert len(bounding_boxes2.shape) == 3
        return np.divide(
            self.intersection_many(bounding_boxes1, bounding_boxes2),
            self.union_many(bounding_boxes1, bounding_boxes2),
        )
