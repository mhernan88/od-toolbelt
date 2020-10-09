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
            bounding_box: NDArray[(2, 2), np.float64]
    ) -> float:
        """Calculates the area of a bounding box.

        Args:
            bounding_box: A bounding box to be evaluated.

        Returns:
            The area of the bounding box.
        """
        return (bounding_box[1, 0] - bounding_box[0, 0]) * (bounding_box[1, 1] - bounding_box[0, 1])

    def area_many(
            self,
            bounding_boxes: NDArray[(Any, 2, 2), np.float64]
    ) -> NDArray[
        (Any,), np.float64
    ]:
        """Calculates the area of multiple bounding boxes.

        Args:
            bounding_boxes: The bounding boxes to be evaluated.

        Returns:
            An array of the areas of the bounding boxes.
        """
        # TODO: Optimize
        areas = np.zeros(bounding_boxes.shape[0])
        for i in np.arange(0, bounding_boxes.shape[0]):
            areas[i] = self.area(bounding_boxes[i, :, :])
        return areas

    @staticmethod
    def contains(
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64]
    ) -> bool:
        """Determines whether a bounding box is fully contained by another bounding box.

        Args:
            bounding_box1: A bounding box to be evaluated.
            bounding_box2: A bounding box to be evaluated against bounding_box1

        Returns:
            True if bounding_box1 fully contains bounding_box2, otherwise False.
        """
        contains_flag = \
            bounding_box1[0, 0] >= bounding_box2[0, 0] and \
            bounding_box1[0, 1] >= bounding_box2[0, 1] and \
            bounding_box1[1, 0] <= bounding_box2[1, 0] and \
            bounding_box1[1, 1] <= bounding_box2[1, 1]
        return contains_flag

    def intersection(
            self,
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64]
    ) -> float:
        """Calculates the intersection (i.e. the overlapping area) of two overlapping boxes.

        Args:
            bounding_box1: A bounding box to be evaluated.
            bounding_box2: A bounding box to be evaluated against bounding_box1.

        Returns:
            The resulting intersection value.
        """
        if (
            bounding_box2[0, 0] >= bounding_box1[1, 0] or
            bounding_box1[0, 0] >= bounding_box2[1, 0] or
            bounding_box2[0, 1] >= bounding_box1[1, 1] or
            bounding_box1[0, 1] >= bounding_box2[1, 1]
        ):
            # If boxes don't overlap, then return 0.0 intersection value.
            return 0.0

        if self.contains(bounding_box1, bounding_box2):
            # If box 1 contains box 2, then return 1.0 intersection value.
            return 1.0

        if self.contains(bounding_box2, bounding_box1):
            # If box 2 contains box 1, then return 1.0 intersection value.
            return 1.0

        common_pt1 = np.max((bounding_box1[0, :], bounding_box2[0, :]), axis=0)
        common_pt2 = np.max((bounding_box1[1, :], bounding_box2[1, :]), axis=0)
        wh = np.abs(np.subtract(common_pt2, common_pt1))
        return np.product(wh)

    def intersection_many(
            self,
            bounding_boxes1: NDArray[(Any, 2, 2), np.float64],
            bounding_boxes2: NDArray[(Any, 2, 2), np.float64],
    ) -> NDArray[
        (Any,), np.float64
    ]:
        """Calculates the intersection of multiple pairs of boxes.

        Args:
            bounding_boxes1: The first array of bounding boxes to compare.
            bounding_boxes2: The second array of bounding boxes to compare.

        Returns:
            An array of intersection values.
        """
        # TODO: Optimize
        intersections = np.zeros(bounding_boxes1.shape[0])
        for i in np.arange(0, bounding_boxes1.shape[0]):
            intersections[i] = self.intersection(bounding_boxes1[i, :, :], bounding_boxes2[i, :, :])
        return intersections

    def union(
            self,
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64]
    ) -> float:
        """Calculates the union (i.e. the overlapping and non-overlapping area) of two overlapping boxes.

        Args:
            bounding_box1: A bounding box to be evaluated.
            bounding_box2: A bounding box to be evaluated against bounding_box1.

        Returns:
            The resulting union value.
        """
        box1_area = self.area(bounding_box1)
        box2_area = self.area(bounding_box2)
        overlap_area = self.intersection(bounding_box1, bounding_box2)
        return np.abs(box1_area + box2_area - overlap_area)

    def union_many(
            self,
            bounding_boxes1: NDArray[(Any, 2, 2), np.float64],
            bounding_boxes2: NDArray[(Any, 2, 2), np.float64]
    ) -> NDArray[
        (Any,), np.float64
    ]:
        """Calculates the union of multiple pairs of boxes.

        Args:
            bounding_boxes1: The first array of bounding boxes to compare.
            bounding_boxes2: The second array of bounding boxes to compare.

        Returns:
            An array of union values.
        """
        # TODO: Optimize
        unions = np.zeros(bounding_boxes1.shape[0])
        for i in np.arange(0, bounding_boxes1.shape[0]):
            unions[i] = self.union(bounding_boxes1[i, :, :], bounding_boxes2[i, :, :])
        return unions

    def compute(
            self,
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64],
    ) -> float:
        """Computes intersection over the union for a single box.

        Args:
            bounding_box1: A bounding box to be evaluated.
            bounding_box2: A bounding box to be evaluated against bounding_box1.

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
        """Computes intersection over the union for multiple boxes.

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
