import numpy as np
from nptyping import NDArray
from typing import Any


class Metric:
    """
    A Metric is how we will measure image overlap.
    """

    def __init__(self, threshold: float, direction: str, *args, **kwargs):
        """Stores a threshold and direction data for use in methods.

        Args:
            threshold: The value that a computed metric must surpass (whether greater
                than or less than) in order to be considered outside of acceptable bounds.
            direction: The direction (either 'lt' for less than, 'gt' for greater than,
                'lte' for less than or equals, or 'gte' for greater than or equals), by which
                the computed metric value is compared against the threshold. For example, with
                a computed metric of 0.7, a threshold of 0.5, and a direction of 'lt', the
                comparison of 0.7 < 0.5 is made, where the resulting False indicates that
                0.7 is outside of acceptable bounds.
            *args:
            **kwargs:
        """
        self.threshold = threshold
        self.direction = direction

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

    def _compare_to_threshold(self, metric):
        if self.direction == "gt":
            return metric > self.threshold
        elif self.direction == "gte":
            return metric >= self.threshold
        elif self.direction == "lt":
            return metric < self.threshold
        elif self.direction == "lte":
            return metric <= self.threshold
        else:
            raise ValueError("Invalid value provided for direction")

    def within_range(
            self,
            bounding_box1: NDArray[(2, 2), np.float64],
            bounding_box2: NDArray[(2, 2), np.float64],
    ) -> bool:
        """Calculates whether a computed metric value is within a normal range.

        The computed value of the metric (from the two bounding boxes) is compared
        against self.threshold in the direction of self.direction. See the __init__
        method for more information and an example.

        Args:
            bounding_box1: The first bounding box to compare.
            bounding_box2: The second bounding box to compare.

        Returns:
            A flag, where True indicates that the computed metric was within a normal range.
        """
        metric = self.compute(bounding_box1, bounding_box2)
        return self._compare_to_threshold(metric)

    def within_range_many(
            self,
            bounding_boxes1: NDArray[(Any, 2, 2), np.float64],
            bounding_boxes2: NDArray[(Any, 2, 2), np.float64],
    ) -> NDArray[(Any,), np.bool]:
        """

        Args:
            bounding_boxes1: The first array of bounding boxes to compare.
            bounding_boxes2: The second array of bounding boxes to compare.

        Returns:
            An array of flags, where True indicates that the computed metric was within a normal range.
        """
        metric = self.compute_many(bounding_boxes1, bounding_boxes2)
        return self._compare_to_threshold(metric)
