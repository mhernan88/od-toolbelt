# Copyright 2020 Michael Hernandez

from typing import Any, Tuple, List  # type: ignore

import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from od_toolbelt.nms.selection import Selector  # type: ignore
from od_toolbelt.nms.metrics import Metric  # type: ignore
from od_toolbelt import BoundingBoxArray  # type: ignore


class Suppressor:
    """
    A base suppressor class to serve as a specification for other suppressor classes.
    """

    def __init__(
            self,
            metric: Metric,
            selector: Selector,
    ):
        self.metric = metric
        self.selector = selector

    def transform(
            self,
            bounding_box_array: BoundingBoxArray,
            *args,
            **kwargs,
    ) -> Tuple[
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        NDArray[(Any,), np.int64],
    ]:
        """A method that filters down the amount of bounding boxes of an image

        Args:
            bounding_box_array: The payload of all bounding_boxes, confidences, and labels.

        Returns:
            A list of the selected bounding boxes.
            A list of the selected confidences.
        """
        raise NotImplementedError

    # def burst(
    #     self,
    #     bounding_box_burst: List[NDArray[(Any, 2, 2), np.float64]],
    #     confidences_burst: List[NDArray[(Any,), np.float64]],
    #     labels_burst: List[NDArray[(Any,), np.int64]],
    #     *args,
    #     **kwargs
    # ) -> Tuple[
    #     NDArray[(Any, 2, 2), np.float64],
    #     NDArray[(Any,), np.float64],
    #     NDArray[(Any,), np.int64],
    # ]:
    #     """A method to run transform() on a burst of images.
    #
    #     All images in the burst should be take of the same subject from the same angle.
    #
    #     Args:
    #         bounding_box_burst: A burst of images. Each element of this list represents an image's worth of
    #             bounding boxes. All of the provided boxes will be included in the non-maximum suppression calculation
    #             together.
    #         confidences_burst: A burst of images. Each element of this list represents an image's worth of
    #             confidences. Each confidence value corresponds to a specific bounding box in bounding_box_burst.
    #
    #     Returns:
    #         A list of the selected bounding boxes.
    #         A list of the selected confidences.
    #     """
    #     raise NotImplementedError
    #
    # def batch(
    #     self,
    #     bounding_box_batch: List[List[NDArray[(Any, 2, 2), np.float64]]],
    #     confidences_batch: List[List[NDArray[(Any,), np.float64]]],
    #     labels_batch: List[List[NDArray[(Any,), np.int64]]],
    #     *args,
    #     **kwargs
    # ) -> Tuple[
    #     List[NDArray[(Any, 2, 2), np.float64]],
    #     List[NDArray[(Any,), np.float64]],
    #     List[NDArray[(Any,), np.int64]],
    # ]:
    #     """A method to run burst() on a batch of different images.
    #
    #     Args:
    #         bounding_box_batch: A batch of bursts of images. Each element of this list represents the bounding boxes
    #         for a burst of images (see documentation for burst() method).
    #         confidences_batch: A batch of bursts of images. Each element of this list represents the confidences
    #         for a burst of images (see documentation for burst() method).
    #
    #     Returns:
    #         A list of bounding box outputs from the burst() method.
    #         A list of confidence outputs from the burst() method.
    #     """
    #     raise NotImplementedError
