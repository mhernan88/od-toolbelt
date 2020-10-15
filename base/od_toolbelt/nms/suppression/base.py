# Copyright 2020 Michael Hernandez

from typing import List, Union  # type: ignore

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
    ) -> BoundingBoxArray:
        """A method that filters down the amount of bounding boxes of an image

        Args:
            bounding_box_array: The payload of all bounding_boxes, confidences, and labels.

        Returns:
            A list of the selected bounding boxes.
            A list of the selected confidences.
        """
        raise NotImplementedError

    def burst(
            self,
            bounding_box_array_burst: List[BoundingBoxArray],
            separate: bool = False
    ) -> Union[
        List[BoundingBoxArray],
        BoundingBoxArray,
    ]:
        """A method to run transform() on a burst of images.

        All images in the burst should be take of the same subject from the same angle.

        Args:
            bounding_box_array_burst: A burst of images. Each element of this list represents an image's worth of
                bounding boxes, confidences, and labels. All of the provided boxes will be included in the
                non-maximum suppression calculation together.
            separate: True if bounding_box_array list elements should each be processed separately.

        Returns:
            If 'separate' argument is True: a list of BoundingBoxArray objects with the resulting bounding boxes,
                confidences and labels. If 'separate' argument is False: a single BoundingBoxArray object with the
                resulting bounding boxes, confidences, and labels.
        """
        raise NotImplementedError

    def batch(
            self,
            bounding_box_array_batch: List[List[BoundingBoxArray]],
            separate: bool = False,
    ) -> Union[
        List[List[BoundingBoxArray]],
        List[BoundingBoxArray],
    ]:
        """A method to run burst() on a batch of different images.

        Args:
            bounding_box_array_batch: A batch of bursts of images. Each element of this list represents the
                bounding boxes, confidences, and labels for a burst of images (see documentation for burst() method).
            separate: True if bounding_box_array list elements (for burst) should each be processed separately.

        Returns:
            If 'separate' argument is True: a list of lists of BoundingBoxArray objects from the burst() method. If
                'separate' argument is False: a list of BoundingBoxArray objects from the burst() method.
        """
        raise NotImplementedError
