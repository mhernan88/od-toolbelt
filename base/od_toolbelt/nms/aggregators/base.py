# Copyright 2020 Michael Hernandez

from typing import List  # type: ignore

from od_toolbelt.nms.suppression.base import Suppressor
from od_toolbelt.nms.metrics.base import Metric
from od_toolbelt import BoundingBoxArray


class Aggregator:
    """
    An aggregator is how one set of bounding boxes is chosen from multiple layered sets of bounding boxes.
    """

    def __init__(self, suppressor: Suppressor, metric: Metric, *args, **kwargs):
        """Any configuration variables can be passed and stored here."""
        self.suppressor = suppressor
        self.metric = metric

    def transform(
        self, bounding_box_arrays: List[BoundingBoxArray]
    ) -> BoundingBoxArray:
        """Ensembles multiple layers of boxes into a single layer.

        Args:
            bounding_box_arrays: The layers of box ids you wish to consider.

        Returns:
            A single layer of box id from the bids argument.
        """
        raise NotImplementedError
