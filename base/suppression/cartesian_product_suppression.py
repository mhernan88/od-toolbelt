from __future__ import annotations  # type: ignore

import itertools  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, List, Tuple  # type: ignore

from metrics.base import Metric  # type: ignore
from selection.base import Selector  # type: ignore
from suppression.base import Suppressor  # type: ignore


class CartesianProductSuppression(Suppressor):
    """
    A suppressor that compares the cartesian product of all boxes.
    """

    def __init__(
        self, metric: Metric, selector: Selector, metric_threshold: float,
    ):
        """Method that simply stores values for use in other methods.

        Args:
            metric: An instance of a "Metric" (see metrics.base.Base for documentation).
            selector: An instance of a "Selector" (see selection.base.Base for documentation).
            metric_threshold: The value a given metric value needs to exceed in order to be considered overlapping
            another bounding box.
        """
        super().__init__()
        self.metric = metric
        self.selector = selector
        self.metric_threshold = metric_threshold

    def transform(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        *args,
        **kwargs
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
        """See base class documentation.
        """
        bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        bounding_box_ids_cp = itertools.product(bounding_box_ids, bounding_box_ids)

        selected_bids = []
        complementary_bids = []
        evaluated_bids = set()
        no_overlap = np.full(bounding_box_ids.shape[0], True, dtype=np.bool)
        last_bid = -1

        for bids in bounding_box_ids_cp:
            metric = self.metric.compute(
                bounding_boxes[bids[0]], bounding_boxes[bids[1]]
            )
            if bids[0] != last_bid and len(complementary_bids) > 0:
                selected_bids.append(self.selector.select(complementary_bids))
                complementary_bids = []
            if (
                (bids[1] not in evaluated_bids)
                and (bids[0] != bids[1])
                and metric > self.metric_threshold
            ):
                complementary_bids.append(bids[1])
                evaluated_bids.add(bids[0])
                evaluated_bids.add(bids[1])

                no_overlap[bids[0]] = False
                no_overlap[bids[1]] = False
            last_bid = bids[0]

        no_overlap_boxes = np.argwhere(no_overlap)
        selected_bids.extend(no_overlap_boxes.ravel().tolist())

        return bounding_boxes[selected_bids, :, :], confidences[selected_bids]

    def burst(
        self,
        bounding_box_burst: List[NDArray[(Any, 2, 2), np.float64]],
        confidences_burst: List[NDArray[(Any,), np.float64]],
        *args,
        **kwargs
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
        """See base class documentation.
        """
        bounding_box = np.concatenate(bounding_box_burst, axis=0)
        confidences = np.concatenate(confidences_burst, axis=0)
        indexes = self.transform(bounding_box, confidences)
        return bounding_box[indexes, :, :], confidences[indexes]

    def batch(
        self,
        bounding_box_batch: List[List[NDArray[(Any, 2, 2), np.float64]]],
        confidences_batch: List[List[NDArray[(Any,), np.float64]]],
        *args,
        **kwargs
    ) -> List[Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]]:
        """See base class documentation.
        """
        return [self.burst(bounding_box_burst, confidences_burst) for bounding_box_burst, confidences_burst in zip(bounding_box_batch, confidences_batch)]