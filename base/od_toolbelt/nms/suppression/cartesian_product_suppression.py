from __future__ import annotations  # type: ignore

import numpy as np  # type: ignore
import itertools  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, List, Tuple, Iterator, Set, Optional, Union  # type: ignore

from od_toolbelt.nms.metrics.base import Metric  # type: ignore
from od_toolbelt.nms.selection.base import Selector  # type: ignore
from od_toolbelt.nms.suppression.base import Suppressor  # type: ignore
from od_toolbelt import BoundingBoxArray  # type: ignore


class CartesianProductSuppression(Suppressor):
    """
    A suppressor that compares the cartesian product of all boxes.
    """

    def __init__(
        self,
        metric: Metric,
        selector: Selector,
    ):
        """Method that simply stores values for use in other methods.

        Args:
            metric: An instance of a "Metric" (see metrics.base.Base for documentation).
            selector: An instance of a "Selector" (see selection.base.Base for documentation).
        """
        super().__init__(metric, selector)

    def transform(
            self,
            bounding_box_array: BoundingBoxArray,
            *args,
            **kwargs
    ) -> Tuple[
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        NDArray[(Any,), np.int64],
    ]:
        """A wrapper for cp_transform."""
        return self._cp_transform(
            bounding_box_array.bounding_boxes,
            bounding_box_array.confidences,
            bounding_box_array.labels,
            *args,
            **kwargs
        )

    def _evaluate_overlap(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        bounding_box_ids: Iterator[
            Tuple[Any, Any]
        ],  # Replace with nested loop instead of CP
        symmetric: bool = False,
    ) -> Tuple[List[int], Set[int]]:
        bounding_box_ids = [x for x in bounding_box_ids]  # TEMP
        boundary_boudning_box_idsx = set([b[0] for b in bounding_box_ids])  # TEMP
        all_boudning_box_idsx = set([b[1] for b in bounding_box_ids])  # TEMP
        non_boundary_bounding_box_ids = all_boudning_box_idsx.difference(
            boundary_boudning_box_idsx
        )

        selected_bids = []
        complementary_bids = []
        evaluated_bids = set()
        no_overlap = np.full(bounding_boxes.shape[0], True, dtype=np.bool)
        last_bid = -1

        empty = True
        for bids in bounding_box_ids:
            empty = False
            if bids[0] != last_bid and len(complementary_bids) > 0:
                selected_bids.append(self.selector.select(complementary_bids))
                complementary_bids = []
            if (
                (bids[1] not in evaluated_bids)
                and (bids[0] != bids[1])
                and not self.metric.within_range(bounding_boxes[bids[0]], bounding_boxes[bids[1]])
            ):
                complementary_bids.append(bids[1])
                evaluated_bids.add(bids[0])
                evaluated_bids.add(bids[1])

                no_overlap[bids[0]] = False
                no_overlap[bids[1]] = False
            last_bid = bids[0]
        if not symmetric and len(non_boundary_bounding_box_ids) > 0:
            no_overlap[list(non_boundary_bounding_box_ids)] = False

        if not empty:
            no_overlap_boxes = np.argwhere(no_overlap).ravel().tolist()
            selected_bids.extend(no_overlap_boxes)
            evaluated_bids.update(selected_bids)

        return selected_bids, evaluated_bids

    def _cp_transform(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
        labels: Union[NDArray[(Any,), np.int64], List[Union[str, int]]],
        bounding_box_ids: Optional[NDArray[(Any,), np.int64]] = None,
        *args,
        **kwargs
    ) -> Tuple[
        NDArray[(Any, 2, 2), np.float64],
        NDArray[(Any,), np.float64],
        NDArray[(Any,), np.int64],
    ]:
        """See base class documentation for transform().

        This method is intended to be called by a wrapper or inherited.
        """
        bb = BoundingBoxArray(
            bounding_boxes=bounding_boxes,
            confidences=confidences,
            labels=labels,
            bounding_box_ids=bounding_box_ids,
        )

        # bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        bounding_box_ids_cp = itertools.product(
            bb.bounding_box_ids, bb.bounding_box_ids
        )

        selected_bids, _ = self._evaluate_overlap(
            bb.bounding_boxes, bounding_box_ids_cp
        )
        return (
            bb.bounding_boxes[selected_bids, :, :],
            bb.confidences[selected_bids],
            bb.labels[selected_bids],
        )

    def burst(
        self,
        bounding_box_burst: List[NDArray[(Any, 2, 2), np.float64]],
        confidences_burst: List[NDArray[(Any,), np.float64]],
        labels_burst: List[NDArray[(Any,), np.int64]],
        *args,
        **kwargs
    ) -> Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]:
        """See base class documentation."""
        bounding_box = np.concatenate(bounding_box_burst, axis=0)
        confidences = np.concatenate(confidences_burst, axis=0)
        indexes = self.transform(bounding_box, confidences, labels_burst)
        return bounding_box[indexes, :, :], confidences[indexes]

    def batch(
        self,
        bounding_box_batch: List[List[NDArray[(Any, 2, 2), np.float64]]],
        confidences_batch: List[List[NDArray[(Any,), np.float64]]],
        labels_batch: List[List[NDArray[(Any,), np.int64]]],
        *args,
        **kwargs
    ) -> List[Tuple[NDArray[(Any, 2, 2), np.float64], NDArray[(Any,), np.float64]]]:
        """See base class documentation."""
        return [
            self.burst(bounding_box_burst, confidences_burst, labels_burst)
            for bounding_box_burst, confidences_burst, labels_burst in zip(
                bounding_box_batch, confidences_batch, labels_batch
            )
        ]
