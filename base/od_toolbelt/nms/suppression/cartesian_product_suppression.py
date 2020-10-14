# Copyright 2020 Michael Hernandez

import numpy as np  # type: ignore
import itertools  # type: ignore
from typing import Any, List, Tuple, Iterator, Set, Optional  # type: ignore

from od_toolbelt.nms.metrics import Metric  # type: ignore
from od_toolbelt.nms.selection import Selector  # type: ignore
from od_toolbelt.nms.suppression import Suppressor  # type: ignore
from od_toolbelt import BoundingBoxArray, concatenate  # type: ignore


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
        self, bounding_box_array: BoundingBoxArray, *args, **kwargs
    ) -> BoundingBoxArray:
        """A wrapper for cp_transform."""
        return self._cp_transform(bounding_box_array, *args, **kwargs)

    def _evaluate_overlap(
        self,
        bounding_box_array: BoundingBoxArray,
        bounding_box_ids: Iterator[
            Tuple[Any, Any]
        ],  # TODO: Replace with nested loop instead of CP
        symmetric: bool = False,
    ) -> Tuple[List[int], Set[int]]:
        """For a given set of bounding boxes, this method applies cartesian product non-maximum suppression to them.

        Args:
            bounding_box_array: The payload of all bounding_boxes, confidences, and labels.
            bounding_box_ids: A list of bounding box pairs to evaluate.
            symmetric: True if bounding_box_ids is a True cartesian product (i.e. comparing each box against every
                other box), otherwise False.

        Returns:
            selected_bids: A list of bounding_box_ids that were selected by our non-maximum suppression selector and
                metric.
            evaluated_bids: A list of bounding_box_ids that were evaluated in selection.
        """
        bounding_box_ids = [
            x for x in bounding_box_ids
        ]  # TODO: Replace with optimized version.
        boundary_boudning_box_idsx = set(
            [b[0] for b in bounding_box_ids]
        )  # TODO: Replace with optimized version.
        all_bounding_box_idsx = set(
            [b[1] for b in bounding_box_ids]
        )  # TODO: Replace with optimized version.
        non_boundary_bounding_box_ids = all_bounding_box_idsx.difference(
            boundary_boudning_box_idsx
        )

        selected_bids = []
        complementary_bids = []
        evaluated_bids = set()
        no_overlap = np.full(
            bounding_box_array.bounding_boxes.shape[0], True, dtype=np.bool
        )
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
                and self.metric.overlap(
                    bounding_box_array.lookup_box(int(bids[0])),
                    bounding_box_array.lookup_box(int(bids[1])),
                )
            ):
                complementary_bids.append(bids[1])
                evaluated_bids.add(bids[0])
                evaluated_bids.add(bids[1])

                no_overlap[bounding_box_array.bounding_box_id_to_ix(bids[0])] = False
                no_overlap[bounding_box_array.bounding_box_id_to_ix(bids[1])] = False
            last_bid = bids[0]
        if not symmetric and len(non_boundary_bounding_box_ids) > 0:
            no_overlap[list(non_boundary_bounding_box_ids)] = False

        if not empty:
            no_overlap_boxes = np.argwhere(no_overlap).ravel().tolist()
            selected_bids.extend(no_overlap_boxes)
            evaluated_bids.update(selected_bids)

        return (
            np.add(
                np.asarray(selected_bids, np.int64), np.min(list(all_bounding_box_idsx))
            ),
            np.add(
                np.asarray(list(evaluated_bids), np.int64),
                np.min(list(all_bounding_box_idsx)),
            ),
        )

    def _cp_transform(
        self, bounding_box_array: Optional[BoundingBoxArray], *args, **kwargs
    ) -> Optional[BoundingBoxArray]:
        """See base class documentation for transform().

        This method is intended to be called by a wrapper or inherited.
        """
        if bounding_box_array is None:
            return None

        bounding_box_ids_cp = itertools.product(
            bounding_box_array.bounding_box_ids, bounding_box_array.bounding_box_ids
        )

        selected_bids, _ = self._evaluate_overlap(
            bounding_box_array, bounding_box_ids_cp
        )

        return bounding_box_array[np.asarray(selected_bids, dtype=np.int64)]

    def burst(
            self,
            bounding_box_array_burst: List[BoundingBoxArray],
    ) -> BoundingBoxArray:
        """See base class documentation."""
        return self.transform(concatenate(bounding_box_array_burst))

    def batch(
        self,
        bounding_box_array_batch: List[List[BoundingBoxArray]],
    ) -> List[BoundingBoxArray]:
        """See base class documentation."""
        return [self.burst(bounding_box_array_burst)for bounding_box_array_burst in bounding_box_array_batch]
