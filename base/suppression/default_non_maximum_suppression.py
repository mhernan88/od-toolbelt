from __future__ import annotations  # type: ignore

import itertools  # type: ignore
import numpy as np  # type: ignore
from nptyping import NDArray  # type: ignore
from typing import Any, Tuple  # type: ignore

from metrics.base import Metric  # type: ignore
from selection.base import Selector  # type: ignore


class DefaultNonMaximumSuppression:
    def __init__(
        self,
        metric: Metric,
        selector: Selector,
        metric_threshold: float,
        confidence_threshold: float
    ):
        self.metric = metric
        self.selector = selector
        self.metric_threshold = metric_threshold
        self.confidence_threshold = confidence_threshold

    def transform(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64],
        confidences: NDArray[(Any,), np.float64],
    ):
        bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        confidence_filter = confidences > self.confidence_threshold
        bounding_boxes_filtered = bounding_boxes[confidence_filter]
        bounding_box_ids_filtered = bounding_box_ids[confidence_filter]

        bounding_boxes_cp_iter = itertools.product(bounding_boxes_filtered, bounding_boxes_filtered)
        bounding_box_ids_cp_iter = itertools.product(bounding_box_ids_filtered, bounding_box_ids_filtered)

        metrics = []

        for boxes in bounding_boxes_cp_iter:
            metrics.append(self.metric.compute(*boxes))

        bounding_box_ids_cp = []
        for bids in bounding_box_ids_cp_iter:
            bounding_box_ids_cp.append(bids)

        ufs = [True if m > self.metric_threshold else False for m in metrics]
        bounding_box_ids_cp_ufs = [box for box, x in zip(bounding_box_ids_cp, ufs) if x]

        unique_bids = list(set([bids[0] for bids in bounding_box_ids_cp]))
        unique_ufs_bids = list(set([bids[0] for bids in bounding_box_ids_cp_ufs]))

        selected_bids = []
        for bid in unique_bids:
            if bid not in unique_ufs_bids:
                # This means a box has zero overlaps. So, it automatically gets selected.
                selected_bids.append(bid)

        evaluated_bids = set()
        for bid in unique_ufs_bids:
            # Find overlapping boxes
            complementary_bids = [bids[1] for bids in bounding_box_ids_cp_ufs]
            cbid_ixs_to_remove = []
            for ix, cbid in enumerate(complementary_bids):
                if (bid, cbid) in evaluated_bids or (cbid, bid) in evaluated_bids or bid == cbid:
                    # If we've already seen this combination, then remove it from consideration
                    cbid_ixs_to_remove.append(ix)
                evaluated_bids.add((bid, cbid))

            complementary_bids = [cbid for ix, cbid in enumerate(complementary_bids) if ix not in cbid_ixs_to_remove]

            if len(complementary_bids) > 0:
                selected_bid = self.selector.select(complementary_bids)
                selected_bids.append(selected_bid)

        selected_bids = list(set(selected_bids))
        return bounding_boxes[selected_bids, :, :], confidences[selected_bids]
