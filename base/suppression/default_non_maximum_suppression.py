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
    ):
        self.metric = metric
        self.selector = selector
        self.metric_threshold = metric_threshold

    def transform(
        self,
        bounding_boxes: NDArray[(Any, 2, 2), np.float64]
    ):
        bounding_box_ids = np.arange(0, bounding_boxes.shape[0])
        bounding_box_ids_cp_iter = itertools.product(bounding_box_ids, bounding_box_ids)

        ufs = []
        bounding_box_ids_cp = []
        for bids in bounding_box_ids_cp_iter:
            ufs.append(self.metric.compute(bounding_boxes[bids[0]], bounding_boxes[bids[1]]) > self.metric_threshold)
            bounding_box_ids_cp.append(bids)

        unique_bids = set()
        unique_ufs_bids = set()
        bounding_box_ids_cp_ufs = []
        for bids, flag in zip(bounding_box_ids_cp, ufs):
            unique_bids.add(bids[0])
            if flag:
                unique_ufs_bids.add(bids[0])
                bounding_box_ids_cp_ufs.append(bids)

        # Boxes here have zero overlaps. So, they are automatically selected
        selected_bids = list(unique_bids.difference(unique_ufs_bids))

        evaluated_bids = set()
        for bid in unique_ufs_bids:
            # Find overlapping boxes
            complementary_bids = [bids[1] for bids in bounding_box_ids_cp_ufs if bids[0] == bid]
            cbid_ixs_to_remove = []
            for ix, cbid in enumerate(complementary_bids):
                if cbid in evaluated_bids or bid == cbid:
                    # If we've already seen this combination, then remove it from consideration
                    cbid_ixs_to_remove.append(ix)
                evaluated_bids.add(cbid)
                evaluated_bids.add(bid)

            complementary_bids = [cbid for ix, cbid in enumerate(complementary_bids) if ix not in cbid_ixs_to_remove]
            if len(complementary_bids) > 0:
                selected_bid = self.selector.select(complementary_bids)
                selected_bids.append(selected_bid)

        selected_bids = list(set(selected_bids))
        return selected_bids
