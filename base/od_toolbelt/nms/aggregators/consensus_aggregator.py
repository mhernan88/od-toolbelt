# Copyright 2020 Michael Hernandez

import numpy as np  # type: ignore
from math import ceil  # type: ignore
from typing import List, Any, Optional  # type: ignore
from nptyping import NDArray  # type: ignore

from od_toolbelt.nms.aggregators import Aggregator  # type: ignore
from od_toolbelt import BoundingBoxArray, concatenate  # type: ignore
from od_toolbelt.nms.suppression import Suppressor  # type: ignore
from od_toolbelt.nms.metrics import Metric  # type: ignore


class ConsensusAggregator(Aggregator):
    """
    Default consensus aggregator. Takes multiple images worth of suppressed bounding boxes and aggregates any
    overlapping boxes down to their average points, ultimately only returning
    """

    def __init__(
        self,
        suppressor: Suppressor,
        metric: Metric,
        vote_threshold: Optional[int] = None,
    ):
        super().__init__(suppressor, metric)
        self.vote_threshold = vote_threshold

    def transform(
        self, bounding_box_arrays: List[BoundingBoxArray]
    ) -> NDArray[(Any, 2, 2), np.float64]:
        # TODO: Carry confidences and labels through
        self.vote_threshold = (
            ceil(len(bounding_box_arrays) / 2)
            if self.vote_threshold is None
            else self.vote_threshold
        )
        assert isinstance(bounding_box_arrays, list)

        new_bounding_box_arrays = []
        for bba in bounding_box_arrays:
            assert isinstance(bba, BoundingBoxArray)
            new_bounding_box_arrays.append(self.suppressor.transform(bba))
        concatenated_bounding_box_array = concatenate(new_bounding_box_arrays)
        overlapping_pairs = self._find_all_pair_overlaps(
            concatenated_bounding_box_array
        )
        box_groups = self._assign_box_groups(
            concatenated_bounding_box_array, overlapping_pairs
        )

        aggregated_box_groups = np.zeros((len(list(box_groups)), 2, 2), np.float64)
        for i, bg in enumerate(box_groups):
            bg = np.asarray(
                [concatenated_bounding_box_array.lookup_box(int(bid)) for bid in bg]
            )
            aggregated_box_groups[i, :, :] = np.mean(bg, axis=0)

        return aggregated_box_groups

    def _find_all_pair_overlaps(self, bounding_box_array: BoundingBoxArray):
        evaluated_pairs = set()
        overlapping_pairs = set()
        for i, bid1 in enumerate(bounding_box_array.bounding_box_ids):
            for j, bid2 in enumerate(bounding_box_array.bounding_box_ids):
                if i == j:
                    continue
                sorted_bid_pair = [bid1, bid2]
                sorted_bid_pair.sort()
                sorted_bid_pair = tuple(sorted_bid_pair)

                if sorted_bid_pair in evaluated_pairs:
                    continue

                # Calculate whether two bounding boxes have an
                # overlap metric greater than the threshold
                # (e.g. if box is 0.95 and threshold is 0.90,
                # then the boxes overlap and store True).
                overlap = self.metric.overlap(
                    bounding_box_array.lookup_box(int(bid1)),
                    bounding_box_array.lookup_box(int(bid2)),
                )

                if overlap:
                    overlapping_pairs.add(sorted_bid_pair)
                evaluated_pairs.add(sorted_bid_pair)
        return overlapping_pairs

    def _assign_box_groups(self, bounding_box_array, overlapping_pairs, vote_threshold):
        flattened_overlapping_pairs = []
        for pair in overlapping_pairs:
            for item in pair:
                flattened_overlapping_pairs.append(item)

        box_groups = [
            set((int(bid),))
            for bid in bounding_box_array.bounding_box_ids
            if bid not in flattened_overlapping_pairs
        ]

        for pair in overlapping_pairs:
            for i, bg in enumerate(box_groups):
                if pair[0] in bg or bg in box_groups[i]:
                    box_groups[i].add(pair[0])
                    box_groups[i].add(pair[1])
                else:
                    box_groups.append(set(pair))
        box_groups = set(
            [tuple(bg) for bg in box_groups if len(tuple(bg)) > self.vote_threshold]
        )
        return box_groups
