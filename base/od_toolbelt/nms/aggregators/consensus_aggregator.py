import numpy as np
from typing import List, Any
from nptyping import NDArray

from od_toolbelt.nms.aggregators import Aggregator
from od_toolbelt import BoundingBoxArray, concatenate


class ConsensusAggregator(Aggregator):
    def transform(
        self, bounding_box_arrays: List[BoundingBoxArray]
    ) -> NDArray((Any, 2, 2), np.float64):
        # TODO: Carry confidences and labels through
        assert isinstance(bounding_box_arrays, list)

        new_bounding_box_arrays = []
        for bba in bounding_box_arrays:
            assert isinstance(bba, BoundingBoxArray)
            new_bounding_box_arrays.append(self.suppressor.transform(bba))
        concatenated_bounding_box_array = concatenate(new_bounding_box_arrays)

        evaluated_pairs = set()
        overlapping_pairs = set()
        for i, bid1 in enumerate(concatenated_bounding_box_array.bounding_box_ids):
            for j, bid2 in enumerate(concatenated_bounding_box_array.bounding_box_ids):
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
                    concatenated_bounding_box_array.lookup_box(int(bid1)),
                    concatenated_bounding_box_array.lookup_box(int(bid2)),
                )

                if overlap:
                    overlapping_pairs.add(sorted_bid_pair)
                evaluated_pairs.add(sorted_bid_pair)

        flattened_overlapping_pairs = []
        for pair in overlapping_pairs:
            for item in pair:
                flattened_overlapping_pairs.append(item)

        box_groups = [
            set((int(bid),))
            for bid in concatenated_bounding_box_array.bounding_box_ids
            if bid not in flattened_overlapping_pairs
        ]
        # Checking that singleton boxes actually don't overlap
        for bg1 in box_groups:
            for bg2 in box_groups:
                bid1 = list(bg1)[0]
                bid2 = list(bg2)[0]
                if bid1 == bid2:
                    continue
                assert not self.metric.overlap(
                    concatenated_bounding_box_array[np.array((bid1,))].bounding_boxes[
                        0, :, :
                    ],
                    concatenated_bounding_box_array[np.array((bid2,))].bounding_boxes[
                        0, :, :
                    ],
                )

        assert len(overlapping_pairs) > 0
        for pair in overlapping_pairs:
            for i, bg in enumerate(box_groups):
                if pair[0] in bg or bg in box_groups[i]:
                    box_groups[i].add(pair[0])
                    box_groups[i].add(pair[1])
                else:
                    box_groups.append(set(pair))

        box_groups = set([tuple(bg) for bg in box_groups])
        aggregated_box_groups = np.zeros((len(list(box_groups)), 2, 2), np.float64)

        for i, bg in enumerate(box_groups):
            bg = np.asarray(
                [concatenated_bounding_box_array.lookup_box(int(bid)) for bid in bg]
            )
            aggregated_box_groups[i, :, :] = np.mean(bg, axis=0)

        return aggregated_box_groups
