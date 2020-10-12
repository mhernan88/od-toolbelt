from typing import List

from od_toolbelt.nms.aggregators import Aggregator
from od_toolbelt import BoundingBoxArray, concatenate


class ConsensusAggregator(Aggregator):
    def transform(self, bounding_box_arrays: List[BoundingBoxArray]):
        bounding_box_arrays = [self.suppressor.transform(bba) for bba in bounding_box_arrays]
        all_bounding_box_arrays = concatenate(bounding_box_arrays)

        overlapping_boxes = dict()
        for i, bid1 in enumerate(all_bounding_box_arrays.bounding_box_ids):
            for j, bid2 in enumerate(all_bounding_box_arrays.bounding_box_ids):
                if bid1 == bid2:
                    continue

                sorted_bid_pair = [bid1, bid2]
                sorted_bid_pair.sort()
                sorted_bid_pair = tuple(sorted_bid_pair)

                if sorted_bid_pair in overlapping_boxes.keys():
                    continue

                # Calculate whether two bounding boxes have an
                # overlap metric greater than the threshold
                # (e.g. if box is 0.95 and threshold is 0.90,
                # then the boxes overlap and store True).
                overlap = not self.metric.within_range(
                    all_bounding_box_arrays[i].lookup_box(bid1),
                    all_bounding_box_arrays[j].lookup_box(bid2),
                )
                overlapping_boxes[sorted_bid_pair] = overlap

        # TODO: Iterate over overlapping_boxes, and link multiple pairs of
        #  overlapping boxes together. Then apply aggregation to each box group.
