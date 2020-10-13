import numpy as np

import od_toolbelt as od


def test_consensus_aggregator():
    boxes1 = np.array(
        (
            ((0.10, 0.10), (0.20, 0.20)),
            ((0.30, 0.30), (0.40, 0.40)),
        ),
        dtype=np.float64,
    )
    confidences1 = np.array((0.6, 0.65), dtype=np.float64)
    labels1 = np.array((1, 1), dtype=np.int64)
    bbids1 = np.array((1, 2), dtype=np.int64)

    bounding_box_array1 = od.BoundingBoxArray(
        bounding_boxes=boxes1,
        confidences=confidences1,
        labels=labels1,
        bounding_box_ids=bbids1,
    )

    boxes2 = np.array(
        (
            ((0.11, 0.11), (0.21, 0.21)),
            ((0.35, 0.35), (0.50, 0.50)),
            ((0.60, 0.60), (0.70, 0.70)),
        ),
        dtype=np.float64,
    )
    confidences2 = np.array((0.7, 0.65, 0.55), dtype=np.float64)
    labels2 = np.array((1, 1, 1), dtype=np.int64)
    bbids2 = np.array((3, 4, 5), dtype=np.int64)

    bounding_box_array2 = od.BoundingBoxArray(
        bounding_boxes=boxes2,
        confidences=confidences2,
        labels=labels2,
        bounding_box_ids=bbids2,
    )

    suppressor_metric = od.nms.metrics.DefaultIntersectionOverTheUnion(
        threshold=0.05, direction="gte"
    )
    suppressor_selector = od.nms.selection.RandomSelector()
    suppressor = od.nms.suppression.CartesianProductSuppression(
        metric=suppressor_metric, selector=suppressor_selector
    )

    aggregator_metric = od.nms.metrics.DefaultIntersectionOverTheUnion(
        threshold=0.05, direction="gte"
    )
    aggregator = od.nms.aggregators.ConsensusAggregator(
        suppressor=suppressor, metric=aggregator_metric
    )
    result = aggregator.transform([bounding_box_array1, bounding_box_array2])

    assert result.shape[0] == 3
