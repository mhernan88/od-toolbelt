import od_toolbelt as od
from .test_utils.setup_tests import setup_test_case_multi


def test_consensus_aggregator():
    bboxes = [od.BoundingBoxArray(*x) for x in setup_test_case_multi()]

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
    result = aggregator.transform(bboxes)

    assert result.shape[0] == 2
