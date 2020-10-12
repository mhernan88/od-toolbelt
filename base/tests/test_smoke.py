from od_toolbelt import BoundingBoxArray
from od_toolbelt.nms.suppression import CartesianProductSuppression, SectorSuppression

from od_toolbelt.nms.metrics.iou import DefaultIntersectionOverTheUnion
from od_toolbelt.nms.selection.random_selector import RandomSelector
from .test_utils.test_setup import setup_test_case


def smoke_test_setup():
    bounding_boxes, confidences, labels = setup_test_case()
    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes, confidences=confidences, labels=labels
    )
    bb.check()

    metric = DefaultIntersectionOverTheUnion(threshold=0.35, direction="lt")
    selector = RandomSelector()
    return bb, metric, selector


def test_smoke1():
    bb, metric, selector = smoke_test_setup()
    suppressor = CartesianProductSuppression(metric=metric, selector=selector)
    result = suppressor.transform(bb)
    result.check()


def test_smoke2():
    bb, metric, selector = smoke_test_setup()
    suppressor = SectorSuppression(metric=metric, selector=selector, sector_divisions=2)
    result = suppressor.transform(bb)
    result.check()
