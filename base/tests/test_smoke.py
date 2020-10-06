import numpy as np

from od_toolbelt import BoundingBoxArray
from od_toolbelt.nms.suppression import CartesianProductSuppression, SectorSuppression

from od_toolbelt.nms.metrics.iou import DefaultIntersectionOverTheUnion
from od_toolbelt.nms.selection.random_selector import RandomSelector
from .test_utils.test_setup import setup

def smoke_test_setup():
    bounding_boxes, confidences, labels = setup()
    bb = BoundingBoxArray(
        bounding_boxes=bounding_boxes, confidences=confidences, labels=labels
    )
    bb.check()

    metric = DefaultIntersectionOverTheUnion(threshold=0.35, direction="lt")
    selector = RandomSelector()
    return bounding_boxes, confidences, labels, metric, selector


def test_smoke1():
    bounding_boxes, confidences, labels, metric, selector = smoke_test_setup()
    suppressor = CartesianProductSuppression(metric=metric, selector=selector)
    suppressor.transform(bounding_boxes, confidences, labels)


def test_smoke2():
    bounding_boxes, confidences, labels, metric, selector = smoke_test_setup()
    suppressor = SectorSuppression(metric=metric, selector=selector, sector_divisions=2)
    suppressor.transform(bounding_boxes, confidences, labels)